use std::{pin::Pin, sync::Arc, time::Duration};

use color_eyre::{
    Result,
    eyre::{WrapErr, ensure, eyre},
};
use serde::{Deserialize, Serialize};

use head_motion::node::{HEAD_MOTION_SERVICE_TOPIC, HeadMotionService};
use kinematics::joints::{
    Joints,
    body::{BodyJoints, LowerBodyJoints, UpperBodyJoints},
};
use linear_algebra::vector;
use motion_inference::{
    inference::{
        GetUpCommand, InferenceCommand, InferenceRequest, KickCommand, PolicyExecution,
        WalkCommand, joints_are_finite,
    },
    locomotion::{KickRequest, leg},
    node::{INFERENCE_SERVICE, InferenceService},
};
use ros_z::{
    Message,
    context::Context,
    parameter::NodeParametersExt,
    qos::{QosDurability, QosProfile},
    service::ServiceClient,
    time::Clock,
};
use types::{
    joint_limits::JointLimits,
    motion_command::{HeadMotion, MotionCommand},
    motor_command::MotorCommand,
    time_wrapper::TimeWrapper,
};

use crate::{
    command::RobotCommand,
    walking::{WalkingParameters, step_from_walk_command},
};

mod arms;
pub mod command;
mod inputs;
mod node;
pub mod walking;

pub const ROBOT_COMMAND_TOPIC: &str = "commands/robot_command";

#[derive(Serialize, Deserialize, Message)]
struct ArmParameters {
    arm_blend_duration: Duration,

    shoulder_pitch_scale: f32,
    shoulder_roll_degrees: f32,
    shoulder_roll_scale: f32,
    knee_lateral_offset: f32,
    elbow_degrees: f32,
    elbow_scale: f32,

    kp: f32,
    kd: f32,
}

#[derive(Serialize, Deserialize, Message)]
struct Parameters {
    arms: ArmParameters,
    walking: WalkingParameters,
    inference_timeout: Duration,
    head_motion_timeout: Duration,
    maximum_command_age: Duration,
    maximum_sensor_age: Duration,
    maximum_hardware_age: Duration,
}

impl Parameters {
    fn validate(&self) -> std::result::Result<(), String> {
        let a = &self.arms;
        let w = &self.walking;
        if [
            a.shoulder_pitch_scale,
            a.shoulder_roll_degrees,
            a.shoulder_roll_scale,
            a.knee_lateral_offset,
            a.elbow_degrees,
            a.elbow_scale,
        ]
        .into_iter()
        .any(|v| !v.is_finite())
            || [a.kp, a.kd, w.max_alignment_rate]
                .into_iter()
                .any(|v| !v.is_finite() || v < 0.0)
            || [w.hybrid_align_distance, w.deceleration_distance]
                .into_iter()
                .any(|v| !v.is_finite() || v <= 0.0)
            || [
                a.arm_blend_duration,
                self.inference_timeout,
                self.head_motion_timeout,
                self.maximum_command_age,
                self.maximum_sensor_age,
                self.maximum_hardware_age,
            ]
            .into_iter()
            .any(|v| v.is_zero())
        {
            return Err("invalid motion coefficients, gains, or durations".into());
        }
        Ok(())
    }
}

pub fn run_boxed(ctx: Arc<Context>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
    Box::pin(node::run(ctx))
}

struct MotionState {
    head_motion_client: ServiceClient<HeadMotionService>,
    inference_client: ServiceClient<InferenceService>,
    generation: u64,
    active: bool,
    last_policy: Option<PolicyExecution>,
    last_arms: TimeWrapper<UpperBodyJoints<f32>>,
}

enum MotionPlan {
    Damping,
    Prepare,
    GetUp {
        command: GetUpCommand,
    },
    Walk {
        head_motion: HeadMotion,
        command: WalkCommand,
    },
    Kick {
        head_motion: HeadMotion,
        command: KickCommand,
    },
}

impl MotionPlan {
    fn from_motion_command(
        motion_command: &MotionCommand,
        parameters: &WalkingParameters,
    ) -> Result<Self> {
        Ok(match motion_command {
            MotionCommand::Damping => Self::Damping,
            MotionCommand::Prepare => Self::Prepare,
            MotionCommand::Stand { head } => Self::Walk {
                head_motion: *head,
                command: WalkCommand::stand(),
            },
            MotionCommand::StandUp { fast } => Self::GetUp {
                command: GetUpCommand { fast: *fast },
            },
            MotionCommand::Kick {
                head,
                ball_position,
                ball_velocity,
                target_speed,
                soft,
                quick,
                kick_direction,
                strong,
            } => Self::Kick {
                head_motion: *head,
                command: KickCommand {
                    soft: *soft,
                    request: KickRequest {
                        ball_position: *ball_position,
                        ball_velocity: *ball_velocity,
                        // TODO: use timestamped odometry to compensate stale ball coordinates
                        // and kick direction for robot motion before inference.
                        direction: kick_direction.angle(),
                        target_speed: *target_speed,
                        strong: *strong,
                        quick: *quick,
                    },
                },
            },
            MotionCommand::Walk {
                head,
                path,
                orientation_mode,
                target_orientation,
                distance_to_be_aligned,
                speed,
            } => {
                let step = step_from_walk_command(
                    path,
                    *orientation_mode,
                    *target_orientation,
                    *distance_to_be_aligned,
                    *speed,
                    parameters,
                )?;

                Self::Walk {
                    head_motion: *head,
                    command: WalkCommand {
                        velocity: vector![step.forward, step.left],
                        angular_velocity: step.turn,
                    },
                }
            }
            MotionCommand::WalkWithVelocity {
                head,
                velocity,
                angular_velocity,
            } => Self::Walk {
                head_motion: *head,
                command: WalkCommand {
                    velocity: *velocity,
                    angular_velocity: *angular_velocity,
                },
            },
        })
    }
}

impl MotionState {
    fn deactivate(&mut self) {
        if self.active {
            self.generation = self.generation.saturating_add(1);
        }
        self.active = false;
        self.last_policy = None;
    }

    async fn infer(
        &mut self,
        motion_plan: MotionPlan,
        clock: &Clock,
        parameters: &Parameters,
        joint_limits: &JointLimits,
    ) -> Result<RobotCommand> {
        let (command, head_motion) = match motion_plan {
            MotionPlan::Damping => {
                self.deactivate();
                return Ok(RobotCommand::Damping);
            }
            MotionPlan::Prepare => {
                self.deactivate();
                return Ok(RobotCommand::Prepare);
            }
            MotionPlan::GetUp { command } => (InferenceCommand::GetUp(command), None),
            MotionPlan::Walk {
                command,
                head_motion,
            } => (InferenceCommand::Walk(command), Some(head_motion)),
            MotionPlan::Kick {
                command,
                head_motion,
            } => (InferenceCommand::Kick(command), Some(head_motion)),
        };
        if !self.active {
            self.generation = self.generation.saturating_add(1);
            self.active = true;
        }
        let now = clock.now();
        let request = InferenceRequest {
            generation: self.generation,
            requested_at: now,
            valid_until: now + parameters.inference_timeout,
            command,
        };
        let body = self
            .inference_client
            .call_with_timeout_async(&request, parameters.inference_timeout);
        let head = async {
            match head_motion {
                Some(head) => self
                    .head_motion_client
                    .call_with_timeout_async(&head, parameters.head_motion_timeout)
                    .await
                    .map(Some),
                None => Ok(None),
            }
        };
        let (output, head) = tokio::join!(body, head);
        let output = output??;
        let head = head?;
        ensure!(
            clock.now() < request.valid_until,
            "motion inference deadline expired before dispatch"
        );
        ensure!(
            output.execution.policy == command.policy(),
            "inference returned the wrong policy"
        );
        let joints_command = match head {
            Some(head) => {
                let legs = LowerBodyJoints::from(BodyJoints::from(*output.joints));
                let arms =
                    self.generate_walking_arm_joints(&legs, clock, &parameters.arms, joint_limits)?;
                Joints::from_head_and_body(head, BodyJoints::from_lower_and_upper(legs, arms))
            }
            None => *output.joints,
        };
        let robot_command = RobotCommand::Custom { joints_command }.clamp(joint_limits)?;
        if let RobotCommand::Custom { joints_command } = &robot_command {
            self.last_arms = TimeWrapper {
                time: now,
                inner: joints_command.upper_body_as_ref().map(|j| j.position),
            };
        }
        self.last_policy = Some(output.execution);
        Ok(robot_command)
    }
}

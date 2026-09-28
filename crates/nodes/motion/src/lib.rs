use std::{pin::Pin, sync::Arc, time::Duration};

use booster::{JointsMotorState, LowState};
use inputs::Latest;

use color_eyre::{
    Result,
    eyre::{WrapErr, ensure, eyre},
};
use serde::{Deserialize, Serialize};
use tracing::{error, warn};

use head_motion::node::{HEAD_MOTION_SERVICE_TOPIC, HeadMotionService};
use kinematics::joints::{
    Joints,
    body::{BodyJoints, LowerBodyJoints, UpperBodyJoints},
    head::HeadJoints,
};
use linear_algebra::vector;
use motion_inference::{
    config::TimingParameters,
    inference::{
        ExecutionState, GetUpCommand, InferenceRequest, KickCommand, WalkCommand, joints_are_finite,
    },
    locomotion::{KickRequest, leg},
    node::{
        EXECUTION_TOPIC, GETUP_INFERENCE_SERVICE, GetUpInferenceService, KICK_INFERENCE_SERVICE,
        KickInferenceService, TIMING_PARAMETERS_TOPIC, WALK_INFERENCE_SERVICE,
        WalkInferenceService,
    },
};
use ros_z::{
    Message,
    context::Context,
    node::Node,
    parameter::NodeParametersExt,
    pubsub::Publisher,
    qos::{QosDurability, QosProfile},
    service::ServiceClient,
    time::{Clock, Time},
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

pub mod command;
mod inputs;
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
}

pub fn run_boxed(ctx: Arc<Context>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
    Box::pin(run(ctx))
}

async fn run(ctx: Arc<Context>) -> Result<()> {
    let node: Arc<Node> = Arc::new(
        ctx.create_node("motion")
            .build()
            .await
            .wrap_err("failed to create motion node")?,
    );

    let parameters = node.bind_parameter_as::<Parameters>("motion")?;

    let inference_timing = node
        .subscriber::<TimingParameters>(TIMING_PARAMETERS_TOPIC)
        .qos(QosProfile {
            durability: QosDurability::TransientLocal,
            ..Default::default()
        })
        .build()
        .await?
        .recv()
        .await
        .wrap_err("failed to receive motion inference timing parameters")?;
    let motion_commands =
        Latest::<MotionCommand>::subscribe(&node, "behavior/motion_command", QosProfile::default())
            .await?;
    let sensors = Latest::<LowState>::subscribe(
        &node,
        "inputs/low_state",
        QosProfile {
            reliability: ros_z::qos::QosReliability::BestEffort,
            ..Default::default()
        },
    )
    .await?;
    let execution_pub = node
        .publisher::<ExecutionState>(EXECUTION_TOPIC)
        .build()
        .await?;

    let motion_emergency_stop_pub = node
        .publisher::<()>("motion/emergency_stop")
        .build()
        .await
        .wrap_err("failed to build emergency stop publisher")?;

    let robot_command_pub = node
        .publisher::<RobotCommand>(ROBOT_COMMAND_TOPIC)
        .build()
        .await
        .wrap_err("failed to build robot_command publisher")?;

    let walk_inference_client = node
        .service_client::<WalkInferenceService>(WALK_INFERENCE_SERVICE)
        .build()
        .await
        .wrap_err("failed to build walk inference service client")?;

    let kick_inference_client = node
        .service_client::<KickInferenceService>(KICK_INFERENCE_SERVICE)
        .build()
        .await
        .wrap_err("failed to build kick inference service client")?;

    let get_up_inference_client = node
        .service_client::<GetUpInferenceService>(GETUP_INFERENCE_SERVICE)
        .build()
        .await
        .wrap_err("failed to build get up inference service client")?;

    let head_motion_client = node
        .service_client::<HeadMotionService>(HEAD_MOTION_SERVICE_TOPIC)
        .build()
        .await
        .wrap_err("failed to build head motion service client")?;

    let joint_limits_sub = node
        .subscriber::<JointLimits>("joint_limits")
        .qos(QosProfile {
            durability: QosDurability::TransientLocal,
            ..Default::default()
        })
        .build()
        .await?;

    let joint_limits = joint_limits_sub
        .recv()
        .await
        .wrap_err("failed to receive joint limits")?;

    joint_limits.validate().map_err(|reason| eyre!(reason))?;

    let clock = node.clock();

    let mut motion_state = MotionState {
        head_motion_client,
        walk_inference_client,
        kick_inference_client,
        get_up_inference_client,
        motion_emergency_stop_pub,

        execution_pub,
        sensors,
        maximum_sensor_age: inference_timing.maximum_sensor_age,
        generation: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos() as u64,
        execution: None,
        arm_blend: None,
    };
    motion_state.deactivate().await?;
    let mut timer = node.create_timer(inference_timing.policy_period);

    loop {
        timer.tick().await;
        // Anchor the next cycle to this actual start, skipping missed deadlines.
        timer.reset();
        let parameters = &parameters.snapshot().typed;
        let sample = motion_commands.fresh(clock, parameters.maximum_command_age);
        let (motion_command, source_time) = match &sample {
            Ok(sample) => (&sample.received.message, sample.received.source_time),
            Err(error) => {
                warn!("Motion command unavailable, damping: {error:#}");
                if motion_state.execution.is_some() {
                    motion_state.send_emergency_stop_signal().await?;
                }
                (&MotionCommand::Damping, clock.now())
            }
        };
        let plan =
            MotionPlan::from_motion_command(motion_command, source_time, &parameters.walking);
        let execution = plan.execution();
        let requested_at = clock.now();
        let valid_until = (requested_at + parameters.inference_timeout)
            .min(source_time + parameters.maximum_command_age);
        let mut robot_command = motion_state
            .infer(
                plan,
                requested_at,
                valid_until,
                clock,
                parameters,
                &joint_limits,
            )
            .await?;

        // Head motion and transport can outlive inference. Recheck the original
        // deadline and the latest command immediately before publishing targets.
        if matches!(robot_command, RobotCommand::Custom { .. }) {
            let still_requested = motion_commands
                .fresh(clock, parameters.maximum_command_age)
                .is_ok_and(|sample| {
                    MotionPlan::from_motion_command(
                        &sample.received.message,
                        sample.received.source_time,
                        &parameters.walking,
                    )
                    .execution()
                        == execution
                });
            if clock.now() >= valid_until || !still_requested {
                motion_state.send_emergency_stop_signal().await?;
                robot_command = RobotCommand::Damping;
            }
        }
        robot_command_pub.publish(&robot_command).await?;
    }
}

struct MotionState {
    head_motion_client: ServiceClient<HeadMotionService>,
    walk_inference_client: ServiceClient<WalkInferenceService>,
    kick_inference_client: ServiceClient<KickInferenceService>,
    get_up_inference_client: ServiceClient<GetUpInferenceService>,
    motion_emergency_stop_pub: Publisher<()>,
    execution_pub: Publisher<ExecutionState>,
    sensors: Latest<LowState>,
    maximum_sensor_age: Duration,
    generation: u64,
    execution: Option<Execution>,
    arm_blend: Option<TimeWrapper<UpperBodyJoints<f32>>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Execution {
    Locomotion,
    SlowGetUp,
    FastGetUp,
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
    fn execution(&self) -> Option<Execution> {
        match self {
            Self::Damping | Self::Prepare => None,
            Self::Walk { .. } | Self::Kick { .. } => Some(Execution::Locomotion),
            Self::GetUp { command } => Some(if command.fast {
                Execution::FastGetUp
            } else {
                Execution::SlowGetUp
            }),
        }
    }

    fn from_motion_command(
        motion_command: &MotionCommand,
        source_time: Time,
        parameters: &WalkingParameters,
    ) -> Self {
        match motion_command {
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
                    reference_time: source_time,
                    request: KickRequest {
                        ball_position: *ball_position,
                        ball_velocity: *ball_velocity,
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
                );

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
        }
    }
}

impl MotionState {
    async fn deactivate(&mut self) -> Result<()> {
        if self.execution.take().is_some() {
            self.generation += 1;
        }
        self.arm_blend = None;
        self.execution_pub
            .publish(&ExecutionState {
                generation: self.generation,
                active: false,
            })
            .await?;
        Ok(())
    }

    async fn infer(
        &mut self,
        motion_plan: MotionPlan,
        requested_at: Time,
        valid_until: Time,
        clock: &Clock,
        parameters: &Parameters,
        joint_limits: &JointLimits,
    ) -> Result<RobotCommand> {
        let execution = motion_plan.execution();
        if execution.is_none() {
            self.deactivate().await?;
        } else if execution != self.execution {
            self.generation += 1;
            self.execution = execution;
            self.arm_blend = None;
        }
        if execution == Some(Execution::Locomotion) && self.arm_blend.is_none() {
            let measured = self
                .sensors
                .fresh(clock, self.maximum_sensor_age)
                .and_then(|sample| sample.received.message.serial_motor_states())
                .map(|motors| motors.positions());
            match measured {
                Ok(joints) if joints.into_iter().all(f32::is_finite) => {
                    self.arm_blend = Some(TimeWrapper {
                        time: clock.now(),
                        inner: joints.upper_body_as_ref().map(|position| *position),
                    });
                }
                _ => {
                    self.send_emergency_stop_signal().await?;
                    return Ok(RobotCommand::Damping);
                }
            }
        }
        let metadata = InferenceRequest {
            generation: self.generation,
            requested_at,
            valid_until,
            command: (),
        };
        if execution.is_some() && !metadata.is_current(clock.now()) {
            self.send_emergency_stop_signal().await?;
            return Ok(RobotCommand::Damping);
        }
        let timeout = valid_until.duration_since(clock.now());

        let robot_command = match motion_plan {
            MotionPlan::Damping => RobotCommand::Damping,
            MotionPlan::Prepare => RobotCommand::Prepare,
            MotionPlan::GetUp { command } => {
                let request = metadata.map_command(|()| command);
                let inference_result = self
                    .get_up_inference_client
                    .call_with_timeout_async(&request, timeout)
                    .await;

                match inference_result {
                    Ok(Ok(joints_command)) => RobotCommand::Custom { joints_command },
                    Ok(Err(inference_error)) => {
                        error!(
                            "GetUp Inference failed, sending RobotCommand::Damping: {inference_error}"
                        );

                        self.send_emergency_stop_signal().await?;

                        RobotCommand::Damping
                    }
                    Err(ros_z_error) => {
                        error!(
                            "Failed to call GetUp inference service, sending RobotCommand::Damping! {ros_z_error}"
                        );

                        RobotCommand::Damping
                    }
                }
            }
            MotionPlan::Walk {
                head_motion,
                command,
            } => {
                let request = metadata.map_command(|()| command);
                let inference_fut = self
                    .walk_inference_client
                    .call_with_timeout_async(&request, timeout);
                let head_motion_fut = self.head_motion_client.call_with_timeout_async(
                    &head_motion,
                    parameters.head_motion_timeout.min(timeout),
                );

                let (inference_result, head_motion_result) =
                    tokio::join!(inference_fut, head_motion_fut);

                let head = match head_motion_result {
                    Ok(Ok(head_joints)) => head_joints,
                    Ok(Err(head_motion_error)) => {
                        error!("Head motion failed, damping head joints: {head_motion_error}");

                        HeadJoints::fill(MotorCommand::damping())
                    }
                    Err(ros_z_error) => {
                        error!(
                            "Failed to call HeadMotion service, damping head motion! {ros_z_error}"
                        );

                        HeadJoints::fill(MotorCommand::damping())
                    }
                };

                let lower_body_command = match inference_result {
                    Ok(Ok(joints_command)) => LowerRobotCommand::Custom {
                        lower_body_joints_command: joints_command,
                    },
                    Ok(Err(inference_error)) => {
                        error!(
                            "Walk inference failed, sending RobotCommand::Damping: {inference_error}"
                        );

                        self.send_emergency_stop_signal().await?;

                        LowerRobotCommand::Damping
                    }
                    Err(ros_z_error) => {
                        error!(
                            "Failed to call walk inference service, sending RobotCommand::Damping! {ros_z_error}"
                        );

                        self.send_emergency_stop_signal().await?;

                        LowerRobotCommand::Damping
                    }
                };

                match lower_body_command {
                    LowerRobotCommand::Custom {
                        lower_body_joints_command,
                    } => {
                        let arms_result = self.generate_walking_arm_joints(
                            &lower_body_joints_command,
                            clock,
                            &parameters.arms,
                            joint_limits,
                        );

                        let arms = match arms_result {
                            Ok(arms) => arms,
                            Err(error) => {
                                error!(
                                    "Failed to generate arm joints, using fallback joints: {error}"
                                );

                                UpperBodyJoints::fill(MotorCommand::damping())
                            }
                        };

                        let body =
                            BodyJoints::from_lower_and_upper(lower_body_joints_command, arms);

                        RobotCommand::Custom {
                            joints_command: Joints::from_head_and_body(head, body),
                        }
                    }
                    LowerRobotCommand::Damping => RobotCommand::Damping,
                }
            }
            MotionPlan::Kick {
                head_motion,
                command,
            } => {
                let request = metadata.map_command(|()| command);
                let inference_fut = self
                    .kick_inference_client
                    .call_with_timeout_async(&request, timeout);
                let head_motion_fut = self.head_motion_client.call_with_timeout_async(
                    &head_motion,
                    parameters.head_motion_timeout.min(timeout),
                );

                let (inference_result, head_motion_result) =
                    tokio::join!(inference_fut, head_motion_fut);

                let head = match head_motion_result {
                    Ok(Ok(head_joints)) => head_joints,
                    Ok(Err(head_motion_error)) => {
                        error!("Head motion failed, damping head joints: {head_motion_error}");

                        HeadJoints::fill(MotorCommand::damping())
                    }
                    Err(ros_z_error) => {
                        error!(
                            "Failed to call HeadMotion service, damping head motion! {ros_z_error}"
                        );

                        HeadJoints::fill(MotorCommand::damping())
                    }
                };

                let lower_body_command = match inference_result {
                    Ok(Ok(joints_command)) => LowerRobotCommand::Custom {
                        lower_body_joints_command: joints_command,
                    },
                    Ok(Err(inference_error)) => {
                        error!(
                            "Walk inference failed, sending RobotCommand::Damping: {inference_error}"
                        );

                        self.send_emergency_stop_signal().await?;

                        LowerRobotCommand::Damping
                    }
                    Err(ros_z_error) => {
                        error!(
                            "Failed to call GetUp inference service, sending RobotCommand::Damping! {ros_z_error}"
                        );

                        self.send_emergency_stop_signal().await?;

                        LowerRobotCommand::Damping
                    }
                };

                match lower_body_command {
                    LowerRobotCommand::Custom {
                        lower_body_joints_command,
                    } => {
                        let arms_result = self.generate_walking_arm_joints(
                            &lower_body_joints_command,
                            clock,
                            &parameters.arms,
                            joint_limits,
                        );

                        let arms = match arms_result {
                            Ok(arms) => arms,
                            Err(error) => {
                                error!(
                                    "Failed to generate arm joints, using fallback joints: {error}"
                                );

                                UpperBodyJoints::fill(MotorCommand::damping())
                            }
                        };

                        let body =
                            BodyJoints::from_lower_and_upper(lower_body_joints_command, arms);

                        RobotCommand::Custom {
                            joints_command: Joints::from_head_and_body(head, body),
                        }
                    }
                    LowerRobotCommand::Damping => RobotCommand::Damping,
                }
            }
        };

        let robot_command = match robot_command.clamp(joint_limits) {
            Ok(robot_command) => robot_command,
            Err(error) => {
                error!("Invalid final robot command, sending RobotCommand::Damping: {error:#}");

                self.send_emergency_stop_signal().await?;

                RobotCommand::Damping
            }
        };

        if matches!(robot_command, RobotCommand::Damping) {
            self.deactivate().await?;
        }

        Ok(robot_command)
    }

    fn generate_walking_arm_joints(
        &self,
        legs: &LowerBodyJoints<MotorCommand>,
        clock: &Clock,
        parameters: &ArmParameters,
        joint_limits: &JointLimits,
    ) -> Result<UpperBodyJoints<MotorCommand>> {
        let blend = self
            .arm_blend
            .as_ref()
            .ok_or_else(|| eyre!("missing arm entry pose"))?;
        walking_arm_joints(blend, legs, clock.now(), parameters, joint_limits)
    }

    async fn send_emergency_stop_signal(&mut self) -> Result<()> {
        self.deactivate().await?;
        self.motion_emergency_stop_pub.publish(&()).await?;

        Ok(())
    }
}

#[allow(clippy::large_enum_variant)]
enum LowerRobotCommand {
    Custom {
        lower_body_joints_command: LowerBodyJoints<MotorCommand>,
    },
    Damping,
}

fn arm_blend_ratio(elapsed: Duration, duration: Duration) -> f32 {
    if duration.is_zero() {
        1.0
    } else {
        (elapsed.as_secs_f32() / duration.as_secs_f32()).clamp(0.0, 1.0)
    }
}

fn walking_arm_joints(
    blend: &TimeWrapper<UpperBodyJoints<f32>>,
    legs: &LowerBodyJoints<MotorCommand>,
    now: Time,
    parameters: &ArmParameters,
    joint_limits: &JointLimits,
) -> Result<UpperBodyJoints<MotorCommand>> {
    let elapsed = now.duration_since(blend.time);

    let legs = legs
        .clone() // I hate this and will fix it later
        .map(|motor_command| motor_command.position)
        .clamp(BodyJoints::from(joint_limits.position).into());

    let ratio = arm_blend_ratio(elapsed, parameters.arm_blend_duration);
    let mut target = Joints::fill(0.0);
    for (left, arm, leg_angles, initial, sign) in [
        (
            true,
            &mut target.left_arm,
            &legs.left_leg,
            blend.inner.left_arm,
            1.0,
        ),
        (
            false,
            &mut target.right_arm,
            &legs.right_leg,
            blend.inner.right_arm,
            -1.0,
        ),
    ] {
        let (sole, knee) = leg(leg_angles, left);
        arm.shoulder_pitch = sole.x() * parameters.shoulder_pitch_scale;
        arm.shoulder_roll = sign
            * (parameters.shoulder_roll_degrees.to_radians()
                + (sign * knee.y() - parameters.knee_lateral_offset).max(0.0)
                    * parameters.shoulder_roll_scale);
        arm.shoulder_yaw = 0.0;
        arm.elbow = sign
            * (parameters.elbow_degrees.to_radians()
                + sole.x() * parameters.shoulder_pitch_scale * parameters.elbow_scale);
        *arm = initial * (1.0 - ratio) + *arm * ratio;
    }
    // let joints = position_targets(target, parameters.kp, parameters.kd);
    let joints = target.map(|position| MotorCommand {
        position,
        kp: parameters.kp,
        kd: parameters.kd,
        ..MotorCommand::zeros()
    });
    ensure!(
        joints_are_finite(&joints),
        "non-finite generated arm joints"
    );
    Ok(UpperBodyJoints {
        left_arm: joints.left_arm,
        right_arm: joints.right_arm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn motion_startup(receive_retained_timing: bool) {
        let context = Arc::new(
            ros_z::context::ContextBuilder::default()
                .with_mode("peer")
                .disable_multicast_scouting()
                .with_connect_endpoints(std::iter::empty::<&str>())
                .with_listen_endpoints(std::iter::empty::<&str>())
                .with_parameter_layer(
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../../../etc/parameters/base"),
                )
                .build()
                .await
                .unwrap(),
        );
        let owner = context
            .create_node("motion_inference")
            .build()
            .await
            .unwrap();
        let parameters = owner
            .bind_parameter_as::<motion_inference::config::Parameters>("motion_inference")
            .unwrap();
        parameters.snapshot().typed.validate().unwrap();
        let retained = QosProfile {
            durability: QosDurability::TransientLocal,
            ..Default::default()
        };
        let timing = owner
            .publisher::<TimingParameters>(TIMING_PARAMETERS_TOPIC)
            .qos(retained)
            .build()
            .await
            .unwrap();
        let limits = owner
            .publisher::<JointLimits>("joint_limits")
            .qos(retained)
            .build()
            .await
            .unwrap();
        limits
            .publish(&JointLimits {
                position: Joints::fill([-3.0, 3.0]),
                maximum_torque: Joints::fill(100.0),
            })
            .await
            .unwrap();
        let commands = owner
            .subscriber::<RobotCommand>(ROBOT_COMMAND_TOPIC)
            .build()
            .await
            .unwrap();
        if receive_retained_timing {
            timing
                .publish(&parameters.snapshot().typed.timing)
                .await
                .unwrap();
        }

        // Run production startup: a second parameter binding would exit here
        // with AlreadyBound, before any robot command can be published.
        let mut motion = tokio::spawn(run(context.clone()));
        if !receive_retained_timing {
            tokio::select! {
                result = &mut motion => panic!("motion exited before timing was available: {result:?}"),
                _ = tokio::time::sleep(Duration::from_millis(50)) => {}
            }
            timing
                .publish(&parameters.snapshot().typed.timing)
                .await
                .unwrap();
        }
        tokio::select! {
            result = &mut motion => panic!("motion exited during startup: {result:?}"),
            command = commands.recv() => assert!(matches!(command.unwrap(), RobotCommand::Damping)),
            _ = tokio::time::sleep(Duration::from_secs(3)) => panic!("motion did not finish startup"),
        }
        motion.abort();
        assert!(motion.await.unwrap_err().is_cancelled());
        context.shutdown().unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn startup_receives_retained_inference_timing_without_rebinding_parameters() {
        motion_startup(true).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn startup_waits_for_inference_timing_without_rebinding_parameters() {
        motion_startup(false).await;
    }

    #[test]
    fn arm_blend_uses_fixed_measured_entry_pose_and_finishes() {
        let parameters = ArmParameters {
            arm_blend_duration: Duration::from_secs(1),
            shoulder_pitch_scale: 1.0,
            shoulder_roll_degrees: 10.0,
            shoulder_roll_scale: 1.0,
            knee_lateral_offset: 0.0,
            elbow_degrees: 15.0,
            elbow_scale: 1.0,
            kp: 20.0,
            kd: 1.0,
        };
        let limits = JointLimits {
            position: Joints::fill([-3.0, 3.0]),
            maximum_torque: Joints::fill(100.0),
        };
        let blend = TimeWrapper {
            time: Time::zero(),
            inner: UpperBodyJoints::fill(0.7),
        };
        let legs = LowerBodyJoints::fill(MotorCommand::zeros());
        let pose = |ms: i64| {
            walking_arm_joints(
                &blend,
                &legs,
                Time::from_nanos(ms * 1_000_000),
                &parameters,
                &limits,
            )
            .unwrap()
            .map(|j| j.position)
        };
        assert_eq!(pose(0), blend.inner);
        let final_pose = pose(1000);
        assert_eq!(pose(2000), final_pose);
        let half = pose(500);
        assert_eq!(
            half.left_arm,
            blend.inner.left_arm * 0.5 + final_pose.left_arm * 0.5
        );
        assert_eq!(
            half.right_arm,
            blend.inner.right_arm * 0.5 + final_pose.right_arm * 0.5
        );
        assert_eq!(arm_blend_ratio(Duration::ZERO, Duration::ZERO), 1.0);
    }
}

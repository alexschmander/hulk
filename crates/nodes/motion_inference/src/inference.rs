use std::{collections::HashMap, path::Path, sync::Arc};

use color_eyre::eyre::{Result, WrapErr, ensure};
use serde::{Deserialize, Serialize};

use coordinate_systems::Ground;
use kinematics::joints::Joints;
use linear_algebra::Vector2;
use ros_z::time::Time;
use types::{
    joint_limits::JointLimits, motor_command::MotorCommand,
    walking_velocity_limits::WalkingVelocityLimits,
};

use crate::{
    config::{Parameters, Policy},
    get_up::{GetUp, fast, slow},
    locomotion::{KickRequest, Locomotion, kick, walk},
    network::Network,
    observation::{SensorFrame, VelocityEstimator},
    trace::{Event, Trace},
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, ros_z::Message)]
pub struct KickCommand {
    pub soft: bool,
    pub request: KickRequest,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, ros_z::Message)]
pub struct WalkCommand {
    pub velocity: Vector2<Ground>,
    pub angular_velocity: f32,
}

impl WalkCommand {
    pub fn stand() -> Self {
        Self {
            velocity: Vector2::zeros(),
            angular_velocity: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, ros_z::Message)]
pub struct GetUpCommand {
    pub fast: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, ros_z::Message)]
pub enum InferenceCommand {
    Walk(WalkCommand),
    Kick(KickCommand),
    GetUp(GetUpCommand),
}

impl InferenceCommand {
    pub fn policy(self) -> Policy {
        match self {
            Self::Walk(_) => Policy::Walk,
            Self::Kick(KickCommand { soft: true, .. }) => Policy::SoftKick,
            Self::Kick(KickCommand { soft: false, .. }) => Policy::Kick,
            Self::GetUp(GetUpCommand { fast: false }) => Policy::SlowGetUp,
            Self::GetUp(GetUpCommand { fast: true }) => Policy::FastGetUp,
        }
    }

    fn validate(self, limits: WalkingVelocityLimits) -> Result<()> {
        match self {
            Self::Walk(WalkCommand {
                velocity,
                angular_velocity,
            }) => {
                ensure!(
                    velocity.inner.iter().all(|v| v.is_finite()) && angular_velocity.is_finite(),
                    "non-finite walking command"
                );
                ensure!(
                    (limits.forward_velocity_limits[0]..=limits.forward_velocity_limits[1])
                        .contains(&velocity.x())
                        && velocity.y().abs() <= limits.lateral_velocity_limit
                        && angular_velocity.abs() <= limits.angular_velocity_limit,
                    "walking command outside trained envelope"
                );
            }
            Self::Kick(KickCommand { request, .. }) => {
                ensure!(request.is_finite(), "invalid kick request")
            }
            _ => {}
        }
        Ok(())
    }
}

pub struct Inference {
    parameters: Arc<Parameters>,
    networks: HashMap<Policy, Network>,
    active: Option<Execution>,
    velocity: VelocityEstimator,
    previous_update: Option<Time>,
}

impl Inference {
    pub fn new(root: &Path, policies: &[Policy], parameters: Arc<Parameters>) -> Result<Self> {
        parameters.validate()?;
        ensure!(!policies.is_empty(), "no policies requested");
        let mut networks = HashMap::new();
        for &policy in policies {
            if let std::collections::hash_map::Entry::Vacant(entry) = networks.entry(policy) {
                entry.insert(Network::load(root, policy, &parameters)?);
            }
        }
        Ok(Self {
            parameters,
            networks,
            active: None,
            velocity: VelocityEstimator::default(),
            previous_update: None,
        })
    }

    pub(crate) fn reset(&mut self) {
        self.active = None;
        self.previous_update = None;
        self.velocity = VelocityEstimator::default();
    }

    pub(crate) fn execute_request(
        &mut self,
        now: Time,
        sensor: &SensorFrame,
        request: InferenceCommand,
        velocity: VelocityEstimator,
        joints: &JointLimits,
        walking_velocity_limits: WalkingVelocityLimits,
        trace: &Trace,
    ) -> Result<Box<Joints<MotorCommand>>> {
        self.validate_update(sensor, request, walking_velocity_limits)?;
        self.velocity = velocity;
        self.activate(now, sensor, request.policy(), joints);
        let standing = self.advance_gait(now, request);
        self.infer(now, sensor, request, standing, joints, trace)
    }

    pub(crate) fn update_parameters(&mut self, parameters: Arc<Parameters>) {
        if Arc::ptr_eq(&self.parameters, &parameters) {
            return;
        }
        if let Some(active) = &mut self.active {
            match &mut active.state {
                State::Locomotion(state) => state.update_parameters(parameters.clone()),
                State::SlowGetUp(state) | State::FastGetUp(state) => {
                    state.update_parameters(parameters.clone());
                }
            }
        }
        self.parameters = parameters;
    }

    fn validate_update(
        &self,
        sensor: &SensorFrame,
        request: InferenceCommand,
        walking_velocity_limits: WalkingVelocityLimits,
    ) -> Result<()> {
        sensor.validate(&self.parameters)?;
        request.validate(walking_velocity_limits)?;
        let policy = request.policy();
        ensure!(
            self.networks.contains_key(&policy),
            "{policy:?} was not initialized"
        );
        Ok(())
    }

    fn activate(&mut self, now: Time, sensor: &SensorFrame, policy: Policy, joints: &JointLimits) {
        if let Some(active) = &mut self.active {
            if active.policy == policy {
                return;
            }
            if active.policy.is_locomotion() && policy.is_locomotion() {
                active.policy = policy;
                return;
            }
        }
        self.active = Some(Execution::new(
            policy,
            sensor,
            now,
            self.parameters.clone(),
            joints,
        ));
        self.previous_update = Some(now);
    }

    fn advance_gait(&mut self, now: Time, request: InferenceCommand) -> bool {
        // Only exact zero requests standing; tiny nonzero commands must retain gait phase.
        let standing = matches!(
            request,
            InferenceCommand::Walk(WalkCommand { velocity, angular_velocity })
                if velocity.x() == 0.0 && velocity.y() == 0.0 && angular_velocity == 0.0
        );
        let elapsed = self
            .previous_update
            .map_or(0.0, |last| now.duration_since(last).as_secs_f32());
        if let Some(active) = &mut self.active {
            active.advance(elapsed, standing);
        }
        self.previous_update = Some(now);
        standing
    }

    fn infer(
        &mut self,
        now: Time,
        sensor: &SensorFrame,
        request: InferenceCommand,
        standing: bool,
        joints: &JointLimits,
        trace: &Trace,
    ) -> Result<Box<Joints<MotorCommand>>> {
        let active = self
            .active
            .as_mut()
            .expect("policy activated before inference");
        let policy = active.policy;
        let observation =
            active.prepare_input(now, sensor, &self.velocity, request, standing, joints);
        trace.record_lazy(|| Event::Input {
            policy,
            shape: vec![1, observation.len() as i64],
            values: observation.clone(),
            sources: trace.inputs().cloned(),
            inference_time: now,
            previous_request: active.previous_request.clone(),
        });
        let raw_output = self
            .networks
            .get_mut(&policy)
            .expect("policy validated before activation")
            .run_with_output_observer(&observation, |shape, values| {
                trace.record_lazy(|| Event::Output {
                    policy,
                    shape: shape.to_vec(),
                    values: values.to_vec(),
                });
            })?;
        let mut commands = active.decode(sensor, &raw_output, joints)?;
        ensure!(joints_are_finite(&commands), "non-finite decoded joints");
        if policy.is_locomotion() || policy == Policy::SlowGetUp {
            for (joint, [minimum, maximum]) in joints.position.enumerate() {
                commands[joint].position = commands[joint].position.clamp(minimum, maximum);
            }
        }
        active.previous_request = trace.request_id().cloned();
        Ok(Box::new(commands))
    }
}

enum State {
    Locomotion(Locomotion),
    SlowGetUp(GetUp),
    FastGetUp(GetUp),
}

struct Execution {
    policy: Policy,
    state: State,
    previous_request: Option<ros_z::service::RequestId>,
}

impl Execution {
    fn new(
        policy: Policy,
        sensor: &SensorFrame,
        now: Time,
        parameters: Arc<Parameters>,
        joints: &JointLimits,
    ) -> Self {
        let state = match policy {
            Policy::Walk | Policy::Kick | Policy::SoftKick => {
                State::Locomotion(Locomotion::new(sensor, parameters, joints))
            }
            Policy::SlowGetUp => State::SlowGetUp(GetUp::new(sensor, now, parameters)),
            Policy::FastGetUp => State::FastGetUp(GetUp::new(sensor, now, parameters)),
        };
        Self {
            policy,
            state,
            previous_request: None,
        }
    }

    fn advance(&mut self, seconds: f32, standing: bool) {
        if let State::Locomotion(state) = &mut self.state {
            state.advance(seconds, standing);
        }
    }

    fn prepare_input(
        &mut self,
        now: Time,
        sensor: &SensorFrame,
        velocity: &VelocityEstimator,
        request: InferenceCommand,
        standing: bool,
        joints: &JointLimits,
    ) -> Vec<f32> {
        match &mut self.state {
            State::Locomotion(state) => match request {
                InferenceCommand::Kick(KickCommand { soft, request }) => {
                    let ball_positions = state.record_kick_sample(sensor, request, joints, soft);
                    kick::Observation::new(
                        state,
                        sensor,
                        // Kick training uses the raw finite-difference velocity.
                        // The walking estimator's slew limit delays a fast swing.
                        &velocity.raw,
                        standing,
                        request,
                        soft,
                        ball_positions,
                        joints,
                    )
                    .to_tensor()
                    .to_vec()
                }
                _ => {
                    state.record_walk_sample(sensor, joints);
                    let (linear_velocity, angular_velocity) = match request {
                        InferenceCommand::Walk(WalkCommand {
                            velocity,
                            angular_velocity,
                        }) => (velocity, angular_velocity),
                        _ => (Vector2::zeros(), 0.0),
                    };
                    walk::Observation::new(
                        state,
                        &velocity.walking,
                        standing,
                        linear_velocity,
                        angular_velocity,
                    )
                    .to_tensor()
                    .to_vec()
                }
            },
            State::SlowGetUp(state) => {
                slow::Observation::new(state, sensor, &velocity.raw, now, joints)
                    .to_tensor()
                    .to_vec()
            }
            State::FastGetUp(state) => fast::Observation::new(state, sensor, &velocity.raw, joints)
                .to_tensor()
                .to_vec(),
        }
    }

    fn decode(
        &mut self,
        sensor: &SensorFrame,
        actions: &[f32],
        joints: &JointLimits,
    ) -> Result<Joints<MotorCommand>> {
        Ok(match &mut self.state {
            State::Locomotion(state) => state.decode(
                self.policy,
                actions
                    .try_into()
                    .wrap_err("invalid locomotion action count")?,
                sensor,
            ),
            State::SlowGetUp(state) | State::FastGetUp(state) => state.decode(
                self.policy,
                actions.try_into().wrap_err("invalid get-up action count")?,
                sensor,
                joints,
            ),
        })
    }
}

pub fn position_targets(
    position: Joints<f32>,
    kp: Joints<f32>,
    kd: Joints<f32>,
) -> Joints<MotorCommand> {
    position
        .enumerate()
        .map(|(joint, position)| MotorCommand {
            position,
            kp: kp[joint],
            kd: kd[joint],
            ..MotorCommand::zeros()
        })
        .collect()
}

pub fn joints_are_finite(joints: &Joints<MotorCommand>) -> bool {
    joints
        .into_iter()
        .flat_map(|joint| {
            [
                joint.position,
                joint.velocity,
                joint.torque,
                joint.kp,
                joint.kd,
            ]
        })
        .all(f32::is_finite)
}

#[cfg(test)]
mod tests {
    use super::*;
    use linear_algebra::{Orientation2, Vector3, point};

    #[test]
    fn kick_observation_uses_the_raw_joint_velocity_from_training() {
        let parameters: Parameters = serde_json::from_value(
            json5::from_str::<serde_json::Value>(include_str!(
                "../../../../etc/parameters/base/motion_inference.json5"
            ))
            .unwrap(),
        )
        .unwrap();
        let position = Policy::Kick.offset(&parameters);
        let sensor = SensorFrame {
            timestamp: Time::zero(),
            position,
            velocity: Joints::fill(10.0),
            orientation: nalgebra::Quaternion::identity(),
            gyro: Vector3::zeros(),
            last_commanded_position: position,
        };
        let joints = JointLimits {
            position: Joints::fill([-3.0, 3.0]),
            maximum_torque: Joints::fill(50.0),
        };
        let mut state = Execution::new(
            Policy::Kick,
            &sensor,
            Time::zero(),
            Arc::new(parameters),
            &joints,
        );
        let mut velocity = VelocityEstimator::default();
        velocity.walking = Joints::fill(1.0);
        velocity.raw = Joints::fill(10.0);
        let command = InferenceCommand::Kick(KickCommand {
            soft: false,
            request: KickRequest {
                ball_position: point![0.5, 0.1],
                ball_velocity: Vector2::zeros(),
                direction: Orientation2::new(0.0),
                target_speed: 1.0,
                strong: false,
                quick: false,
            },
        });
        let tensor = state.prepare_input(Time::zero(), &sensor, &velocity, command, false, &joints);
        assert_eq!(
            tensor[23], 1.0,
            "kick training uses raw finite-difference velocity scaled by 0.1"
        );
        assert_eq!(
            tensor[56], 0.1,
            "target speed in m/s is scaled by 0.1 exactly once"
        );
        for (angle, normalized, sine, cosine) in [
            (std::f32::consts::FRAC_PI_2, 0.5, 1.0, 0.0),
            (-std::f32::consts::FRAC_PI_2, -0.5, -1.0, 0.0),
            (std::f32::consts::PI, -1.0, 0.0, -1.0),
        ] {
            let InferenceCommand::Kick(mut kick) = command else {
                unreachable!()
            };
            kick.request.direction = Orientation2::new(angle);
            kick.request.ball_velocity = linear_algebra::vector![1.2, -0.4];
            let tensor = state.prepare_input(
                Time::zero(),
                &sensor,
                &velocity,
                InferenceCommand::Kick(kick),
                false,
                &joints,
            );
            for (actual, expected) in [
                (tensor[8], normalized),
                (tensor[54], sine),
                (tensor[55], cosine),
                (tensor[51], 0.12),
                (tensor[52], -0.04),
            ] {
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "direction and velocity retain the robot-yaw frame"
                );
            }
        }
    }
}

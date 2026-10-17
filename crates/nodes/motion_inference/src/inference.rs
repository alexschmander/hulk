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
    get_up::{GetUp, observation::Observation as GetUpObservation},
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
#[serde(deny_unknown_fields)]
pub struct GetUpCommand {}

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
            Self::GetUp(_) => Policy::GetUp,
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

    #[allow(clippy::too_many_arguments)]
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
                State::GetUp(state) => {
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
        let mut commands = active.decode(sensor, &raw_output)?;
        ensure!(joints_are_finite(&commands), "non-finite decoded joints");
        for (joint, [minimum, maximum]) in joints.position.enumerate() {
            commands[joint].position = commands[joint].position.clamp(minimum, maximum);
        }
        active.previous_request = trace.request_id().cloned();
        Ok(Box::new(commands))
    }
}

enum State {
    Locomotion(Locomotion),
    GetUp(GetUp),
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
            Policy::GetUp => State::GetUp(GetUp::new(sensor, now, parameters)),
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
                        // This existing estimator is raw finite-difference velocity.
                        &velocity.get_up,
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
            State::GetUp(state) => {
                GetUpObservation::new(state, sensor, &velocity.get_up, now, joints)
                    .to_tensor()
                    .to_vec()
            }
        }
    }

    fn decode(&mut self, sensor: &SensorFrame, actions: &[f32]) -> Result<Joints<MotorCommand>> {
        Ok(match &mut self.state {
            State::Locomotion(state) => state.decode(
                self.policy,
                actions
                    .try_into()
                    .wrap_err("invalid locomotion action count")?,
                sensor,
            ),
            State::GetUp(state) => {
                state.decode(actions.try_into().wrap_err("invalid get-up action count")?)
            }
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
    use linear_algebra::{Orientation2, Vector3, point, vector};

    fn fixture() -> (Parameters, SensorFrame, JointLimits) {
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
        (parameters, sensor, joints)
    }

    #[test]
    fn calibrated_kick_tensor_preserves_training_inputs_and_ball_history() {
        let (mut parameters, sensor, joints) = fixture();
        parameters.kick.shift_distance = 0.0;
        let parameters = Arc::new(parameters);
        let mut velocity = VelocityEstimator::default();
        velocity.walking = Joints::fill(1.0);
        velocity.get_up = Joints::fill(10.0);
        for (angle, soft, speed, incoming, extension) in [
            (0.0, false, 3.4, false, 0.12),
            (std::f32::consts::FRAC_PI_4, false, 3.4, false, 0.12),
            (-std::f32::consts::FRAC_PI_2, true, 0.3, false, 0.12),
            (0.0, true, 0.1, false, 0.0),
            (0.0, true, 0.2, false, 0.06),
            (0.0, false, 3.4, true, 0.0),
            (0.0, true, 0.3, true, 0.0),
        ] {
            let mut request = KickRequest {
                ball_position: point![0.4, 0.05],
                ball_velocity: vector![if incoming { -0.3 } else { 0.2 }, 0.0],
                direction: Orientation2::new(angle),
                target_speed: speed,
                strong: false,
                quick: false,
            };
            let expected =
                request.ball_position + vector![angle.cos() * extension, angle.sin() * extension];
            let policy = if soft { Policy::SoftKick } else { Policy::Kick };
            let mut state =
                Execution::new(policy, &sensor, Time::zero(), parameters.clone(), &joints);
            let State::Locomotion(locomotion) = &mut state.state else {
                unreachable!()
            };
            let mut actions = [0.0; 13];
            actions[12] = 1.25;
            locomotion.decode(policy, &actions, &sensor);
            locomotion.advance(0.02, false);
            assert!((locomotion.phase - 0.04).abs() < 1e-6);
            let command = |request| InferenceCommand::Kick(KickCommand { soft, request });
            let first = state.prepare_input(
                Time::zero(),
                &sensor,
                &velocity,
                command(request),
                false,
                &joints,
            );
            assert_eq!(first[23], 1.0, "raw joint velocity, scaled once");
            assert_eq!(first[47], 1.25, "frequency feedback precedes gait clipping");
            assert!((first[56] - speed * 0.1).abs() < 1e-6);
            assert!((first[6] - expected.x() * 0.1).abs() < 1e-6);
            assert!((first[7] - expected.y() * 0.1).abs() < 1e-6);
            request.ball_position += vector![0.01, 0.0];
            let second = state.prepare_input(
                Time::zero(),
                &sensor,
                &velocity,
                command(request),
                false,
                &joints,
            );
            assert_eq!(&second[57..59], &first[6..8]);
            assert!((second[6] - first[6] - 0.001).abs() < 1e-6);
        }
    }

    #[test]
    fn foot_bias_is_invariant_to_translating_both_soles_relative_to_the_torso() {
        let (parameters, mut sensor, joints) = fixture();
        let parameters = Arc::new(parameters);
        let command = InferenceCommand::Kick(KickCommand {
            soft: false,
            request: KickRequest {
                ball_position: point![0.5, 0.08],
                ball_velocity: Vector2::zeros(),
                direction: Orientation2::new(0.0),
                target_speed: 3.4,
                strong: false,
                quick: false,
            },
        });
        let input = |sensor: &SensorFrame| {
            Execution::new(
                Policy::Kick,
                sensor,
                Time::zero(),
                parameters.clone(),
                &joints,
            )
            .prepare_input(
                Time::zero(),
                sensor,
                &VelocityEstimator::default(),
                command,
                false,
                &joints,
            )
        };
        let before = input(&sensor);
        sensor.position.left_leg.hip_roll = 0.25;
        sensor.position.right_leg.hip_roll = 0.25;
        sensor.position.left_leg.ankle_down = -0.25;
        sensor.position.right_leg.ankle_down = -0.25;
        let after = input(&sensor);
        assert!((before[6] - after[6]).abs() < 1e-6);
        assert!((before[7] - after[7]).abs() < 1e-6);
    }

    #[test]
    fn follow_through_distance_is_bounded_and_optional_in_old_configs() {
        let (mut parameters, _, _) = fixture();
        parameters.validate().unwrap();
        for distance in [-0.01, 0.16, f32::NAN] {
            parameters.kick.follow_through_distance = distance;
            assert!(parameters.validate().is_err());
        }
        let mut old = serde_json::to_value(parameters).unwrap();
        old["kick"]
            .as_object_mut()
            .unwrap()
            .remove("follow_through_distance");
        let old: Parameters = serde_json::from_value(old).unwrap();
        assert_eq!(old.kick.follow_through_distance, 0.0);
        old.validate().unwrap();
    }
}

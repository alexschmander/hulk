use std::{collections::HashMap, path::Path, sync::Arc};

use color_eyre::eyre::{Result, ensure};
use serde::{Deserialize, Serialize};

use coordinate_systems::Ground;
use kinematics::joints::Joints;
use linear_algebra::Vector2;
use ros_z::{Message, time::Time};
use types::joint_limits::JointLimits;
use types::motor_command::MotorCommand;

use crate::{
    config::{Parameters, Policy},
    get_up::{GetUp, fast, slow},
    locomotion::{KickRequest, Locomotion, kick, walk},
    network::Network,
    observation::{SensorFrame, VelocityEstimator},
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

#[derive(Clone, Serialize, Deserialize, Message)]
pub struct InferenceOutput {
    pub joints: Box<Joints<MotorCommand>>,
    pub mode: Mode,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Message)]
pub enum Mode {
    Full,
    Body,
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

    fn validate(self, parameters: &Parameters) -> Result<()> {
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
                    (parameters.locomotion.forward_velocity_limits[0]
                        ..=parameters.locomotion.forward_velocity_limits[1])
                        .contains(&velocity.x())
                        && velocity.y().abs() <= parameters.locomotion.lateral_velocity_limit
                        && angular_velocity.abs() <= parameters.locomotion.angular_velocity_limit,
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
    /// Switch tuning at a request boundary, preserving gait phase and history. Model
    /// changes load a complete replacement before replacing any existing network.
    pub(crate) fn update_parameters(&mut self, parameters: Arc<Parameters>) -> Result<()> {
        if Arc::ptr_eq(&self.parameters, &parameters) {
            return Ok(());
        }
        parameters.validate()?;
        let reload = self.parameters.neural_networks_folder != parameters.neural_networks_folder
            || self.parameters.inference_threads != parameters.inference_threads
            || self
                .networks
                .keys()
                .any(|policy| policy.file(&self.parameters) != policy.file(&parameters));
        if reload {
            let networks = self
                .networks
                .keys()
                .map(|&policy| {
                    Network::load(&parameters.neural_networks_folder, policy, &parameters)
                        .map(|network| (policy, network))
                })
                .collect::<Result<HashMap<_, _>>>()?;
            self.networks = networks;
        }
        if let Some(active) = &mut self.active {
            match &mut active.state {
                State::Locomotion(state) => state.update_parameters(parameters.clone()),
                State::SlowGetUp(state) | State::FastGetUp(state) => {
                    state.update_parameters(parameters.clone())
                }
            }
        }
        self.parameters = parameters;
        Ok(())
    }

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

    pub(crate) fn execute_request(
        &mut self,
        now: Time,
        sensor: &SensorFrame,
        request: InferenceCommand,
        velocity: VelocityEstimator,
        joints: &JointLimits,
        parameters: Arc<Parameters>,
    ) -> Result<InferenceOutput> {
        self.update_parameters(parameters)?;
        self.validate_update(now, sensor, request)?;
        self.velocity = velocity;
        self.activate(now, sensor, request.policy(), joints);
        let standing = self.advance_gait(now, request);
        self.infer(now, sensor, request, standing, joints)
    }

    fn validate_update(
        &self,
        now: Time,
        sensor: &SensorFrame,
        request: InferenceCommand,
    ) -> Result<()> {
        sensor.validate(&self.parameters)?;
        request.validate(&self.parameters)?;
        let policy = request.policy();
        ensure!(
            self.networks.contains_key(&policy),
            "{policy:?} was not initialized"
        );
        if let Some(last) = self.previous_update {
            ensure!(now >= last, "controller time moved backwards");
        }
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
    ) -> Result<InferenceOutput> {
        let active = self
            .active
            .as_mut()
            .expect("policy activated before inference");
        let policy = active.policy;
        let observation =
            active.prepare_input(now, sensor, &self.velocity, request, standing, joints);
        let raw_output = self
            .networks
            .get_mut(&policy)
            .expect("policy validated before activation")
            .run(&observation)?;
        let joints = active.decode(sensor, &raw_output, joints);
        ensure!(joints_are_finite(&joints), "non-finite decoded joints");
        Ok(InferenceOutput {
            joints: Box::new(joints),
            mode: if policy.is_locomotion() {
                Mode::Body
            } else {
                Mode::Full
            },
        })
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
        Self { policy, state }
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
                    let ball_positions = state.record_kick_sample(sensor, request, joints);
                    kick::Observation::new(
                        state,
                        sensor,
                        &velocity.walking,
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
                slow::Observation::new(state, sensor, &velocity.get_up, now, joints)
                    .to_tensor()
                    .to_vec()
            }
            State::FastGetUp(state) => {
                fast::Observation::new(state, sensor, &velocity.get_up, joints)
                    .to_tensor()
                    .to_vec()
            }
        }
    }

    fn decode(
        &mut self,
        sensor: &SensorFrame,
        actions: &[f32],
        joints: &JointLimits,
    ) -> Joints<MotorCommand> {
        match &mut self.state {
            State::Locomotion(state) => state.decode(self.policy, actions, sensor),
            State::SlowGetUp(state) | State::FastGetUp(state) => {
                state.decode(self.policy, actions, sensor, joints)
            }
        }
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
    use ros_z::prelude::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn tuning_updates_active_policies_without_resetting_gait_or_time() {
        let context = ContextBuilder::default()
            .with_mode("peer")
            .disable_multicast_scouting()
            .with_connect_endpoints(std::iter::empty::<&str>())
            .with_listen_endpoints(std::iter::empty::<&str>())
            .with_parameter_layer(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../etc/parameters/base"),
            )
            .build()
            .await
            .unwrap();
        let node = context
            .create_node("inference_tuning_test")
            .build()
            .await
            .unwrap();
        let binding = node
            .bind_parameter_as::<Parameters>("motion_inference")
            .unwrap();
        let original = binding.snapshot().typed.clone();
        let sensor = SensorFrame {
            timestamp: Time::zero(),
            position: Joints::fill(0.0),
            velocity: Joints::fill(0.0),
            orientation: nalgebra::Quaternion::identity(),
            gyro: linear_algebra::Vector3::zeros(),
            last_commanded_position: Joints::fill(0.0),
        };
        let limits = JointLimits {
            position: Joints::fill([-3.0, 3.0]),
            maximum_torque: Joints::fill(100.0),
        };
        for policy in Policy::ALL {
            let mut inference = Inference {
                parameters: original.clone(),
                networks: HashMap::new(),
                active: Some(Execution::new(
                    policy,
                    &sensor,
                    Time::zero(),
                    original.clone(),
                    &limits,
                )),
                velocity: VelocityEstimator::default(),
                previous_update: Some(Time::from_nanos(1_000_000_000)),
            };
            if let State::Locomotion(state) = &mut inference.active.as_mut().unwrap().state {
                state.phase = 0.37;
            }
            let mut tuned = (*original).clone();
            tuned.policies.get_mut(&policy).unwrap().kp = Joints::fill(31.0);
            tuned.policies.get_mut(&policy).unwrap().kd = Joints::fill(2.0);
            inference
                .update_parameters(Arc::new(tuned.clone()))
                .unwrap();
            let active = inference.active.as_mut().unwrap();
            if let State::Locomotion(state) = &active.state {
                assert_eq!(state.phase, 0.37);
            }
            let command = active.decode(&sensor, &[0.0; 22], &limits);
            assert_eq!(command.left_leg.hip_pitch.kp, 31.0);
            assert_eq!(command.left_leg.hip_pitch.kd, 2.0);
            assert_eq!(
                inference.previous_update,
                Some(Time::from_nanos(1_000_000_000))
            );
            tuned.policies.get_mut(&policy).unwrap().kp = Joints::fill(-1.0);
            assert!(inference.update_parameters(Arc::new(tuned)).is_err());
            assert_eq!(
                inference.parameters.policies[&policy].kp.left_leg.hip_pitch,
                31.0
            );
        }
        context.shutdown().unwrap();
    }
}

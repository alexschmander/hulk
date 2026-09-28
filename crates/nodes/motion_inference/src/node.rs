use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};

use booster::{JointsMotorState, LowState};
use color_eyre::{Report, Result};
use kinematics::joints::{Joints, body::LowerBodyJoints};
use nalgebra::UnitQuaternion;
use ros_z::{
    Message,
    prelude::*,
    qos::{QosDurability, QosHistory, QosReliability},
    time::Time,
};
use serde::{Deserialize, Serialize};
use types::joint_limits::JointLimits;
use types::motor_command::MotorCommand;

pub use crate::config::Parameters;
use crate::{
    config::{Policy, TimingParameters},
    execution::{ExecutionGate, PolicyCadence},
    inference::{
        ExecutionState, GetUpCommand, Inference, InferenceCommand, InferenceRequest, KickCommand,
        WalkCommand,
    },
    observation::{SensorFrame, VelocityEstimator, YawHistory},
};

pub const GETUP_INFERENCE_SERVICE: &str = "motion_inference/infer_getup";
pub const KICK_INFERENCE_SERVICE: &str = "motion_inference/infer_kick";
pub const WALK_INFERENCE_SERVICE: &str = "motion_inference/infer_walk";
pub const EXECUTION_TOPIC: &str = "motion_inference/execution";
pub const TIMING_PARAMETERS_TOPIC: &str = "motion_inference/timing_parameters";

macro_rules! inference_service {
    ($name:ident, $command:ty, $joints:ty) => {
        pub struct $name;
        impl Service for $name {
            type Request = InferenceRequest<$command>;
            type Response = InferenceResult<$joints>;
        }
        impl ServiceTypeInfo for $name {
            fn service_type_info() -> TypeInfo {
                let descriptor = ros_z_schema::ServiceDef::new(
                    concat!(module_path!(), "::", stringify!($name)),
                    <Self as Service>::Request::type_name(),
                    <Self as Service>::Response::type_name(),
                )
                .expect("static inference service descriptor");
                TypeInfo::new(
                    descriptor.type_name.as_str(),
                    ros_z_schema::compute_hash(&descriptor).expect("static service hash"),
                )
            }
        }
    };
}

inference_service!(
    WalkInferenceService,
    WalkCommand,
    LowerBodyJoints<MotorCommand>
);
inference_service!(
    KickInferenceService,
    KickCommand,
    LowerBodyJoints<MotorCommand>
);
inference_service!(GetUpInferenceService, GetUpCommand, Joints<MotorCommand>);

use crate::services::{InferenceReply, Requests};
pub const STATUS_TOPIC: &str = "motion_inference/status";
pub const TIMING_TOPIC: &str = "motion_inference/timing";

pub type InferenceResult<T> = std::result::Result<T, InferenceError>;

#[derive(Clone, Debug, Serialize, Deserialize, Message, thiserror::Error)]
pub enum InferenceError {
    #[error("inference request or selected sensor frame expired")]
    Expired,
    #[error("inference request was superseded")]
    Superseded,
    #[error("inference is waiting for fresh sensors and validated joint limits")]
    Unavailable,
    #[error("motion inference fault: {source:#}")]
    Fault {
        #[serde(with = "ros_z::message::report")]
        source: Arc<Report>,
    },
}

#[derive(Clone, Serialize, Deserialize, Message)]
pub struct Status {
    pub time: Time,
    pub state: State,
}
#[derive(Clone, Serialize, Deserialize, Message)]
pub enum State {
    Idle,
    Initialized,
    Fault { reason: String },
}

#[derive(Clone, Serialize, Deserialize, Message)]
pub struct Timing {
    pub generation: u64,
    pub requested_at: Time,
    pub policy: Policy,
    pub received_at: Time,
    pub started_at: Time,
    pub completed_at: Time,
    pub sensor_time: Time,
    pub compute_duration: Duration,
    pub accepted: bool,
    pub error: Option<String>,
}

struct Pending {
    request: InferenceRequest<InferenceCommand>,
    reply: InferenceReply,
    received_at: Time,
}
impl Pending {
    async fn reject(self, error: InferenceError) {
        self.reply.respond(Err(error)).await;
    }
}

pub fn run_boxed(ctx: Arc<Context>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
    Box::pin(run(ctx))
}

async fn run(ctx: Arc<Context>) -> Result<()> {
    let node = ctx.create_node("motion_inference").build().await?;
    let parameters = node.bind_parameter_as::<Parameters>("motion_inference")?;
    let startup = parameters.snapshot().typed.clone();
    startup.validate()?;
    parameters.add_validation_hook(move |candidate| {
        candidate.validate().map_err(|error| format!("{error:#}"))?;
        if candidate != startup.as_ref() {
            return Err("motion inference parameter changes require a restart".into());
        }
        Ok(())
    })?;
    // The inference node owns this configuration. Retain its timing settings
    // so motion can start before or after the owner without binding them again.
    let timing_parameters = node
        .publisher::<TimingParameters>(TIMING_PARAMETERS_TOPIC)
        .qos(QosProfile {
            durability: QosDurability::TransientLocal,
            ..Default::default()
        })
        .build()
        .await?;
    timing_parameters
        .publish(&parameters.snapshot().typed.timing)
        .await?;
    let qos = QosProfile {
        history: QosHistory::from_depth(1),
        reliability: QosReliability::BestEffort,
        ..Default::default()
    };
    let statuses = node
        .publisher::<Status>(STATUS_TOPIC)
        .qos(QosProfile {
            durability: QosDurability::TransientLocal,
            ..qos
        })
        .build()
        .await?;
    let timings = node
        .publisher::<Timing>(TIMING_TOPIC)
        .qos(qos)
        .build()
        .await?;
    statuses
        .publish(&Status {
            time: node.clock().now(),
            state: State::Idle,
        })
        .await?;
    let inference_parameters = parameters.snapshot().typed.clone();
    let initialized = tokio::task::spawn_blocking(move || {
        Inference::new(
            &inference_parameters.neural_networks_folder,
            &Policy::ALL,
            inference_parameters.clone(),
        )
    })
    .await?;
    let controller = match initialized {
        Ok(controller) => Some(controller),
        Err(error) => {
            statuses
                .publish(&Status {
                    time: node.clock().now(),
                    state: State::Fault {
                        reason: format!("{error:#}"),
                    },
                })
                .await?;
            return Err(error);
        }
    };
    let sensors = node
        .subscriber::<LowState>("inputs/low_state")
        .qos(qos)
        .build()
        .await?;
    let limits = node
        .subscriber::<JointLimits>("joint_limits")
        .qos(QosProfile {
            durability: QosDurability::TransientLocal,
            ..qos
        })
        .build()
        .await?;
    let mut requests = Requests::new(&node, qos).await?;
    let execution = node
        .subscriber::<ExecutionState>(EXECUTION_TOPIC)
        .build()
        .await?;
    statuses
        .publish(&Status {
            time: node.clock().now(),
            state: State::Initialized,
        })
        .await?;
    Runtime::new(
        parameters.snapshot().typed.clone(),
        node.clock().clone(),
        controller,
    )
    .run(
        sensors,
        limits,
        execution,
        &mut requests,
        &statuses,
        &timings,
    )
    .await
}

type WorkerResult = (
    Inference,
    Result<Box<Joints<MotorCommand>>>,
    Policy,
    Time,
    Duration,
);

struct Runtime {
    parameters: Arc<Parameters>,
    clock: ros_z::time::Clock,
    controller: Option<Inference>,
    worker: Option<tokio::task::JoinHandle<WorkerResult>>,
    active: Option<(Pending, Time)>,
    pending: Option<Pending>,
    sensor: Option<SensorFrame>,
    joint_limits: Option<Arc<JointLimits>>,
    velocity: VelocityEstimator,
    last_position: Option<(Policy, Joints<f32>)>,
    fault: Option<Arc<Report>>,
    execution: ExecutionGate,
    cadence: PolicyCadence,
    yaw_history: YawHistory,
}

impl Runtime {
    fn new(
        parameters: Arc<Parameters>,
        clock: ros_z::time::Clock,
        controller: Option<Inference>,
    ) -> Self {
        Self {
            cadence: PolicyCadence::new(parameters.timing.policy_period),
            parameters,
            clock,
            controller,
            worker: None,
            active: None,
            pending: None,
            sensor: None,
            joint_limits: None,
            velocity: VelocityEstimator::default(),
            last_position: None,
            fault: None,
            execution: ExecutionGate::default(),
            yaw_history: YawHistory::default(),
        }
    }

    async fn run(
        mut self,
        sensors: Subscriber<LowState>,
        limits: Subscriber<JointLimits>,
        execution: Subscriber<ExecutionState>,
        requests: &mut Requests,
        statuses: &Publisher<Status>,
        timings: &Publisher<Timing>,
    ) -> Result<()> {
        let mut published_fault: Option<String> = None;
        loop {
            let wake_at = self.dispatch_deadline();
            let clock = self.clock.clone();
            tokio::select! {
                received = sensors.recv_with_metadata() => {
                    let received = received?;
                    self.receive_sensor(&received, received.source_time);
                }
                received = limits.recv() => {
                    self.receive_limits(received?);
                }
                received = execution.recv() => {
                    self.observe_execution(received?);
                }
                received = requests.receive() => {
                    let (request, reply) = received?;
                    self.enqueue(Pending {
                        request,
                        reply,
                        received_at: self.clock.now(),
                    })
                    .await;
                }
                completed = async { self.worker.as_mut().expect("active worker").await }, if self.worker.is_some() => {
                    self.complete(completed?, timings).await?;
                }
                _ = async { clock.sleep_until(wake_at.expect("pending dispatch")).await }, if wake_at.is_some() => {
                    self.dispatch().await;
                }
            }
            let fault = self.fault.as_ref().map(|error| format!("{error:#}"));

            if fault != published_fault {
                let state = match &fault {
                    Some(reason) => State::Fault {
                        reason: reason.clone(),
                    },
                    None => State::Initialized,
                };

                statuses
                    .publish(&Status {
                        time: self.clock.now(),
                        state,
                    })
                    .await?;

                published_fault = fault;
            }
        }
    }

    fn receive_sensor(&mut self, low: &LowState, time: Time) {
        let sensor = match sensor_frame(low, time) {
            Ok(sensor) => sensor,
            Err(error) => {
                self.sensor = None;
                self.fault = Some(Arc::new(error));
                return;
            }
        };
        if let Err(error) = sensor.validate(&self.parameters) {
            self.sensor = None;
            self.fault = Some(Arc::new(error));
            return;
        }
        let now = self.clock.now();
        if sensor.timestamp > now {
            self.sensor = None;
            self.fault = Some(Arc::new(Report::msg("sensor timestamp is in the future")));
            return;
        }
        if now.duration_since(sensor.timestamp) > self.parameters.timing.maximum_sensor_age
            || self
                .sensor
                .as_ref()
                .is_some_and(|old| sensor.timestamp <= old.timestamp)
        {
            return;
        }
        if let Err(error) = self.velocity.update(&sensor, &self.parameters) {
            self.fault = Some(Arc::new(error));
        }
        self.sensor = Some(sensor);
        self.yaw_history
            .record(time, low.imu_state.roll_pitch_yaw.z());
    }

    fn receive_limits(&mut self, limits: JointLimits) {
        match limits.validate() {
            Ok(()) => self.joint_limits = Some(Arc::new(limits)),
            Err(reason) => {
                self.joint_limits = None;
                self.fault = Some(Arc::new(Report::msg(reason)));
            }
        }
    }

    async fn enqueue(&mut self, pending: Pending) {
        if !pending.request.is_current(self.clock.now()) {
            pending.reject(InferenceError::Expired).await;
            return;
        }
        self.observe_execution(ExecutionState {
            generation: pending.request.generation,
            active: true,
        });
        if !self.execution.accepts(&pending.request, self.clock.now()) {
            pending.reject(InferenceError::Superseded).await;
            return;
        }
        if let Some(old) = self.pending.replace(pending) {
            old.reject(InferenceError::Superseded).await;
        }
    }

    fn observe_execution(&mut self, state: ExecutionState) {
        if self.execution.observe(state) {
            if let Some(controller) = &mut self.controller {
                controller.reset();
            }
            self.last_position = None;
            // Sensor derivatives remain valid while motion is inactive.
            self.fault = None;
        }
    }

    fn dispatch_deadline(&self) -> Option<Time> {
        if self.worker.is_some() {
            return None;
        }
        let pending = self.pending.as_ref()?;
        let now = self.clock.now();
        if !self.execution.accepts(&pending.request, now)
            || self.fault.is_some()
            || self.sensor.is_none()
            || self.joint_limits.is_none()
        {
            return Some(now);
        }
        let sensor = self.sensor.as_ref().expect("checked sensor");
        Some(if self.cadence.has_new_sensor(sensor.timestamp) {
            self.cadence
                .next_start(now)
                .min(pending.request.valid_until)
        } else {
            pending.request.valid_until
        })
    }

    async fn dispatch(&mut self) {
        let pending = self.pending.take().expect("pending request");
        let now = self.clock.now();
        if !self.execution.accepts(&pending.request, now) {
            pending.reject(InferenceError::Expired).await;
            return;
        }
        if let Some(source) = &self.fault {
            pending
                .reject(InferenceError::Fault {
                    source: source.clone(),
                })
                .await;
            return;
        }
        let (Some(sensor), Some(limits)) = (&self.sensor, &self.joint_limits) else {
            pending.reject(InferenceError::Unavailable).await;
            return;
        };
        if sensor.validate_at(now, &self.parameters).is_err() {
            pending.reject(InferenceError::Expired).await;
            return;
        }
        let mut sensor = sensor.clone();
        sensor.last_commanded_position = match self.last_position {
            Some((policy, _))
                if policy.is_locomotion()
                    && matches!(pending.request.command, InferenceCommand::GetUp(_)) =>
            {
                sensor.position
            }
            Some((_, position)) => position,
            None => sensor.position,
        };
        let limits = limits.clone();
        let velocity = self.velocity.clone();
        let mut request = pending.request;
        if let InferenceCommand::Kick(command) = &mut request.command {
            let Some(reference_yaw) = self.yaw_history.at(
                command.reference_time,
                self.parameters.timing.maximum_sensor_age,
            ) else {
                pending.reject(InferenceError::Unavailable).await;
                return;
            };
            let sensor_yaw = sensor.rotation().euler_angles().2;
            command.request = command.request.rotated(reference_yaw - sensor_yaw);
        }
        let mut controller = self.controller.take().expect("idle controller");
        let clock = self.clock.clone();
        let parameters = self.parameters.clone();
        self.active = Some((pending, sensor.timestamp));
        self.worker = Some(tokio::task::spawn_blocking(move || {
            let start = Instant::now();
            let started_at = clock.now();
            let result = if request.is_current(started_at) {
                controller.execute_request(
                    started_at,
                    &sensor,
                    request.command,
                    velocity,
                    &limits,
                    parameters,
                )
            } else {
                Err(Report::msg("request expired before worker start"))
            };
            (
                controller,
                result,
                request.command.policy(),
                started_at,
                start.elapsed(),
            )
        }));
    }

    async fn complete(
        &mut self,
        completed: WorkerResult,
        timings: &Publisher<Timing>,
    ) -> Result<()> {
        self.worker = None;
        let (pending, sensor_time) = self.active.take().expect("active request");
        let (timing, result) = self.accept_output(
            &pending.request,
            pending.received_at,
            sensor_time,
            completed,
        );
        timings.publish(&timing).await?;
        pending.reply.respond(result).await;
        Ok(())
    }

    fn accept_output(
        &mut self,
        request: &InferenceRequest<InferenceCommand>,
        received_at: Time,
        sensor_time: Time,
        completed: WorkerResult,
    ) -> (Timing, InferenceResult<Box<Joints<MotorCommand>>>) {
        let (mut controller, result, policy, started_at, compute_duration) = completed;
        self.cadence.started(started_at, sensor_time);
        let now = self.clock.now();
        let valid = self.execution.accepts(request, now)
            && now >= sensor_time
            && now.duration_since(sensor_time) <= self.parameters.timing.maximum_sensor_age;
        let result = if let Some(source) = &self.fault {
            controller.reset();
            self.last_position = None;
            Err(InferenceError::Fault {
                source: source.clone(),
            })
        } else if !valid {
            controller.reset();
            self.last_position = None;
            Err(InferenceError::Expired)
        } else {
            result.map_err(|error| {
                let source = Arc::new(error);
                self.fault = Some(source.clone());
                InferenceError::Fault { source }
            })
        };
        if let Ok(output) = &result {
            self.last_position = Some((
                policy,
                output
                    .as_ref()
                    .into_iter()
                    .map(|joint| joint.position)
                    .collect(),
            ));
        } else {
            controller.reset();
            self.last_position = None;
        }
        self.controller = Some(controller);
        let timing = Timing {
            generation: request.generation,
            requested_at: request.requested_at,
            policy: request.command.policy(),
            received_at,
            started_at,
            completed_at: now,
            sensor_time,
            compute_duration,
            accepted: result.is_ok(),
            error: result.as_ref().err().map(ToString::to_string),
        };
        (timing, result)
    }
}

fn sensor_frame(low_state: &LowState, timestamp: Time) -> Result<SensorFrame> {
    let motors = low_state.serial_motor_states()?;
    let angles = low_state.imu_state.roll_pitch_yaw;
    let orientation = UnitQuaternion::from_euler_angles(angles.x(), angles.y(), angles.z());
    Ok(SensorFrame {
        timestamp,
        position: motors.positions(),
        velocity: motors.velocities(),
        orientation: orientation.into_inner(),
        gyro: low_state.imu_state.angular_velocity,
        last_commanded_position: motors.positions(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensor_frame_seeds_previous_targets_from_measured_positions() {
        let positions: Joints<f32> = (0..crate::config::JOINT_COUNT)
            .map(|index| index as f32 * 0.01)
            .collect();
        let low_state = LowState {
            motor_state_serial: positions
                .into_iter()
                .map(|position| booster::MotorState {
                    position,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        };

        let sensor = sensor_frame(&low_state, Time::zero()).unwrap();

        assert_eq!(sensor.position, positions);
        assert_eq!(sensor.last_commanded_position, positions);
    }
}

#[cfg(test)]
mod completion_tests {
    use super::*;
    use crate::{
        inference::tests::controller,
        test_support::{parameters, time},
    };

    #[test]
    fn late_or_superseded_worker_results_do_not_commit_targets() {
        for superseded in [false, true] {
            let clock = ros_z::time::Clock::logical(time(30));
            let mut runtime = Runtime::new(parameters(), clock.clone(), None);
            runtime.observe_execution(ExecutionState {
                generation: 1,
                active: true,
            });
            let request = InferenceRequest {
                generation: 1,
                requested_at: time(0),
                valid_until: time(40),
                command: InferenceCommand::Walk(WalkCommand::stand()),
            };
            if superseded {
                runtime.observe_execution(ExecutionState {
                    generation: 2,
                    active: false,
                });
            } else {
                clock.set_time(time(40)).unwrap();
            }
            let (_, result) = runtime.accept_output(
                &request,
                time(0),
                time(20),
                (
                    controller(),
                    Ok(Box::new(Joints::fill(MotorCommand::zeros()))),
                    Policy::Walk,
                    time(20),
                    Duration::from_millis(2),
                ),
            );
            assert!(matches!(result, Err(InferenceError::Expired)));
            assert!(runtime.last_position.is_none());
            assert!(runtime.controller.is_some());
            assert_eq!(runtime.cadence.next_start(time(30)), time(40));
        }
    }

    #[test]
    fn current_result_commits_targets_then_inactive_generation_clears_them() {
        let mut runtime = Runtime::new(parameters(), ros_z::time::Clock::logical(time(30)), None);
        runtime.observe_execution(ExecutionState {
            generation: 1,
            active: true,
        });
        let request = InferenceRequest {
            generation: 1,
            requested_at: time(0),
            valid_until: time(40),
            command: InferenceCommand::Walk(WalkCommand::stand()),
        };
        let target = MotorCommand {
            position: 0.2,
            ..MotorCommand::zeros()
        };
        let (timing, result) = runtime.accept_output(
            &request,
            time(0),
            time(20),
            (
                controller(),
                Ok(Box::new(Joints::fill(target))),
                Policy::Walk,
                time(20),
                Duration::from_millis(2),
            ),
        );
        assert!(result.is_ok());
        assert!(timing.accepted);
        assert_eq!(runtime.last_position.as_ref().unwrap().1, Joints::fill(0.2));
        runtime.observe_execution(ExecutionState {
            generation: 2,
            active: false,
        });
        assert!(runtime.last_position.is_none());
    }
}

use super::*;
use crate::inputs::{Latest, Sample};
use crate::recovery::Recovery;
use booster::{JointsMotorState, LowState};
use ros_z::{
    prelude::*,
    qos::{QosHistory, QosReliability},
    time::Time,
};
use tracing::error;
use types::fall_detection::{FALL_DETECTION_TOPIC, FallDetection};
use types::hardware_status::{ControlMode, HARDWARE_STATUS_TOPIC, HardwareStatus};
use types::motion_execution::{MOTION_EXECUTION_TOPIC, MotionExecution, MotionPhase};

struct Inputs {
    commands: Latest<MotionCommand>,
    sensors: Latest<LowState>,
    falls: Latest<FallDetection>,
    limits: Latest<JointLimits>,
    hardware: Latest<HardwareStatus>,
    inference: Latest<motion_inference::node::Status>,
}
struct Frame {
    command: Arc<Sample<MotionCommand>>,
    position: Joints<f32>,
    fall: FallDetection,
    limits: Arc<Sample<JointLimits>>,
    hardware: Arc<Sample<HardwareStatus>>,
}
impl Inputs {
    async fn new(node: &Node, qos: QosProfile) -> Result<Self> {
        let retained = QosProfile {
            durability: QosDurability::TransientLocal,
            ..qos
        };
        Ok(Self {
            commands: Latest::subscribe(node, "behavior/motion_command", qos).await?,
            sensors: Latest::subscribe(node, "inputs/low_state", qos).await?,
            falls: Latest::subscribe(node, FALL_DETECTION_TOPIC, qos).await?,
            limits: Latest::subscribe(node, "joint_limits", retained).await?,
            hardware: Latest::subscribe(node, HARDWARE_STATUS_TOPIC, qos).await?,
            inference: Latest::subscribe(node, motion_inference::node::STATUS_TOPIC, retained)
                .await?,
        })
    }
    fn frame(&self, now: Time, p: &Parameters) -> Result<Frame> {
        let command = self
            .commands
            .fresh(now, p.maximum_command_age)
            .wrap_err("behavior command")?;
        let sensor = self
            .sensors
            .fresh(now, p.maximum_sensor_age)
            .wrap_err("body sensors")?;
        let motors = sensor.received.serial_motor_states()?;
        let position = motors.positions();
        ensure!(
            position
                .into_iter()
                .chain(motors.velocities())
                .chain(
                    sensor
                        .received
                        .imu_state
                        .roll_pitch_yaw
                        .inner
                        .iter()
                        .copied()
                )
                .chain(
                    sensor
                        .received
                        .imu_state
                        .angular_velocity
                        .inner
                        .iter()
                        .copied()
                )
                .all(f32::is_finite),
            "invalid body sensors"
        );
        let limits = self
            .limits
            .latest()
            .ok_or_else(|| eyre!("joint limits unavailable"))?;
        limits.received.validate().map_err(|e| eyre!(e))?;
        let hardware = self
            .hardware
            .fresh(now, p.maximum_hardware_age)
            .wrap_err("hardware state")?;
        let fall = self.falls.fresh(now, p.maximum_sensor_age)?;
        ensure!(
            fall.received.is_fresh(now, p.maximum_sensor_age),
            "fall estimate unavailable or stale"
        );
        Ok(Frame {
            fall: fall.received.message,
            command,
            position,
            limits,
            hardware,
        })
    }
}

pub(super) async fn run(ctx: Arc<Context>) -> Result<()> {
    let node = ctx.create_node("motion").build().await?;
    let parameters = node.bind_parameter_as::<Parameters>("motion")?;
    parameters.add_validation_hook(Parameters::validate)?;
    let qos = QosProfile {
        reliability: QosReliability::BestEffort,
        history: QosHistory::from_depth(1),
        ..Default::default()
    };
    let inputs = Inputs::new(&node, qos).await?;
    let outputs = node
        .publisher::<RobotCommand>(ROBOT_COMMAND_TOPIC)
        .qos(qos)
        .build()
        .await?;
    let statuses = node
        .publisher::<MotionExecution>(MOTION_EXECUTION_TOPIC)
        .qos(qos)
        .build()
        .await?;
    let mut motion = MotionState {
        inference_client: node
            .service_client::<InferenceService>(INFERENCE_SERVICE)
            .qos(qos)
            .build()
            .await?,
        head_motion_client: node
            .service_client::<HeadMotionService>(HEAD_MOTION_SERVICE_TOPIC)
            .build()
            .await?,
        generation: node.clock().now().as_nanos() as u64,
        active: false,
        last_policy: None,
        last_arms: TimeWrapper {
            time: node.clock().now(),
            inner: UpperBodyJoints::fill(0.0),
        },
    };
    let mut safety = ControlSafety::default();
    let mut timer = node.create_timer(Duration::from_millis(20));
    loop {
        timer.tick().await;
        let p = parameters.snapshot();
        let command = cycle(&node, &inputs, &mut motion, &mut safety, p.typed()).await;
        outputs.publish(&command).await?;
        statuses
            .publish(&safety.status(&command, &motion, node.clock().now()))
            .await?;
    }
}

#[derive(Default)]
struct ControlSafety {
    fault: Option<String>,
    saw_damping: bool,
    has_actuated: bool,
    recovery: Recovery,
}
impl ControlSafety {
    fn stop(&mut self, motion: &mut MotionState) {
        motion.deactivate();
        self.recovery.reset();
    }
    fn status(&self, command: &RobotCommand, motion: &MotionState, now: Time) -> MotionExecution {
        let phase = if self.fault.is_some() {
            MotionPhase::Fault
        } else {
            match command {
                RobotCommand::Damping => MotionPhase::Damping,
                RobotCommand::Prepare => MotionPhase::Preparing,
                _ => self.recovery.phase(),
            }
        };
        let execution = self.recovery.execution();
        MotionExecution {
            time: now,
            generation: motion.generation,
            phase,
            recovery_started_at: execution.map(|e| e.started_at),
            recovery_progress: execution.and_then(|e| e.progress),
            fault: self.fault.clone(),
        }
    }

    fn fail(&mut self, error: impl std::fmt::Display) {
        if self.fault.is_none() {
            error!("motion safety fault: {error}");
            self.fault = Some(error.to_string());
            self.saw_damping = false;
        }
    }
    fn rearm(&mut self, command: &MotionCommand) -> bool {
        match command {
            MotionCommand::Damping => {
                self.saw_damping = true;
                false
            }
            MotionCommand::Prepare if self.saw_damping => {
                self.fault = None;
                self.saw_damping = false;
                self.has_actuated = false;
                true
            }
            _ => false,
        }
    }
}

async fn cycle(
    node: &Node,
    inputs: &Inputs,
    motion: &mut MotionState,
    safety: &mut ControlSafety,
    p: &Parameters,
) -> RobotCommand {
    let now = node.clock().now();
    if let Ok(request) = inputs.commands.fresh(now, p.maximum_command_age)
        && matches!(request.received.message, MotionCommand::Damping)
    {
        if safety.has_actuated
            && let Err(error) = inputs.frame(now, p)
        {
            safety.fail(error);
        }
        if safety.fault.is_some() {
            safety.rearm(&request.received);
        }
        safety.stop(motion);
        return RobotCommand::Damping;
    }
    let frame = match inputs.frame(now, p) {
        Ok(frame) => frame,
        Err(error) => {
            if safety.has_actuated {
                safety.fail(error);
            }
            safety.stop(motion);
            return RobotCommand::Damping;
        }
    };
    if safety.fault.is_some() && !safety.rearm(&frame.command.received) {
        safety.stop(motion);
        return RobotCommand::Damping;
    }
    if matches!(frame.command.received.message, MotionCommand::Prepare) {
        safety.stop(motion);
        return RobotCommand::Prepare;
    }
    match run_policy(node, inputs, motion, safety, &frame, p).await {
        Ok(command) => {
            safety.has_actuated |= matches!(command, RobotCommand::Custom { .. });
            command
        }
        Err(error) => {
            safety.fail(error);
            safety.stop(motion);
            RobotCommand::Damping
        }
    }
}

async fn run_policy(
    node: &Node,
    inputs: &Inputs,
    motion: &mut MotionState,
    safety: &mut ControlSafety,
    frame: &Frame,
    p: &Parameters,
) -> Result<RobotCommand> {
    if let Some(fault) = &frame.hardware.received.fault {
        return Err(eyre!("hardware fault: {fault}"));
    }
    let initialized = inputs
        .inference
        .latest()
        .is_some_and(|s| !matches!(s.received.state, motion_inference::node::State::Idle));
    if !initialized {
        safety.stop(motion);
        return Ok(RobotCommand::Damping);
    }
    let previous_phase = safety.recovery.phase();
    let plan = safety.recovery.select(
        &frame.command.received,
        frame.command.received.source_time,
        &frame.fall,
        node.clock().now(),
        p,
    )?;
    if previous_phase != safety.recovery.phase() && safety.recovery.phase() != MotionPhase::Normal {
        motion.deactivate();
    }
    if matches!(plan, MotionPlan::Damping) {
        safety.stop(motion);
        return Ok(RobotCommand::Damping);
    }
    if frame.hardware.received.acknowledged != Some(ControlMode::Custom)
        || frame.hardware.received.desired != ControlMode::Custom
    {
        ensure!(!motion.active, "lost Custom mode acknowledgement");
        return Ok(RobotCommand::EnableCustom);
    }
    if !motion.active {
        motion.last_arms = TimeWrapper {
            time: node.clock().now(),
            inner: frame.position.upper_body_as_ref().map(|v| *v),
        };
    }
    infer_and_validate(node, inputs, motion, safety, frame, plan, p).await
}

async fn infer_and_validate(
    node: &Node,
    inputs: &Inputs,
    motion: &mut MotionState,
    safety: &mut ControlSafety,
    frame: &Frame,
    plan: MotionPlan,
    p: &Parameters,
) -> Result<RobotCommand> {
    let command = motion
        .infer(plan, node.clock(), p, &frame.limits.received)
        .await?;
    let now = node.clock().now();
    let fresh = inputs.frame(now, p)?;
    if matches!(
        fresh.command.received.message,
        MotionCommand::Damping | MotionCommand::Prepare
    ) {
        safety.stop(motion);
        return Ok(
            if matches!(fresh.command.received.message, MotionCommand::Prepare) {
                RobotCommand::Prepare
            } else {
                RobotCommand::Damping
            },
        );
    }
    ensure!(
        fresh.hardware.received.fault.is_none()
            && fresh.hardware.received.acknowledged == Some(ControlMode::Custom)
            && fresh.hardware.received.desired == ControlMode::Custom,
        "hardware no longer authorizes Custom"
    );
    if !safety.recovery.allows_output(&fresh.fall) {
        safety.stop(motion);
        return Ok(RobotCommand::Damping);
    }
    let command = command.clamp(&fresh.limits.received)?;
    if let Some(execution) = motion.last_policy {
        safety.recovery.observe(execution, now);
    }
    Ok(command)
}

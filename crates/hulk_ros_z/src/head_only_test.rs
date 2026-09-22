//! Seated bench experiment with the production head controller and actuator path.
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use clap::ValueEnum;
use color_eyre::{
    Result,
    eyre::{WrapErr, bail, ensure, eyre},
};
use ros_z::{
    prelude::*,
    qos::{QosHistory, QosReliability},
};
use serde_json::json;
use tokio::{sync::oneshot, task::JoinSet};
use types::{
    hardware_status::{ControlMode, HARDWARE_STATUS_TOPIC, HardwareStatus},
    motion_command::{HeadMotion, MotionCommand},
    motion_execution::{MOTION_EXECUTION_TOPIC, MotionExecution, MotionPhase},
};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Pattern {
    Zero,
    Hold,
    /// Left -> center -> right -> center, with pitch changes (LookAround).
    Scan,
    LookAround,
    /// Original center-once, then left/right lost-ball search.
    Search,
}

impl Pattern {
    fn head_motion(self) -> HeadMotion {
        match self {
            Self::Zero | Self::Hold => HeadMotion::ZeroAngles,
            Self::Scan | Self::LookAround => HeadMotion::LookAround,
            Self::Search => HeadMotion::SearchForLostBall,
        }
    }
}

#[derive(Debug, clap::Args)]
pub struct Args {
    /// Run only head motion, with every body joint damped. Robot must be supported.
    #[arg(long, value_enum)]
    pub head_only_test: Option<Pattern>,
    /// Active test duration (seconds), starting after head control becomes active.
    #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(u64).range(1..=600))]
    head_test_seconds: u64,
    /// Fixed hold yaw, in radians; used only by the hold pattern.
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    head_test_yaw: f32,
    /// Fixed hold pitch, in radians; used only by the hold pattern.
    #[arg(long, default_value_t = 0.7, allow_hyphen_values = true)]
    head_test_pitch: f32,
    /// Head evaluation/output rate for this experiment (50, 100, or 200 Hz).
    #[arg(long, requires = "head_only_test", value_parser = parse_rate)]
    head_test_rate_hz: Option<u32>,
    /// Override only the active yaw proportional gain for this experiment.
    #[arg(long, requires = "head_only_test", value_parser = parse_gain)]
    head_test_yaw_kp: Option<f32>,
    /// Override only the active yaw derivative gain for this experiment.
    #[arg(long, requires = "head_only_test", value_parser = parse_gain)]
    head_test_yaw_kd: Option<f32>,
    /// Override only the active pitch proportional gain for this experiment.
    #[arg(long, requires = "head_only_test", value_parser = parse_gain)]
    head_test_pitch_kp: Option<f32>,
    /// Override only the active pitch derivative gain for this experiment.
    #[arg(long, requires = "head_only_test", value_parser = parse_gain)]
    head_test_pitch_kd: Option<f32>,
}

fn parse_rate(value: &str) -> Result<u32, String> {
    match value.parse::<u32>() {
        Ok(rate @ (50 | 100 | 200)) => Ok(rate),
        _ => Err("rate must be 50, 100, or 200 Hz".into()),
    }
}

fn parse_gain(value: &str) -> Result<f32, String> {
    let gain: f32 = value.parse().map_err(|_| "gain must be a number")?;
    if !gain.is_finite() || gain < 0.0 {
        return Err("gain must be finite and nonnegative".into());
    }
    Ok(gain)
}

pub fn prepare(
    args: &Args,
    log: &Path,
    layers: &[PathBuf],
    hardware_id: &str,
    namespace: &str,
) -> Result<Vec<PathBuf>> {
    ensure!(
        args.head_test_yaw.is_finite() && args.head_test_pitch.is_finite(),
        "head target must be finite"
    );
    if let Some(parent) = log.parent() {
        fs::create_dir_all(parent)?;
    }
    // A new directory prevents overwriting a previous experiment.
    fs::create_dir(log).wrap_err("test log directory must not already exist")?;
    let mut snapshots = Vec::new();
    for (index, layer) in layers.iter().enumerate() {
        let target = log.join(format!("parameters/{index}"));
        fs::create_dir_all(&target)?;
        for key in [
            "global",
            "motion",
            "hardware_interface",
            "head_motion",
            "mcap_recorder",
        ] {
            let filename = format!("{key}.json5");
            let source = layer.join(&filename);
            if source.exists() {
                fs::copy(source, target.join(filename))?;
            }
        }
        snapshots.push(target);
    }
    let overrides = log.join("parameters/test");
    fs::create_dir_all(&overrides)?;
    let injected = match args.head_only_test {
        Some(Pattern::Hold) => json!({"yaw": args.head_test_yaw, "pitch": args.head_test_pitch}),
        _ => serde_json::Value::Null,
    };
    let mut head_overrides = json!({"injected_head_joints": injected});
    let mut gain_overrides = json!({});
    for (joint, gain, value) in [
        ("yaw", "kp", args.head_test_yaw_kp),
        ("yaw", "kd", args.head_test_yaw_kd),
        ("pitch", "kp", args.head_test_pitch_kp),
        ("pitch", "kd", args.head_test_pitch_kd),
    ] {
        if let Some(value) = value {
            ensure!(value.is_finite() && value >= 0.0, "invalid {joint} {gain}");
            gain_overrides[gain][joint] = json!(value);
        }
    }
    if gain_overrides != json!({}) {
        head_overrides["joint_control"] = gain_overrides.clone();
    }
    fs::write(
        overrides.join("head_motion.json5"),
        serde_json::to_vec_pretty(&head_overrides)?,
    )?;
    if let Some(rate) = args.head_test_rate_hz {
        ensure!([50, 100, 200].contains(&rate), "unsupported head rate");
        let period = json!({"secs": 0, "nanos": 1_000_000_000 / rate});
        for (key, field) in [
            ("motion", "control_period"),
            ("hardware_interface", "joint_control_message_interval"),
        ] {
            fs::write(
                overrides.join(format!("{key}.json5")),
                serde_json::to_vec_pretty(&json!({field: period}))?,
            )?;
        }
    }
    let recording = mcap_recorder::McapRecorderParameters {
        enable: true,
        max_duration: None,
        include_raw_images: false,
        raw_image_min_interval: None,
        queue_depth: 1024,
        schema_discovery_timeout: Duration::from_secs(5),
        topics: [
            "inputs/low_state",
            "joint_limits",
            "behavior/motion_command",
            motion::ROBOT_COMMAND_TOPIC,
            MOTION_EXECUTION_TOPIC,
            HARDWARE_STATUS_TOPIC,
            "hardware_interface/joint_command",
            hardware_interface::COMMAND_TIMING_TOPIC,
            head_motion::diagnostics::TOPIC,
            motion::TIMING_TOPIC,
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
    };
    fs::write(
        overrides.join("mcap_recorder.json5"),
        serde_json::to_vec_pretty(&recording)?,
    )?;
    fs::write(
        log.join("experiment.json"),
        serde_json::to_vec_pretty(&json!({
            "mode": "head_only", "pattern": format!("{:?}", args.head_only_test),
            "head_motion": args.head_only_test.map(Pattern::head_motion),
            "seconds": args.head_test_seconds, "yaw": args.head_test_yaw, "pitch": args.head_test_pitch,
            "gain_overrides": gain_overrides,
            "rate_hz_override": args.head_test_rate_hz,
            "hardware_id": hardware_id, "namespace": namespace,
            "arguments": std::env::args().collect::<Vec<_>>(), "source_parameter_layers": layers,
            "body": {"kp": 0, "kd": 1, "velocity": 0, "torque": 0},
        }))?,
    )?;
    snapshots.push(overrides);
    Ok(snapshots)
}

pub async fn run(ctx: Arc<Context>, args: &Args, log: PathBuf, namespace: &str) -> Result<()> {
    let pattern = args
        .head_only_test
        .ok_or_else(|| eyre!("head-only pattern missing"))?;
    let head = pattern.head_motion();
    let node = ctx.create_node("head_only_test").build().await?;
    let qos = QosProfile {
        history: QosHistory::from_depth(1),
        reliability: QosReliability::BestEffort,
        ..Default::default()
    };
    let commands = node
        .publisher::<MotionCommand>("behavior/motion_command")
        .qos(qos)
        .build()
        .await?;
    let hardware = node
        .subscriber::<HardwareStatus>(HARDWARE_STATUS_TOPIC)
        .qos(qos)
        .build()
        .await?;
    let execution = node
        .subscriber::<MotionExecution>(MOTION_EXECUTION_TOPIC)
        .qos(qos)
        .build()
        .await?;
    // Register shutdown before any actuator task starts.
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let mut tasks = JoinSet::new();
    tasks.spawn(global_parameter_provider::run_boxed(ctx.clone()));
    tasks.spawn(low_state_bridge::run_boxed(ctx.clone()));
    tasks.spawn(head_motion::node::run_boxed(ctx.clone()));
    tasks.spawn(motion::run_head_only_boxed(ctx.clone()));
    tasks.spawn(hardware_interface::run_boxed(ctx.clone()));
    let (ready_tx, mut ready_rx) = oneshot::channel();
    let (stop_tx, stop_rx) = oneshot::channel();
    let mut recorder = tokio::spawn(mcap_recorder::run_controlled(
        ctx.clone(),
        log.clone(),
        ready_tx,
        stop_rx,
    ));
    let mut recorder_finished = false;
    let mut recording_ready = false;
    let mut hardware_state: Option<HardwareStatus> = None;
    let mut execution_state: Option<MotionExecution> = None;
    let mut arming = false;
    let mut active_since = None;
    let startup = Instant::now();
    let mut timer = tokio::time::interval(Duration::from_millis(20));
    timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    eprintln!(
        "Head-only {pattern:?}: body damping, {} s; Ctrl-C stops. Recording: {}",
        args.head_test_seconds,
        log.display()
    );
    let mut stop_reason = "error";
    let result: Result<()> = async {
        loop {
            tokio::select! {
                result = tokio::signal::ctrl_c() => { result?; stop_reason = "ctrl_c"; break; },
                _ = terminate.recv() => { stop_reason = "sigterm"; break; },
                result = tasks.join_next() => bail!("head test node stopped: {result:?}"),
                result = &mut recorder => {
                    recorder_finished = true;
                    result??;
                    bail!("recorder ended before the test completed");
                },
                result = &mut ready_rx, if !recording_ready => {
                    result.wrap_err("recorder failed before becoming ready")?;
                    recording_ready = true;
                },
                status = hardware.recv() => { hardware_state = Some(status?); },
                status = execution.recv() => { execution_state = Some(status?); },
                _ = timer.tick() => {
                    let now = node.clock().now();
                    for topic in ["behavior/motion_command", motion::ROBOT_COMMAND_TOPIC] {
                        ensure!(node.graph().lock().publishers_on(&format!("{namespace}/{topic}")).count() <= 1,
                            "competing publisher on {topic}; stop the regular HULK service");
                    }
                    if let Some(status) = &hardware_state { ensure!(status.fault.is_none(), "hardware fault: {:?}", status.fault); }
                    if let Some(status) = &execution_state { ensure!(status.fault.is_none(), "motion fault: {:?}", status.fault); }
                    if active_since.is_none() { ensure!(startup.elapsed() < Duration::from_secs(30), "head test startup timed out"); }
                    if arming {
                        ensure!(hardware_state.as_ref().is_some_and(|s| s.time <= now && now.duration_since(s.time) < Duration::from_millis(200)), "hardware status stopped");
                        ensure!(execution_state.as_ref().is_some_and(|s| s.is_fresh(now)), "motion status stopped");
                    }
                    if !arming && recording_ready && startup.elapsed() >= Duration::from_secs(1) {
                        arming = hardware_state.as_ref().is_some_and(|s| s.desired == ControlMode::Damping && s.acknowledged == Some(ControlMode::Damping) && s.time <= now && now.duration_since(s.time) < Duration::from_millis(100))
                            && execution_state.as_ref().is_some_and(|s| s.phase == MotionPhase::Damping && s.is_fresh(now));
                    }
                    if active_since.is_none() && arming && execution_state.as_ref().is_some_and(|s| s.phase == MotionPhase::HeadOnly && s.is_fresh(now)) {
                        active_since = Some(Instant::now());
                        eprintln!("Head control active; timed recording started.");
                    }
                    if active_since.is_some_and(|start| start.elapsed() >= Duration::from_secs(args.head_test_seconds)) { stop_reason = "completed"; break; }
                    commands.publish(&if arming { MotionCommand::HeadOnly { head } } else { MotionCommand::Damping }).await?;
                }
            }
        }
        Ok(())
    }.await;

    let active_seconds = active_since.map(|start| start.elapsed().as_secs_f64());
    // Keep the actuator and recording alive long enough to observe the stop.
    eprintln!("Stopping head test and requesting firmware damping...");
    for _ in 0..25 {
        let _ = commands.publish(&MotionCommand::Damping).await;
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    tasks.abort_all();
    while tasks.join_next().await.is_some() {}
    // Also works if Motion or the actuator exited before seeing the stop command.
    let damping_result = hardware_interface::request_damping(&ctx).await;
    let _ = stop_tx.send(());
    let recording_result = if recorder_finished {
        Ok(())
    } else {
        match tokio::time::timeout(Duration::from_secs(10), &mut recorder).await {
            Ok(joined) => joined.wrap_err("recorder task failed").and_then(|r| r),
            Err(error) => {
                recorder.abort();
                Err(error.into())
            }
        }
    };
    fs::write(
        log.join("result.json"),
        serde_json::to_vec_pretty(&json!({
            "test_error": result.as_ref().err().map(|e| format!("{e:#}")),
            "damping_error": damping_result.as_ref().err().map(|e| format!("{e:#}")),
            "recording_error": recording_result.as_ref().err().map(|e| format!("{e:#}")),
            "active_seconds": active_seconds, "stop_reason": stop_reason,
        }))?,
    )?;
    damping_result.wrap_err("firmware damping was not acknowledged")?;
    recording_result?;
    result
}

#[cfg(test)]
mod tests;

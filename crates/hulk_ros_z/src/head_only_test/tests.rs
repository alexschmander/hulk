use super::*;
use booster::{LowCommand, LowState, MotorState, RpcReqMsg, RpcRespMsg};
use cdr::{CdrLe, Infinite};
use kinematics::joints::Joints;
use ros2::sensor_msgs::joint_state::JointState;
use std::{collections::HashSet, sync::Mutex, time::SystemTime};

// Exercise the complete launch path against an isolated fake SDK transport. No robot
// network, inference model, walking readiness, or fall estimate is available.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn timed_seated_test_records_and_returns_firmware_to_damping() {
    let temp = tempfile::tempdir().unwrap();
    let log = temp.path().join("experiment");
    let args = Args {
        head_only_test: Some(Pattern::Scan),
        head_test_seconds: 1,
        head_test_yaw: 0.0,
        head_test_pitch: 0.7,
    };
    let layers = prepare(
        &args,
        &log,
        &[Path::new(env!("CARGO_MANIFEST_DIR")).join("../../etc/parameters/base")],
        "fake",
        "/bench",
    )
    .unwrap();
    let ctx = Arc::new(
        ContextBuilder::default()
            .with_mode("peer")
            .disable_multicast_scouting()
            .with_namespace("/bench")
            .with_parameter_layers(layers)
            .build()
            .await
            .unwrap(),
    );
    let rpc_requests = ctx
        .session()
        .declare_subscriber("rt/LocoApiTopicReq")
        .await
        .unwrap();
    let rpc_responses = ctx
        .session()
        .declare_publisher("rt/LocoApiTopicResp")
        .await
        .unwrap();
    let commands = ctx
        .session()
        .declare_subscriber("rt/joint_ctrl")
        .await
        .unwrap();
    let low_states = ctx
        .session()
        .declare_publisher("rt/low_state")
        .await
        .unwrap();
    let joint_states = ctx
        .session()
        .declare_publisher("rt/joint_states")
        .await
        .unwrap();
    let modes = Arc::new(Mutex::new(Vec::new()));
    let sent = Arc::new(Mutex::new(Vec::<LowCommand>::new()));
    let mut fake = JoinSet::new();
    let seen_modes = modes.clone();
    fake.spawn(async move {
        loop {
            let sample = rpc_requests.recv_async().await.unwrap();
            let request: RpcReqMsg = cdr::deserialize(&sample.payload().to_bytes()).unwrap();
            let header: serde_json::Value = serde_json::from_str(&request.header).unwrap();
            let body: serde_json::Value = serde_json::from_str(&request.body).unwrap();
            assert_eq!(header["api_id"], 2000, "only mode RPCs are allowed");
            let mode = body["mode"].as_i64().unwrap();
            assert!(matches!(mode, 0 | 3), "Prepare and Walking are forbidden");
            seen_modes.lock().unwrap().push(mode);
            let response = RpcRespMsg {
                uuid: request.uuid,
                header: r#"{"status":0}"#.into(),
                body: String::new(),
            };
            rpc_responses
                .put(cdr::serialize::<_, _, CdrLe>(&response, Infinite).unwrap())
                .await
                .unwrap();
        }
    });
    let seen_commands = sent.clone();
    fake.spawn(async move {
        loop {
            let sample = commands.recv_async().await.unwrap();
            let command: LowCommand = cdr::deserialize(&sample.payload().to_bytes()).unwrap();
            seen_commands.lock().unwrap().push(command);
        }
    });
    fake.spawn(async move {
        let mut low = LowState {
            motor_state_serial: Joints::fill(MotorState::default()).into_iter().collect(),
            ..Default::default()
        };
        low.imu_state.roll_pitch_yaw.inner.x = 0.8;
        let low_bytes = cdr::serialize::<_, _, CdrLe>(&low, Infinite).unwrap();
        let mut joints = JointState {
            position: vec![0.0; low.motor_state_serial.len()],
            velocity: vec![0.0; low.motor_state_serial.len()],
            effort: vec![0.0; low.motor_state_serial.len()],
            ..Default::default()
        };
        loop {
            joints.header.stamp = SystemTime::now().into();
            joint_states
                .put(cdr::serialize::<_, _, CdrLe>(&joints, Infinite).unwrap())
                .await
                .unwrap();
            low_states.put(low_bytes.clone()).await.unwrap();
            tokio::time::sleep(Duration::from_millis(4)).await;
        }
    });
    tokio::time::timeout(
        Duration::from_secs(15),
        run(ctx.clone(), &args, log.clone(), "/bench"),
    )
    .await
    .unwrap()
    .unwrap();
    // Give the raw command subscriber one final scheduling opportunity.
    tokio::task::yield_now().await;
    assert!(fake.try_join_next().is_none(), "fake transport task failed");
    fake.abort_all();
    while fake.join_next().await.is_some() {}
    let modes = modes.lock().unwrap();
    assert!(modes.contains(&3));
    assert_eq!(modes.last(), Some(&0));
    let sent = sent.lock().unwrap();
    assert!(sent.iter().any(|c| c.motor_commands[0].kp > 0.0));
    for command in sent.iter() {
        for motor in command.motor_commands.iter().skip(2) {
            assert_eq!(
                (motor.kp, motor.kd, motor.velocity, motor.torque),
                (0.0, 1.0, 0.0, 0.0)
            );
        }
    }
    assert!(
        sent.last()
            .unwrap()
            .motor_commands
            .iter()
            .all(|m| m.kp == 0.0)
    );
    let bytes = fs::read(log.join("recording.mcap")).unwrap();
    let summary = mcap::Summary::read(&bytes)
        .unwrap()
        .expect("MCAP must be finalized");
    let topics: HashSet<_> = summary
        .channels
        .values()
        .map(|c| c.topic.as_str())
        .collect();
    for topic in [
        "inputs/low_state",
        "commands/robot_command",
        "head_motion/diagnostics",
        "hardware_interface/command_timing",
        "hardware_interface/joint_command",
    ] {
        assert!(
            topics.contains(topic),
            "missing recording topic {topic}: {topics:?}"
        );
    }
    assert!(
        mcap::MessageStream::new(&bytes)
            .unwrap()
            .all(|message| message.is_ok())
    );
    let result: serde_json::Value =
        serde_json::from_slice(&fs::read(log.join("result.json")).unwrap()).unwrap();
    assert!(result["test_error"].is_null());
    assert!(result["damping_error"].is_null());
    assert!(result["recording_error"].is_null());
    ctx.shutdown().unwrap();
}

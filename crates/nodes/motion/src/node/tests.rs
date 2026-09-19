use super::*;
use booster::MotorState;
use kinematics::joints::head::HeadJoints;
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};
use tokio::{
    sync::watch,
    task::JoinSet,
    time::{sleep, timeout},
};
use types::motor_command::MotorCommand;

struct Bench {
    tasks: JoinSet<()>,
    request: watch::Sender<MotionCommand>,
    sensors_enabled: Arc<AtomicBool>,
    commands_enabled: Arc<AtomicBool>,
    custom_acknowledged: Arc<AtomicBool>,
    head_replies: Arc<AtomicBool>,
    outputs: Subscriber<RobotCommand>,
    statuses: Subscriber<MotionExecution>,
}

impl Bench {
    async fn new(head_only: bool) -> Self {
        #[derive(Deserialize)]
        struct Global {
            joint_limits: JointLimits,
        }
        let global: Global = json5::from_str(include_str!(
            "../../../../../etc/parameters/base/global.json5"
        ))
        .unwrap();
        let ctx = Arc::new(
            ContextBuilder::default()
                .with_mode("peer")
                .disable_multicast_scouting()
                .with_parameter_layer(
                    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../etc/parameters/base"),
                )
                .build()
                .await
                .unwrap(),
        );
        let node = ctx.create_node("bench_test").build().await.unwrap();
        let qos = QosProfile {
            reliability: QosReliability::BestEffort,
            history: QosHistory::from_depth(1),
            ..Default::default()
        };
        let commands = node
            .publisher::<MotionCommand>("behavior/motion_command")
            .qos(qos)
            .build()
            .await
            .unwrap();
        let sensors = node
            .publisher::<LowState>("inputs/low_state")
            .qos(qos)
            .build()
            .await
            .unwrap();
        let hardware = node
            .publisher::<HardwareStatus>(HARDWARE_STATUS_TOPIC)
            .qos(qos)
            .build()
            .await
            .unwrap();
        let limits = node
            .publisher::<JointLimits>("joint_limits")
            .qos(QosProfile {
                durability: QosDurability::TransientLocal,
                ..qos
            })
            .build()
            .await
            .unwrap();
        limits.publish(&global.joint_limits).await.unwrap();
        let outputs = node
            .subscriber::<RobotCommand>(ROBOT_COMMAND_TOPIC)
            .qos(qos)
            .build()
            .await
            .unwrap();
        let statuses = node
            .subscriber::<MotionExecution>(MOTION_EXECUTION_TOPIC)
            .qos(qos)
            .build()
            .await
            .unwrap();
        let mut head = node
            .service_server::<HeadMotionService>(HEAD_MOTION_SERVICE_TOPIC)
            .build()
            .await
            .unwrap();
        let (request, requests) = watch::channel(MotionCommand::Damping);
        let sensors_enabled = Arc::new(AtomicBool::new(true));
        let commands_enabled = Arc::new(AtomicBool::new(true));
        let custom_acknowledged = Arc::new(AtomicBool::new(false));
        let head_replies = Arc::new(AtomicBool::new(true));
        let mut tasks = JoinSet::new();
        let enabled = head_replies.clone();
        tasks.spawn(async move {
            while let Ok(request) = head.take_request_async().await {
                if enabled.load(Ordering::Relaxed) {
                    request
                        .into_parts()
                        .1
                        .reply_async(&HeadJoints::fill(MotorCommand {
                            position: 0.3,
                            velocity: 0.1,
                            torque: 0.0,
                            kp: 10.0,
                            kd: 1.2,
                        }))
                        .await
                        .unwrap();
                }
            }
        });
        let sensor_flag = sensors_enabled.clone();
        let command_flag = commands_enabled.clone();
        let ack = custom_acknowledged.clone();
        tasks.spawn(async move {
            let mut sample = LowState {
                motor_state_serial: Joints::fill(MotorState::default()).into_iter().collect(),
                ..Default::default()
            };
            // Explicitly non-upright: no ready-for-walk gate is needed in the bench runtime.
            sample.imu_state.roll_pitch_yaw = vector![0.8, 0.6, 0.0];
            loop {
                let now = node.clock().now();
                if sensor_flag.load(Ordering::Relaxed) {
                    sensors.publish(&sample).await.unwrap();
                }
                if command_flag.load(Ordering::Relaxed) {
                    let command = requests.borrow().clone();
                    commands.publish(&command).await.unwrap();
                }
                let mode = if ack.load(Ordering::Relaxed) {
                    ControlMode::Custom
                } else {
                    ControlMode::Damping
                };
                hardware
                    .publish(&HardwareStatus {
                        time: now,
                        desired: mode,
                        acknowledged: Some(mode),
                        command_time: Some(now),
                        fault: None,
                    })
                    .await
                    .unwrap();
                sleep(Duration::from_millis(5)).await;
            }
        });
        tasks.spawn(async move {
            run(ctx, head_only).await.unwrap();
        });
        let bench = Self {
            tasks,
            request,
            sensors_enabled,
            commands_enabled,
            custom_acknowledged,
            head_replies,
            outputs,
            statuses,
        };
        bench.wait_phase(MotionPhase::Damping).await;
        bench
    }

    async fn wait_phase(&self, phase: MotionPhase) -> MotionExecution {
        timeout(Duration::from_secs(3), async {
            loop {
                let status = self.statuses.recv().await.unwrap();
                if status.phase == phase {
                    return status;
                }
                assert!(
                    status.fault.is_none(),
                    "unexpected fault: {:?}",
                    status.fault
                );
            }
        })
        .await
        .expect("motion phase did not arrive")
    }

    async fn start(&self) {
        self.request.send_replace(MotionCommand::HeadOnly {
            head: HeadMotion::SearchForLostBall,
        });
        timeout(Duration::from_secs(3), async {
            loop {
                match self.outputs.recv().await.unwrap() {
                    RobotCommand::EnableCustom => break,
                    RobotCommand::Damping => {}
                    _ => panic!("active output before hardware acknowledged Custom"),
                }
            }
        })
        .await
        .unwrap();
        self.custom_acknowledged.store(true, Ordering::Relaxed);
        self.wait_phase(MotionPhase::HeadOnly).await;
    }

    async fn assert_damping_latched(&self) {
        self.wait_phase(MotionPhase::Fault).await;
        // Drain the output associated with the status and verify several later cycles.
        for _ in 0..5 {
            assert!(matches!(
                self.outputs.recv().await.unwrap(),
                RobotCommand::Damping
            ));
        }
        self.request.send_replace(MotionCommand::Damping);
        sleep(Duration::from_millis(40)).await;
        self.request.send_replace(MotionCommand::Prepare);
        self.wait_phase(MotionPhase::Fault).await;
        assert!(matches!(
            self.outputs.recv().await.unwrap(),
            RobotCommand::Damping
        ));
    }
}

impl Drop for Bench {
    fn drop(&mut self) {
        self.tasks.abort_all();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seated_head_only_requires_custom_ack_and_damps_every_body_joint() {
    let bench = Bench::new(true).await;
    bench.start().await;
    for _ in 0..5 {
        let RobotCommand::Custom { joints_command } = bench.outputs.recv().await.unwrap() else {
            panic!("expected head command");
        };
        assert_eq!(joints_command.head.yaw.kp, 10.0);
        assert_eq!(joints_command.head.pitch.position, 0.3);
        for (joint, motor) in joints_command.enumerate() {
            if matches!(joint, kinematics::joints::JointsName::Head(_)) {
                continue;
            }
            assert_eq!(
                (motor.kp, motor.kd, motor.velocity, motor.torque),
                (0.0, 1.0, 0.0, 0.0)
            );
        }
    }
    bench.request.send_replace(MotionCommand::Damping);
    bench.wait_phase(MotionPhase::Damping).await;
    assert!(matches!(
        bench.outputs.recv().await.unwrap(),
        RobotCommand::Damping
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stale_sensors_latch_head_only_damping() {
    let bench = Bench::new(true).await;
    bench.start().await;
    bench.sensors_enabled.store(false, Ordering::Relaxed);
    bench.assert_damping_latched().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lost_command_lease_latches_head_only_damping() {
    let bench = Bench::new(true).await;
    bench.start().await;
    bench.commands_enabled.store(false, Ordering::Relaxed);
    bench.assert_damping_latched().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn head_service_timeout_latches_damping() {
    let bench = Bench::new(true).await;
    bench.start().await;
    bench.head_replies.store(false, Ordering::Relaxed);
    bench.assert_damping_latched().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn loss_of_custom_ack_latches_damping() {
    let bench = Bench::new(true).await;
    bench.start().await;
    bench.custom_acknowledged.store(false, Ordering::Relaxed);
    bench.assert_damping_latched().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn head_only_runtime_rejects_body_motion() {
    for command in [
        MotionCommand::Prepare,
        MotionCommand::Stand {
            head: HeadMotion::ZeroAngles,
        },
        MotionCommand::StandUp { fast: true },
    ] {
        let bench = Bench::new(true).await;
        bench.request.send_replace(command);
        bench.assert_damping_latched().await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn normal_runtime_rejects_head_only() {
    let bench = Bench::new(false).await;
    bench.request.send_replace(MotionCommand::HeadOnly {
        head: HeadMotion::ZeroAngles,
    });
    let status = bench.wait_phase(MotionPhase::Fault).await;
    assert!(status.fault.unwrap().contains("dedicated head-only"));
    assert!(matches!(
        bench.outputs.recv().await.unwrap(),
        RobotCommand::Damping
    ));
}

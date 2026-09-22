use super::*;
use kinematics::joints::head::HeadJoints;
use motion_inference::config::Policy;
use ros_z::prelude::*;
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

struct Bench {
    motion: MotionState,
    clock: Clock,
    parameters: Parameters,
    limits: JointLimits,
    requests: Arc<Mutex<Vec<(Time, u64)>>>,
    stalled: Arc<AtomicBool>,
    heads: Arc<AtomicUsize>,
    tasks: JoinSet<()>,
    _context: Context,
}

impl Bench {
    async fn new() -> Self {
        let clock = Clock::logical(Time::from_nanos(1_000_000_000));
        let ctx = ContextBuilder::default()
            .with_mode("peer")
            .disable_multicast_scouting()
            .with_clock(clock.clone())
            .build()
            .await
            .unwrap();
        let node = ctx.create_node("body_schedule_test").build().await.unwrap();
        let mut walk = node
            .service_server::<WalkInferenceService>(WALK_INFERENCE_SERVICE)
            .build()
            .await
            .unwrap();
        let mut getup = node
            .service_server::<GetUpInferenceService>(GETUP_INFERENCE_SERVICE)
            .build()
            .await
            .unwrap();
        let mut head = node
            .service_server::<HeadMotionService>(HEAD_MOTION_SERVICE_TOPIC)
            .build()
            .await
            .unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stalled = Arc::new(AtomicBool::new(false));
        let heads = Arc::new(AtomicUsize::new(0));
        let mut tasks = JoinSet::new();
        let seen = requests.clone();
        let blocked = stalled.clone();
        tasks.spawn(async move {
            let mut delayed = Vec::new();
            loop {
                let (request, reply) = walk.take_request_async().await.unwrap().into_parts();
                seen.lock()
                    .unwrap()
                    .push((request.requested_at, request.generation));
                let response = Ok(InferenceResponse {
                    joints: Box::new(LowerBodyJoints::fill(MotorCommand {
                        position: 0.01 * seen.lock().unwrap().len() as f32,
                        velocity: 0.13,
                        torque: 0.27,
                        kp: 20.0,
                        kd: 2.0,
                    })),
                    execution: PolicyExecution {
                        policy: Policy::Walk,
                        started_at: request.requested_at,
                        progress: None,
                        sensor_time: request.requested_at,
                    },
                });
                if blocked.load(Ordering::Relaxed) {
                    // Keep the reply alive: simulate a slow computation, not an immediate error.
                    delayed.push((reply, response));
                } else {
                    for (reply, response) in delayed.drain(..) {
                        let _ = reply.reply_async(&response).await;
                    }
                    reply.reply_async(&response).await.unwrap();
                }
            }
        });
        tasks.spawn(async move {
            loop {
                let (request, reply) = getup.take_request_async().await.unwrap().into_parts();
                reply
                    .reply_async(&Ok(InferenceResponse {
                        joints: Box::new(Joints::fill(MotorCommand {
                            position: 0.42,
                            velocity: 0.0,
                            torque: 0.0,
                            kp: 30.0,
                            kd: 3.0,
                        })),
                        execution: PolicyExecution {
                            policy: if request.command.fast {
                                Policy::FastGetUp
                            } else {
                                Policy::SlowGetUp
                            },
                            started_at: request.requested_at,
                            progress: Some(0.1),
                            sensor_time: request.requested_at,
                        },
                    }))
                    .await
                    .unwrap();
            }
        });
        let count = heads.clone();
        tasks.spawn(async move {
            loop {
                let (_, reply) = head.take_request_async().await.unwrap().into_parts();
                let position = count.fetch_add(1, Ordering::Relaxed) as f32 * 0.001;
                reply
                    .reply_async(&HeadJoints::fill(MotorCommand {
                        position,
                        ..MotorCommand::damping()
                    }))
                    .await
                    .unwrap();
            }
        });
        #[derive(Deserialize)]
        struct Global {
            joint_limits: JointLimits,
        }
        let global: Global = json5::from_str(include_str!(
            "../../../../../etc/parameters/base/global.json5"
        ))
        .unwrap();
        Self {
            motion: node::motion_state(&node, QosProfile::default())
                .await
                .unwrap(),
            clock,
            parameters: json5::from_str(include_str!(
                "../../../../../etc/parameters/base/motion.json5"
            ))
            .unwrap(),
            limits: global.joint_limits,
            requests,
            stalled,
            heads,
            tasks,
            _context: ctx,
        }
    }

    async fn tick(&mut self, millis: u64, plan: MotionPlan) -> Result<RobotCommand> {
        self.clock.advance(Duration::from_millis(millis)).unwrap();
        let result = self
            .motion
            .infer(plan, &self.clock, &self.parameters, &self.limits)
            .await;
        // Let local service replies finish without advancing logical robot time.
        tokio::time::sleep(Duration::from_millis(2)).await;
        result
    }
}

impl Drop for Bench {
    fn drop(&mut self) {
        self.tasks.abort_all();
    }
}

fn walk() -> MotionPlan {
    MotionPlan::Walk {
        command: WalkCommand::stand(),
        head_motion: HeadMotion::LookAround,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn head_rates_hold_complete_body_commands_and_keep_inference_at_50_hz() {
    for period in [20, 10, 5] {
        let mut bench = Bench::new().await;
        let mut previous: Option<(Time, Joints<MotorCommand>)> = None;
        let mut repeated = 0;
        for _ in 0..(200 / period) {
            if let RobotCommand::Custom { joints_command } =
                bench.tick(period, walk()).await.unwrap()
            {
                let body_time = bench.motion.body.cached.as_ref().unwrap().requested_at;
                if let Some((old_time, old)) = previous
                    && old_time == body_time
                {
                    repeated += 1;
                    assert_ne!(old.head.yaw.position, joints_command.head.yaw.position);
                    for (joint, _) in Joints::fill(()).enumerate() {
                        let motor = &joints_command[joint];
                        if matches!(joint, kinematics::joints::JointsName::Head(_)) {
                            continue;
                        }
                        let old = &old[joint];
                        assert_eq!(
                            (
                                motor.position,
                                motor.velocity,
                                motor.torque,
                                motor.kp,
                                motor.kd
                            ),
                            (old.position, old.velocity, old.torque, old.kp, old.kd)
                        );
                    }
                }
                previous = Some((body_time, joints_command));
            }
        }
        assert_eq!(bench.heads.load(Ordering::Relaxed), (200 / period) as usize);
        let requests = bench.requests.lock().unwrap();
        assert_eq!(requests.len(), 10);
        assert!(
            requests
                .windows(2)
                .all(|pair| pair[1].0.duration_since(pair[0].0) == POLICY_PERIOD)
        );
        if period < 20 {
            assert!(repeated >= 200 / period - 11);
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stalled_inference_does_not_block_head_or_freshen_the_held_body() {
    let mut bench = Bench::new().await;
    bench.tick(5, walk()).await.unwrap();
    bench.tick(5, walk()).await.unwrap();
    bench.stalled.store(true, Ordering::Relaxed);
    let body_time = bench.motion.body.cached.as_ref().unwrap().requested_at;
    for _ in 0..6 {
        assert!(matches!(
            bench.tick(5, walk()).await.unwrap(),
            RobotCommand::Custom { .. }
        ));
        assert_eq!(
            bench.motion.body.cached.as_ref().unwrap().requested_at,
            body_time
        );
    }
    assert_eq!(bench.heads.load(Ordering::Relaxed), 8);
    // Live timeout changes must not extend an already issued request's lease.
    bench.parameters.inference_timeout = Duration::from_millis(100);
    assert!(
        bench
            .tick(5, walk())
            .await
            .err()
            .unwrap()
            .to_string()
            .contains("expired")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn switching_policy_discards_pending_body_and_getup_owns_head() {
    let mut bench = Bench::new().await;
    bench.stalled.store(true, Ordering::Relaxed);
    bench.tick(5, walk()).await.unwrap();
    let generation = bench.motion.generation;
    let getup = MotionPlan::GetUp {
        command: GetUpCommand { fast: true },
    };
    assert!(matches!(
        bench.tick(5, getup).await.unwrap(),
        RobotCommand::EnableCustom
    ));
    assert!(bench.motion.generation > generation);
    let RobotCommand::Custom { joints_command } = bench.tick(5, getup).await.unwrap() else {
        panic!("getup result missing");
    };
    assert_eq!(joints_command.head.yaw.position, 0.42);
    assert_eq!(bench.heads.load(Ordering::Relaxed), 1);
    bench.motion.deactivate();
    assert!(bench.motion.body.cached.is_none());
    assert!(bench.motion.body.workers.is_empty());
    bench.stalled.store(false, Ordering::Relaxed);
    // The fake service sends the previous generation's delayed walk reply here too.
    bench.tick(5, walk()).await.unwrap();
    bench.tick(5, walk()).await.unwrap();
    assert_eq!(
        bench.motion.body.cached.as_ref().unwrap().requested_at,
        bench.clock.now() - Duration::from_millis(5)
    );
}

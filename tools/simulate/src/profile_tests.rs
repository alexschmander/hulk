use super::*;
use crate::{
    bevy_mujoco::{MjcfObject, MujocoWorldPlugin},
    robotics::Configuration,
};
use ros_z::prelude::*;
use types::primary_state::PrimaryState;

#[test]
#[ignore = "requires ONNX Runtime, motion models, and free GameController UDP ports"]
fn startup_pause_resume_and_scene_edits_preserve_time() {
    for profile in crate::Profile::ALL {
        if std::env::var("SIMULATOR_TEST_PROFILE").is_ok_and(|requested| {
            requested != serde_json::to_value(profile).unwrap().as_str().unwrap()
        }) {
            continue;
        }
        validate_profile(profile, false);
    }
}

fn validate_profile(profile: crate::Profile, require_acquisition: bool) {
    eprintln!("Validating {}", profile.label());
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let router = runtime
        .block_on(
            ContextBuilder::default()
                .with_mode("router")
                .disable_multicast_scouting()
                .with_connect_endpoints(std::iter::empty::<&str>())
                .with_listen_endpoints(["tcp/127.0.0.1:0"])
                .build(),
        )
        .unwrap();
    let endpoint = runtime
        .block_on(async { router.session().info().locators().await })
        .first()
        .unwrap()
        .to_string();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let io = runtime
        .block_on(Robotics::new(
            runtime.handle().clone(),
            Configuration {
                parameter_root: root.join("../../etc/parameters"),
                model_directory: root.join("../../etc/neural_networks"),
                router: Some(endpoint),
                namespace: "/simulator/startup_test".into(),
                location: None,
                robot_count: 1,
                opponent_count: 0,
                field_configuration: None,
                profile,
                controller: crate::ControllerSource::External,
            },
        ))
        .unwrap();
    let (whistle, reference_camera, camera, reference_pose, pose, detected, ball_output) = runtime.block_on(async {
        let node = io.node();
        (
            node.subscriber::<types::filtered_whistle::FilteredWhistle>("filtered_whistle").cache(32).build().await.unwrap(),
            node.subscriber::<types::time_wrapper::TimeWrapper<projection::camera_matrix::CameraMatrix>>("ground_truth/camera_matrix").cache(128).build().await.unwrap(),
            node.subscriber::<types::time_wrapper::TimeWrapper<projection::camera_matrix::CameraMatrix>>("camera_matrix").cache(128).build().await.unwrap(),
            node.subscriber::<linear_algebra::Isometry2<coordinate_systems::Ground,coordinate_systems::Field>>("ground_truth/ground_to_field").cache(10).build().await.unwrap(),
            node.subscriber::<linear_algebra::Isometry2<coordinate_systems::Ground,coordinate_systems::Field>>("ground_to_field").cache(10).build().await.unwrap(),
            node.subscriber::<types::time_wrapper::TimeWrapper<Vec<types::object_detection::Object<types::object_detection::RobocupObjectLabel>>>>("ground_truth/detected_objects").cache(10).build().await.unwrap(),
            node.subscriber::<Option<types::ball_position::BallPosition<coordinate_systems::Ground>>>("ball_filter/ball_position").cache(10).build().await.unwrap(),
        )
    });
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, MujocoWorldPlugin));
    let robot = app
        .world_mut()
        .spawn((
            MjcfObject::new(root.join("assets/k1_robot.xml"), "Trunk")
                .with_free_joint("world_joint")
                .grounded(),
            Transform::default(),
        ))
        .id();
    app.update();
    let physics = app.world().resource::<SharedPhysics>().clone();
    let mut worker = PhysicsWorker::start(physics.clone(), io.clone());
    let started = Instant::now();
    while !worker.ready() {
        assert!(worker.poll().is_none(), "{}", io.status());
        assert!(
            started.elapsed() < Duration::from_secs(25),
            "{}",
            io.status()
        );
        thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(*io.primary.get_latest().unwrap(), PrimaryState::Initial);
    runtime.block_on(async {
        tokio::time::timeout(Duration::from_secs(5), async {
            while io.led_color()
                != Some(<booster::LedColor as led_handler::DefaultLEDColors>::MAGENTA)
            {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("Initial LED command must reach the simulator");
    });
    assert_eq!(physics.lock().mode, SimulationMode::Paused);
    validate_pause(&runtime, &io, &physics);
    let scene_pose = physics.lock().object_pose(robot).unwrap();
    assert!(scene_pose.translation.y > 0.4);
    assert!((scene_pose.translation.x + io.field_dimensions.length / 2.0).abs() < 0.1);
    assert!((scene_pose.translation.z - io.field_dimensions.width / 2.0).abs() < 0.1);
    assert!((scene_pose.rotation * Vec3::X).distance(Vec3::NEG_Z) < 0.1);
    // Check all production owners through the same graph that Twix observes.
    for topic in [
        "inputs/low_state",
        "inputs/imu_state",
        "inputs/serial_motor_states",
        "robot_kinematics",
        "support_foot_state",
        "robot_to_ground",
        "ground_to_robot",
        "camera_matrix",
        "camera_geometry",
        "inputs/odometry",
        "ground_to_field",
        "ball_filter/ball_position",
        "visual_kick/ball_position",
        "obstacles",
    ] {
        let name = format!("/simulator/startup_test/{topic}");
        let graph = io.node().graph().lock();
        assert_eq!(
            graph.publishers_on(&name).count(),
            1,
            "{profile:?}: {topic}"
        );
        if !topic.starts_with("inputs/") || topic == "inputs/odometry" {
            assert_eq!(
                graph
                    .publishers_on(&format!("/simulator/startup_test/ground_truth/{topic}"))
                    .count(),
                1,
                "reference {topic}"
            );
        }
    }
    assert_eq!(
        io.node()
            .graph()
            .lock()
            .publishers_on("/simulator/startup_test/current_odometry_to_last_odometry")
            .count(),
        0
    );
    assert!(reference_pose.get_latest().is_some());
    assert!(reference_camera.get_latest().is_some());
    let predicted = camera.get_latest().unwrap();
    let actual = reference_camera.get_nearest(predicted.time).unwrap();
    let error = (actual.inner.ground_to_camera.inner.translation.vector
        - predicted.inner.ground_to_camera.inner.translation.vector)
        .norm();
    eprintln!("{profile:?}: camera translation difference {error:.4} m");
    assert!(error < 0.05, "camera geometry differs by {error} m");
    physics.lock().mode = SimulationMode::Running;
    if profile == crate::Profile::Localization {
        validate_localization(&runtime, &io, &physics, require_acquisition);
        drop(worker);
        drop(io);
        router.shutdown().unwrap();
        return;
    }
    // The pulse goes through the real whistle filter; idle samples clear it.
    io.whistle();
    let started = Instant::now();
    while !whistle.get_latest().is_some_and(|v| v.is_detected) {
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "whistle was not detected"
        );
        thread::sleep(Duration::from_millis(10));
    }
    let first_detection = whistle.get_latest().unwrap().last_detection;
    runtime.block_on(async {
        tokio::time::timeout(
            Duration::from_secs(5),
            io.node()
                .clock()
                .sleep(crate::whistle::PULSE_DURATION + Duration::from_millis(500)),
        )
        .await
        .unwrap();
    });
    assert!(!whistle.get_latest().unwrap().is_detected);
    io.whistle();
    let started = Instant::now();
    while !whistle
        .get_latest()
        .is_some_and(|v| v.last_detection > first_detection)
    {
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "second whistle was not detected"
        );
        thread::sleep(Duration::from_millis(10));
    }
    eprintln!(
        "{profile:?}: initial truth detections {}",
        detected.get_latest().map_or(0, |v| v.inner.len())
    );
    if !profile.localization() {
        assert!(pose.get_latest().is_some());
    }
    // Place a ball in front of the physical camera, on the field plane.
    let parameters = io.parameters.snapshot().typed().ball.clone();
    let radius = f64::from(io.field_dimensions.ball_radius);
    let ball_entity = app
        .world_mut()
        .spawn((
            MjcfObject::from_factory(
                move || crate::scene::ball::ball_spec(radius, &parameters),
                "ball",
            )
            .with_free_joint("ball_free_joint")
            .grounded(),
            Transform::from_xyz(
                scene_pose.translation.x,
                0.0,
                scene_pose.translation.z - 2.0,
            ),
            crate::scene::ball::Ball,
        ))
        .id();
    app.update();

    let started = Instant::now();
    while !ball_output.get_latest().is_some_and(|v| v.is_some()) {
        assert!(worker.poll().is_none(), "{}", io.status());
        assert!(
            started.elapsed() < Duration::from_secs(8),
            "{profile:?}: ball not acquired; detections {:?}",
            detected.get_latest()
        );
        thread::sleep(Duration::from_millis(20));
    }
    eprintln!("{profile:?}: ball acquired after {:?}", started.elapsed());
    if profile.filtering() {
        validate_filtering(&runtime, &io, &physics, &mut app, scene_pose, &ball_output);
    }
    use types::{
        controller_input::{Axis, Button, ControllerAxis, ControllerButton, ControllerInput},
        motion_command::MotionCommand,
    };
    let controller = runtime
        .block_on(
            io.node()
                .publisher::<ControllerInput>("inputs/controller_input")
                .build(),
        )
        .unwrap();
    let mut input = ControllerInput {
        connected: true,
        device_name: "integration fixture".into(),
        axes: vec![ControllerAxis {
            name: Axis::LeftStickY,
            value: 0.15,
        }],
        buttons: vec![ControllerButton {
            name: Button::Start,
            pressed: true,
            value: 1.0,
        }],
    };
    let started = Instant::now();
    loop {
        runtime
            .block_on(
                controller.publish_with_source_time(&input, ros_z::time::Clock::wallclock().now()),
            )
            .unwrap();
        if matches!(io.active_motion(), MotionCommand::WalkWithVelocity { velocity, .. } if (velocity.x()-0.15).abs() < 1e-5)
        {
            break;
        }
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "external gamepad did not reach behavior: {:?}",
            io.active_motion()
        );
        thread::sleep(Duration::from_millis(20));
    }
    input.buttons.clear();
    runtime
        .block_on(
            controller.publish_with_source_time(&input, ros_z::time::Clock::wallclock().now()),
        )
        .unwrap();
    // Losing the remote source must stop a commanded walk after its 250 ms freshness window.
    thread::sleep(Duration::from_millis(400));
    assert!(matches!(io.active_motion(), MotionCommand::Stand { .. }));
    eprintln!("{profile:?}: remote control and disconnect passed");
    app.world_mut().despawn(ball_entity);
    app.update();
    let started = Instant::now();
    while ball_output.get_latest().is_some_and(|v| v.is_some()) {
        assert!(
            started.elapsed() < Duration::from_secs(25),
            "deleted ball did not expire"
        );
        thread::sleep(Duration::from_millis(20));
    }
    eprintln!(
        "{profile:?}: deleted ball expired after {:?}",
        started.elapsed()
    );
    // Scene recompilation freezes both clocks and sampling, even beyond the
    // 40/50 ms sensor deadlines. Resume must not age the cached sensors.
    let timing = runtime
        .block_on(
            io.node()
                .subscriber::<crate::observations::FrameTiming>("diagnostics/camera_frames")
                .cache(8)
                .build(),
        )
        .unwrap();
    thread::sleep(Duration::from_millis(80));
    let scene_edit = physics.lock();
    thread::sleep(Duration::from_millis(80));
    let captured = timing.get_latest().unwrap().captured;
    let frozen_time = io.now();
    thread::sleep(Duration::from_millis(220));
    assert_eq!(
        timing.get_latest().unwrap().captured,
        captured,
        "scene rebuild republished stale camera geometry"
    );
    assert!(
        io.safe_pose
            .latest_stamp()
            .is_some_and(|stamp| io.now().duration_since(stamp) < Duration::from_millis(50))
    );
    assert_eq!(
        io.now(),
        frozen_time,
        "robotics time advanced during scene rebuild"
    );
    assert_eq!(*io.primary.get_latest().unwrap(), PrimaryState::Initial);
    assert!(!io.status().contains("Emergency stop"));
    drop(scene_edit);
    let start = Instant::now();
    while timing.get_latest().unwrap().captured == captured {
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "camera frames did not resume"
        );
        thread::sleep(Duration::from_millis(10));
    }
    assert!(
        timing
            .get_latest()
            .unwrap()
            .captured
            .duration_since(captured)
            < Duration::from_millis(100),
        "scene rebuild caused a catch-up time jump"
    );
    assert!(worker.poll().is_none());
    if profile == crate::Profile::BodyStateOdometry {
        validate_body_state(&runtime, &io, &physics);
    }
    drop(worker);
    drop(io);
    router.shutdown().unwrap();
}

fn validate_pause(runtime: &tokio::runtime::Runtime, io: &Robotics, physics: &SharedPhysics) {
    let (low, motion, frames, whistle) = runtime.block_on(async {
        let node = io.node();
        (
            node.subscriber::<booster::LowState>("inputs/low_state")
                .cache(1)
                .build()
                .await
                .unwrap(),
            node.subscriber::<types::motion_command::MotionCommand>("behavior/motion_command")
                .cache(1)
                .build()
                .await
                .unwrap(),
            node.subscriber::<crate::observations::FrameTiming>("diagnostics/camera_frames")
                .cache(1)
                .build()
                .await
                .unwrap(),
            node.subscriber::<types::filtered_whistle::FilteredWhistle>("filtered_whistle")
                .cache(1)
                .build()
                .await
                .unwrap(),
        )
    });
    physics.lock().mode = SimulationMode::Running;
    let start = Instant::now();
    while low.latest_stamp().is_none()
        || motion.latest_stamp().is_none()
        || frames.get_latest().is_none()
        || whistle.get_latest().is_none()
    {
        assert!(
            start.elapsed() < Duration::from_secs(3),
            "timing probes did not receive data"
        );
        thread::sleep(Duration::from_millis(10));
    }
    physics.lock().mode = SimulationMode::Paused;
    // Let already-published samples and in-flight node work drain.
    thread::sleep(Duration::from_millis(150));
    let before = io.now();
    let physical_before = physics.lock().data().time();
    let low_before = low.latest_stamp();
    let motion_before = motion.latest_stamp();
    let frame_before = frames.get_latest().unwrap().captured;
    let clock = io.node().clock().clone();
    let deadline = before + Duration::from_millis(100);
    let timer = runtime.spawn(async move { clock.sleep_until(deadline).await });
    io.whistle();
    thread::sleep(crate::whistle::PULSE_DURATION + Duration::from_millis(100));
    assert_eq!(io.now(), before, "robotics time advanced while paused");
    assert_eq!(
        physics.lock().data().time(),
        physical_before,
        "physics time advanced while paused"
    );
    assert_eq!(
        low.latest_stamp(),
        low_before,
        "paused sensors were restamped"
    );
    assert_eq!(
        motion.latest_stamp(),
        motion_before,
        "behavior timer advanced while paused"
    );
    assert_eq!(
        frames.get_latest().unwrap().captured,
        frame_before,
        "paused camera kept capturing"
    );
    assert!(!timer.is_finished(), "robotics timer expired in wall time");
    assert!(
        !whistle.get_latest().unwrap().is_detected,
        "paused whistle was processed early"
    );
    physics.lock().mode = SimulationMode::Running;
    runtime.block_on(async {
        tokio::time::timeout(Duration::from_secs(3), timer)
            .await
            .unwrap()
            .unwrap();
    });
    physics.lock().mode = SimulationMode::Paused;
    thread::sleep(Duration::from_millis(100));
    assert!(
        whistle.get_latest().unwrap().is_detected,
        "paused whistle was lost before resume"
    );
    let elapsed = io.now().duration_since(before);
    let physical_elapsed = physics.lock().data().time() - physical_before;
    assert!(
        elapsed < Duration::from_millis(300),
        "resume caught up wall time: {elapsed:?}"
    );
    assert!(
        (elapsed.as_secs_f64() - physical_elapsed).abs() < 1e-6,
        "robotics/physics clocks diverged"
    );
    assert!(!io.status().contains("Emergency stop"), "{}", io.status());
    eprintln!(
        "{:?}: pause froze clocks, sensors and behavior; resume advanced {elapsed:?} without catch-up",
        io.profile
    );
}

fn validate_filtering(
    runtime: &tokio::runtime::Runtime,
    io: &Robotics,
    physics: &SharedPhysics,
    app: &mut App,
    robot_pose: Transform,
    ball_output: &ros_z::cache::Cache<
        Option<types::ball_position::BallPosition<coordinate_systems::Ground>>,
    >,
) {
    use coordinate_systems::Ground;
    use types::{
        ball_position::BallPosition,
        object_detection::{Object, RobocupObjectLabel},
        time_wrapper::TimeWrapper,
    };
    let (truth, detected, hypotheses, kick, obstacles, search) = runtime.block_on(async {
        let node = io.node();
        (
            node.subscriber::<Option<BallPosition<Ground>>>(
                "ground_truth/ball_filter/ball_position",
            )
            .cache(1)
            .build()
            .await
            .unwrap(),
            node.subscriber::<TimeWrapper<Vec<Object<RobocupObjectLabel>>>>("detected_objects")
                .cache(1)
                .build()
                .await
                .unwrap(),
            node.subscriber::<ball_filter::BallFilter>("ball_filter/ball_filter_state")
                .cache(1)
                .build()
                .await
                .unwrap(),
            node.subscriber::<Option<BallPosition<Ground>>>("visual_kick/ball_position")
                .cache(1)
                .build()
                .await
                .unwrap(),
            node.subscriber::<Vec<types::obstacles::Obstacle>>("obstacles")
                .cache(1)
                .build()
                .await
                .unwrap(),
            node.subscriber::<Option<linear_algebra::Point2<coordinate_systems::Field>>>(
                "suggested_search_position",
            )
            .cache(1)
            .build()
            .await
            .unwrap(),
        )
    });
    // Give the physical ball a sustained rolling velocity, then compare the real
    // filter with the independently sampled position and velocity.
    physics.lock().mode = SimulationMode::Running;
    let start = io.now();
    let deadline = Instant::now();
    while io.now().duration_since(start) < Duration::from_secs(2) {
        assert!(
            deadline.elapsed() < Duration::from_secs(10),
            "rolling fixture stopped advancing"
        );
        let mut world = physics.lock();
        let ball = world.balls[0];
        let joint = world
            .data()
            .joint(&format!("object_{}_ball_free_joint", ball.to_bits()))
            .unwrap();
        let mut view = joint.view_mut(world.data_mut());
        view.qvel[0] = 0.3;
        view.qvel[1] = 0.0;
        // Rolling about field y for positive field x motion.
        view.qvel[4] = 0.3 / f64::from(io.field_dimensions.ball_radius);
        drop(world);
        thread::sleep(Duration::from_millis(10));
    }
    let expected = truth.get_latest().unwrap().unwrap();
    let estimate = ball_output.get_latest().unwrap().unwrap();
    let velocity_error = (expected.velocity - estimate.velocity).norm();
    assert!(
        expected.velocity.norm() > 0.2,
        "moving-ball fixture stopped"
    );
    assert!(
        velocity_error < 0.15,
        "moving ball velocity error {velocity_error} m/s"
    );
    assert!((expected.position - estimate.position).norm() < 0.15);
    physics.lock().mode = SimulationMode::Paused;
    thread::sleep(Duration::from_millis(500));
    eprintln!(
        "{:?}: rolling ball velocity error {velocity_error:.4} m/s",
        io.profile
    );
    physics.lock().mode = SimulationMode::Running;
    let layer = io.parameters.snapshot().layers.last().unwrap().clone();
    io.parameters
        .set_json(
            "observations.pixel_noise_std_dev",
            serde_json::json!(2.0),
            layer.clone(),
        )
        .unwrap();
    thread::sleep(Duration::from_secs(1));
    let expected = truth.get_latest().unwrap().unwrap();
    let estimate = ball_output.get_latest().unwrap().unwrap();
    let error = (expected.position - estimate.position).norm();
    assert!(error < 0.15, "2 px noise: ball position error {error} m");
    assert!(obstacles.get_latest().is_some_and(|o| !o.is_empty()));
    io.parameters
        .set_json(
            "observations.detection_dropout_probability",
            serde_json::json!(1.0),
            layer.clone(),
        )
        .unwrap();
    let start = Instant::now();
    while ball_output.get_latest().is_some_and(|b| b.is_some()) {
        assert!(
            start.elapsed() < Duration::from_secs(25),
            "ball filter did not expire missing observations"
        );
        thread::sleep(Duration::from_millis(20));
    }
    assert!(
        truth.get_latest().unwrap().is_some(),
        "observation loss removed truth"
    );
    assert!(detected.get_latest().unwrap().inner.is_empty());
    let target = search
        .get_latest()
        .expect("no search output")
        .expect("no search target after ball loss");
    assert!(target.x().is_finite() && target.y().is_finite());
    assert!(
        target.x().abs() <= io.field_dimensions.length / 2.0
            && target.y().abs() <= io.field_dimensions.width / 2.0
    );
    io.parameters.set_json("observations",serde_json::json!({"pixel_noise_std_dev":0.0,"detection_dropout_probability":0.0,"seed":0}),layer).unwrap();
    let start = Instant::now();
    while !ball_output.get_latest().is_some_and(|b| b.is_some()) {
        assert!(
            start.elapsed() < Duration::from_secs(8),
            "ball filter did not reacquire observations"
        );
        thread::sleep(Duration::from_millis(20));
    }
    let old_epoch = physics.lock().motion_epoch;
    let parameters = io.parameters.snapshot().typed().ball.clone();
    let radius = f64::from(io.field_dimensions.ball_radius);
    let second = app
        .world_mut()
        .spawn((
            MjcfObject::from_factory(
                move || crate::scene::ball::ball_spec(radius, &parameters),
                "ball",
            )
            .with_free_joint("ball_free_joint")
            .grounded(),
            Transform::from_xyz(
                robot_pose.translation.x + 0.6,
                0.0,
                robot_pose.translation.z - 3.0,
            ),
        ))
        .id();
    app.update();
    assert_eq!(
        physics.lock().motion_epoch,
        old_epoch,
        "adding a ball reset VO"
    );
    let start = Instant::now();
    while !hypotheses
        .get_latest()
        .is_some_and(|filter| filter.hypotheses.len() >= 2)
    {
        assert!(
            start.elapsed() < Duration::from_secs(8),
            "two visible balls did not produce two hypotheses"
        );
        thread::sleep(Duration::from_millis(20));
    }
    let nearest = kick.get_latest().unwrap().unwrap();
    assert!(
        (nearest.position - truth.get_latest().unwrap().unwrap().position).norm() < 0.15,
        "visual kick did not select nearest ball"
    );
    app.world_mut().despawn(second);
    app.update();
    assert_eq!(
        physics.lock().motion_epoch,
        old_epoch,
        "deleting a ball reset VO"
    );
    eprintln!(
        "{:?}: noisy ball error {error:.4} m; dropout, reacquisition, multiple hypotheses, kick selection, obstacles and search passed",
        io.profile
    );
}

fn validate_body_state(runtime: &tokio::runtime::Runtime, io: &Robotics, physics: &SharedPhysics) {
    use coordinate_systems::{Ground, Odometry, Robot};
    use linear_algebra::{Isometry3, Pose2};
    use types::{
        controller_input::{Axis, ControllerAxis, ControllerInput},
        fall_detection::{FallDetection, Posture},
        time_wrapper::TimeWrapper,
    };
    let (ground, odometry, truth, posture, controller) = runtime.block_on(async {
        let node = io.node();
        (
            node.subscriber::<TimeWrapper<Option<Isometry3<Robot, Ground>>>>("robot_to_ground")
                .cache(32)
                .with_stamp(|v| v.time)
                .build()
                .await
                .unwrap(),
            node.subscriber::<Pose2<Odometry>>("inputs/odometry")
                .cache(32)
                .build()
                .await
                .unwrap(),
            node.subscriber::<Pose2<Odometry>>("ground_truth/inputs/odometry")
                .cache(32)
                .build()
                .await
                .unwrap(),
            node.subscriber::<FallDetection>("fall_detection/status")
                .cache(32)
                .with_stamp(|v| v.time)
                .build()
                .await
                .unwrap(),
            node.publisher::<ControllerInput>("inputs/controller_input")
                .build()
                .await
                .unwrap(),
        )
    });
    let (obstacles, reference_obstacles) = runtime.block_on(async {
        let node = io.node();
        (
            node.subscriber::<Vec<types::obstacles::Obstacle>>("obstacles")
                .cache(16)
                .build()
                .await
                .unwrap(),
            node.subscriber::<Vec<types::obstacles::Obstacle>>("ground_truth/obstacles")
                .cache(16)
                .build()
                .await
                .unwrap(),
        )
    });
    let start = Instant::now();
    while odometry.get_latest().is_none()
        || truth.get_latest().is_none()
        || obstacles.get_latest().is_none()
        || reference_obstacles.get_latest().is_none()
    {
        assert!(
            start.elapsed() < Duration::from_secs(3),
            "body-state probes did not receive data"
        );
        thread::sleep(Duration::from_millis(10));
    }
    physics.lock().mode = SimulationMode::Paused;
    thread::sleep(Duration::from_millis(100));
    let beginning = *odometry.get_latest().unwrap();
    thread::sleep(Duration::from_millis(500));
    let stationary = *odometry.get_latest().unwrap();
    assert!(
        (stationary.position() - beginning.position()).norm() < 0.001,
        "paused odometry drift"
    );
    let initial_truth = *truth.get_latest().unwrap();
    let input = ControllerInput {
        connected: true,
        device_name: "walking fixture".into(),
        axes: vec![ControllerAxis {
            name: Axis::LeftStickY,
            value: 0.15,
        }],
        ..Default::default()
    };
    // Remote mode was enabled earlier and remains enabled across a disconnect.
    physics.lock().mode = SimulationMode::Running;
    let start = io.now();
    let deadline = Instant::now();
    let mut max_position_error = 0.0f32;
    let mut max_yaw_error = 0.0f32;
    let mut max_obstacle_error = 0.0f32;
    while io.now().duration_since(start) < Duration::from_secs(3) {
        assert!(
            deadline.elapsed() < Duration::from_secs(15),
            "walking fixture stopped advancing"
        );
        runtime
            .block_on(
                controller.publish_with_source_time(&input, ros_z::time::Clock::wallclock().now()),
            )
            .unwrap();
        let estimate = *odometry.get_latest().unwrap();
        let actual = *truth.get_nearest(odometry.latest_stamp().unwrap()).unwrap();
        max_position_error = max_position_error.max(
            ((estimate.position() - stationary.position())
                - (actual.position() - initial_truth.position()))
            .norm(),
        );
        max_yaw_error = max_yaw_error.max(
            ((estimate.angle() - stationary.angle()) - (actual.angle() - initial_truth.angle()))
                .abs(),
        );
        let reference = reference_obstacles
            .get_nearest(obstacles.latest_stamp().unwrap())
            .unwrap();
        assert!(obstacles.get_latest().unwrap().len() >= 4);
        assert!(
            obstacles
                .get_latest()
                .unwrap()
                .iter()
                .all(|o| o.position.x().is_finite() && o.position.y().is_finite())
        );
        for obstacle in obstacles
            .get_latest()
            .unwrap()
            .iter()
            .filter(|o| o.kind == types::obstacles::ObstacleKind::GoalPost)
        {
            let error = reference
                .iter()
                .map(|truth| (truth.position - obstacle.position).norm())
                .fold(f32::INFINITY, f32::min);
            max_obstacle_error = max_obstacle_error.max(error);
        }
        assert!(
            ground.get_latest().is_some_and(|g| g.inner.is_some()),
            "ground unavailable during slow walking"
        );
        assert!(!io.status().contains("Emergency stop"), "{}", io.status());
        thread::sleep(Duration::from_millis(20));
    }
    physics.lock().mode = SimulationMode::Paused;
    let distance = (truth.get_latest().unwrap().position() - initial_truth.position()).norm();
    assert!(
        distance > 0.05,
        "walking fixture did not move: {distance} m"
    );
    assert!(
        max_position_error < 0.15,
        "walking odometry error {max_position_error} m"
    );
    assert!(
        max_yaw_error < 0.05,
        "walking yaw error {max_yaw_error} rad"
    );
    eprintln!(
        "BodyStateOdometry: 3 s simulated walking maximum odometry error {max_position_error:.4} m, yaw {max_yaw_error:.4} rad"
    );
    // Compare arrival-time output against truth; this includes processing delay.
    eprintln!(
        "BodyStateOdometry: moving-observer goalpost maximum arrival-time discrepancy {max_obstacle_error:.4} m; production filter uses absolute odometry"
    );
    let mut world = physics.lock();
    let robot = world.robots[0];
    let mut fallen = initial_pose(&io.field_dimensions);
    fallen.rotation *= Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    world.ground_object(robot, fallen).unwrap();
    world.mode = SimulationMode::Running;
    drop(world);
    let start = Instant::now();
    while !posture
        .get_latest()
        .is_some_and(|state| matches!(state.posture, Posture::Fallen { .. }))
    {
        assert!(
            start.elapsed() < Duration::from_secs(3),
            "fallen fixture not detected: {:?}",
            posture.get_latest()
        );
        thread::sleep(Duration::from_millis(20));
    }
    let start = Instant::now();
    while ground.get_latest().is_some_and(|g| g.inner.is_some()) {
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "real ground provider retained support while fallen"
        );
        thread::sleep(Duration::from_millis(20));
    }
    let mut world = physics.lock();
    let binding = RobotBinding::new(world.data(), &format!("object_{}_", robot.to_bits())).unwrap();
    binding.reset_joints(world.data_mut());
    world
        .ground_object(robot, initial_pose(&io.field_dimensions))
        .unwrap();
    drop(world);
    let start = Instant::now();
    while !posture
        .get_latest()
        .is_some_and(|state| state.posture == Posture::Upright)
        || !ground.get_latest().is_some_and(|g| g.inner.is_some())
    {
        assert!(
            start.elapsed() < Duration::from_secs(3),
            "reset did not restore body estimates"
        );
        thread::sleep(Duration::from_millis(20));
    }
    eprintln!("BodyStateOdometry: falling removes ground support; upright reset restores it");
}

fn validate_localization(
    runtime: &tokio::runtime::Runtime,
    io: &Robotics,
    physics: &SharedPhysics,
    require_acquisition: bool,
) {
    use coordinate_systems::{Field, Ground};
    use linear_algebra::Isometry2;
    use types::{time_wrapper::TimeWrapper, visual_localization::VisualLocalizationFrame};
    let (localized, pose, truth, associations, diagnostics) = runtime.block_on(async {
        let node = io.node();
        (
            node.subscriber::<types::localization::LocalizationEstimate>("localization/estimate")
                .cache(10)
                .build()
                .await
                .unwrap(),
            node.subscriber::<Isometry2<Ground, Field>>("ground_to_field")
                .cache(128)
                .build()
                .await
                .unwrap(),
            node.subscriber::<Isometry2<Ground, Field>>("ground_truth/ground_to_field")
                .cache(128)
                .build()
                .await
                .unwrap(),
            node.subscriber::<TimeWrapper<VisualLocalizationFrame>>(
                "field_mark_association/visual_localization_local",
            )
            .cache(32)
            .build()
            .await
            .unwrap(),
            node.subscriber::<localization_3d::SolveDiagnostics>("debug/solve_diagnostics")
                .cache(32)
                .build()
                .await
                .unwrap(),
        )
    });
    if !require_acquisition {
        let start = Instant::now();
        while !diagnostics
            .get_latest()
            .is_some_and(|d| d.measurement_count > 0 && d.failure.is_none())
            || associations.get_latest().is_none()
            || localized.get_latest().is_none()
        {
            assert!(io.poll().is_none(), "{}", io.status());
            assert!(
                start.elapsed() < Duration::from_secs(3),
                "localization outputs did not arrive: associations {:?}, diagnostics {:?}",
                associations.get_latest(),
                diagnostics.get_latest()
            );
            thread::sleep(Duration::from_millis(20));
        }
        assert!(
            associations.get_latest().is_some(),
            "association node did not emit frames"
        );
        assert!(io.ready());
        if !localized
            .get_latest()
            .is_some_and(|pose| pose.robot_to_field.is_some())
        {
            assert!(
                io.status().contains("acquiring"),
                "startup prior was presented as an acquired pose"
            );
        }
        eprintln!(
            "Localization: raw sensors, body/filter outputs, association frames and VO ingestion live; acquisition/tracking checked separately"
        );
        return;
    }
    let start = Instant::now();
    while !localized
        .get_latest()
        .is_some_and(|pose| pose.robot_to_field.is_some())
        || !associations
            .get_latest()
            .is_some_and(|frame| frame.inner.associations.len() >= 5)
        || pose.get_latest().is_none()
        || truth.get_latest().is_none()
    {
        assert!(io.poll().is_none(), "{}", io.status());
        assert!(
            start.elapsed() < Duration::from_secs(15),
            "localization did not acquire; associations: {:?}; diagnostics: {:?}",
            associations.get_latest(),
            diagnostics.get_latest()
        );
        thread::sleep(Duration::from_millis(30));
    }
    assert!(
        associations
            .get_latest()
            .is_some_and(|v| v.inner.associations.len() >= 5)
    );
    let estimated = pose.get_latest().unwrap();
    let actual = truth.get_nearest(pose.latest_stamp().unwrap()).unwrap();
    let error = (estimated.inner.translation.vector - actual.inner.translation.vector).norm();
    let yaw_error = estimated
        .inner
        .rotation
        .angle_to(&actual.inner.rotation)
        .abs();
    assert!(error < 0.15, "stationary localization error {error} m");
    assert!(
        yaw_error < 0.05,
        "stationary localization yaw error {yaw_error} rad"
    );
    eprintln!(
        "Localization: acquired; position error {error:.4} m, yaw error {yaw_error:.4} rad; associations {}",
        associations.get_latest().unwrap().inner.associations.len()
    );
    // Test ordinary motion and articulated head VO without teleporting or changing field side.
    use types::controller_input::{
        Axis, Button, ControllerAxis, ControllerButton, ControllerInput,
    };
    let controller = runtime
        .block_on(
            io.node()
                .publisher::<ControllerInput>("inputs/controller_input")
                .build(),
        )
        .unwrap();
    let mut input = ControllerInput {
        connected: true,
        device_name: "localization tracking fixture".into(),
        axes: vec![ControllerAxis {
            name: Axis::LeftStickY,
            value: 0.15,
        }],
        buttons: vec![ControllerButton {
            name: Button::DPadLeft,
            pressed: true,
            value: 0.15,
        }],
    };
    let enable = ControllerInput {
        connected: true,
        device_name: input.device_name.clone(),
        buttons: vec![ControllerButton {
            name: Button::Start,
            pressed: true,
            value: 1.0,
        }],
        ..Default::default()
    };
    for _ in 0..5 {
        runtime
            .block_on(
                controller.publish_with_source_time(&enable, ros_z::time::Clock::wallclock().now()),
            )
            .unwrap();
        thread::sleep(Duration::from_millis(20));
    }
    let initial_position = actual.inner.translation.vector;
    physics.lock().mode = SimulationMode::Running;
    let start = io.now();
    let deadline = Instant::now();
    let mut max_error = error;
    let mut max_yaw_error = yaw_error;
    while io.now().duration_since(start) < Duration::from_secs(3) {
        assert!(
            deadline.elapsed() < Duration::from_secs(15),
            "localization fixture stopped advancing"
        );
        if io.now().duration_since(start) > Duration::from_secs(1) {
            input.buttons.clear();
        }
        runtime
            .block_on(
                controller.publish_with_source_time(&input, ros_z::time::Clock::wallclock().now()),
            )
            .unwrap();
        assert!(io.poll().is_none(), "{}", io.status());
        assert!(!io.status().contains("Emergency stop"), "{}", io.status());
        if let Some(stamp) = pose.latest_stamp() {
            assert!(
                io.now().duration_since(stamp) < Duration::from_millis(500),
                "localization stopped publishing fresh poses"
            );
            if let (Some(estimated), Some(actual)) = (pose.get_latest(), truth.get_nearest(stamp)) {
                max_error = max_error.max(
                    (estimated.inner.translation.vector - actual.inner.translation.vector).norm(),
                );
                max_yaw_error = max_yaw_error.max(
                    estimated
                        .inner
                        .rotation
                        .angle_to(&actual.inner.rotation)
                        .abs(),
                );
            }
        }
        thread::sleep(Duration::from_millis(20));
    }
    physics.lock().mode = SimulationMode::Paused;
    assert!(
        (truth.get_latest().unwrap().inner.translation.vector - initial_position).norm() > 0.05,
        "tracking fixture did not walk"
    );
    assert!(
        max_error < 0.20,
        "tracking localization error {max_error} m"
    );
    assert!(
        max_yaw_error < 0.10,
        "tracking localization yaw error {max_yaw_error} rad"
    );
    let diagnostics = diagnostics
        .get_latest()
        .expect("localization solver did not emit diagnostics");
    assert!(
        diagnostics.measurement_count > 0,
        "no measurements in localization solve"
    );
    assert!(diagnostics.failure.is_none(), "{diagnostics:?}");
    eprintln!(
        "Localization: 3 s simulated walking/head tracking max error {max_error:.4} m, yaw {max_yaw_error:.4} rad; measurements {}, solve {:?}",
        diagnostics.measurement_count, diagnostics.duration
    );
}

#[test]
fn localization_startup_frame_acquires_with_production_parameters() {
    let captured: serde_json::Value =
        serde_json::from_str(include_str!("../tests/fixtures/localization_startup.json")).unwrap();
    let objects: Vec<types::object_detection::Object<types::object_detection::RobocupObjectLabel>> =
        serde_json::from_value(captured["objects"].clone()).unwrap();
    let camera: projection::camera_matrix::CameraMatrix =
        serde_json::from_value(captured["camera"].clone()).unwrap();
    let field = serde_json::from_value(captured["field"].clone()).unwrap();
    let params: field_mark_association::FieldMarkAssociationParameters = json5::from_str(
        include_str!("../../../etc/parameters/base/field_mark_association.json5"),
    )
    .unwrap();
    let features = field_mark_association::find_detected_visual_features(&objects);
    let result = field_mark_association::associate_global_visual_features(
        field_mark_association::GlobalAssociationInput {
            visual_features: &features,
            robot_to_ground: linear_algebra::Rotation3::wrap(
                camera.ground_to_robot.inner.rotation.inverse(),
            ),
            robot_to_camera: camera.head_to_camera * camera.robot_to_head,
            camera_intrinsic: camera.intrinsics,
            field_dimensions: &field,
            parameters: &params.global_localizer,
            heading: None,
        },
    );
    assert!(
        result.associations.len() >= 3,
        "visible startup landmarks did not acquire localization"
    );
}

#[test]
#[ignore = "requires ONNX Runtime, motion models, and free GameController UDP ports"]
fn localization_acquisition_and_tracking() {
    validate_profile(crate::Profile::Localization, true);
}

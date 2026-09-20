use super::*;
use crate::bevy_mujoco::{MujocoWorld, MujocoWorldPlugin, SimulationMode};
use bevy::prelude::*;
use std::time::Duration;
use types::{motion_command::HeadMotion, motion_execution::MotionPhase};

#[test]
fn supported_head_moves_without_behavior_or_inference_and_stops_and_restarts() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let router = format!("tcp/127.0.0.1:{}", listener.local_addr().unwrap().port());
    drop(listener);
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let clock = Clock::logical(ros_z::time::Time::zero());
    let (server, mut io) = runtime.block_on(async {
        let server = ContextBuilder::default()
            .with_mode("router")
            .disable_multicast_scouting()
            .with_connect_endpoints(std::iter::empty::<&str>())
            .with_listen_endpoints([router.as_str()])
            .build()
            .await
            .unwrap();
        let io = Robotics::new(
            runtime.handle().clone(),
            StackConfiguration {
                router,
                namespace: "/head_only_sim_test".into(),
                parameter_layers: vec![
                    root.join("tools/simulate/parameters"),
                    root.join("etc/parameters/base"),
                    root.join("etc/parameters/location/simulator"),
                ],
                launch_nodes: true,
                head_only: true,
            },
            clock.clone(),
        )
        .await
        .unwrap();
        (server, io)
    });
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, MujocoWorldPlugin));
    app.insert_resource(SimulationMode::Paused);
    let robot = app
        .world_mut()
        .spawn(crate::scene::robot::head_only_model())
        .id();
    app.update();
    let mut world = app.world_mut().resource_mut::<MujocoWorld>();
    let prefix = format!("object_{}_", robot.to_bits());
    let binding = RobotBinding::new(world.data(), &prefix).unwrap();
    assert!(
        world
            .data()
            .joint(&format!("{prefix}world_joint"))
            .is_none()
    );
    let torso_position = world
        .data()
        .body(&format!("{prefix}Trunk"))
        .unwrap()
        .view(world.data())
        .xpos
        .to_vec();
    io.publish_field_dimensions(&FieldDimensions::SPL_2025)
        .unwrap();
    io.publish_observation(binding.observe(world.data()), clock.now())
        .unwrap();
    io.publish_inputs().unwrap();

    let step = |io: &Robotics, world: &mut MujocoWorld| {
        for _ in 0..8 {
            binding.apply(world.data_mut(), io.latest_command().as_ref());
            world.data_mut().step();
            world.data_mut().forward();
            let time = ros_z::time::Time::from_nanos((world.data().time() * 1e9).round() as i64);
            io.publish_observation(binding.observe(world.data()), time)
                .unwrap();
            io.publish_inputs().unwrap();
        }
        runtime.block_on(async { tokio::time::sleep(Duration::from_millis(16)).await });
        assert!(
            io.execution
                .get_latest()
                .is_none_or(|status| status.fault.is_none()),
            "{}",
            io.status()
        );
    };
    for _ in 0..25 {
        step(&io, &mut world);
    }
    assert_eq!(
        io.execution.get_latest().unwrap().phase,
        MotionPhase::Damping
    );
    io.input_motion = MotionCommand::HeadOnly {
        head: HeadMotion::LookAround,
    };
    io.inject_current_motion().unwrap();
    let mut active_samples = 0;
    let mut maximum_yaw = 0.0_f32;
    for _ in 0..125 {
        step(&io, &mut world);
        let command = io.latest_command().unwrap();
        if command.motor_commands[0].kp > 0.0 {
            active_samples += 1;
            for motor in &command.motor_commands[2..] {
                assert_eq!(
                    (motor.kp, motor.kd, motor.velocity, motor.torque),
                    (0.0, 1.0, 0.0, 0.0)
                );
            }
        }
        maximum_yaw = maximum_yaw.max(
            binding.observe(world.data()).low_state.motor_state_serial[0]
                .position
                .abs(),
        );
    }
    assert!(active_samples > 100);
    assert!(
        maximum_yaw > 0.3,
        "head did not physically move: {maximum_yaw}"
    );
    assert_eq!(
        io.execution.get_latest().unwrap().phase,
        MotionPhase::HeadOnly
    );
    assert_eq!(
        world
            .data()
            .body(&format!("{prefix}Trunk"))
            .unwrap()
            .view(world.data())
            .xpos
            .to_vec(),
        torso_position
    );
    for command in [
        MotionCommand::Prepare,
        MotionCommand::Stand {
            head: HeadMotion::ZeroAngles,
        },
    ] {
        assert!(io.validate_motion(&command).is_err());
    }
    // Clearing while paused publishes a stop without advancing the simulation clock.
    let paused_at = clock.now();
    io.clear_injection().unwrap();
    assert_eq!(clock.now(), paused_at);
    for _ in 0..20 {
        step(&io, &mut world);
    }
    assert_eq!(
        io.execution.get_latest().unwrap().phase,
        MotionPhase::Damping
    );
    assert!(
        io.latest_command()
            .unwrap()
            .motor_commands
            .iter()
            .all(|motor| motor.kp == 0.0)
    );

    io.input_motion = MotionCommand::HeadOnly {
        head: HeadMotion::ZeroAngles,
    };
    io.inject_current_motion().unwrap();
    for _ in 0..20 {
        step(&io, &mut world);
    }
    // The simulator reset recreates the dedicated runtime, retaining the selected command.
    binding.reset_joints(world.data_mut());
    io.restart().unwrap();
    io.publish_observation(binding.observe(world.data()), clock.now())
        .unwrap();
    for _ in 0..30 {
        step(&io, &mut world);
    }
    assert!(io.is_head_only());
    assert_eq!(
        io.execution.get_latest().unwrap().phase,
        MotionPhase::HeadOnly
    );
    let graph = io._node.graph();
    let graph = graph.lock();
    assert_eq!(
        graph
            .publishers_on("/head_only_sim_test/motion_inference/status")
            .count(),
        0
    );
    assert_eq!(
        graph
            .publishers_on("/head_only_sim_test/behavior/blackboard")
            .count(),
        0
    );
    assert_eq!(
        graph
            .publishers_on("/head_only_sim_test/behavior/motion_command")
            .count(),
        1
    );
    drop(graph);
    eprintln!(
        "Head-only measured yaw {maximum_yaw:.3} rad; body damping verified across {active_samples} samples"
    );
    drop(io);
    server.shutdown().unwrap();
}

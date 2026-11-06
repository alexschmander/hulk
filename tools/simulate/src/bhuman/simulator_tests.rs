use std::{
    sync::{Arc, atomic::AtomicBool},
    thread,
    time::{Duration, Instant},
};

use bevy::prelude::*;

use super::*;
use crate::{
    bevy_mujoco::{MjcfObject, MujocoWorldPlugin, SharedPhysics, SimulationMode},
    simulation::{PhysicsWorker, Speed},
    team::Team,
};

#[test]
#[ignore = "Requires native B-Human worker, source, simulator assets and HULK motion models"]
fn mixed_match_steps_native_players_and_preserves_pause() {
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
    let endpoint =
        runtime.block_on(async { router.session().info().locators().await[0].to_string() });
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let parameter_root = root.join("../../etc/parameters");
    let field = crate::FieldConfiguration::load(&parameter_root, "incheon_small").unwrap();
    let configuration = crate::Configuration {
        referee: crate::RefereeMode::Automatic,
        competition: crate::Competition::default(),
        parameter_root,
        model_directory: root.join("../../etc/neural_networks"),
        router: Some(endpoint),
        namespace: crate::robot_namespace(crate::RobotId::FIRST),
        location: Some("incheon_small".into()),
        robot_count: 1,
        opponent_count: 2,
        opponent: crate::Opponent::BHuman {
            executable: std::env::var_os("BHUMAN_WORKER")
                .expect("Set BHUMAN_WORKER")
                .into(),
            source: std::env::var_os("BHUMAN_SOURCE")
                .expect("Set BHUMAN_SOURCE")
                .into(),
        },
        field_configuration: Some(field),
        profile: std::env::var("SIMULATOR_TEST_PROFILE")
            .ok()
            .map(|profile| {
                serde_json::from_str(&profile)
                    .or_else(|_| serde_json::from_value(serde_json::Value::String(profile)))
                    .unwrap()
            })
            .unwrap_or(crate::Profile::MotionBehavior),
        mcap_enabled: Some(false),
    };
    let team = runtime
        .block_on(Team::new_with_controller(
            runtime.handle().clone(),
            configuration,
            Arc::new(AtomicBool::new(false)),
            None,
            false,
        ))
        .unwrap();
    for member in team.members() {
        assert_eq!(member.io.now(), team.now());
        assert!(
            member.io.actuator_control().supports_standing(),
            "{}: {}",
            member.id,
            member.io.status()
        );
    }
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, MujocoWorldPlugin));
    bind_players(&mut app, &team);
    app.update();
    let physics = app.world().resource::<SharedPhysics>().clone();
    physics.lock().mode = SimulationMode::Paused;
    let mut worker = PhysicsWorker::start_team(physics.clone(), team.clone());
    worker.set_speed(Speed::Quadruple);
    let wait = |worker: &mut PhysicsWorker, condition: &dyn Fn() -> bool| {
        let deadline = Instant::now() + Duration::from_secs(30);
        while !condition() {
            if let Some(error) = worker.poll() {
                panic!("{error}");
            }
            assert!(Instant::now() < deadline, "mixed match condition timed out");
            thread::sleep(Duration::from_millis(10));
        }
    };
    wait(&mut worker, &|| {
        team.members().iter().all(|member| member.entity.is_some())
    });
    let native: Vec<_> = team
        .members()
        .into_iter()
        .filter(|member| member.io.is_bhuman())
        .collect();
    assert_eq!(native.len(), 2);
    assert_ne!(
        native[0]
            .io
            .bhuman()
            .unwrap()
            .state
            .lock()
            .unwrap()
            .robot
            .child
            .id(),
        native[1]
            .io
            .bhuman()
            .unwrap()
            .state
            .lock()
            .unwrap()
            .robot
            .child
            .id()
    );
    let native_start: Vec<_> = native
        .iter()
        .map(|member| {
            let time = member.io.bhuman().unwrap().state.lock().unwrap().time_ns;
            assert!(time >= crate::simulation::STARTUP_WARMUP.as_nanos() as u64);
            time
        })
        .collect();
    {
        let mut world = physics.lock();
        world.remaining_steps = Some(1000);
        world.mode = SimulationMode::Running;
    }
    wait(&mut worker, &|| physics.lock().remaining_steps == Some(0));
    for (member, start) in native.iter().zip(&native_start) {
        let state = member.io.bhuman().unwrap().state.lock().unwrap();
        assert_eq!(state.time_ns, start + 1000 * STEP_NS);
        assert_eq!(state.output.as_ref().unwrap().player_state, "active");
        assert_eq!(state.output.as_ref().unwrap().motion, "stand");
        assert!(
            state.packet.is_some(),
            "native player never received GameController data"
        );
        assert!(member.io.primary.get_latest().is_some());
    }
    let paused = team.now();
    thread::sleep(Duration::from_millis(100));
    assert_eq!(team.now(), paused);
    for (member, start) in native.iter().zip(&native_start) {
        assert_eq!(
            member.io.bhuman().unwrap().state.lock().unwrap().time_ns,
            start + 1000 * STEP_NS
        );
    }
    let upright = || {
        let world = physics.lock();
        for member in team.members() {
            assert_eq!(member.io.now(), team.now(), "{} clock diverged", member.id);
            let pose = world.object_pose(member.entity.unwrap()).unwrap();
            assert!(
                pose.translation.y > 0.4 && (pose.rotation * Vec3::Y).dot(Vec3::Y) > 0.5,
                "{} fell after handover: {pose:?}",
                member.id
            );
        }
    };
    upright();
    if std::env::var_os("BHUMAN_TEST_ADDITIONS").is_some() {
        let add_running = std::env::var_os("SIMULATOR_TEST_ADD_RUNNING").is_some();
        if add_running {
            let mut world = physics.lock();
            world.remaining_steps = None;
            world.mode = SimulationMode::Running;
        }
        for side in crate::TeamId::ALL {
            while team
                .members()
                .iter()
                .filter(|member| member.id.team == side)
                .count()
                < 5
            {
                let (time, occupied) = {
                    let mut world = physics.lock();
                    if !add_running {
                        // The last step integrates qpos after capturing sensors.
                        // Compare current transforms, not that prior sample.
                        world.forward();
                    }
                    (
                        team.now(),
                        world
                            .robots
                            .iter()
                            .map(|entity| world.object_pose(*entity).unwrap())
                            .collect::<Vec<_>>(),
                    )
                };
                let pose = crate::team::vacant_spawn(
                    &team
                        .configuration()
                        .field_configuration
                        .as_ref()
                        .unwrap()
                        .dimensions,
                    &occupied,
                    team.away(side),
                )
                .unwrap();
                let id = runtime.block_on(team.add(side, pose)).unwrap();
                let member = team
                    .members()
                    .into_iter()
                    .find(|member| member.id == id)
                    .unwrap();
                assert!(
                    member.io.actuator_control().supports_standing(),
                    "{id} joined without joint support"
                );
                bind_players(&mut app, &team);
                app.update();
                thread::sleep(Duration::from_millis(100));
                wait(&mut worker, &|| worker_ready(&physics, &team));
                if add_running {
                    assert!(team.now() > time, "Adding {id} froze the running match");
                } else {
                    assert_eq!(team.now(), time, "Adding {id} advanced the paused match");
                    let world = physics.lock();
                    for pose in &occupied {
                        assert!(
                            world
                                .robots
                                .iter()
                                .filter_map(|entity| world.object_pose(*entity))
                                .any(
                                    |current| current.translation.distance(pose.translation) < 1e-6
                                ),
                            "Adding {id} moved an existing paused robot"
                        );
                    }
                }
                upright();
            }
        }
        assert_eq!(team.members().len(), 10);
        {
            let mut world = physics.lock();
            world.remaining_steps = Some(1000);
            world.mode = SimulationMode::Running;
        }
        wait(&mut worker, &|| {
            upright();
            physics.lock().remaining_steps == Some(0)
        });
        upright();
        let frozen = team.now();
        let native_times: Vec<_> = team
            .members()
            .into_iter()
            .filter(|member| member.io.is_bhuman())
            .map(|member| {
                let time = member.io.bhuman().unwrap().state.lock().unwrap().time_ns;
                (member, time)
            })
            .collect();
        thread::sleep(Duration::from_millis(100));
        assert_eq!(team.now(), frozen);
        for (member, time) in native_times {
            assert_eq!(
                member.io.bhuman().unwrap().state.lock().unwrap().time_ns,
                time
            );
        }
    }
    // Both edges arrive while paused. The bridge must preserve the tap until
    // separate cognition frames can consume it.
    native[0]
        .io
        .button_event(0, ButtonEventType::PressDown)
        .unwrap();
    native[0]
        .io
        .button_event(0, ButtonEventType::PressUp)
        .unwrap();
    {
        let mut world = physics.lock();
        world.remaining_steps = Some(50);
        world.mode = SimulationMode::Running;
    }
    wait(&mut worker, &|| physics.lock().remaining_steps == Some(0));
    {
        let state = native[0].io.bhuman().unwrap().state.lock().unwrap();
        let output = state.output.as_ref().unwrap();
        assert_eq!(output.player_state, "unstiff");
        assert_eq!(
            output.leds,
            [LedState::Off, LedState::Off, LedState::Blinking]
        );
        assert!(output.joints.iter().all(|joint| joint.kp == 0.0));
    }
    let color = native[0].io.led_color();
    thread::sleep(Duration::from_millis(100));
    assert_eq!(
        native[0].io.led_color(),
        color,
        "native LED blink advanced during pause"
    );
    drop(worker);
    // Inputs may arrive while a native frame computes. Cancelling that frame
    // must close the process, rather than leave a reply for the next tick.
    {
        let member = &native[1];
        let observation = {
            let world = physics.lock();
            crate::robot_io::RobotBinding::new(
                world.data(),
                &format!("object_{}_", member.entity.unwrap().to_bits()),
            )
            .unwrap()
            .observe(world.data())
        };
        let robot = member.io.bhuman().unwrap();
        let pending = robot
            .begin_step(
                &crate::robot_io::MotionSample::from(&observation),
                Some(crate::bhuman::World::default()),
            )
            .unwrap();
        robot
            .button(0, booster::ButtonEventType::PressDown)
            .unwrap();
        drop(pending);
        assert!(robot.poll().is_some(), "cancelled native process survived");
        assert!(robot.state.lock().unwrap().robot.stream.is_none());
    }
    drop(native);
    drop(team);
    drop(app);
    assert!(crate::finish_recordings().is_empty());
    router.shutdown().unwrap();
}

fn bind_players(app: &mut App, team: &Team) {
    for member in team
        .members()
        .into_iter()
        .filter(|member| member.entity.is_none())
    {
        let entity = app
            .world_mut()
            .spawn((
                MjcfObject::from_shared_file(
                    concat!(env!("CARGO_MANIFEST_DIR"), "/assets/k1_robot.xml"),
                    "Trunk",
                )
                .with_free_joint("world_joint")
                .grounded(),
                member.pose,
            ))
            .id();
        team.bind(member.id, entity);
    }
}

fn worker_ready(physics: &SharedPhysics, team: &Team) -> bool {
    let world = physics.lock();
    team.members().iter().all(|member| {
        member
            .entity
            .is_some_and(|entity| world.contains_object(entity))
    })
}

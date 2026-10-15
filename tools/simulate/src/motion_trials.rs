//! Opt-in measurements through the production behavior, motion and inference nodes.
use super::*;
use crate::bevy_mujoco::{MjcfObject, MujocoWorldPlugin};
use crate::{Competition, Configuration, RefereeMode};
use ros_z::{parameter::RemoteParameterClient, prelude::*};
use types::motion_command::{HeadMotion, MotionCommand};

#[path = "motion_trials/contacts.rs"]
mod contacts;
#[path = "motion_trials/coordinates.rs"]
mod coordinates;
#[path = "motion_trials/kickoff.rs"]
mod kickoff;

#[test]
#[ignore = "requires MuJoCo, ONNX Runtime and motion models; runs three live-policy trials"]
fn kick_contact_regressions() {
    for (scenario, speed, duration, ratio, moving) in [
        ("normal", "3.4", "3", "0.9", false),
        ("soft", "0.3", "3", "0.8", false),
        ("behavior", "0.3", "9", "0.75", true),
    ] {
        // Separate processes isolate native runtimes and fixture environment.
        let mut trial = std::process::Command::new(std::env::current_exe().unwrap());
        for (name, _) in std::env::vars_os() {
            if name
                .to_str()
                .is_some_and(|name| name.starts_with("MOTION_TRIAL"))
            {
                trial.env_remove(name);
            }
        }
        trial
            .args(["motion_policy_trial", "--ignored", "--nocapture"])
            .env("MOTION_TRIAL", scenario)
            .env("MOTION_TRIAL_SPEED", speed)
            .env("MOTION_TRIAL_DURATION", duration)
            .env("MOTION_TRIAL_CONTACT_AUDIT", "1")
            .env("MOTION_TRIAL_MIN_CONTACT_SPEED_RATIO", ratio)
            .env("MOTION_TRIAL_MIN_FOLLOWTHROUGH_MM", "0")
            .env("MOTION_TRIAL_MAX_CONTACT_LINE_OFFSET_MM", "30")
            .env("MOTION_TRIAL_REQUIRE_FOOT_FIRST", "1");
        if moving {
            trial
                .env("MOTION_TRIAL_PROFILE", "filtering")
                .env("MOTION_TRIAL_VX", "0.6");
        }
        assert!(
            trial.status().unwrap().success(),
            "{scenario} contact regression"
        );
    }
}

#[test]
#[ignore = "requires MuJoCo, ONNX Runtime and motion models"]
fn motion_policy_trial() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("error")
        .with_test_writer()
        .try_init();
    let scenario = std::env::var("MOTION_TRIAL").unwrap_or_else(|_| "normal".into());
    assert!(
        matches!(
            scenario.as_str(),
            "normal" | "soft" | "strong" | "quick" | "shot" | "behavior" | "kickoff" | "walk"
        ),
        "unknown motion trial: {scenario}"
    );
    let parse = |name: &str, default: f64| {
        std::env::var(name)
            .ok()
            .map(|v| v.parse::<f64>().unwrap())
            .unwrap_or(default)
    };
    let speed = parse("MOTION_TRIAL_SPEED", 3.4) as f32;
    let velocity = [parse("MOTION_TRIAL_VX", 0.0), parse("MOTION_TRIAL_VY", 0.0)];
    let is_kickoff = scenario == "kickoff";
    let autonomous = scenario == "behavior" || is_kickoff;
    let duration = parse("MOTION_TRIAL_DURATION", if is_kickoff { 25.0 } else { 6.0 });
    let position = [
        parse("MOTION_TRIAL_X", if is_kickoff { 0.0 } else { 0.8 }),
        parse("MOTION_TRIAL_Y", if is_kickoff { 0.0 } else { 0.08 }),
    ];
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
        runtime.block_on(async { router.session().info().locators().await })[0].to_string();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let policy_overrides = tempfile::tempdir().unwrap();
    let mut location = std::env::var("MOTION_TRIAL_POLICY").ok().map(|overrides| {
        let value: serde_json::Value = serde_json::from_str(&overrides).unwrap();
        std::fs::write(
            policy_overrides.path().join("motion_inference.json5"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
        policy_overrides.path().to_string_lossy().into_owned()
    });
    if let Ok(mass) = std::env::var("MOTION_TRIAL_BALL_MASS") {
        let mass: f32 = mass.parse().unwrap();
        std::fs::write(
            policy_overrides.path().join("simulator.json5"),
            serde_json::to_vec(&serde_json::json!({"ball":{"mass":mass}})).unwrap(),
        )
        .unwrap();
        location = Some(policy_overrides.path().to_string_lossy().into_owned());
    }
    let io = runtime
        .block_on(Robotics::new(
            runtime.handle().clone(),
            Configuration {
                referee: RefereeMode::External,
                competition: Competition::default(),
                parameter_root: root.join("../../etc/parameters"),
                model_directory: root.join("../../etc/neural_networks"),
                router: Some(endpoint),
                namespace: "/motion_trial".into(),
                location,
                robot_count: 1,
                opponent_count: 0,
                field_configuration: None,
                profile: std::env::var("MOTION_TRIAL_PROFILE")
                    .ok()
                    .map(|s| serde_json::from_value(serde_json::Value::String(s)).unwrap())
                    .unwrap_or_default(),
            },
        ))
        .unwrap();
    let mut kickoff_trial = is_kickoff.then(|| kickoff::Trial::new(&runtime, &io));
    let traces = runtime
        .block_on(
            io.node()
                .subscriber::<motion_inference::trace::Record>(motion_inference::trace::TRACE_TOPIC)
                .cache(20000)
                .build(),
        )
        .unwrap();
    let commands = runtime
        .block_on(
            io.node()
                .subscriber::<MotionCommand>("behavior/motion_command")
                .cache(10000)
                .build(),
        )
        .unwrap();
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
    let radius = f64::from(io.field_dimensions.ball_radius);
    let parameters = io.parameters.snapshot().typed().ball.clone();
    let ball = app
        .world_mut()
        .spawn((
            MjcfObject::from_factory(
                move || crate::scene::ball::ball_spec(radius, &parameters),
                "ball",
            )
            .with_free_joint("ball_free_joint")
            .grounded(),
            Transform::from_xyz(position[0] as f32, 0.0, -position[1] as f32),
            crate::scene::ball::Ball,
        ))
        .id();
    app.update();
    let physics = app.world().resource::<SharedPhysics>().clone();
    let contact_audit =
        (std::env::var("MOTION_TRIAL_CONTACT_AUDIT").as_deref() == Ok("1")).then(|| {
            Arc::new(std::sync::Mutex::new(contacts::Audit::new(
                physics.lock().data(),
                &format!("object_{}_", robot.to_bits()),
                &format!("object_{}_", ball.to_bits()),
            )))
        });
    let observed = contact_audit.clone();
    let mut worker =
        PhysicsWorker::start_observed(physics.clone(), io.clone(), move |data, time| {
            if let Some(audit) = &observed {
                audit.lock().unwrap().capture(data, time);
            }
        });
    let start = Instant::now();
    while !worker.ready() {
        assert!(
            start.elapsed() < Duration::from_secs(25),
            "startup: {}",
            io.status()
        );
        assert!(worker.poll().is_none(), "{}", io.status());
        thread::sleep(Duration::from_millis(10));
    }
    // The normal parameter API writes only the simulator's temporary live layer.
    let client =
        RemoteParameterClient::new(io.node().clone(), "/motion_trial/behavior_node").unwrap();
    let snapshot = runtime.block_on(client.get_snapshot()).unwrap();
    let layer = snapshot.layers.last().unwrap().clone();
    let set = |path: &str, value: serde_json::Value| {
        let response = runtime
            .block_on(client.set_json(path, &value, layer.clone(), None))
            .unwrap();
        assert!(response.success, "{}", response.message);
    };
    let head = HeadMotion::ZeroAngles;
    if let Ok(strength) = std::env::var("MOTION_TRIAL_STRENGTH") {
        assert!(matches!(strength.as_str(), "soft" | "medium" | "strong"));
        set("kicking.soft", serde_json::json!(strength == "soft"));
        set(
            "kicking.allow_strong_kicks",
            serde_json::json!(strength == "strong"),
        );
        set("kicking.target_speed", serde_json::json!(speed));
        set("kicking.soft_target_speed", serde_json::json!(speed));
        if strength == "strong" {
            set(
                "kicking.strong_kick_min_target_distance",
                serde_json::json!(0.0),
            );
            set("kicking.strong_target_speed", serde_json::json!(speed));
        }
    }
    set(
        "control.injected_motion_command",
        serde_json::to_value(MotionCommand::Stand { head }).unwrap(),
    );
    {
        let mut world = physics.lock();
        let mut pose = world.object_pose(robot).unwrap();
        pose.translation.x =
            parse("MOTION_TRIAL_ROBOT_X", if is_kickoff { -1.5 } else { 0.0 }) as f32;
        pose.translation.z = -parse("MOTION_TRIAL_ROBOT_Y", 0.0) as f32;
        pose.rotation = Quat::from_rotation_y(parse("MOTION_TRIAL_YAW", 0.0).to_radians() as f32);
        world.set_object_pose(robot, pose).unwrap();
        world.mode = SimulationMode::Running;
    }
    let wait = |seconds: f64| {
        let start = io.now();
        let deadline = Instant::now();
        while io.now().duration_since(start).as_secs_f64() < seconds {
            assert!(
                deadline.elapsed() < Duration::from_secs(30),
                "clock stopped: {}",
                io.status()
            );
            thread::sleep(Duration::from_millis(2));
        }
    };
    wait(0.5);
    physics.lock().mode = SimulationMode::Paused;
    {
        let mut world = physics.lock();
        let data = world.data_mut();
        let joint = data
            .joint(&format!("object_{}_ball_free_joint", ball.to_bits()))
            .unwrap();
        joint.view_mut(data).qvel[..2].copy_from_slice(&if scenario == "walk" {
            [0.0; 2]
        } else {
            velocity
        });
        data.forward();
    }
    if let Some(trial) = &mut kickoff_trial {
        physics.lock().mode = SimulationMode::Running;
        set("control.injected_motion_command", serde_json::Value::Null);
        trial.start(&runtime, &io);
    } else if scenario == "behavior" {
        physics.lock().mode = SimulationMode::Running;
        for event in [
            booster::ButtonEventType::PressDown,
            booster::ButtonEventType::LongPressStart,
            booster::ButtonEventType::LongPressEnd,
            booster::ButtonEventType::PressUp,
        ] {
            io.button_event(2, event).unwrap();
            thread::sleep(Duration::from_millis(30));
        }
        wait(0.1);
        assert_eq!(
            *io.primary.get_latest().unwrap(),
            types::primary_state::PrimaryState::Playing
        );
        set("control.injected_motion_command", serde_json::Value::Null);
    }
    let started = io.now();
    let deadline = Instant::now();
    let mut maximum_speed = 0.0f64;
    let mut maximum_height = 0.0f64;
    let mut final_position = position;
    let mut first_contact = None;
    let mut contact_position = None;
    let mut contact_direction = 0.0;
    let mut followthrough_finished = false;
    let mut position_errors = Vec::new();
    let mut final_velocity = velocity;
    let mut next_command = started;
    let mut cruise_start = None;
    let mut minimum_robot_height = f32::INFINITY;
    let mut maximum_position_error = 0.0f32;
    let mut coordinate_audit = (std::env::var("MOTION_TRIAL_AUDIT").as_deref() == Ok("1"))
        .then(coordinates::Audit::default);
    assert!(
        coordinate_audit.is_none() || scenario == "behavior",
        "coordinate audit requires autonomous behavior"
    );
    physics.lock().mode = SimulationMode::Running;
    while io.now().duration_since(started).as_secs_f64() < duration {
        assert!(
            deadline.elapsed() < Duration::from_secs(60),
            "trial stopped: {}",
            io.status()
        );
        let (ball_position, ball_velocity, contact) = {
            let world = physics.lock();
            let data = world.data();
            if let Some(audit) = &mut coordinate_audit {
                audit.capture(
                    data,
                    &format!("object_{}_", robot.to_bits()),
                    &format!("object_{}_", ball.to_bits()),
                    io.now(),
                );
            }
            let body = data
                .body(&format!("object_{}_ball", ball.to_bits()))
                .unwrap();
            let p = body.view(data).xpos;
            let joint = data
                .joint(&format!("object_{}_ball_free_joint", ball.to_bits()))
                .unwrap();
            let v = joint.view(data).qvel;
            let ball_geom = data
                .model()
                .name_to_id(
                    mujoco_rs::prelude::MjtObj::mjOBJ_GEOM,
                    &format!("object_{}_ball", ball.to_bits()),
                )
                .unwrap() as i32;
            let contact = data.contact().iter().any(|c| {
                c.geom.contains(&ball_geom) && c.geom.iter().any(|g| *g != ball_geom && *g != 0)
            });
            ([p[0], p[1], p[2]], [v[0], v[1]], contact)
        };
        let elapsed = io.now().duration_since(started).as_secs_f64();
        if let Some(trial) = &mut kickoff_trial {
            if trial.observe(
                &runtime,
                &io,
                elapsed,
                ball_position,
                ball_velocity,
                contact,
                &physics,
                robot,
            ) && !followthrough_finished
            {
                set(
                    "control.injected_motion_command",
                    serde_json::to_value(MotionCommand::Stand { head }).unwrap(),
                );
                followthrough_finished = true;
            }
        }
        let current_robot = physics.lock().object_pose(robot).unwrap();
        minimum_robot_height = minimum_robot_height.min(current_robot.translation.y);
        if scenario == "walk"
            && elapsed >= parse("MOTION_TRIAL_DURATION", 6.0) * 0.5
            && cruise_start.is_none()
        {
            cruise_start = Some((elapsed, current_robot.translation));
        }
        if contact && first_contact.is_none() {
            first_contact = Some(elapsed);
            contact_position = Some([ball_position[0], ball_position[1]]);
            if let MotionCommand::Kick { kick_direction, .. } = io.active_motion() {
                let world = physics.lock();
                let binding = crate::robot_io::RobotBinding::new(
                    world.data(),
                    &format!("object_{}_", robot.to_bits()),
                )
                .unwrap();
                let direction = kick_direction.as_unit_vector();
                let aim = binding.ground_to_world(world.data()).rotation
                    * nalgebra::vector![direction.x(), direction.y(), 0.0];
                contact_direction = aim.y.atan2(aim.x).to_degrees() as f64;
            }
        }
        maximum_speed = maximum_speed.max(ball_velocity[0].hypot(ball_velocity[1]));
        maximum_height = maximum_height.max(ball_position[2] - radius);
        final_position = [ball_position[0], ball_position[1]];
        final_velocity = ball_velocity;
        if scenario == "behavior"
            && first_contact.is_none()
            && let MotionCommand::Kick {
                ball_position: commanded,
                ..
            } = io.active_motion()
        {
            let world = physics.lock();
            let binding = crate::robot_io::RobotBinding::new(
                world.data(),
                &format!("object_{}_", robot.to_bits()),
            )
            .unwrap();
            let p = binding
                .ground_to_world(world.data())
                .inverse_transform_point(&nalgebra::point![
                    ball_position[0] as f32,
                    ball_position[1] as f32,
                    0.0
                ]);
            maximum_position_error =
                maximum_position_error.max((commanded - linear_algebra::point![p.x, p.y]).norm());
            position_errors.push((commanded - linear_algebra::point![p.x, p.y]).norm() as f64);
        }
        if scenario == "behavior"
            && !followthrough_finished
            && first_contact
                .is_some_and(|contact| elapsed - contact > parse("MOTION_TRIAL_FOLLOWTHROUGH", 1.0))
        {
            set(
                "control.injected_motion_command",
                serde_json::to_value(MotionCommand::Stand { head }).unwrap(),
            );
            followthrough_finished = true;
        }
        if !autonomous && io.now() >= next_command {
            next_command = io.now() + Duration::from_millis(20);
            let world = physics.lock();
            let binding = crate::robot_io::RobotBinding::new(
                world.data(),
                &format!("object_{}_", robot.to_bits()),
            )
            .unwrap();
            let ground = binding.ground_to_world(world.data());
            let p = ground.inverse_transform_point(&nalgebra::point![
                ball_position[0] as f32,
                ball_position[1] as f32,
                0.0
            ]);
            let v = ground.rotation.inverse()
                * nalgebra::vector![ball_velocity[0] as f32, ball_velocity[1] as f32, 0.0];
            let aim = ground.rotation.inverse() * nalgebra::Vector3::x();
            drop(world);
            if first_contact.is_none()
                && let MotionCommand::Kick { ball_position, .. } = io.active_motion()
            {
                maximum_position_error = maximum_position_error
                    .max((ball_position - linear_algebra::point![p.x, p.y]).norm());
            }
            let command = if scenario == "walk" {
                MotionCommand::WalkWithVelocity {
                    head,
                    velocity: linear_algebra::vector![velocity[0] as f32, velocity[1] as f32],
                    angular_velocity: 0.0,
                }
            } else if first_contact
                .is_some_and(|contact| elapsed - contact > parse("MOTION_TRIAL_FOLLOWTHROUGH", 1.0))
            {
                MotionCommand::Stand { head }
            } else {
                MotionCommand::Kick {
                    head,
                    target_speed: speed,
                    soft: scenario == "soft",
                    quick: scenario == "quick" || scenario == "shot",
                    strong: scenario == "strong" || scenario == "shot",
                    ball_position: linear_algebra::point![p.x, p.y],
                    ball_velocity: linear_algebra::vector![v.x, v.y],
                    kick_direction: linear_algebra::Orientation2::from_vector(
                        linear_algebra::vector![aim.x, aim.y],
                    ),
                }
            };
            set(
                "control.injected_motion_command",
                serde_json::to_value(command).unwrap(),
            );
        }
        thread::sleep(Duration::from_millis(2));
    }
    physics.lock().mode = SimulationMode::Paused;
    let origin = contact_position.unwrap_or(position);
    let distance = (final_position[0] - origin[0]).hypot(final_position[1] - origin[1]);
    let outgoing_direction = (final_position[1] - origin[1])
        .atan2(final_position[0] - origin[0])
        .to_degrees();
    let direction_error =
        (outgoing_direction - contact_direction + 180.0).rem_euclid(360.0) - 180.0;
    let kick_commands = commands
        .get_interval(started, io.now())
        .iter()
        .filter(|command| matches!(***command, MotionCommand::Kick { .. }))
        .count();
    let records = traces.get_interval(started, io.now());
    let mut input_times = Vec::new();
    let mut sensor_ages = Vec::new();
    let mut command_ages = Vec::new();
    let mut request_times = std::collections::HashMap::new();
    let mut service_delays = Vec::new();
    let mut target_speed_features = Vec::new();
    let mut request_samples = Vec::new();
    let mut next_request_sample = started;
    for record in records {
        if let Some(audit) = &mut coordinate_audit {
            audit.inspect(
                &record,
                started + Duration::from_millis(250),
                started
                    + Duration::from_secs_f64(
                        first_contact.unwrap_or(parse("MOTION_TRIAL_DURATION", 6.0)),
                    ),
                f64::from(io.field_dimensions.length) * 0.5,
            );
        }
        match &record.event {
            motion_inference::trace::Event::Request(command) => {
                request_times.insert(record.request_id.clone(), record.time);
                if std::env::var("MOTION_TRIAL_INPUT_SAMPLES").as_deref() == Ok("1")
                    && record.time >= next_request_sample
                    && record.time.duration_since(started) < Duration::from_secs(4)
                {
                    request_samples.push(serde_json::json!({"time":record.time.duration_since(started).as_secs_f64(),"command":command}));
                    next_request_sample = record.time + Duration::from_millis(250);
                }
            }
            motion_inference::trace::Event::Input {
                sources,
                inference_time,
                policy,
                values,
                ..
            } => {
                input_times.push(*inference_time);
                if matches!(
                    policy,
                    motion_inference::config::Policy::Kick
                        | motion_inference::config::Policy::SoftKick
                ) {
                    target_speed_features.push(values[56] as f64);
                }
                if let Some(sources) = sources {
                    sensor_ages.push(
                        inference_time
                            .duration_since(sources.sensor.source_time)
                            .as_secs_f64()
                            * 1000.0,
                    );
                }
                if let Some(request_time) = request_times.get(&record.request_id) {
                    service_delays
                        .push(inference_time.duration_since(*request_time).as_secs_f64() * 1000.0);
                }
                if let Some((stamp, _)) = commands.get_nearest_with_stamp(*inference_time)
                    && stamp <= *inference_time
                {
                    command_ages.push(inference_time.duration_since(stamp).as_secs_f64() * 1000.0);
                }
            }
            _ => {}
        }
    }
    let intervals: Vec<_> = input_times
        .windows(2)
        .map(|pair| pair[1].duration_since(pair[0]).as_secs_f64() * 1000.0)
        .collect();
    let stats = |mut samples: Vec<f64>| {
        samples.sort_by(f64::total_cmp);
        serde_json::json!({"count":samples.len(), "median":samples.get(samples.len()/2),"p95":samples.get(samples.len()*95/100),"max":samples.last()})
    };
    let robot_pose = physics.lock().object_pose(robot).unwrap();
    let coordinate_errors = coordinate_audit
        .as_mut()
        .map(|audit| audit.finish(io.profile.filtering()));
    let cruise_velocity = cruise_start.map(|(time, pose)| {
        let elapsed = io.now().duration_since(started).as_secs_f64() - time;
        [
            (robot_pose.translation.x - pose.x) as f64 / elapsed,
            -(robot_pose.translation.z - pose.z) as f64 / elapsed,
        ]
    });
    eprintln!(
        "MOTION_TRIAL {}",
        serde_json::json!({"scenario":scenario,"profile":io.profile,"strength_override":std::env::var("MOTION_TRIAL_STRENGTH").ok(),"requested_speed":speed,"ball_mass":io.parameters.snapshot().typed().ball.mass,"target_speed_feature":stats(target_speed_features),"input_position":position,"input_velocity":velocity,
        "first_contact_seconds":first_contact,"maximum_ball_speed":maximum_speed,"distance":distance,"lift":maximum_height,"final_position":final_position,
        "final_velocity":final_velocity,"final_ball_speed":final_velocity[0].hypot(final_velocity[1]),"duration_seconds":io.now().duration_since(started).as_secs_f64(),"maximum_command_position_error":maximum_position_error,
        "command_position_error":stats(position_errors),"direction_error_degrees":direction_error,"contact_direction_degrees":contact_direction,"contact_position":contact_position,
        "policy_interval_ms":stats(intervals),"sensor_age_ms":stats(sensor_ages),"command_age_ms":stats(command_ages),"service_delay_ms":stats(service_delays),"request_samples":request_samples,
        "robot_position":robot_pose.translation.to_array(),"coordinate_errors":coordinate_errors,"cruise_velocity":cruise_velocity,"minimum_robot_height":minimum_robot_height,"kick_commands":kick_commands,"status":io.status()})
    );
    if let Some(trial) = kickoff_trial {
        trial.finish(&commands, started, io.now());
    }
    if let Some(audit) = contact_audit {
        audit.lock().unwrap().finish(
            started,
            &scenario,
            f64::from(io.field_dimensions.length) * 0.5,
        );
    }
    assert!(
        !io.status().contains("Emergency stop"),
        "invalid trial: {}",
        io.status()
    );
    drop(worker);
    assert!(
        robot_pose.translation.y > 0.35,
        "robot fell during the trial"
    );
    assert!(minimum_robot_height > 0.30, "robot fell during the trial");
    if scenario == "walk" {
        return;
    }
    assert!(first_contact.is_some(), "policy missed the ball");
    assert!(
        distance >= parse("MOTION_TRIAL_MIN_DISTANCE", 0.0),
        "kick travelled only {distance:.2} m"
    );
    assert!(
        maximum_speed >= parse("MOTION_TRIAL_MIN_BALL_SPEED", 0.0),
        "ball reached only {maximum_speed:.2} m/s"
    );
    let effective_speed = if scenario == "soft" {
        speed.min(1.7)
    } else {
        speed
    };
    assert!(
        maximum_speed >= f64::from(effective_speed) * 0.65,
        "kick too weak: {maximum_speed:.3} m/s for {effective_speed:.3} m/s request"
    );
}

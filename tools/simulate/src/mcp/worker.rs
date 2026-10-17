use crate::{
    bevy_mujoco::{MjcfObject, SharedPhysics, SimulationMode},
    simulation::PhysicsWorker,
    team::Team,
};
use bevy::prelude::*;
use color_eyre::{
    Result,
    eyre::{bail, ensure, eyre},
};
use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Write},
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant},
};

struct Session {
    physics: PhysicsWorker,
    app: App,
    team: Team,
    _router: zenoh::Session,
    inputs: std::collections::HashMap<
        String,
        (
            tokio_util::sync::CancellationToken,
            tokio::task::JoinHandle<Result<()>>,
        ),
    >,
    endpoint: String,
    id: String,
    origin: i64,
    revision: u64,
    rendering: bool,
}
impl Drop for Session {
    fn drop(&mut self) {
        for (cancel, _) in self.inputs.values() {
            cancel.cancel();
        }
        self.team.cancel();
    }
}
pub(super) fn run(socket: &str) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let _guard = runtime.enter();
    let stream = std::os::unix::net::UnixStream::connect(socket)?;
    let mut output = stream.try_clone()?;
    let mut input = BufReader::new(stream);
    let mut session: Option<Session> = None;
    loop {
        let mut line = String::new();
        if input.read_line(&mut line)? == 0 {
            break;
        }
        let request = serde_json::from_str::<Value>(&line);
        let stop = request
            .as_ref()
            .is_ok_and(|request| request["method"] == "stop");
        let result = (|| -> Result<Value> {
            let request = request?;
            if request["method"] == "start" {
                ensure!(session.is_none(), "Session already started");
                session = Some(Session::start(&runtime, &request["arguments"])?);
                return session.as_mut().unwrap().status();
            }
            let current = session.as_mut().ok_or_else(|| eyre!("No session"))?;
            current.command(&runtime, &request)
        })();
        let response = match result {
            Ok(value) => json!({"result":value}),
            Err(error) => json!({"error":format!("{error:#}")}),
        };
        writeln!(output, "{response}")?;
        output.flush()?;
        if stop {
            break;
        }
    }
    drop(session);
    crate::finish_recordings();
    Ok(())
}
impl Session {
    fn start(runtime: &tokio::runtime::Runtime, args: &Value) -> Result<Self> {
        let configuration = &args["configuration"];
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let parameter_root = configuration["parameter_root"]
            .as_str()
            .map(Into::into)
            .unwrap_or_else(|| root.join("etc/parameters"));
        let model_directory = configuration["model_directory"]
            .as_str()
            .map(Into::into)
            .unwrap_or_else(|| root.join("etc/neural_networks"));
        let location = configuration["location"]
            .as_str()
            .ok_or_else(|| eyre!("Missing location"))?;
        let field = crate::FieldConfiguration::load(&parameter_root, location)?.with_ball(
            configuration
                .get("ball")
                .map(|value| serde_json::from_value(value.clone()))
                .transpose()?
                .unwrap_or_default(),
        );
        let mut router_config = zenoh::Config::default();
        router_config
            .insert_json5("mode", "\"router\"")
            .map_err(|e| eyre!("{e}"))?;
        router_config
            .insert_json5("listen/endpoints", "[\"tcp/127.0.0.1:0\"]")
            .map_err(|e| eyre!("{e}"))?;
        router_config
            .insert_json5("scouting/multicast/enabled", "false")
            .map_err(|e| eyre!("{e}"))?;
        let router = runtime
            .block_on(async { zenoh::open(router_config).await })
            .map_err(|e| eyre!("{e}"))?;
        let endpoint = runtime
            .block_on(async { router.info().locators().await })
            .first()
            .ok_or_else(|| eyre!("Router has no endpoint"))?
            .to_string();
        let profile = if configuration["profile"].is_object() {
            crate::Profile::Custom(serde_json::from_value(configuration["profile"].clone())?)
        } else {
            serde_json::from_value(configuration["profile"].clone())?
        };
        let config = crate::Configuration {
            referee: serde_json::from_value(configuration["referee"].clone())?,
            competition: configuration
                .get("competition")
                .map(|value| serde_json::from_value(value.clone()))
                .transpose()?
                .unwrap_or_default(),
            parameter_root,
            model_directory,
            router: Some(endpoint.clone()),
            namespace: "/hulks/1".into(),
            location: Some(location.into()),
            robot_count: configuration["robot_count"].as_u64().unwrap_or(1) as u8,
            opponent_count: configuration["opponent_count"].as_u64().unwrap_or(0) as u8,
            field_configuration: Some(field),
            profile,
        };
        let team = runtime.block_on(Team::new_with_controller(
            runtime.handle().clone(),
            config,
            Arc::new(AtomicBool::new(false)),
            None,
            false,
        ))?;
        let clock = Arc::new(super::clock::SharedClock::open(std::path::Path::new(
            args["clock_path"]
                .as_str()
                .ok_or_else(|| eyre!("Missing clock path"))?,
        ))?);
        team.set_clock_sink(Arc::new(move |time| clock.store(time)));
        let origin = team.now().as_nanos();
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, crate::bevy_mujoco::MujocoWorldPlugin));
        for member in team.members() {
            spawn_robot(&mut app, &team, &member);
        }
        if team.referee().is_some() {
            let radius = team.members()[0].io.field_dimensions.ball_radius;
            spawn_ball(&mut app, &team, Transform::from_xyz(0.0, radius, 0.0));
        }
        let dimensions = team.members()[0].io.field_dimensions;
        app.world_mut().spawn((
            MjcfObject::from_factory(
                move || crate::scene::markings::mujoco_field(dimensions),
                "field",
            ),
            Transform::default(),
        ));
        for (x, angle) in [
            (-dimensions.length / 2.0, 0.0),
            (dimensions.length / 2.0, std::f32::consts::PI),
        ] {
            let goal = crate::scene::goal::GoalDimensions::from(&dimensions);
            app.world_mut().spawn((
                MjcfObject::from_factory(move || crate::scene::goal::goal_spec(goal), "goal")
                    .with_mocap_body("goal"),
                Transform::from_xyz(x, 0.0, 0.0).with_rotation(Quat::from_rotation_y(angle)),
            ));
        }
        app.update();
        let physics = PhysicsWorker::start_team(
            app.world().resource::<SharedPhysics>().clone(),
            team.clone(),
        );
        let mut session = Self {
            physics,
            app,
            team,
            _router: router,
            inputs: Default::default(),
            endpoint,
            id: args["session_id"].as_str().unwrap().into(),
            origin,
            revision: 0,
            rendering: configuration["rendering"] != "disabled",
        };
        let deadline = Instant::now() + Duration::from_secs(30);
        while !session.physics.ready() {
            ensure!(Instant::now() < deadline, "Physics startup timed out");
            if let Some(error) = session.physics.poll() {
                bail!("{error}");
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        Ok(session)
    }
    fn target(&self, id: crate::RobotId) -> String {
        format!("{}:{}:{}", self.id, id.team.name(), id.number)
    }
    fn member(&self, target: &Value) -> Result<crate::team::Member> {
        self.team
            .members()
            .into_iter()
            .find(|member| target.as_str() == Some(&self.target(member.id)))
            .ok_or_else(|| eyre!("Unknown target {target}"))
    }
    fn shared(&self) -> SharedPhysics {
        self.app.world().resource::<SharedPhysics>().clone()
    }
    fn status(&mut self) -> Result<Value> {
        let failure = self.physics.poll();
        let shared = self.shared();
        let world = shared.lock();
        let targets: Vec<_> = self.team.members().iter().map(|member| json!({"target_id":self.target(member.id),"session_id":self.id,"namespace":crate::robot_namespace(member.id),"transport_scope":crate::transport_scope(member.id),"live":failure.is_none(),"kind":"robot","recording_directory":member.io.recording_status().map(|(path,_)|path)})).collect();
        Ok(
            json!({"session_id":self.id,"state":if failure.is_some(){"failed"}else if world.mode == SimulationMode::Paused {"paused"}else{"running"},"simulation_time_ns":(self.team.now().as_nanos()-self.origin).to_string(),"clock_origin_ns":self.origin.to_string(),"timestep_ns":Duration::from_secs_f64(world.data().model_opt().timestep).as_nanos().to_string(),"remaining_steps":world.remaining_steps,"scene_revision":self.revision,"speed":self.physics.speed().factor(),"targets":targets,"endpoint":self.endpoint,"error":failure}),
        )
    }
    fn command(&mut self, runtime: &tokio::runtime::Runtime, request: &Value) -> Result<Value> {
        let args = &request["arguments"];
        match request["method"].as_str().unwrap_or("") {
            "status" => self.status(),
            "stop" => {
                for (cancel, task) in self.inputs.drain().map(|(_, input)| input) {
                    cancel.cancel();
                    let _ = runtime.block_on(task);
                }
                self.shared().lock().mode = SimulationMode::Paused;
                self.team.cancel();
                self.status()
            }
            "control" => {
                let shared = self.shared();
                let mut world = shared.lock();
                match args["kind"].as_str().unwrap_or("") {
                    "pause" => {
                        world.mode = SimulationMode::Paused;
                        world.remaining_steps = None;
                    }
                    "resume" => {
                        ensure!(
                            world.remaining_steps.is_none_or(|n| n == 0),
                            "Advance in progress"
                        );
                        world.remaining_steps = None;
                        world.mode = SimulationMode::Running;
                    }
                    "advance" => {
                        ensure!(
                            world.mode == SimulationMode::Paused,
                            "Advance requires paused simulation"
                        );
                        let step =
                            Duration::from_secs_f64(world.data().model_opt().timestep).as_nanos();
                        let steps = if let Some(steps) = args["steps"].as_u64() {
                            steps
                        } else {
                            u64::try_from(
                                args["duration_ns"]
                                    .as_str()
                                    .ok_or_else(|| eyre!("Missing duration"))?
                                    .parse::<u128>()?
                                    .div_ceil(step),
                            )?
                        };
                        ensure!(steps > 0, "Advance requires positive steps");
                        world.remaining_steps = Some(steps);
                        world.mode = SimulationMode::Running;
                    }
                    "set_speed" => {
                        let factor = args["factor"].as_f64().unwrap_or(0.0);
                        self.physics.set_speed(
                            crate::simulation::Speed::ALL
                                .into_iter()
                                .find(|speed| speed.factor() == factor)
                                .ok_or_else(|| eyre!("Invalid speed"))?,
                        );
                    }
                    other => bail!("Unknown control {other}"),
                }
                drop(world);
                self.status()
            }
            "scene" => self.scene(),
            "edit" => self.edit(runtime, args),
            "input" => self.input(runtime, args),
            "input_status" => {
                let key = args["id"]
                    .as_str()
                    .ok_or_else(|| eyre!("Missing input ID"))?;
                let (cancel, task) = self.inputs.get(key).ok_or_else(|| eyre!("Unknown input"))?;
                if args["cancel"] == true {
                    cancel.cancel();
                }
                if !task.is_finished() {
                    return Ok(json!({"done":false}));
                }
                let (_, task) = self.inputs.remove(key).unwrap();
                runtime.block_on(task)??;
                Ok(json!({"done":true,"effect":"applied"}))
            }
            "checkpoint" => {
                let member = self.member(&args["target_id"])?;
                runtime.block_on(member.io.recording_checkpoint())
            }
            "render" => self.render(args),
            method => bail!("Unknown worker method {method}"),
        }
    }
    fn scene(&self) -> Result<Value> {
        let shared = self.shared();
        let world = shared.lock();
        let mut objects = Vec::new();
        for member in self.team.members() {
            if let Some(entity) = member.entity
                && let Some(pose) = world.object_pose(entity)
            {
                objects.push(json!({"entity_id":format!("{}:{}",self.id,entity.to_bits()),"kind":"robot","target_id":self.target(member.id),"pose":pose_json(pose)}));
            }
        }
        for entity in &world.balls {
            if let Some(pose) = world.object_pose(*entity) {
                objects.push(json!({"entity_id":format!("{}:{}",self.id,entity.to_bits()),"kind":"ball","pose":pose_json(pose)}));
            }
        }
        Ok(
            json!({"session_id":self.id,"scene_revision":self.revision,"simulation_time_ns":(self.team.now().as_nanos()-self.origin).to_string(),"objects":objects,"field":self.team.configuration().field_configuration.as_ref().unwrap().dimensions}),
        )
    }
    fn edit(&mut self, runtime: &tokio::runtime::Runtime, args: &Value) -> Result<Value> {
        ensure!(
            args["expected_scene_revision"].as_u64() == Some(self.revision),
            "Scene revision conflict"
        );
        let shared = self.shared();
        ensure!(
            shared.lock().mode == SimulationMode::Paused,
            "Scene edits require paused simulation"
        );
        let edit = &args["edit"];
        match edit["kind"].as_str().unwrap_or("") {
            "add_robot" => {
                let side = if edit["team"] == "hulks" {
                    crate::TeamId::Hulks
                } else {
                    crate::TeamId::Opponents
                };
                let members = self.team.members();
                let pose = crate::team::vacant_spawn(
                    &members[0].io.field_dimensions,
                    &members.iter().map(|m| m.pose).collect::<Vec<_>>(),
                    side.default_away(),
                )
                .ok_or_else(|| eyre!("No vacant spawn"))?;
                let id = runtime.block_on(self.team.add(side, pose))?;
                let member = self
                    .team
                    .members()
                    .into_iter()
                    .find(|member| member.id == id)
                    .unwrap();
                spawn_robot(&mut self.app, &self.team, &member);
            }
            "add_ball" => {
                let entity = spawn_ball(&mut self.app, &self.team, parse_pose(&edit["pose"])?);
                self.app.update();
                if let Some(velocity) = edit.get("linear_velocity_m_s") {
                    let velocity: [f64; 3] = serde_json::from_value(velocity.clone())?;
                    shared
                        .lock()
                        .set_object_velocity(entity, velocity)
                        .map_err(|e| eyre!(e))?;
                }
            }
            "move_object" => {
                let entity = entity(edit, &self.id)?;
                shared
                    .lock()
                    .set_object_pose_with_velocity(
                        entity,
                        parse_pose(&edit["pose"])?,
                        edit["clear_velocity"].as_bool().unwrap_or(true),
                    )
                    .map_err(|e| eyre!(e))?;
            }
            "remove_ball" => {
                let entity = entity(edit, &self.id)?;
                ensure!(
                    shared.lock().balls.contains(&entity),
                    "Entity is not a ball"
                );
                self.app.world_mut().despawn(entity);
            }
            "reset_robot_pose" => {
                let member = self.member(&edit["target_id"])?;
                let mut world = shared.lock();
                let entity = member.entity.unwrap();
                let binding = crate::robot_io::RobotBinding::new(
                    world.data(),
                    &format!("object_{}_", entity.to_bits()),
                )?;
                binding.reset_joints(world.data_mut());
                world
                    .ground_object(entity, member.pose)
                    .map_err(|e| eyre!(e))?;
            }
            kind => bail!("Unknown edit {kind}"),
        }
        self.app.update();
        self.revision += 1;
        self.scene()
    }
    fn input(&mut self, runtime: &tokio::runtime::Runtime, input: &Value) -> Result<Value> {
        ensure!(self.inputs.len() < 32, "Too many pending inputs");
        match input["kind"].as_str().unwrap_or("") {
            "whistle" => {
                for member in self.team.members() {
                    if input["target_ids"]
                        .as_array()
                        .is_none_or(|ids| ids.contains(&json!(self.target(member.id))))
                    {
                        member.io.whistle();
                    }
                }
            }
            "body_button" => {
                let member = self.member(&input["target_id"])?;
                let button = match input["button"].as_str() {
                    Some("f1") => 0,
                    Some("stand") => 1,
                    Some("walk") => 2,
                    _ => bail!("Unknown button"),
                };
                let hold = input["gesture"] == "hold";
                let cancel = tokio_util::sync::CancellationToken::new();
                let cancelled = cancel.clone();
                let task = runtime.spawn(async move {
                    use booster::ButtonEventType::*;
                    member.io.button_event(button, PressDown)?;
                    let result = async {
                        if hold {
                            tokio::select! {_ = tokio::time::sleep(Duration::from_secs(1)) => {
                                member.io.button_event(button, LongPressStart)?;
                                member.io.button_event(button, LongPressHold)?;
                                member.io.button_event(button, LongPressEnd)?;
                            }, _ = cancelled.cancelled() => {}}
                        }
                        Ok::<_, color_eyre::Report>(())
                    }
                    .await;
                    member.io.button_event(button, PressUp)?;
                    if !hold {
                        member.io.button_event(button, SingleClick)?;
                    }
                    result
                });
                let id = super::server::id();
                self.inputs.insert(id.clone(), (cancel, task));
                return Ok(json!({"id":id,"done":false}));
            }

            "referee" => self
                .team
                .referee()
                .ok_or_else(|| eyre!("External referee mode"))?
                .command(input["name"].as_str().unwrap_or(""), &input["arguments"])
                .map_err(|e| eyre!(e))?,
            "controller" => {
                let member = self.member(&input["target_id"])?;
                let value = serde_json::from_value(input["value"].clone())?;
                let duration =
                    Duration::from_millis(input["wall_duration_ms"].as_u64().unwrap_or(1));
                let cancel = tokio_util::sync::CancellationToken::new();
                let cancelled = cancel.clone();
                let task = runtime.spawn(async move {
                    member.io.controller_input(value, duration, cancelled).await
                });
                let id = super::server::id();
                self.inputs.insert(id.clone(), (cancel, task));
                return Ok(json!({"id":id,"done":false}));
            }

            kind => bail!("Unknown input {kind}"),
        }
        Ok(json!({"done":true,"effect":"applied"}))
    }
    fn render(&self, args: &Value) -> Result<Value> {
        ensure!(self.rendering, "Rendering disabled for session");
        let shared = self.shared();
        let mut world = shared.lock();
        let mut camera = mujoco_rs::prelude::MjvCamera::default();
        let field = &self
            .team
            .configuration()
            .field_configuration
            .as_ref()
            .unwrap()
            .dimensions;
        camera.lookat = [0.0, 0.0, 0.0];
        camera.distance = f64::from(field.length) * 1.2;
        camera.azimuth = 90.0;
        camera.elevation = -45.0;
        let specification = &args["camera"];
        if specification["name"] == "top" {
            camera.elevation = -90.0;
        }
        if specification["name"] == "follow_robot" {
            let member = self.member(&specification["target_id"])?;
            let pose = world.object_pose(member.entity.unwrap()).unwrap();
            camera.lookat = [
                pose.translation.x as f64,
                -pose.translation.z as f64,
                pose.translation.y as f64,
            ];
            camera.distance = 3.0;
        }
        if specification["kind"] == "world" {
            let position: [f64; 3] = serde_json::from_value(specification["position_m"].clone())?;
            let look: [f64; 3] = serde_json::from_value(specification["look_at_m"].clone())?;
            let delta = [
                position[0] - look[0],
                position[1] - look[1],
                position[2] - look[2],
            ];
            camera.lookat = look;
            camera.distance = (delta.iter().map(|x| x * x).sum::<f64>()).sqrt();
            ensure!(camera.distance > 0.01, "Camera position equals look-at");
            camera.azimuth = delta[1].atan2(delta[0]).to_degrees();
            camera.elevation = -(delta[2] / camera.distance).asin().to_degrees();
        }
        let width = args["size"]["width"].as_u64().unwrap_or(1280) as u32;
        let height = args["size"]["height"].as_u64().unwrap_or(720) as u32;
        world.data_mut().model_vis_mut().headlight.ambient = [0.5, 0.5, 0.5];
        world.data_mut().model_vis_mut().headlight.diffuse = [0.7, 0.7, 0.7];
        world.data_mut().model_vis_mut().global.offwidth = width as i32;
        world.data_mut().model_vis_mut().global.offheight = height as i32;
        let mut renderer = mujoco_rs::renderer::MjRenderer::builder()
            .width(width)
            .height(height)
            .rgb(true)
            .camera(camera)
            .build(world.data().model())?;
        renderer.sync_data(world.data_mut())?;
        renderer.render()?;
        let path = args["path"]
            .as_str()
            .ok_or_else(|| eyre!("Missing render destination"))?;
        renderer.save_rgb(path)?;
        Ok(
            json!({"path":path,"width":width,"height":height,"simulation_time_ns":(self.team.now().as_nanos()-self.origin).to_string()}),
        )
    }
}
fn spawn_robot(app: &mut App, team: &Team, member: &crate::team::Member) {
    let entity = app
        .world_mut()
        .spawn((
            MjcfObject::new(
                concat!(env!("CARGO_MANIFEST_DIR"), "/assets/k1_robot.xml"),
                "Trunk",
            )
            .with_free_joint("world_joint")
            .grounded(),
            member.pose,
            crate::simulation::ControlledRobot,
        ))
        .id();
    team.bind(member.id, entity);
}
fn spawn_ball(app: &mut App, team: &Team, pose: Transform) -> Entity {
    let member = &team.members()[0];
    let radius = member.io.field_dimensions.ball_radius as f64;
    let parameters = member.io.parameters.snapshot().typed().ball.clone();
    app.world_mut()
        .spawn((
            MjcfObject::from_factory(
                move || crate::scene::ball::ball_spec(radius, &parameters),
                "ball",
            )
            .with_free_joint("ball_free_joint"),
            pose,
            crate::scene::ball::Ball,
        ))
        .id()
}
fn entity(edit: &Value, session: &str) -> Result<Entity> {
    let id = edit["entity_id"]
        .as_str()
        .ok_or_else(|| eyre!("Missing entity"))?;
    let bits = id
        .strip_prefix(&format!("{session}:"))
        .ok_or_else(|| eyre!("Entity belongs to another session"))?;
    Ok(Entity::from_bits(bits.parse()?))
}

fn parse_pose(value: &Value) -> Result<Transform> {
    let p: [f32; 3] = serde_json::from_value(value["position_m"].clone())?;
    let q: [f32; 4] = serde_json::from_value(value["orientation_xyzw"].clone())?;
    ensure!(
        p.iter().chain(q.iter()).all(|x| x.is_finite()),
        "Pose must be finite"
    );
    let rotation = Quat::from_xyzw(q[0], q[2], -q[1], q[3]);
    ensure!(
        (rotation.length() - 1.0).abs() < 1e-4,
        "Quaternion must be normalized"
    );
    Ok(Transform::from_xyz(p[0], p[2], -p[1]).with_rotation(rotation))
}
fn pose_json(pose: Transform) -> Value {
    let p = pose.translation;
    let q = pose.rotation;
    json!({"frame":"world","position_m":[p.x,-p.z,p.y],"orientation_xyzw":[q.x,-q.z,q.y,q.w]})
}

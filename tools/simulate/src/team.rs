use crate::{
    network::Network,
    robotics::{Configuration, RobotSettings, Robotics},
};
use bevy::prelude::*;
use color_eyre::{
    Result,
    eyre::{ensure, eyre},
};
use ros_z::time::Clock;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
use tokio::runtime::Handle;

#[derive(Clone)]
pub(crate) struct Member {
    pub number: u8,
    pub io: Robotics,
    pub entity: Option<Entity>,
    pub pose: Transform,
    _network: Arc<crate::network::Connection>,
}
#[derive(Resource, Clone)]
pub(crate) struct Team(Arc<Inner>);
struct Inner {
    cancelled: Arc<std::sync::atomic::AtomicBool>,
    runtime: Handle,
    configuration: Configuration,
    clock: Clock,
    network: tokio::sync::Mutex<Network>,
    members: Mutex<BTreeMap<u8, Member>>,
    selected: std::sync::atomic::AtomicU8,
    controller: Mutex<Option<crate::controller::Controller>>,
    _lease: Option<crate::Lease>,
}
impl Team {
    #[cfg(test)]
    pub async fn new(runtime: Handle, configuration: Configuration) -> Result<Self> {
        Self::new_cancellable(
            runtime,
            configuration,
            Arc::new(std::sync::atomic::AtomicBool::new(false)),
            None,
        )
        .await
    }
    pub async fn new_cancellable(
        runtime: Handle,
        configuration: Configuration,
        cancelled: Arc<std::sync::atomic::AtomicBool>,
        lease: Option<crate::Lease>,
    ) -> Result<Self> {
        ensure!(
            (1..=5).contains(&configuration.robot_count),
            "Choose one to five robots"
        );
        let field = configuration
            .field_configuration
            .as_ref()
            .ok_or_else(|| eyre!("Missing field configuration"))?;
        field.validate().map_err(|error| eyre!(error))?;
        let team = Self(Arc::new(Inner {
            cancelled,
            _lease: lease,
            runtime,
            clock: Clock::logical(Clock::wallclock().now()),
            network: tokio::sync::Mutex::new(Network::new(3838, 3939).await?),
            members: Mutex::new(BTreeMap::new()),
            selected: std::sync::atomic::AtomicU8::new(1),
            controller: Mutex::new(None),
            configuration: configuration.clone(),
        }));
        for index in 0..configuration.robot_count {
            team.add(spawn_pose(&field.dimensions, index)).await?;
        }
        if team.members().iter().any(|member| member.io.away()) {
            for member in team.0.members.lock().unwrap().values_mut() {
                member.pose = on_field_side(member.pose, true);
            }
        }
        if configuration.controller == crate::ControllerSource::Local {
            let members = team.members();
            let controller = crate::controller::Controller::new(&configuration, members).await?;
            *team.0.controller.lock().unwrap() = Some(controller);
        }
        Ok(team)
    }
    pub fn poll_controller(&self) -> Option<String> {
        self.0
            .controller
            .lock()
            .unwrap()
            .as_mut()
            .and_then(|controller| controller.poll())
    }
    pub fn now(&self) -> ros_z::time::Time {
        self.0.clock.now()
    }
    pub fn advance(&self, period: std::time::Duration) -> Result<()> {
        self.0.clock.advance(period)?;
        Ok(())
    }
    pub fn cancel(&self) {
        self.0
            .cancelled
            .store(true, std::sync::atomic::Ordering::Release);
    }
    pub fn runtime(&self) -> &Handle {
        &self.0.runtime
    }
    pub fn configuration(&self) -> &Configuration {
        &self.0.configuration
    }
    pub fn members(&self) -> Vec<Member> {
        self.0.members.lock().unwrap().values().cloned().collect()
    }
    pub fn selected(&self) -> Option<Member> {
        self.0
            .members
            .lock()
            .unwrap()
            .get(&self.0.selected.load(std::sync::atomic::Ordering::Relaxed))
            .cloned()
    }
    pub fn select(&self, number: u8) {
        if let Some(controller) = self.0.controller.lock().unwrap().as_ref() {
            controller.select(number);
        }
        if self.0.members.lock().unwrap().contains_key(&number) {
            self.0
                .selected
                .store(number, std::sync::atomic::Ordering::Relaxed);
        }
    }
    pub fn bind(&self, number: u8, entity: Entity) {
        self.0
            .members
            .lock()
            .unwrap()
            .get_mut(&number)
            .unwrap()
            .entity = Some(entity);
    }
    pub async fn add(&self, pose: Transform) -> Result<u8> {
        ensure!(
            !self.0.cancelled.load(std::sync::atomic::Ordering::Acquire),
            "Simulator stopped"
        );
        // Serializes allocation and startup, including dynamic additions.
        let mut network = self.0.network.lock().await;
        let number = (1..=5)
            .find(|number| !self.0.members.lock().unwrap().contains_key(number))
            .ok_or_else(|| eyre!("All five player numbers are in use"))?;
        let player = [
            hsl_network_messages::PlayerNumber::One,
            hsl_network_messages::PlayerNumber::Two,
            hsl_network_messages::PlayerNumber::Three,
            hsl_network_messages::PlayerNumber::Four,
            hsl_network_messages::PlayerNumber::Five,
        ][usize::from(number - 1)];
        let mut configuration = self.0.configuration.clone();
        configuration.namespace = format!("/{number}");
        configuration.controller = crate::ControllerSource::External;
        let connection = Arc::new(network.add_robot().await?);
        let io = Robotics::new_robot(
            self.0.runtime.clone(),
            configuration,
            RobotSettings {
                player,
                scope: transport_scope(number),
                clock: self.0.clock.clone(),
                ports: connection.ports,
            },
        )
        .await?;
        crate::simulation::bootstrap(&io, &self.0.cancelled)?;
        let member = Member {
            number,
            io,
            entity: None,
            pose,
            _network: connection,
        };
        if let Some(controller) = self.0.controller.lock().unwrap().as_ref() {
            controller.add(member.clone());
        }
        self.0.members.lock().unwrap().insert(number, member);
        Ok(number)
    }
}
pub fn transport_scope(number: u8) -> String {
    format!("{}/{number}", crate::ZENOH_NAMESPACE)
}

pub(crate) fn spawn_pose(field: &types::field_dimensions::FieldDimensions, index: u8) -> Transform {
    let side = if index.is_multiple_of(2) { 1.0 } else { -1.0 };
    Transform::from_xyz(
        -field.length / 2.0 + 0.5 + f32::from(index / 2) * 0.9,
        0.0,
        side * field.width / 2.0,
    )
    .with_rotation(Quat::from_rotation_y(side * std::f32::consts::FRAC_PI_2))
}
fn on_field_side(mut pose: Transform, away: bool) -> Transform {
    if away {
        pose.translation.x = -pose.translation.x;
        pose.translation.z = -pose.translation.z;
        pose.rotation = Quat::from_rotation_y(std::f32::consts::PI) * pose.rotation;
    }
    pose
}
pub(crate) fn vacant_spawn(
    field: &types::field_dimensions::FieldDimensions,
    occupied: &[Transform],
    away: bool,
) -> Option<Transform> {
    for side in 0..2 {
        for slot in 0..((field.length / 2.0 - 0.5) / 0.9).ceil() as u8 {
            let pose = on_field_side(spawn_pose(field, slot * 2 + side), away);
            if occupied.iter().all(|other| {
                let delta = pose.translation - other.translation;
                Vec2::new(delta.x, delta.z).length() >= 0.8
            }) {
                return Some(pose);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn initial_and_dynamic_slots_stay_in_own_half_and_avoid_occupied_positions() {
        let field = crate::FieldConfiguration::load(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../etc/parameters")
                .as_path(),
            "hsl_small",
        )
        .unwrap()
        .dimensions;
        let occupied: Vec<_> = (0..5).map(|i| spawn_pose(&field, i)).collect();
        assert_eq!(occupied.iter().filter(|p| p.translation.z > 0.0).count(), 3);
        assert!(occupied.iter().all(|p| p.translation.x < 0.0));
        let next = vacant_spawn(&field, &occupied, false).unwrap();
        assert!(next.translation.x < 0.0);
        assert!(
            occupied
                .iter()
                .all(|p| p.translation.distance(next.translation) >= 0.8)
        );
        assert!(vacant_spawn(&field, &[], true).unwrap().translation.x > 0.0);
    }
}

#[cfg(test)]
mod runtime_tests {
    use super::*;
    use crate::{
        bevy_mujoco::{MjcfObject, MujocoWorldPlugin, SharedPhysics, SimulationMode},
        simulation::PhysicsWorker,
    };
    use ros_z::prelude::*;
    use std::{
        thread,
        time::{Duration, Instant},
    };
    fn bind_scene(app: &mut App, team: &Team) {
        for member in team.members() {
            if member.entity.is_some() {
                continue;
            }
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
                ))
                .id();
            team.bind(member.number, entity);
        }
        app.update();
    }
    fn wait(worker: &mut PhysicsWorker, mut predicate: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            assert!(worker.poll().is_none(), "physics worker failed");
            if predicate() {
                break;
            }
            assert!(Instant::now() < deadline, "condition timed out");
            thread::sleep(Duration::from_millis(10));
        }
    }
    fn validate_game_controller(
        runtime: &tokio::runtime::Runtime,
        team: &Team,
        physics: &SharedPhysics,
        worker: &mut PhysicsWorker,
        exchange: &std::path::Path,
    ) {
        use types::filtered_game_state::FilteredGameState;
        let members = team.members();
        let observations = runtime.block_on(async {
            let mut observations = Vec::new();
            for member in &members {
                let node = member.io.node();
                let filtered = node.subscriber::<types::filtered_game_controller_state::FilteredGameControllerState>("filtered_game_controller_state").cache(1).build().await.unwrap();
                let raw = node.subscriber::<Option<types::game_controller_state::GameControllerState>>("game_controller_state").cache(1).build().await.unwrap();
                observations.push((filtered,raw));
            }
            observations
        });
        physics.lock().mode = SimulationMode::Running;
        for (stage, penalized) in [
            ("ready", false),
            ("penalized", true),
            ("unpenalized", false),
        ] {
            wait(worker, || {
                observations.iter().all(|(state, _)| {
                    state.get_latest().is_some_and(|state| {
                        state.game_state == FilteredGameState::Ready
                            && state.penalties[hsl_network_messages::PlayerNumber::Three].is_some()
                                == penalized
                    })
                })
            });
            if stage == "penalized" {
                wait(worker, || {
                    members[2].io.primary.get_latest().is_some_and(|state| {
                        *state == types::primary_state::PrimaryState::Penalized
                    })
                });
                for member in [&members[0], &members[1], &members[3], &members[4]] {
                    assert_ne!(
                        *member.io.primary.get_latest().unwrap(),
                        types::primary_state::PrimaryState::Penalized
                    );
                }
            }
            std::fs::write(exchange.join(stage), "observed by five robots").unwrap();
        }
        wait(worker, || {
            observations.iter().all(|(state, _)| {
                state
                    .get_latest()
                    .is_some_and(|state| state.game_state == FilteredGameState::Set)
            })
        });
        for member in &members {
            member.io.whistle();
        }
        wait(worker, || {
            observations.iter().all(|(state, _)| {
                state.get_latest().is_some_and(|state| {
                    matches!(state.game_state, FilteredGameState::Playing { .. })
                })
            })
        });
        std::fs::write(exchange.join("whistle_in_set"), "observed").unwrap();
        wait(worker, || {
            observations.iter().all(|(_, state)| {
                state.get_latest().is_some_and(|state| {
                    state.as_ref().as_ref().is_some_and(|state| {
                        state.game_state == hsl_network_messages::GameState::Playing
                    })
                })
            })
        });
        physics.lock().mode = SimulationMode::Paused;
        thread::sleep(Duration::from_millis(50));
        let frozen = team.now();
        let remaining = observations[0]
            .1
            .get_latest()
            .unwrap()
            .as_ref()
            .as_ref()
            .unwrap()
            .remaining_time_in_half;
        wait(worker, || {
            observations.iter().all(|(_, state)| {
                state.get_latest().is_some_and(|state| {
                    state.as_ref().as_ref().is_some_and(|state| {
                        remaining.saturating_sub(state.remaining_time_in_half)
                            >= Duration::from_secs(2)
                    })
                })
            })
        });
        assert_eq!(team.now(), frozen);
        std::fs::write(exchange.join("paused_clock"), "observed").unwrap();
        wait(worker, || exchange.join("returns_paused").exists());
        physics.lock().mode = SimulationMode::Running;
        wait(worker, || exchange.join("returns_resumed").exists());
    }

    #[test]
    #[ignore = "requires motion models, ONNX Runtime and GameController ports"]
    fn five_players_dynamic_spawn_namespaces_and_shared_time() {
        let _ = tracing_subscriber::fmt().with_env_filter("motion=warn,hardware_interface=warn,motion_inference=warn,primary_state_filter=warn").try_init();
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
        let profile = std::env::var("SIMULATOR_TEST_PROFILE")
            .ok()
            .map(|name| serde_json::from_value(serde_json::Value::String(name)).unwrap())
            .unwrap_or(crate::Profile::MotionBehavior);
        let configuration = Configuration {
            field_configuration: Some(
                crate::FieldConfiguration::load(&parameter_root, "incheon_small").unwrap(),
            ),
            parameter_root,
            model_directory: root.join("../../etc/neural_networks"),
            router: Some(endpoint),
            namespace: "/1".into(),
            location: Some("incheon_small".into()),
            robot_count: std::env::var("SIMULATOR_TEST_ROBOTS")
                .ok()
                .map(|n| n.parse().unwrap())
                .unwrap_or(2),
            profile,
            controller: crate::ControllerSource::External,
        };
        let team = runtime
            .block_on(Team::new(runtime.handle().clone(), configuration))
            .unwrap();
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, MujocoWorldPlugin));
        bind_scene(&mut app, &team);
        let physics = app.world().resource::<SharedPhysics>().clone();
        physics.lock().mode = SimulationMode::Paused;
        let mut worker = PhysicsWorker::start_team(physics.clone(), team.clone());
        wait(&mut worker, || {
            team.members().iter().all(|member| member.entity.is_some())
        });
        let first = team.members()[0].entity.unwrap();
        thread::sleep(Duration::from_millis(100));
        let initial_pose = physics.lock().object_pose(first).unwrap();
        let time = team.now();
        let add_while_running = std::env::var_os("SIMULATOR_TEST_ADD_RUNNING").is_some();
        if add_while_running {
            physics.lock().mode = SimulationMode::Running;
        }
        for number in team.members().len() as u8 + 1..=5 {
            let occupied = {
                let world = physics.lock();
                world
                    .robots
                    .iter()
                    .filter_map(|id| world.object_pose(*id))
                    .collect::<Vec<_>>()
            };
            let pose =
                vacant_spawn(&team.members()[0].io.field_dimensions, &occupied, false).unwrap();
            assert_eq!(runtime.block_on(team.add(pose)).unwrap(), number);
            bind_scene(&mut app, &team);
            thread::sleep(Duration::from_millis(100));
            wait(&mut worker, || {
                team.members().iter().all(|member| member.entity.is_some())
            });
            if !add_while_running {
                assert_eq!(
                    team.now(),
                    time,
                    "Adding a robot must not advance a paused simulation"
                );
                assert!(
                    physics
                        .lock()
                        .object_pose(first)
                        .unwrap()
                        .translation
                        .distance(initial_pose.translation)
                        < 1e-6
                );
            }
        }
        assert!(runtime.block_on(team.add(Transform::default())).is_err());
        let members = team.members();
        let ids = runtime.block_on(async {
            let mut ids = Vec::new();
            for member in &members {
                let node = member.io.node();
                let player = node
                    .subscriber::<hsl_network_messages::PlayerNumber>("player_number")
                    .qos(QosProfile {
                        durability: ros_z::qos::QosDurability::TransientLocal,
                        ..Default::default()
                    })
                    .build()
                    .await
                    .unwrap();
                let player = tokio::time::timeout(Duration::from_secs(3), player.recv())
                    .await
                    .unwrap()
                    .unwrap();
                ids.push(player.to_string());
            }
            ids
        });
        assert_eq!(ids, ["1", "2", "3", "4", "5"]);
        if let Ok(exchange) = std::env::var("HSL_EXCHANGE") {
            validate_game_controller(
                &runtime,
                &team,
                &physics,
                &mut worker,
                std::path::Path::new(&exchange),
            );
        } else {
            for member in &members {
                assert_eq!(member.io.now(), team.now());
                assert_eq!(
                    *member.io.primary.get_latest().unwrap(),
                    types::primary_state::PrimaryState::Initial,
                    "player {}: {}",
                    member.number,
                    member.io.status()
                );
                assert_eq!(member.io.field_dimensions.length, 8.95);
            }
            if profile == crate::Profile::Localization {
                let estimates = runtime.block_on(async {
                    let mut caches = Vec::new();
                    for member in &members {
                        caches.push((
                            member
                                .io
                                .node()
                                .subscriber::<types::localization::LocalizationEstimate>(
                                    "localization/estimate",
                                )
                                .cache(1)
                                .build()
                                .await
                                .unwrap(),
                            member
                                .io
                                .node()
                                .subscriber::<linear_algebra::Isometry2<
                                    coordinate_systems::Ground,
                                    coordinate_systems::Field,
                                >>("ground_truth/ground_to_field")
                                .cache(1)
                                .build()
                                .await
                                .unwrap(),
                        ));
                    }
                    caches
                });
                physics.lock().mode = SimulationMode::Running;
                wait(&mut worker, || {
                    estimates.iter().all(|(estimate, _)| {
                        estimate
                            .get_latest()
                            .is_some_and(|estimate| estimate.robot_to_field.is_some())
                    })
                });
                for (index, (estimate, truth)) in estimates.iter().enumerate() {
                    let estimate = estimate.get_latest().unwrap();
                    let actual = estimate
                        .robot_to_field
                        .as_ref()
                        .unwrap()
                        .pose
                        .inner
                        .translation
                        .vector;
                    let truth = truth.get_latest().unwrap();
                    let expected = truth.inner.translation.vector;
                    assert!(
                        (actual.x as f32 - expected.x).hypot(actual.y as f32 - expected.y) < 0.3,
                        "Player {} localization: {actual:?} vs {expected:?}",
                        index + 1
                    );
                }
            }
            eprintln!("Five {profile:?} players ready; running additions: {add_while_running}");
            members[0]
                .io
                .button_event(0, booster::ButtonEventType::PressDown)
                .unwrap();
            members[0]
                .io
                .button_event(0, booster::ButtonEventType::PressUp)
                .unwrap();
            physics.lock().mode = SimulationMode::Running;
            wait(&mut worker, || {
                members[0]
                    .io
                    .primary
                    .get_latest()
                    .is_some_and(|state| *state == types::primary_state::PrimaryState::Damping)
                    && team.now().duration_since(time) >= Duration::from_secs(1)
            });
            assert_eq!(
                *members[0].io.primary.get_latest().unwrap(),
                types::primary_state::PrimaryState::Damping
            );
            for member in &members[1..] {
                assert_eq!(
                    *member.io.primary.get_latest().unwrap(),
                    types::primary_state::PrimaryState::Initial,
                    "player {}: {}",
                    member.number,
                    member.io.status()
                );
            }
        }
        physics.lock().mode = SimulationMode::Paused;
        thread::sleep(Duration::from_millis(50));
        let paused = team.now();
        thread::sleep(Duration::from_millis(150));
        assert_eq!(team.now(), paused);
        for member in &members {
            assert_eq!(member.io.now(), paused);
        }
        drop(worker);
        drop(members);
        drop(team);
        drop(app);
        router.shutdown().unwrap();
        std::net::UdpSocket::bind("0.0.0.0:3838").unwrap();
    }
}

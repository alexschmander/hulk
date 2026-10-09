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
    pub id: RobotId,
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
    referee: Option<crate::autoref::AutoRef>,
    network: tokio::sync::Mutex<Network>,
    members: Mutex<BTreeMap<RobotId, Member>>,
    selected: Mutex<RobotId>,
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
        ensure!(
            configuration.opponent_count <= 5,
            "Choose zero to five opponents"
        );
        let field = configuration
            .field_configuration
            .as_ref()
            .ok_or_else(|| eyre!("Missing field configuration"))?;
        field.validate().map_err(|error| eyre!(error))?;
        let clock = Clock::logical(Clock::wallclock().now());
        let mut referee = if configuration.referee == crate::RefereeMode::Automatic {
            Some(crate::autoref::AutoRef::new(&configuration).await?)
        } else {
            None
        };
        let network = if let Some(referee) = &mut referee {
            let network = Network::automatic(referee.network()).await?;
            referee.connect(network.address()?, clock.clone());
            network
        } else {
            Network::new(3838, 3939).await?
        };
        let team = Self(Arc::new(Inner {
            cancelled,
            _lease: lease,
            runtime,
            clock,
            network: tokio::sync::Mutex::new(network),
            referee,
            members: Mutex::new(BTreeMap::new()),
            selected: Mutex::new(RobotId::FIRST),
            controller: Mutex::new(None),
            configuration: configuration.clone(),
        }));
        for (side, count) in [
            (TeamId::Hulks, configuration.robot_count),
            (TeamId::Opponents, configuration.opponent_count),
        ] {
            for index in 0..count {
                team.add(
                    side,
                    on_field_side(spawn_pose(&field.dimensions, index), side.default_away()),
                )
                .await?;
            }
        }
        for member in team.0.members.lock().unwrap().values_mut() {
            if member.io.away() != member.id.team.default_away() {
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
    pub fn referee(&self) -> Option<&crate::autoref::AutoRef> {
        self.0.referee.as_ref()
    }
    pub fn poll_controller(&self) -> Option<String> {
        if let Some(error) = self.referee().and_then(|r| r.poll()) {
            return Some(error);
        }
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
            .get(&*self.0.selected.lock().unwrap())
            .cloned()
    }
    pub fn select(&self, id: Option<RobotId>) {
        if let Some(controller) = self.0.controller.lock().unwrap().as_ref() {
            controller.select(id);
        }
        if let Some(id) = id.filter(|id| self.0.members.lock().unwrap().contains_key(id)) {
            *self.0.selected.lock().unwrap() = id;
        }
    }
    pub fn away(&self, side: TeamId) -> bool {
        let members = self.members();
        members
            .iter()
            .find(|member| member.id.team == side)
            .map(|member| member.io.away())
            .or_else(|| members.first().map(|member| !member.io.away()))
            .unwrap_or(side.default_away())
    }
    pub fn bind(&self, id: RobotId, entity: Entity) {
        self.0.members.lock().unwrap().get_mut(&id).unwrap().entity = Some(entity);
    }
    pub async fn add(&self, side: TeamId, pose: Transform) -> Result<RobotId> {
        ensure!(
            !self.0.cancelled.load(std::sync::atomic::Ordering::Acquire),
            "Simulator stopped"
        );
        // Serializes allocation and startup, including dynamic additions.
        let mut network = self.0.network.lock().await;
        let number = (1..=5)
            .find(|number| {
                !self.0.members.lock().unwrap().contains_key(&RobotId {
                    team: side,
                    number: *number,
                })
            })
            .ok_or_else(|| eyre!("All five player numbers are in use"))?;
        let id = RobotId { team: side, number };
        let player = [
            hsl_network_messages::PlayerNumber::One,
            hsl_network_messages::PlayerNumber::Two,
            hsl_network_messages::PlayerNumber::Three,
            hsl_network_messages::PlayerNumber::Four,
            hsl_network_messages::PlayerNumber::Five,
        ][usize::from(number - 1)];
        let mut configuration = self.0.configuration.clone();
        configuration.namespace = robot_namespace(id);
        configuration.controller = crate::ControllerSource::External;
        let connection = Arc::new(network.add_robot(side).await?);
        let io = Robotics::new_robot(
            self.0.runtime.clone(),
            configuration,
            RobotSettings {
                player,
                default_away: side.default_away(),
                scope: transport_scope(id),
                clock: self.0.clock.clone(),
                ports: connection.ports,
            },
        )
        .await?;
        crate::simulation::bootstrap(&io, &self.0.cancelled)?;
        let member = Member {
            id,
            io,
            entity: None,
            pose,
            _network: connection,
        };
        if let Some(controller) = self.0.controller.lock().unwrap().as_ref() {
            controller.add(member.clone());
        }
        self.0.members.lock().unwrap().insert(id, member);
        Ok(id)
    }
}
/// A team keeps its wire identity separate from its readable transport namespace.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TeamId {
    #[default]
    Hulks,
    Opponents,
}
impl TeamId {
    pub const ALL: [Self; 2] = [Self::Hulks, Self::Opponents];
    pub fn name(self) -> &'static str {
        match self {
            Self::Hulks => "hulks",
            Self::Opponents => "opponents",
        }
    }
    pub fn wire_number(self) -> u8 {
        match self {
            Self::Hulks => 24,
            Self::Opponents => 5,
        }
    }
    pub fn default_away(self) -> bool {
        self == Self::Opponents
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RobotId {
    pub team: TeamId,
    pub number: u8,
}
impl RobotId {
    pub const FIRST: Self = Self {
        team: TeamId::Hulks,
        number: 1,
    };
}
impl std::fmt::Display for RobotId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.team.name(), self.number)
    }
}
pub fn robot_namespace(id: RobotId) -> String {
    format!("/{id}")
}
pub fn robot_id(namespace: &str) -> Option<RobotId> {
    let (team, number) = namespace.trim_start_matches('/').split_once('/')?;
    let team = TeamId::ALL
        .into_iter()
        .find(|candidate| candidate.name() == team)?;
    let number = number
        .parse::<u8>()
        .ok()
        .filter(|number| (1..=5).contains(number))?;
    Some(RobotId { team, number })
}
pub fn transport_scope(id: RobotId) -> String {
    format!("{}{}", crate::ZENOH_NAMESPACE, robot_namespace(id))
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
pub(crate) fn on_field_side(mut pose: Transform, away: bool) -> Transform {
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
            team.bind(member.id, entity);
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
    fn validate_team_channels(runtime: &tokio::runtime::Runtime, members: &[Member]) {
        use hsl_network_messages::{HulkMessage, PlayerNumber, StateMessage};
        use types::{
            messages::{IncomingMessage, OutgoingMessage},
            time_wrapper::TimeWrapper,
        };
        runtime.block_on(async {
            let mut readers = tokio::task::JoinSet::new();
            let mut publishers = Vec::new();
            for member in members {
                let node = member.io.node();
                let subscriber = node
                    .subscriber::<TimeWrapper<IncomingMessage>>("filtered_message")
                    .build()
                    .await
                    .unwrap();
                let id = member.id;
                readers.spawn(async move {
                    let deadline = tokio::time::Instant::now() + Duration::from_secs(1);
                    let mut received = 0;
                    while let Ok(Ok(message)) =
                        tokio::time::timeout_at(deadline, subscriber.recv()).await
                    {
                        if let IncomingMessage::Hsl(HulkMessage::State(state)) = message.inner
                            && state.head_yaw.abs() == 1.234
                        {
                            assert_ne!(
                                id.number, 1,
                                "production filter must discard its own player number"
                            );
                            assert_eq!(
                                state.head_yaw > 0.0,
                                id.team == TeamId::Hulks,
                                "cross-team message at {id}"
                            );
                            received += 1;
                        }
                    }
                    if id.number != 1 {
                        assert!(received > 0, "{id} missed teammate messages");
                    }
                });
                if id.number == 1 {
                    publishers.push((
                        id.team,
                        node.publisher::<OutgoingMessage>("outputs/message")
                            .build()
                            .await
                            .unwrap(),
                    ));
                }
            }
            for _ in 0..10 {
                for (side, publisher) in &publishers {
                    publisher
                        .publish(&OutgoingMessage::Hsl(HulkMessage::State(StateMessage {
                            player_number: PlayerNumber::One,
                            head_yaw: if *side == TeamId::Hulks {
                                1.234
                            } else {
                                -1.234
                            },
                            ..Default::default()
                        })))
                        .await
                        .unwrap();
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            while let Some(result) = readers.join_next().await {
                result.unwrap();
            }
        });
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
        let selfplay = members
            .iter()
            .any(|member| member.id.team == TeamId::Opponents);
        let mut stages = vec![
            ("ready", None),
            ("penalized", Some(TeamId::Hulks)),
            ("unpenalized", None),
        ];
        if selfplay {
            stages.extend([
                ("opponent_penalized", Some(TeamId::Opponents)),
                ("opponent_unpenalized", None),
            ]);
        }
        for (stage, penalized) in stages {
            wait(worker, || {
                observations
                    .iter()
                    .zip(&members)
                    .all(|((state, _), member)| {
                        state.get_latest().is_some_and(|state| {
                            state.game_state == FilteredGameState::Ready
                                && state.penalties[hsl_network_messages::PlayerNumber::Three]
                                    .is_some()
                                    == (penalized == Some(member.id.team))
                        })
                    })
            });
            if let Some(side) = penalized {
                wait(worker, || {
                    members.iter().all(|member| {
                        member.io.primary.get_latest().is_some_and(|state| {
                            (*state == types::primary_state::PrimaryState::Penalized)
                                == (member.id.team == side && member.id.number == 3)
                        })
                    })
                });
            }
            for (member, (_, raw)) in members.iter().zip(&observations) {
                let state = raw.get_latest().unwrap();
                let state = state.as_ref().as_ref().unwrap();
                assert_eq!(
                    state.global_field_side == types::field_dimensions::GlobalFieldSide::Away,
                    member.id.team == TeamId::Opponents
                );
                assert_eq!(
                    state.kicking_team,
                    Some(if member.id.team == TeamId::Hulks {
                        hsl_network_messages::Team::Hulks
                    } else {
                        hsl_network_messages::Team::Opponent
                    })
                );
            }
            std::fs::write(exchange.join(stage), "observed by all robots").unwrap();
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
        wait(worker, || {
            TeamId::ALL
                .into_iter()
                .filter(|side| *side == TeamId::Hulks || selfplay)
                .all(|side| {
                    members
                        .iter()
                        .filter(|member| member.id.team == side)
                        .any(|member| {
                            matches!(
                                member.io.active_motion(),
                                types::motion_command::MotionCommand::Walk { .. }
                                    | types::motion_command::MotionCommand::WalkWithVelocity { .. }
                            )
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
            referee: crate::RefereeMode::External,
            competition: crate::Competition::default(),
            field_configuration: Some(
                crate::FieldConfiguration::load(&parameter_root, "incheon_small").unwrap(),
            ),
            parameter_root,
            model_directory: root.join("../../etc/neural_networks"),
            router: Some(endpoint),
            namespace: robot_namespace(RobotId::FIRST),
            location: Some("incheon_small".into()),
            opponent_count: std::env::var("SIMULATOR_TEST_OPPONENTS")
                .ok()
                .map(|n| n.parse().unwrap())
                .unwrap_or(0),
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
        if std::env::var_os("HSL_EXCHANGE").is_some() {
            let member = &team.members()[0];
            let parameters = member.io.parameters.snapshot().typed().ball.clone();
            let radius = f64::from(member.io.field_dimensions.ball_radius);
            app.world_mut().spawn((
                MjcfObject::from_factory(
                    move || crate::scene::ball::ball_spec(radius, &parameters),
                    "ball",
                )
                .with_free_joint("ball_free_joint")
                .grounded(),
                Transform::from_xyz(0.0, 0.0, 0.0),
                crate::scene::ball::Ball,
            ));
        }
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
        let selfplay = team.configuration().opponent_count > 0;
        for side in TeamId::ALL {
            if side == TeamId::Opponents && !selfplay {
                continue;
            }
            let count = team
                .members()
                .iter()
                .filter(|member| member.id.team == side)
                .count() as u8;
            for number in count + 1..=5 {
                let occupied = {
                    let world = physics.lock();
                    world
                        .robots
                        .iter()
                        .filter_map(|id| world.object_pose(*id))
                        .collect::<Vec<_>>()
                };
                let pose = vacant_spawn(
                    &team.members()[0].io.field_dimensions,
                    &occupied,
                    team.away(side),
                )
                .unwrap();
                assert_eq!(
                    runtime.block_on(team.add(side, pose)).unwrap().number,
                    number
                );
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
            assert!(
                runtime
                    .block_on(team.add(side, Transform::default()))
                    .is_err()
            );
        }
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
        let expected: Vec<_> = (0..if selfplay { 2 } else { 1 })
            .flat_map(|_| (1..=5).map(|number| number.to_string()))
            .collect();
        assert_eq!(ids, expected);
        for member in &members {
            assert_eq!(robot_id(&robot_namespace(member.id)), Some(member.id));
            assert_eq!(member.pose.translation.x > 0.0, member.io.away());
        }
        validate_team_channels(&runtime, &members);
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
                    member.id,
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
            eprintln!(
                "{} {profile:?} players ready; running additions: {add_while_running}",
                members.len()
            );
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
                    member.id,
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

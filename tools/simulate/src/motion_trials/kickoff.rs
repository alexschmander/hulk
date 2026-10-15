//! Script the referee inputs; leave approach, aim, strength and recovery to behavior.
use super::*;
use hsl_network_messages::{
    GameControllerStateMessage, GamePhase, GameState, Half, Player, PlayerNumber, Team, TeamColor,
    TeamState,
};
use ros_z::time::Time;
use types::{
    filtered_game_controller_state::FilteredGameControllerState, messages::IncomingMessage,
    time_wrapper::TimeWrapper,
};

pub(super) struct Trial {
    publisher: ros_z::pubsub::Publisher<TimeWrapper<IncomingMessage>>,
    game: ros_z::cache::Cache<FilteredGameControllerState>,
    blackboard: ros_z::cache::Cache<behavior_node::node::Blackboard>,
    next_publish: Time,
    last_contact: bool,
    last_speed: f64,
    contacts: Vec<serde_json::Value>,
    stages: Vec<serde_json::Value>,
    samples: Vec<serde_json::Value>,
    next_sample: f64,
    last_stage: &'static str,
    soft_contact: Option<(f64, [f64; 2])>,
    strong_contact: Option<f64>,
    soft_direction: Option<f64>,
    soft_distance: Option<f64>,
    strong_origin: Option<[f64; 2]>,
    strong_aim: f64,
    final_position: [f64; 2],
    shot_peak: f64,
}

impl Trial {
    pub(super) fn new(runtime: &tokio::runtime::Runtime, io: &Robotics) -> Self {
        let (publisher, game, blackboard) = runtime.block_on(async {
            (
                io.node()
                    .publisher("filtered_message")
                    .build()
                    .await
                    .unwrap(),
                io.node()
                    .subscriber("filtered_game_controller_state")
                    .cache(1)
                    .build()
                    .await
                    .unwrap(),
                io.node()
                    .subscriber("behavior/blackboard")
                    .cache(1)
                    .build()
                    .await
                    .unwrap(),
            )
        });
        Self {
            publisher,
            game,
            blackboard,
            next_publish: Time::zero(),
            last_contact: false,
            last_speed: 0.0,
            contacts: Vec::new(),
            stages: Vec::new(),
            samples: Vec::new(),
            next_sample: 0.0,
            last_stage: "Idle",
            soft_contact: None,
            strong_contact: None,
            soft_direction: None,
            soft_distance: None,
            strong_origin: None,
            strong_aim: 0.0,
            final_position: [0.0; 2],
            shot_peak: 0.0,
        }
    }

    fn publish(&mut self, runtime: &tokio::runtime::Runtime, io: &Robotics, state: GameState) {
        let team = |number| TeamState {
            team_number: number,
            field_player_color: TeamColor::Blue,
            goal_keeper_color: TeamColor::Blue,
            goal_keeper_player_number: Some(PlayerNumber::One),
            score: 0,
            penalty_shoot_index: 0,
            penalty_shoots: Vec::new(),
            remaining_amount_of_messages: 1000,
            players: (0..5)
                .map(|_| Player {
                    penalty: None,
                    caution: 0,
                })
                .collect(),
        };
        let message = GameControllerStateMessage {
            competition_type: serde_json::from_value(serde_json::json!("Small")).unwrap(),
            stopped: false,
            game_phase: GamePhase::Normal,
            game_state: state,
            sub_state: None,
            half: Half::First,
            remaining_time_in_half: Duration::from_secs(600),
            secondary_time: Duration::ZERO,
            hulks_team: team(24),
            opponent_team: team(25),
            kicking_team: Some(Team::Hulks),
            hulks_team_is_home_after_coin_toss: true,
        };
        runtime
            .block_on(self.publisher.publish(&TimeWrapper {
                time: io.now(),
                inner: IncomingMessage::GameController("127.0.0.1:3838".parse().unwrap(), message),
            }))
            .unwrap();
        self.next_publish = io.now() + Duration::from_millis(100);
    }

    pub(super) fn start(&mut self, runtime: &tokio::runtime::Runtime, io: &Robotics) {
        for state in [GameState::Ready, GameState::Set] {
            let start = io.now();
            let deadline = Instant::now();
            while io.now().duration_since(start) < Duration::from_secs(1) {
                assert!(
                    deadline.elapsed() < Duration::from_secs(5),
                    "kickoff clock stopped: {}",
                    io.status()
                );
                self.publish(runtime, io, state);
                thread::sleep(Duration::from_millis(20));
            }
            assert_eq!(
                self.game.get_latest().unwrap().game_state,
                match state {
                    GameState::Ready => types::filtered_game_state::FilteredGameState::Ready,
                    _ => types::filtered_game_state::FilteredGameState::Set,
                }
            );
        }
        io.whistle();
        let deadline = Instant::now();
        while *io.primary.get_latest().unwrap() != types::primary_state::PrimaryState::Playing {
            assert!(
                deadline.elapsed() < Duration::from_secs(5),
                "whistle did not start Playing"
            );
            self.publish(runtime, io, GameState::Set);
            thread::sleep(Duration::from_millis(20));
        }
    }

    pub(super) fn observe(
        &mut self,
        runtime: &tokio::runtime::Runtime,
        io: &Robotics,
        elapsed: f64,
        position: [f64; 3],
        velocity: [f64; 2],
        contact: bool,
        physics: &SharedPhysics,
        robot: Entity,
    ) -> bool {
        if io.now() >= self.next_publish {
            self.publish(runtime, io, GameState::Playing);
        }
        if let Some(bb) = self.blackboard.get_latest() {
            if elapsed >= self.next_sample {
                self.next_sample = elapsed + 1.0;
                self.samples.push(serde_json::json!({"time":elapsed,"pose":bb.world_state.robot.ground_to_field,"ball":bb.world_state.ball,"motion":bb.last_motion_command}));
            }
            use behavior_node::kickoff::KickoffState;
            let stage = match bb.kickoff {
                KickoffState::Idle => "Idle",
                KickoffState::Armed => "Armed",
                KickoffState::Touch { .. } => "Touch",
                KickoffState::FollowUp { .. } => "FollowUp",
                KickoffState::Settle { .. } => "Settle",
                KickoffState::Shot { .. } => "Shot",
                KickoffState::Done => "Done",
            };
            if stage != self.last_stage {
                self.stages
                    .push(serde_json::json!({"time":elapsed,"stage":stage,"ball":position}));
                self.last_stage = stage;
                if stage == "Shot"
                    && let Some((_, origin)) = self.soft_contact
                {
                    self.soft_distance =
                        Some((position[0] - origin[0]).hypot(position[1] - origin[1]));
                }
            }
        }
        let speed = velocity[0].hypot(velocity[1]);
        // A foot contact can last a single 2 ms physics step between observer polls.
        let impact = speed - self.last_speed > 0.15;
        if (contact && !self.last_contact) || impact {
            if let MotionCommand::Kick {
                soft,
                target_speed,
                kick_direction,
                ..
            } = io.active_motion()
            {
                let world = physics.lock();
                let binding = crate::robot_io::RobotBinding::new(
                    world.data(),
                    &format!("object_{}_", robot.to_bits()),
                )
                .unwrap();
                let direction = kick_direction.as_unit_vector();
                let aim = binding.ground_to_world(world.data()).rotation
                    * nalgebra::vector![direction.x(), direction.y(), 0.0];
                let expected_aim = if soft {
                    45.0
                } else {
                    (-position[1])
                        .atan2(f64::from(io.field_dimensions.length) / 2.0 - position[0])
                        .to_degrees()
                };
                self.contacts.push(serde_json::json!({"time":elapsed,"soft":soft,"target_speed":target_speed,"aim_degrees":expected_aim,"command_aim_degrees":aim.y.atan2(aim.x).to_degrees(),"ball":position}));
                if soft && self.last_stage == "Touch" && self.soft_contact.is_none() {
                    self.soft_contact = Some((elapsed, [position[0], position[1]]));
                } else if !soft && self.soft_contact.is_some() && self.strong_contact.is_none() {
                    self.strong_contact = Some(elapsed);
                    self.strong_origin = Some([position[0], position[1]]);
                    self.strong_aim = expected_aim;
                }
            }
        }
        self.last_contact = contact;
        self.last_speed = speed;
        self.final_position = [position[0], position[1]];
        if let Some((_, origin)) = self.soft_contact
            && self.soft_direction.is_none()
            && (position[0] - origin[0]).hypot(position[1] - origin[1]) >= 0.25
        {
            self.soft_direction = Some(
                (position[1] - origin[1])
                    .atan2(position[0] - origin[0])
                    .to_degrees(),
            );
        }
        if self.strong_contact.is_some() {
            self.shot_peak = self.shot_peak.max(velocity[0].hypot(velocity[1]));
        }
        self.strong_contact.is_some_and(|time| elapsed - time > 1.0)
    }

    pub(super) fn finish(
        self,
        commands: &ros_z::cache::Cache<MotionCommand>,
        start: Time,
        end: Time,
    ) {
        let shot_displacement = self.strong_origin.map(|origin| {
            [
                self.final_position[0] - origin[0],
                self.final_position[1] - origin[1],
            ]
        });
        let shot_distance = shot_displacement.map(|d| d[0].hypot(d[1]));
        let shot_direction_error = shot_displacement.map(|d| {
            (d[1].atan2(d[0]).to_degrees() - self.strong_aim + 180.0).rem_euclid(360.0) - 180.0
        });
        eprintln!(
            "KICKOFF_TRIAL {}",
            serde_json::json!({"stages":self.stages,"samples":self.samples,"contacts":self.contacts,"soft_direction_degrees":self.soft_direction,"soft_distance_before_shot":self.soft_distance,"strong_peak_speed":self.shot_peak,"strong_distance":shot_distance,"strong_direction_error_degrees":shot_direction_error})
        );
        let (soft_time, _) = self.soft_contact.expect("kickoff missed its soft touch");
        let strong_time = self
            .strong_contact
            .expect("kickoff missed its strong follow-up");
        assert!(strong_time > soft_time);
        let between = commands.get_interval(
            start + Duration::from_secs_f64(soft_time),
            start + Duration::from_secs_f64(strong_time),
        );
        assert!(
            between.iter().any(|command| matches!(
                **command,
                MotionCommand::Walk { .. } | MotionCommand::WalkWithVelocity { .. }
            )),
            "kick policy chased the ball between kicks"
        );
        assert!(
            (self.soft_direction.unwrap() - 45.0).abs() < 15.0,
            "soft touch missed its diagonal target"
        );
        assert!(self.shot_peak > 2.5, "strong follow-up too weak");
        assert!(
            shot_direction_error.unwrap().abs() < 15.0,
            "strong follow-up missed its target direction"
        );
        assert!(end > start);
    }
}

//! Adapt simulator observations and physical handling to the upstream referee.
use super::{
    Competition,
    calls::{self, Call},
    side, team,
};
use crate::{FieldConfiguration, RobotId, TeamId};
use enum_map::enum_map;
use game_controller_autoreferee::{
    Autoreferee, Command, Config, Features, FieldDimensions, PlayerPose, World,
};
use game_controller_core::{
    GameController,
    action::VAction,
    actions::*,
    log::{LogEntry, Logger, TimestampedLogEntry},
    types::*,
};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::{Arc, Mutex},
    time::Duration,
};

#[derive(Clone, Debug)]
pub(crate) struct Ball {
    pub entity: u64,
    pub position: [f64; 3],
    pub epoch: u64,
}
#[derive(Clone, Debug)]
pub(crate) struct Robot {
    pub id: RobotId,
    pub position: [f64; 3],
    pub yaw: f64,
    pub speed: f64,
    pub fallen: bool,
    pub penalized: bool,
}
#[derive(Clone, Debug, Default)]
pub(crate) struct Snapshot {
    pub ball: Option<Ball>,
    pub robots: Vec<Robot>,
    pub contacts: BTreeSet<RobotId>,
}
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Effect {
    Ball([f64; 2]),
    Robot(RobotId, [f64; 2], f64),
    Whistle,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Origin {
    Automatic,
    Operator,
}
#[derive(Clone, Debug)]
pub(crate) struct Event {
    pub time: Duration,
    pub origin: Origin,
    pub accepted: bool,
    pub text: String,
    pub logged: std::time::Instant,
}
// Observe actions through the core's logger, including actions applied inside Autoreferee.
#[derive(Clone, Default)]
struct Journal(Arc<Mutex<Vec<(Duration, VAction)>>>);
impl Logger for Journal {
    fn append(&mut self, entry: TimestampedLogEntry) {
        if let LogEntry::Action(action) = entry.entry
            && !matches!(action.action, VAction::TeamMessage(_))
        {
            self.0
                .lock()
                .unwrap()
                .push((entry.timestamp, action.action));
        }
    }
}
#[derive(Clone, Copy)]
enum Handling {
    Waiting(f64),
    Placed(f64),
    Serving,
    Gone,
}
struct Handler {
    target: ([f64; 2], f64),
    phase: Handling,
}

pub(crate) struct Engine {
    pub core: GameController,
    pub enabled: bool,
    pub events: VecDeque<Event>,
    pub returns: BTreeMap<(u8, u8), Duration>,
    field: FieldConfiguration,
    referee: Autoreferee,
    stuck: super::stuck::Stuck,
    journal: Journal,
    snapshot: Snapshot,
    pending: Vec<Effect>,
    handlers: BTreeMap<RobotId, Handler>,
    placement: Option<([f64; 2], f64)>,
    resume_at: Option<f64>,
    // Referee placements must not be mistaken for manual edits of the trajectory.
    commanded_ball: Option<[f64; 2]>,
    manual_ball_outside: bool,
    next_half_at: f64,
    mapping: SideMapping,
    whistles: VecDeque<f64>,
}
impl Engine {
    pub fn new(competition: Competition, field: FieldConfiguration) -> color_eyre::Result<Self> {
        let params = Params {
            competition: serde_yaml::from_str(competition.yaml())?,
            game: GameParams {
                teams: enum_map! {
                    Side::Home => TeamParams { number: 24, field_player_color: Color::Blue, goalkeeper_color: Color::Blue },
                    Side::Away => TeamParams { number: 5, field_player_color: Color::Red, goalkeeper_color: Color::Red },
                },
                kick_off_side: Side::Home,
                side_mapping: SideMapping::HomeDefendsLeftGoal,
                test: TestParams::default(),
            },
        };
        let journal = Journal::default();
        Ok(Self {
            core: GameController::new(params, Box::new(journal.clone())),
            referee: Autoreferee::new(Self::config(&field)),
            stuck: super::stuck::Stuck::default(),
            field,
            journal,
            enabled: true,
            events: VecDeque::new(),
            returns: BTreeMap::new(),
            snapshot: Snapshot::default(),
            pending: Vec::new(),
            handlers: BTreeMap::new(),
            placement: None,
            resume_at: None,
            commanded_ball: None,
            manual_ball_outside: false,
            next_half_at: 0.0,
            mapping: SideMapping::HomeDefendsLeftGoal,
            whistles: VecDeque::new(),
        })
    }
    fn config(field: &FieldConfiguration) -> Config {
        let f = field.dimensions;
        // Hulk measures between line centers, upstream measures outer line edges.
        Config {
            features: Features::empty(),
            field: FieldDimensions {
                length: f.length + f.line_width,
                width: f.width + f.line_width,
                goal_width: f.goal_inner_width,
                goal_height: field.goal_height,
                goal_area_length: f.goal_box_area_length + f.line_width,
                goal_area_width: f.goal_box_area_width + f.line_width,
                penalty_area_length: f.penalty_area_length + f.line_width,
                penalty_area_width: f.penalty_area_width + f.line_width,
                penalty_mark_distance: f.penalty_marker_distance + f.line_width / 2.0,
                center_circle_radius: (f.center_circle_diameter + f.line_width) / 2.0,
                border_strip_width: f.border_strip_width,
                mark_radius: f.penalty_marker_size / 2.0,
                line_width: f.line_width,
                ball_radius: f.ball_radius,
            },
            time_until_switch_to_set: Duration::from_secs(5),
            time_until_switch_to_playing: Duration::from_secs(2),
            penalty_placement_distance: 0.75,
        }
    }
    fn world(&self) -> World {
        let mut world = World {
            ball: self.snapshot.ball.as_ref().map_or([0.0; 3], |b| {
                [
                    -b.position[0] as f32,
                    -b.position[1] as f32,
                    b.position[2] as f32,
                ]
            }),
            players: Default::default(),
        };
        for robot in &self.snapshot.robots {
            // Upstream's left goal is +x; Hulk's is -x. Rotate the entire plane by pi.
            world.players[side(robot.id.team)][usize::from(robot.id.number - 1)] =
                Some(PlayerPose {
                    x: -robot.position[0] as f32,
                    y: -robot.position[1] as f32,
                    theta: (robot.yaw + std::f64::consts::PI) as f32,
                    // Standing K1 footprint. Upstream judges positions and motion from this pose.
                    radius: 0.20,
                });
        }
        // Reserve destinations while waiting for robots to acknowledge their penalties.
        for (id, handler) in &self.handlers {
            if matches!(handler.phase, Handling::Waiting(_))
                && self.penalty(*id) != Penalty::NoPenalty
                && let Some(pose) = &mut world.players[side(id.team)][usize::from(id.number - 1)]
            {
                pose.x = -handler.target.0[0] as f32;
                pose.y = -handler.target.0[1] as f32;
            }
        }
        world
    }
    fn now(&self) -> f64 {
        self.core.get_time().as_secs_f64()
    }
    fn log(&mut self, time: Duration, origin: Origin, accepted: bool, text: String) {
        self.events.push_back(Event {
            time,
            origin,
            accepted,
            text,
            logged: std::time::Instant::now(),
        });
        while self.events.len() > 128 {
            self.events.pop_front();
        }
    }
    fn collect_events(&mut self, origin: Origin) {
        let actions = std::mem::take(&mut *self.journal.0.lock().unwrap());
        for (time, action) in actions {
            if matches!(action, VAction::FinishHalf(_)) {
                self.next_half_at = time.as_secs_f64()
                    + self
                        .core
                        .params
                        .competition
                        .half_time_break_duration
                        .as_secs_f64();
                for i in 0..if self.core.get_game(false).phase == Phase::FirstHalf {
                    2
                } else {
                    3
                } {
                    self.whistles.push_back(time.as_secs_f64() + f64::from(i));
                }
            }
            self.log(
                time,
                origin,
                true,
                calls::describe(&action, self.core.get_game(false)),
            );
        }
    }
    pub fn allowed(&mut self, call: Call) -> bool {
        if matches!(call, Call::Restart(..)) && self.core.get_game(false).state == State::Playing {
            let mut game = self.core.get_game(false).clone();
            game.set_play = SetPlay::NoSetPlay;
            game.kicking_side = None;
            return call
                .action(&game)
                .is_legal(&game_controller_core::action::ActionContext::new(
                    &mut game,
                    &self.core.params,
                    None,
                    None,
                ));
        }
        call.action(self.core.get_game(false))
            .is_legal(&self.core.get_context(false))
    }
    pub fn call(&mut self, call: Call) -> Result<(), String> {
        self.collect_events(Origin::Automatic);
        if !self.allowed(call) {
            let reason = call.refusal(self.core.get_game(false));
            self.log(
                self.core.get_time(),
                Origin::Operator,
                false,
                format!("{}: {reason}", call.label()),
            );
            return Err(reason);
        }
        // Observe each intervention immediately, including several calls while paused.
        self.observe(false);
        if matches!(call, Call::Restart(..)) && self.core.get_game(false).state == State::Playing {
            self.core
                .apply(VAction::FinishSetPlay(FinishSetPlay), ActionSource::Network);
            self.observe(false);
        }
        self.core.apply(
            call.action(self.core.get_game(false)),
            ActionSource::Network,
        );
        let mut world = self.world();
        if let Call::Restart(team, kind @ (SetPlay::GoalKick | SetPlay::CornerKick)) = call {
            // An operator can award either team a restart with the ball anywhere.
            // Supply its half to upstream's placement policy instead of the old ball's half.
            let own = if self.away(team) { -1.0 } else { 1.0 };
            world.ball[0] =
                world.ball[0]
                    .abs()
                    .copysign(if kind == SetPlay::GoalKick { own } else { -own });
        }
        self.observe_world(false, &world);
        self.collect_events(Origin::Operator);
        if call == Call::SecondHalf {
            self.next_half_at = self.now();
        }
        Ok(())
    }
    fn observe(&mut self, automatic: bool) {
        self.observe_world(automatic, &self.world());
    }
    fn features(&self, automatic: bool) -> Features {
        let mut features = Features::PLACE_BALL | Features::PLACE_FOR_PENALTY;
        if automatic {
            features |=
                Features::all() - Features::UNPENALIZE - Features::PLACE_FOR_PENALTY_SHOOT_OUT;
        }
        if self.snapshot.ball.is_none() || self.manual_ball_outside || self.placement.is_some() {
            features.remove(
                Features::BALL_OUT
                    | Features::BALL_FREE
                    | Features::DOUBLE_TOUCH
                    | Features::GAME_STUCK
                    | Features::CLEAR_BALL,
            );
        }
        features
    }
    fn observe_world(&mut self, automatic: bool, world: &World) {
        self.observe_features(self.features(automatic), world);
    }
    fn observe_features(&mut self, features: Features, world: &World) {
        // The upstream stuck timer counts ball age even without a nearby player.
        // Keep player-specific timing in the simulator while retaining its other judgments.
        self.referee.set_features(features - Features::GAME_STUCK);
        let commands = self.referee.update(&mut self.core, world);
        for command in commands {
            match command {
                Command::PlaceBall {
                    position: [x, y, _],
                } => {
                    let p = [-f64::from(x), -f64::from(y)];
                    if self.core.get_game(false).state == State::Playing {
                        // Upstream resumes immediately. Wait until the physical handling completes.
                        self.core.apply(
                            VAction::StopPlay(StopPlay { resume: false }),
                            ActionSource::Network,
                        );
                        self.placement = Some((p, self.now() + 0.5));
                    } else {
                        self.commanded_ball = Some(p);
                        self.pending.push(Effect::Ball(p));
                    }
                }
                Command::PlacePlayer {
                    side: s,
                    player,
                    x,
                    y,
                    theta,
                } => {
                    let id = RobotId {
                        team: team(s),
                        number: u8::from(player),
                    };
                    let target = (
                        [-f64::from(x), -f64::from(y)],
                        f64::from(theta) - std::f64::consts::PI,
                    );
                    if self.penalty(id) == Penalty::NoPenalty {
                        self.pending.push(Effect::Robot(id, target.0, target.1));
                    } else {
                        self.handlers.insert(
                            id,
                            Handler {
                                target,
                                phase: Handling::Waiting(self.now()),
                            },
                        );
                    }
                }
                Command::Whistle => self.pending.push(Effect::Whistle),
            }
        }
    }
    fn penalty(&self, id: RobotId) -> Penalty {
        self.core.get_game(false).teams[side(id.team)][PlayerNumber::new(id.number)].penalty
    }
    fn away(&self, team: TeamId) -> bool {
        (team == TeamId::Opponents)
            == (self.core.get_game(false).sides == SideMapping::HomeDefendsLeftGoal)
    }
    pub(super) fn update(&mut self, snapshot: &Snapshot, dt: Duration) -> Vec<Effect> {
        self.core.seek(dt);
        self.collect_events(Origin::Automatic);
        let edited = match (&self.snapshot.ball, &snapshot.ball) {
            (Some(old), Some(new)) => old.entity != new.entity || old.epoch != new.epoch,
            (None, None) => false,
            _ => true,
        };
        if edited {
            let commanded = self.commanded_ball.take().is_some_and(|p| {
                snapshot
                    .ball
                    .as_ref()
                    .is_some_and(|b| (b.position[0] - p[0]).hypot(b.position[1] - p[1]) < 0.05)
            });
            if !commanded {
                // A manual edit defines a new scenario, clearing previous touches/restart history.
                if self.core.get_game(false).state == State::Playing {
                    self.core
                        .apply(VAction::FinishSetPlay(FinishSetPlay), ActionSource::Network);
                }
                self.referee = Autoreferee::new(Self::config(&self.field));
                self.stuck = super::stuck::Stuck::default();
                self.manual_ball_outside = true;
            }
        }
        self.snapshot = snapshot.clone();
        if let Some(ball) = &snapshot.ball {
            let f = self.field.dimensions;
            if ball.position[0].abs() < f.length as f64 / 2.0
                && ball.position[1].abs() < f.width as f64 / 2.0
            {
                self.manual_ball_outside = false;
            }
        }
        let now = self.now();
        if self.core.get_game(false).sides != self.mapping {
            self.mapping = self.core.get_game(false).sides;
            self.handlers.clear();
            for robot in &snapshot.robots {
                let pose = crate::team::on_field_side(
                    crate::team::spawn_pose(&self.field.dimensions, robot.id.number - 1),
                    self.away(robot.id.team),
                );
                self.pending.push(Effect::Robot(
                    robot.id,
                    [pose.translation.x as f64, -pose.translation.z as f64],
                    if pose.translation.z > 0.0 {
                        std::f64::consts::FRAC_PI_2
                    } else {
                        -std::f64::consts::FRAC_PI_2
                    },
                ));
            }
        }
        // Establish the initial observation before applying the first kickoff.
        if self.core.get_game(false).state == State::Initial {
            self.observe(false);
        }
        let game = self.core.get_game(false);
        if self.enabled
            && snapshot.ball.is_some()
            && ((game.state == State::Initial && now >= self.next_half_at)
                || (game.state == State::Timeout
                    && game.secondary_timer.get_remaining()
                        <= game_controller_core::timer::SignedDuration::ZERO))
        {
            self.core.apply(
                VAction::StartSetPlay(StartSetPlay {
                    side: game.kicking_side,
                    set_play: SetPlay::KickOff,
                }),
                ActionSource::Network,
            );
        }
        let world = self.world();
        let features = self.features(self.enabled);
        // Feed the current ball pose before contacts, then judge exits with this frame's last touch.
        self.observe_features(
            if snapshot.contacts.is_empty() {
                features
            } else {
                features - Features::BALL_OUT
            },
            &world,
        );
        if self.enabled
            && !self.manual_ball_outside
            && self.placement.is_none()
            && snapshot.ball.is_some()
            && self.core.get_game(false).state == State::Playing
            && !self.core.get_game(false).stopped
        {
            for id in &snapshot.contacts {
                self.referee.ball_contact(
                    &mut self.core,
                    side(id.team),
                    PlayerNumber::new(id.number),
                );
            }
            if !snapshot.contacts.is_empty() {
                self.observe_features(
                    Features::PLACE_BALL
                        | Features::PLACE_FOR_PENALTY
                        | (features & Features::BALL_OUT),
                    &world,
                );
            }
        }
        if let Some(action) =
            self.stuck
                .update(&self.core, &world, features.contains(Features::GAME_STUCK))
        {
            self.core.apply(action, ActionSource::Network);
            // Let upstream supply placements for penalties and global-stuck restarts.
            self.observe(false);
        }
        self.collect_events(Origin::Automatic);
        self.handle_penalties();
        if let Some((p, after)) = self.placement
            && now >= after
            && snapshot
                .robots
                .iter()
                .all(|r| r.speed < 0.06 || r.fallen || self.penalty(r.id) != Penalty::NoPenalty)
        {
            self.pending.push(Effect::Ball(p));
            self.commanded_ball = Some(p);
            self.placement = None;
            self.resume_at = Some(now + 0.3);
        }
        if self.resume_at.is_some_and(|time| now >= time) {
            self.core.apply(
                VAction::StopPlay(StopPlay { resume: true }),
                ActionSource::Network,
            );
            self.resume_at = None;
            self.observe(false);
        }
        while self.whistles.front().is_some_and(|time| now >= *time) {
            self.whistles.pop_front();
            self.pending.push(Effect::Whistle);
        }
        self.collect_events(Origin::Automatic);
        std::mem::take(&mut self.pending)
    }
    fn handle_penalties(&mut self) {
        let now = self.now();
        let ids: Vec<_> = self.handlers.keys().copied().collect();
        for id in ids {
            let penalty = self.penalty(id);
            if penalty == Penalty::NoPenalty {
                self.handlers.remove(&id);
                continue;
            }
            let Some(robot) = self.snapshot.robots.iter().find(|r| r.id == id) else {
                self.handlers.remove(&id);
                continue;
            };
            let handler = self.handlers.get_mut(&id).unwrap();
            let removed = matches!(penalty, Penalty::SentOff | Penalty::Substitute);
            if removed && !matches!(handler.phase, Handling::Waiting(_) | Handling::Gone) {
                handler.phase = Handling::Waiting(now);
            }
            match handler.phase {
                Handling::Waiting(since) if robot.penalized || now - since > 1.0 => {
                    let (mut p, yaw) = handler.target;
                    if removed {
                        p[1] = p[1].signum()
                            * (self.field.dimensions.width as f64 / 2.0
                                + self.field.dimensions.border_strip_width as f64
                                + 0.6);
                    }
                    self.pending.push(Effect::Robot(id, p, yaw));
                    handler.phase = if removed {
                        Handling::Gone
                    } else {
                        Handling::Placed(now + 0.3)
                    };
                }
                Handling::Placed(after) if now >= after => {
                    self.core.apply(
                        VAction::Unpenalize(Unpenalize {
                            side: side(id.team),
                            player: PlayerNumber::new(id.number),
                            force: false,
                        }),
                        ActionSource::Network,
                    );
                    handler.phase = Handling::Serving;
                }
                Handling::Serving
                    if self.core.get_game(false).teams[side(id.team)]
                        [PlayerNumber::new(id.number)]
                    .penalty_timer
                    .get_remaining()
                    .is_zero() =>
                {
                    self.core.apply(
                        VAction::Unpenalize(Unpenalize {
                            side: side(id.team),
                            player: PlayerNumber::new(id.number),
                            force: false,
                        }),
                        ActionSource::Network,
                    );
                    self.handlers.remove(&id);
                }
                _ => {}
            }
        }
    }
}

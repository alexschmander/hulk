use super::{
    Competition,
    calls::{self, Call},
    side,
};
use crate::{FieldConfiguration, RobotId, TeamId};
use enum_map::enum_map;
use game_controller_core::{
    GameController, action::VAction, actions::*, log::NullLogger, types::*,
};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    time::Duration,
};

#[derive(Clone, Debug)]
pub(crate) struct Ball {
    pub entity: u64,
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    pub epoch: u64,
}
#[derive(Clone, Debug)]
pub(crate) struct Robot {
    pub id: RobotId,
    pub position: [f64; 3],
    pub feet: Vec<[f64; 2]>,
    pub speed: f64,
    pub leg_speed: f64,
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
#[derive(Clone)]
struct Restart {
    kind: SetPlay,
    side: Option<Side>,
    position: [f64; 2],
    kicker: Option<RobotId>,
    touched: BTreeSet<RobotId>,
    outside_circle: bool,
    players: usize,
    taken: bool,
    started: f64,
}
/// Who made a referee decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Origin {
    Automatic,
    Operator,
}
#[derive(Clone, Debug)]
pub(crate) struct Event {
    /// Core time, which advances with simulation time.
    pub time: Duration,
    pub origin: Origin,
    /// Whether the core accepted the decision.
    pub accepted: bool,
    pub text: String,
    /// Wall time, so the panel can briefly mark a new decision.
    pub logged: std::time::Instant,
}
#[derive(Clone, Copy)]
enum Handling {
    Waiting(f64),
    Placed(f64),
    Serving,
    Gone,
}
pub(crate) struct Engine {
    pub core: GameController,
    pub enabled: bool,
    pub events: VecDeque<Event>,
    pub returns: BTreeMap<(u8, u8), Duration>,
    field: FieldConfiguration,
    // Operator calls apply immediately, also while paused, against the last physics
    // observation; their physical effects follow on the next simulation step.
    snapshot: Snapshot,
    pending: Vec<Effect>,
    operator: bool,
    whistle: bool,
    restart: Option<Restart>,
    placement: Option<([f64; 2], f64)>,
    resume_at: Option<f64>,
    last_ball: Option<Ball>,
    last_touch: Option<RobotId>,
    contacts: BTreeSet<RobotId>,
    state: State,
    state_since: f64,
    stopped: bool,
    stopped_since: f64,
    handlers: BTreeMap<RobotId, Handling>,
    violations: BTreeMap<RobotId, f64>,
    goal_entries: BTreeMap<RobotId, f64>,
    still_ball: Option<([f64; 2], f64)>,
    no_near_robot: Option<f64>,
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
        Ok(Self {
            core: GameController::new(params, Box::new(NullLogger)),
            enabled: true,
            events: VecDeque::new(),
            returns: BTreeMap::new(),
            field,
            snapshot: Snapshot::default(),
            pending: Vec::new(),
            operator: false,
            whistle: false,
            restart: None,
            placement: None,
            resume_at: None,
            last_ball: None,
            last_touch: None,
            contacts: BTreeSet::new(),
            state: State::Initial,
            state_since: 0.0,
            stopped: false,
            stopped_since: 0.0,
            handlers: BTreeMap::new(),
            violations: BTreeMap::new(),
            goal_entries: BTreeMap::new(),
            still_ball: None,
            no_near_robot: None,
            next_half_at: 0.0,
            mapping: SideMapping::HomeDefendsLeftGoal,
            whistles: VecDeque::new(),
        })
    }
    fn now(&self) -> f64 {
        self.core.get_time().as_secs_f64()
    }
    fn log(&mut self, accepted: bool, text: String) {
        self.events.push_back(Event {
            time: self.core.get_time(),
            origin: if self.operator {
                Origin::Operator
            } else {
                Origin::Automatic
            },
            accepted,
            text,
            logged: std::time::Instant::now(),
        });
        while self.events.len() > 128 {
            self.events.pop_front();
        }
    }
    fn event(&mut self, message: impl Into<String>) {
        self.log(true, message.into());
    }
    fn apply(&mut self, action: VAction) -> bool {
        // Referee actions need no core undo history: undoing a game state alone
        // cannot undo physical robot/ball handling. Keep the bounded event log.
        let accepted = self.core.apply(action.clone(), ActionSource::Network);
        let text = calls::describe(&action, self.core.get_game(false));
        self.log(accepted, text);
        if accepted {
            self.observe();
        }
        accepted
    }
    /// Records state changes as they happen, so several transitions between two physics
    /// steps still each restart their grace periods and place the ball on entering Set.
    fn observe(&mut self) {
        let now = self.now();
        let game = self.core.get_game(false);
        let (stopped, state) = (game.stopped, game.state);
        if stopped != self.stopped {
            self.stopped = stopped;
            self.stopped_since = now;
            self.violations.clear();
        }
        if state != self.state {
            self.state = state;
            self.state_since = now;
            self.violations.clear();
            if state == State::Set {
                let position = self.restart.as_ref().map_or([0.0; 2], |r| r.position);
                self.pending.push(Effect::Ball(position));
                self.last_ball = None;
            }
        }
    }
    /// Whether the core currently accepts an operator call.
    pub fn allowed(&mut self, call: Call) -> bool {
        if matches!(call, Call::Restart(..)) && self.core.get_game(false).state == State::Playing {
            // A new restart first ends the running set play, as in begin_restart.
            let mut game = self.core.get_game(false).clone();
            game.set_play = SetPlay::NoSetPlay;
            game.kicking_side = None;
            let action = call.action(&game);
            return action.is_legal(&game_controller_core::action::ActionContext::new(
                &mut game,
                &self.core.params,
                None,
                None,
            ));
        }
        let action = call.action(self.core.get_game(false));
        action.is_legal(&self.core.get_context(false))
    }
    /// Applies an operator call now, with the same physical handling as automatic decisions.
    pub fn call(&mut self, call: Call) -> Result<(), String> {
        self.operator = true;
        if !self.allowed(call) {
            let reason = call.refusal(self.core.get_game(false));
            self.log(false, format!("{}: {reason}", call.label()));
            self.operator = false;
            return Err(reason);
        }
        let snapshot = std::mem::take(&mut self.snapshot);
        let now = self.now();
        match call {
            Call::Kickoff => {
                let owner = self.core.get_game(false).kicking_side;
                self.begin_restart(SetPlay::KickOff, owner, [0.0; 2], &snapshot);
            }
            Call::Restart(team, kind) => {
                let position = self.restart_position(team, kind, &snapshot);
                self.begin_restart(kind, Some(side(team)), position, &snapshot);
            }
            Call::Whistle => {
                self.free_set_play(&snapshot, &mut Vec::new());
            }
            Call::BallFree => {
                // The set play ended without a kick; its touch restrictions no longer apply.
                if self.apply(call.action(self.core.get_game(false))) {
                    self.restart = None;
                }
            }
            Call::Goal(_) | Call::DroppedBall => {
                if self.apply(call.action(self.core.get_game(false))) {
                    self.kickoff_after_goal(&snapshot);
                }
            }
            Call::FinishHalf => self.finish_half(now),
            Call::SecondHalf => {
                if self.apply(call.action(self.core.get_game(false))) {
                    self.next_half_at = now;
                }
            }
            _ => {
                self.apply(call.action(self.core.get_game(false)));
            }
        }
        self.operator = false;
        self.snapshot = snapshot;
        Ok(())
    }
    fn kickoff_after_goal(&mut self, snapshot: &Snapshot) {
        let game = self.core.get_game(false);
        if game.state == State::Ready {
            self.record_restart(SetPlay::KickOff, game.kicking_side, [0.0; 2], snapshot);
        }
    }
    fn finish_half(&mut self, now: f64) {
        let first = self.core.get_game(false).phase == Phase::FirstHalf;
        self.next_half_at = now
            + self
                .core
                .params
                .competition
                .half_time_break_duration
                .as_secs_f64();
        if self.apply(VAction::FinishHalf(FinishHalf)) {
            for i in 0..if first { 2 } else { 3 } {
                self.whistles.push_back(now + i as f64);
            }
        }
    }
    fn free_set_play(&mut self, snapshot: &Snapshot, effects: &mut Vec<Effect>) {
        if !self.apply(VAction::FreeSetPlay(FreeSetPlay)) {
            return;
        }
        let owner = self.restart.as_ref().and_then(|r| r.side);
        let players = snapshot
            .robots
            .iter()
            .filter(|r| Some(side(r.id.team)) == owner && self.penalty(r.id) == Penalty::NoPenalty)
            .count();
        if let Some(restart) = &mut self.restart {
            restart.players = players;
        }
        if self.operator {
            self.pending.push(Effect::Whistle);
        } else {
            effects.push(Effect::Whistle);
        }
    }
    fn restart_position(&self, team: TeamId, kind: SetPlay, snapshot: &Snapshot) -> [f64; 2] {
        let f = self.field.dimensions;
        let ball = snapshot.ball.as_ref().map_or([0.0; 3], |b| b.position);
        let own = if self.away(team) { 1.0 } else { -1.0 };
        let y = if ball[1] >= 0.0 { 1.0 } else { -1.0 };
        match kind {
            SetPlay::GoalKick => [
                own * (f.length / 2.0 - f.goal_box_area_length) as f64,
                y * f.goal_box_area_width as f64 / 2.0,
            ],
            SetPlay::CornerKick => [-own * f.length as f64 / 2.0, y * f.width as f64 / 2.0],
            SetPlay::ThrowIn => [
                ball[0].clamp(-f.length as f64 / 2.0, f.length as f64 / 2.0),
                y * f.width as f64 / 2.0,
            ],
            SetPlay::PenaltyKick => [
                -own * (f.length / 2.0 - f.penalty_marker_distance) as f64,
                0.0,
            ],
            _ => [ball[0], ball[1]],
        }
    }
    fn away(&self, team: TeamId) -> bool {
        (team == TeamId::Opponents)
            == (self.core.get_game(false).sides == SideMapping::HomeDefendsLeftGoal)
    }
    fn penalty(&self, id: RobotId) -> Penalty {
        self.core.get_game(false).teams[side(id.team)][PlayerNumber::new(id.number)].penalty
    }
    fn penalize(&mut self, id: RobotId, call: PenaltyCall) {
        self.apply(VAction::Penalize(Penalize {
            side: side(id.team),
            player: PlayerNumber::new(id.number),
            call,
        }));
    }
    fn begin_restart(
        &mut self,
        kind: SetPlay,
        owner: Option<Side>,
        position: [f64; 2],
        snapshot: &Snapshot,
    ) {
        if self.core.get_game(false).set_play != SetPlay::NoSetPlay
            && self.core.get_game(false).state == State::Playing
        {
            self.apply(VAction::FinishSetPlay(FinishSetPlay));
        }
        let accepted = if kind == SetPlay::KickOff
            && owner.is_none()
            && self.core.get_game(false).state == State::Playing
        {
            self.apply(VAction::GlobalGameStuck(GlobalGameStuck))
        } else {
            self.apply(VAction::StartSetPlay(StartSetPlay {
                side: owner,
                set_play: kind,
            }))
        };
        if accepted {
            self.record_restart(kind, owner, position, snapshot);
        }
    }
    fn record_restart(
        &mut self,
        kind: SetPlay,
        owner: Option<Side>,
        position: [f64; 2],
        snapshot: &Snapshot,
    ) {
        let players = snapshot
            .robots
            .iter()
            .filter(|r| Some(side(r.id.team)) == owner && self.penalty(r.id) == Penalty::NoPenalty)
            .count();
        self.restart = Some(Restart {
            kind,
            side: owner,
            position,
            kicker: None,
            touched: BTreeSet::new(),
            outside_circle: false,
            players,
            taken: false,
            started: self.now(),
        });
        self.placement = None;
        self.resume_at = None;
        self.last_touch = None;
        self.contacts.clear();
        self.still_ball = None;
        self.no_near_robot = None;
        if self.core.get_game(false).stopped {
            self.placement = Some((position, self.now() + 0.5));
        }
    }
    pub(super) fn update(&mut self, snapshot: &Snapshot, dt: Duration) -> Vec<Effect> {
        self.core.seek(dt);
        self.observe();
        self.snapshot = snapshot.clone();
        let now = self.now();
        let mut effects = std::mem::take(&mut self.pending);
        let game = self.core.get_game(false).clone();
        if game.sides != self.mapping {
            self.mapping = game.sides;
            for robot in &snapshot.robots {
                let pose = crate::team::on_field_side(
                    crate::team::spawn_pose(&self.field.dimensions, robot.id.number - 1),
                    self.away(robot.id.team),
                );
                // Bevy z is negative field y.
                effects.push(Effect::Robot(
                    robot.id,
                    [pose.translation.x as f64, -pose.translation.z as f64],
                    if pose.translation.z > 0.0 {
                        std::f64::consts::FRAC_PI_2
                    } else {
                        -std::f64::consts::FRAC_PI_2
                    },
                ));
            }
            self.handlers.clear();
            self.event("Teams changed ends");
        }
        self.handle_penalties(snapshot, &mut effects);
        if self.enabled {
            let break_over = match game.state {
                State::Initial => now >= self.next_half_at,
                State::Timeout => {
                    game.secondary_timer.get_remaining()
                        <= game_controller_core::timer::SignedDuration::ZERO
                }
                _ => false,
            };
            if break_over && snapshot.ball.is_some() {
                self.begin_restart(SetPlay::KickOff, game.kicking_side, [0.0; 2], snapshot);
            }
            self.check_positions(snapshot);
            if game.state == State::Set && now - self.state_since >= 2.0 {
                self.whistle = true;
            }
        }
        // Physical handling follows accepted calls even with automatic judgments off.
        if let Some((position, after)) = self.placement
            && now >= after
            && snapshot
                .robots
                .iter()
                .all(|r| r.speed < 0.06 || r.fallen || self.penalty(r.id) != Penalty::NoPenalty)
        {
            effects.push(Effect::Ball(position));
            self.placement = None;
            self.resume_at = Some(now + 0.3);
            self.last_ball = None;
            self.event("Ball placed for restart");
        }
        if self.resume_at.is_some_and(|t| now >= t) {
            self.apply(VAction::StopPlay(StopPlay { resume: true }));
            self.resume_at = None;
        }
        if std::mem::take(&mut self.whistle) {
            self.free_set_play(snapshot, &mut effects);
        }
        if self.enabled
            && self.core.get_game(false).state == State::Playing
            && !self.core.get_game(false).stopped
        {
            self.judge_ball(snapshot);
            self.check_stuck(snapshot);
            let game = self.core.get_game(false);
            if game.primary_timer.get_remaining()
                <= game_controller_core::timer::SignedDuration::ZERO
                && game.set_play != SetPlay::PenaltyKick
                && snapshot
                    .ball
                    .as_ref()
                    .is_none_or(|b| length(b.velocity) < 0.02 || self.outside(b.position))
            {
                self.finish_half(now);
            }
        }
        while self.whistles.front().is_some_and(|time| now >= *time) {
            self.whistles.pop_front();
            effects.push(Effect::Whistle);
        }
        if !effects
            .iter()
            .any(|effect| matches!(effect, Effect::Ball(_)))
        {
            self.last_ball = snapshot.ball.clone();
        }
        self.contacts = snapshot.contacts.clone();
        effects
    }
    fn outside(&self, p: [f64; 3]) -> bool {
        let f = &self.field.dimensions;
        let extra = f.ball_radius as f64 + f.line_width as f64 / 2.0;
        p[0].abs() > f.length as f64 / 2.0 + extra || p[1].abs() > f.width as f64 / 2.0 + extra
    }
    fn judge_ball(&mut self, snapshot: &Snapshot) {
        let Some(ball) = &snapshot.ball else {
            return;
        };
        if self
            .last_ball
            .as_ref()
            .is_some_and(|old| old.entity != ball.entity || old.epoch != ball.epoch)
        {
            self.last_touch = None;
            self.contacts.clear();
            self.still_ball = None;
            self.no_near_robot = None;
            // A manual edit is a new trajectory, not a kick or a scored goal.
            self.last_ball = None;
            return;
        }
        if snapshot.contacts.iter().any(|id| id.team == TeamId::Hulks)
            && snapshot
                .contacts
                .iter()
                .any(|id| id.team == TeamId::Opponents)
        {
            self.last_touch = None;
        }
        for &id in &snapshot.contacts {
            if snapshot.contacts.iter().all(|other| other.team == id.team) {
                self.last_touch = Some(id);
            }
            if let Some(restart) = &mut self.restart
                && restart.kicker == Some(id)
                && ball.position[0].hypot(ball.position[1])
                    > self.field.dimensions.center_circle_diameter as f64 / 2.0
            {
                restart.outside_circle = true;
            }
            if self.contacts.contains(&id)
                && self.restart.as_ref().is_none_or(|r| {
                    r.kicker.is_some()
                        || distance(r.position, [ball.position[0], ball.position[1]]) < 0.05
                })
            {
                continue;
            }
            let mut second_touch = false;
            if let Some(restart) = &mut self.restart {
                restart.touched.insert(id);
                if restart.kicker.is_none() && Some(side(id.team)) == restart.side {
                    restart.kicker = Some(id);
                }
                if restart.kicker == Some(id)
                    && ball.position[0].hypot(ball.position[1])
                        > self.field.dimensions.center_circle_diameter as f64 / 2.0
                {
                    restart.outside_circle = true;
                }
                second_touch = restart.taken
                    && restart.kicker == Some(id)
                    && restart.touched.len() == 1
                    && restart.players >= 3
                    && matches!(
                        restart.kind,
                        SetPlay::ThrowIn | SetPlay::GoalKick | SetPlay::CornerKick
                    );
            }
            if second_touch {
                self.event(format!("Second touch by {}", calls::short(id)));
                self.begin_restart(
                    SetPlay::IndirectFreeKick,
                    Some(-side(id.team)),
                    [ball.position[0], ball.position[1]],
                    snapshot,
                );
                return;
            }
        }
        if let Some(restart) = &mut self.restart
            && !restart.taken
            && restart.kicker.is_some()
            && distance(restart.position, [ball.position[0], ball.position[1]]) > 0.05
            && snapshot
                .robots
                .iter()
                .any(|r| Some(r.id) == restart.kicker && !r.fallen)
        {
            restart.taken = true;
            self.apply(VAction::FinishSetPlay(FinishSetPlay));
        }
        let Some(previous) = &self.last_ball else {
            return;
        };
        if self.outside(previous.position) || !self.outside(ball.position) {
            return;
        }
        let f = self.field.dimensions;
        let extra = f.ball_radius as f64 + f.line_width as f64 / 2.0;
        let hx = f.length as f64 / 2.0 + extra;
        let hy = f.width as f64 / 2.0 + extra;
        let crossing = |axis: usize, edge: f64| {
            let delta = ball.position[axis] - previous.position[axis];
            if ball.position[axis].abs() <= edge || delta.abs() < 1e-9 {
                f64::INFINITY
            } else {
                ((ball.position[axis].signum() * edge - previous.position[axis]) / delta)
                    .clamp(0.0, 1.0)
            }
        };
        let tx = crossing(0, hx);
        let ty = crossing(1, hy);
        let t = tx.min(ty);
        let p: [f64; 3] = std::array::from_fn(|i| {
            previous.position[i] + t * (ball.position[i] - previous.position[i])
        });
        if ty < tx {
            if let Some(last) = self.last_touch {
                self.begin_restart(
                    SetPlay::ThrowIn,
                    Some(-side(last.team)),
                    [
                        p[0].clamp(-f.length as f64 / 2.0, f.length as f64 / 2.0),
                        p[1].signum() * f.width as f64 / 2.0,
                    ],
                    snapshot,
                );
            } else {
                self.begin_restart(SetPlay::KickOff, None, [0.0; 2], snapshot);
            }
            return;
        }
        let defender = if p[0] > 0.0 {
            if self.away(TeamId::Hulks) {
                Side::Home
            } else {
                Side::Away
            }
        } else if self.away(TeamId::Hulks) {
            Side::Away
        } else {
            Side::Home
        };
        let in_goal = p[1].abs() + (f.ball_radius as f64) < f.goal_inner_width as f64 / 2.0
            && p[2] + (f.ball_radius as f64) < self.field.goal_height as f64;
        if in_goal {
            let scorer = -defender;
            let valid = self.goal_allowed(scorer);
            if valid && self.apply(VAction::Goal(Goal { side: scorer })) {
                self.kickoff_after_goal(snapshot);
                return;
            }
            self.event("Goal disallowed by restart touch restrictions");
        }
        let Some(last) = self.last_touch else {
            self.begin_restart(SetPlay::KickOff, None, [0.0; 2], snapshot);
            return;
        };
        if side(last.team) == defender {
            self.begin_restart(
                SetPlay::CornerKick,
                Some(-defender),
                [
                    p[0].signum() * f.length as f64 / 2.0,
                    p[1].signum() * f.width as f64 / 2.0,
                ],
                snapshot,
            );
        } else {
            self.begin_restart(
                SetPlay::GoalKick,
                Some(defender),
                [
                    p[0].signum() * (f.length / 2.0 - f.goal_box_area_length) as f64,
                    if p[1] >= 0.0 { 1.0 } else { -1.0 } * f.goal_box_area_width as f64 / 2.0,
                ],
                snapshot,
            );
        }
    }
    fn goal_allowed(&self, scorer: Side) -> bool {
        let Some(r) = &self.restart else {
            return true;
        };
        let Some(owner) = r.side else {
            return true;
        };
        let Some(kicker) = r.kicker else {
            return true;
        };
        if r.touched.len() == 1 && scorer != owner {
            return false;
        } // Direct own goal.
        if scorer != owner {
            return true;
        }
        match r.kind {
            SetPlay::KickOff => {
                if r.players >= 3 {
                    r.touched.iter().filter(|id| id.team == kicker.team).count() >= 2
                } else {
                    r.outside_circle
                }
            }
            SetPlay::ThrowIn | SetPlay::IndirectFreeKick => r.touched.len() >= 2,
            _ => true,
        }
    }
    fn check_stuck(&mut self, snapshot: &Snapshot) {
        if self.core.get_game(false).set_play != SetPlay::NoSetPlay {
            return;
        }
        let Some(ball) = &snapshot.ball else {
            return;
        };
        let p = [ball.position[0], ball.position[1]];
        let nearest = snapshot
            .robots
            .iter()
            .filter(|r| self.penalty(r.id) == Penalty::NoPenalty)
            .map(|r| (r.id, distance([r.position[0], r.position[1]], p)))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        let now = self.now();
        if nearest.is_none_or(|(_, d)| d > 1.0) {
            let since = *self.no_near_robot.get_or_insert(now);
            self.still_ball = None;
            if now - since >= 30.0 {
                self.begin_restart(SetPlay::KickOff, None, [0.0; 2], snapshot);
            }
        } else {
            self.no_near_robot = None;
            if self
                .still_ball
                .is_none_or(|(old, _)| distance(old, p) > 0.05)
            {
                self.still_ball = Some((p, now));
            }
            if self
                .still_ball
                .is_some_and(|(_, since)| now - since >= 10.0)
            {
                self.penalize(nearest.unwrap().0, PenaltyCall::LocalGameStuck);
                self.still_ball = None;
            }
        }
    }
    fn check_positions(&mut self, snapshot: &Snapshot) {
        let game = self.core.get_game(false).clone();
        let f = self.field.dimensions;
        let now = self.now();
        // Keep entry order during play; in Set remove the players nearest the border.
        let mut occupants = BTreeMap::<TeamId, Vec<(RobotId, f64)>>::new();
        let mut present = BTreeSet::new();
        for robot in &snapshot.robots {
            if self.penalty(robot.id) != Penalty::NoPenalty {
                continue;
            }
            let own = if self.away(robot.id.team) { 1.0 } else { -1.0 };
            let depth = robot
                .feet
                .iter()
                .filter_map(|p| {
                    let dx = p[0] * own - (f.length / 2.0 - f.goal_box_area_length) as f64;
                    let dy = f.goal_box_area_width as f64 / 2.0 - p[1].abs();
                    (dx >= -f.line_width as f64 / 2.0 && dy >= -f.line_width as f64 / 2.0)
                        .then_some(dx.min(dy))
                })
                .reduce(f64::max);
            if let Some(depth) = depth {
                present.insert(robot.id);
                let entered = *self.goal_entries.entry(robot.id).or_insert(now);
                occupants.entry(robot.id.team).or_default().push((
                    robot.id,
                    if game.state == State::Set {
                        -depth
                    } else {
                        entered
                    },
                ));
            }
        }
        self.goal_entries.retain(|id, _| present.contains(id));
        let mut excess = BTreeSet::new();
        for robots in occupants.values_mut() {
            robots.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            excess.extend(robots.iter().skip(3).map(|(id, _)| *id));
        }
        let mut forward_kicker = None;
        for robot in &snapshot.robots {
            if self.penalty(robot.id) != Penalty::NoPenalty {
                continue;
            }
            let own = if self.away(robot.id.team) { 1.0 } else { -1.0 };
            let in_own_half = robot.feet.iter().all(|p| p[0] * own >= -0.05);
            let in_field = robot.feet.iter().any(|p| {
                p[0].abs() <= f.length as f64 / 2.0 + 0.025
                    && p[1].abs() <= f.width as f64 / 2.0 + 0.025
            });
            let near_circle = robot
                .feet
                .iter()
                .any(|p| p[0].hypot(p[1]) < f.center_circle_diameter as f64 / 2.0 - 0.05);
            let mut call = None;
            if game.stopped && now - self.stopped_since > 1.0 && robot.leg_speed > 0.4 {
                call = Some(PenaltyCall::MotionInStop);
            } else if (game.state == State::Set && now - self.state_since > 1.0)
                || (game.state == State::Playing && game.set_play == SetPlay::KickOff)
            {
                let kickoff = game.set_play == SetPlay::KickOff;
                if game.state == State::Set && !robot.fallen && robot.leg_speed > 0.4 {
                    call = Some(PenaltyCall::MotionInSet);
                } else if kickoff
                    && (!in_field
                        || (!in_own_half
                            && !(game.kicking_side == Some(side(robot.id.team)) && near_circle))
                        || (game.kicking_side != Some(side(robot.id.team)) && near_circle))
                {
                    call = Some(PenaltyCall::IllegalPosition);
                }
                if kickoff
                    && game.kicking_side == Some(side(robot.id.team))
                    && !in_own_half
                    && near_circle
                    && forward_kicker.replace(robot.id).is_some()
                {
                    call = Some(PenaltyCall::IllegalPosition);
                }
            } else if game.state == State::Playing && !game.stopped {
                let on_carpet = robot.feet.iter().any(|p| {
                    p[0].abs() <= (f.length / 2.0 + f.border_strip_width) as f64
                        && p[1].abs() <= (f.width / 2.0 + f.border_strip_width) as f64
                });
                if !on_carpet {
                    call = Some(PenaltyCall::LeavingTheField);
                }
                if let Some(r) = &self.restart
                    && game.set_play != SetPlay::NoSetPlay
                    && r.side.is_some_and(|s| s != side(robot.id.team))
                    && now - r.started > 2.0
                {
                    let forbidden = if r.kind == SetPlay::GoalKick {
                        let attack = -own;
                        robot.feet.iter().any(|p| {
                            p[0] * attack >= (f.length / 2.0 - f.penalty_area_length) as f64
                                && p[1].abs() <= f.penalty_area_width as f64 / 2.0
                        })
                    } else {
                        robot.feet.iter().any(|p| {
                            distance(*p, r.position) < f.center_circle_diameter as f64 / 2.0 - 0.1
                        })
                    };
                    // Allow robots already encroaching to retreat after a restart.
                    if forbidden && robot.speed < 0.04 {
                        call = Some(PenaltyCall::IllegalPosition);
                    }
                }
            }
            if matches!(game.state, State::Set | State::Playing) && excess.contains(&robot.id) {
                call = Some(PenaltyCall::IllegalPosition);
            }
            if let Some(call) = call {
                let since = *self.violations.entry(robot.id).or_insert(now);
                if now - since >= 0.5 {
                    self.penalize(robot.id, call);
                    self.violations.remove(&robot.id);
                }
            } else {
                self.violations.remove(&robot.id);
            }
        }
    }
    fn handle_penalties(&mut self, snapshot: &Snapshot, effects: &mut Vec<Effect>) {
        let now = self.now();
        let mut occupied: Vec<_> = snapshot
            .robots
            .iter()
            .map(|r| [r.position[0], r.position[1]])
            .collect();
        for robot in &snapshot.robots {
            let penalty = self.penalty(robot.id);
            if penalty == Penalty::NoPenalty {
                self.handlers.remove(&robot.id);
                continue;
            }
            if penalty == Penalty::MotionInSet {
                continue;
            }
            let phase = self
                .handlers
                .entry(robot.id)
                .or_insert(Handling::Waiting(now));
            if matches!(penalty, Penalty::SentOff | Penalty::Substitute)
                && !matches!(phase, Handling::Waiting(_) | Handling::Gone)
            {
                *phase = Handling::Waiting(now);
            }
            match *phase {
                Handling::Waiting(since) if robot.penalized || now - since > 1.0 => {
                    if let Some((p, yaw)) = self.penalty_slot(
                        robot.id,
                        &occupied,
                        penalty == Penalty::SentOff || penalty == Penalty::Substitute,
                    ) {
                        occupied.push(p);
                        effects.push(Effect::Robot(robot.id, p, yaw));
                        self.handlers.insert(
                            robot.id,
                            if matches!(penalty, Penalty::SentOff | Penalty::Substitute) {
                                Handling::Gone
                            } else {
                                Handling::Placed(now + 0.3)
                            },
                        );
                        self.event(format!(
                            "{} placed beside the field",
                            calls::short(robot.id)
                        ));
                    }
                }
                Handling::Placed(after) if now >= after => {
                    self.apply(VAction::Unpenalize(Unpenalize {
                        side: side(robot.id.team),
                        player: PlayerNumber::new(robot.id.number),
                        force: false,
                    }));
                    self.handlers.insert(robot.id, Handling::Serving);
                }
                Handling::Serving => {
                    let timer = &self.core.get_game(false).teams[side(robot.id.team)]
                        [PlayerNumber::new(robot.id.number)]
                    .penalty_timer;
                    if timer.get_remaining().is_zero() {
                        self.apply(VAction::Unpenalize(Unpenalize {
                            side: side(robot.id.team),
                            player: PlayerNumber::new(robot.id.number),
                            force: false,
                        }));
                        self.handlers.remove(&robot.id);
                    }
                }
                _ => {}
            }
        }
    }
    fn penalty_slot(
        &self,
        id: RobotId,
        occupied: &[[f64; 2]],
        sent_off: bool,
    ) -> Option<([f64; 2], f64)> {
        let f = self.field.dimensions;
        let own = if self.away(id.team) { 1.0 } else { -1.0 };
        let x = own * (f.length / 2.0 - f.penalty_marker_distance) as f64;
        let y = f.width as f64 / 2.0 + if sent_off { 1.6 } else { 0.4 };
        for n in 0..20 {
            let dx = if n % 2 == 0 { 1.0 } else { -1.0 } * ((n + 1) / 2) as f64 * 0.75;
            let px = x + dx;
            if px * own < 0.4 || px.abs() > f.length as f64 / 2.0 - 0.3 {
                continue;
            }
            for sign in [1.0, -1.0] {
                let p = [px, sign * y];
                if occupied.iter().all(|q| distance(p, *q) > 0.7) {
                    return Some((p, -sign * std::f64::consts::FRAC_PI_2));
                }
            }
        }
        None
    }
}
fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}
fn length(v: [f64; 3]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

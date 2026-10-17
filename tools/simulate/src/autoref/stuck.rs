//! Time local stuck play per player, rather than penalizing whoever next approaches an old ball.
use std::time::Duration;

use enum_map::EnumMap;
use game_controller_autoreferee::{NUM_PLAYERS, World};
use game_controller_core::{
    GameController,
    action::VAction,
    actions::{GlobalGameStuck, Penalize},
    types::{Penalty, PenaltyCall, Phase, PlayerNumber, SetPlay, Side, State},
};

// Match the pinned upstream referee's distances, footprint convention and durations.
const BALL_MOVEMENT: f32 = 0.1;
const LOCAL_PROXIMITY: f32 = 0.5;
const LOCAL_DURATION: Duration = Duration::from_secs(10);
const GLOBAL_PROXIMITY: f32 = 1.0;
const GLOBAL_DURATION: Duration = Duration::from_secs(30);

#[derive(Default)]
pub(super) struct Stuck {
    clock: Duration,
    last_update: Option<Duration>,
    was_stopped: bool,
    last_near_ball: Duration,
    ball_reference: Option<[f32; 2]>,
    near_since: EnumMap<Side, [Option<Duration>; NUM_PLAYERS]>,
}

impl Stuck {
    pub fn update(&mut self, gc: &GameController, world: &World, enabled: bool) -> Option<VAction> {
        let now = gc.get_time();
        let game = gc.get_game(false);
        if !self.was_stopped && !game.stopped {
            self.clock += now.saturating_sub(self.last_update.unwrap_or(now));
        }
        self.last_update = Some(now);
        self.was_stopped = game.stopped;

        if !enabled
            || game.phase == Phase::PenaltyShootout
            || game.state != State::Playing
            || game.set_play != SetPlay::NoSetPlay
        {
            self.reset_local();
            self.last_near_ball = self.clock;
            return None;
        }
        if game.stopped {
            return None;
        }

        let [ball_x, ball_y, _] = world.ball;
        if self
            .ball_reference
            .is_none_or(|[x, y]| (ball_x - x).hypot(ball_y - y) > BALL_MOVEMENT)
        {
            self.reset_local();
            self.ball_reference = Some([ball_x, ball_y]);
        }

        let mut player_near_ball = false;
        let mut closest: Option<(Side, PlayerNumber, f32)> = None;
        for (side, players) in &world.players {
            for (index, pose) in players.iter().enumerate() {
                let player = PlayerNumber::new(index as u8 + PlayerNumber::MIN);
                let since = &mut self.near_since[side][index];
                let Some(pose) =
                    pose.filter(|_| game.teams[side][player].penalty == Penalty::NoPenalty)
                else {
                    *since = None;
                    continue;
                };
                let distance = (pose.x - ball_x).hypot(pose.y - ball_y) - pose.radius;
                player_near_ball |= distance <= GLOBAL_PROXIMITY;
                if distance > LOCAL_PROXIMITY {
                    *since = None;
                    continue;
                }
                let since = *since.get_or_insert(self.clock);
                if self.clock.saturating_sub(since) >= LOCAL_DURATION
                    && closest.is_none_or(|(_, _, previous)| distance < previous)
                {
                    closest = Some((side, player, distance));
                }
            }
        }

        if player_near_ball {
            self.last_near_ball = self.clock;
        } else if self.clock.saturating_sub(self.last_near_ball) >= GLOBAL_DURATION {
            return Some(VAction::GlobalGameStuck(GlobalGameStuck));
        }
        let (side, player, _) = closest?;
        self.reset_local();
        Some(VAction::Penalize(Penalize {
            side,
            player,
            call: PenaltyCall::LocalGameStuck,
        }))
    }

    fn reset_local(&mut self) {
        self.ball_reference = None;
        self.near_since = EnumMap::default();
    }
}

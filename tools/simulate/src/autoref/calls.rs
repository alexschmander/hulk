//! Referee interventions offered to the operator. The upstream core decides their legality.
use super::{side, team};
use crate::{RobotId, TeamId};
use game_controller_core::{
    action::VAction,
    actions::*,
    types::{Game, Penalty, PenaltyCall, Phase, PlayerNumber, SetPlay, Side, State},
};

/// One operator call. Each maps to a single core action; the engine adds physical handling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Call {
    StopPlay,
    ResumePlay,
    /// Ready for the kickoff owned by the core's kicking side.
    Kickoff,
    Set,
    /// Free the set play and blow the start whistle.
    Whistle,
    BallFree,
    DroppedBall,
    FinishHalf,
    SecondHalf,
    RefereeTimeout,
    AddMinute,
    Goal(TeamId),
    Restart(TeamId, SetPlay),
    Timeout(TeamId),
    Penalize(RobotId, PenaltyCall),
    /// Release a penalty before its timer expires.
    Release(RobotId),
}

/// Free kicks the operator can award, in the order of the HSL GameController.
pub(crate) const RESTARTS: [SetPlay; 6] = [
    SetPlay::ThrowIn,
    SetPlay::CornerKick,
    SetPlay::GoalKick,
    SetPlay::DirectFreeKick,
    SetPlay::IndirectFreeKick,
    SetPlay::PenaltyKick,
];

/// Penalty calls in the HSL GameController's order, with the card calls last.
pub(crate) const PENALTIES: [PenaltyCall; 12] = [
    PenaltyCall::Pushing,
    PenaltyCall::IncapableRobot,
    PenaltyCall::LeavingTheField,
    PenaltyCall::IllegalPosition,
    PenaltyCall::MotionInSet,
    PenaltyCall::MotionInStop,
    PenaltyCall::BallHolding,
    PenaltyCall::LocalGameStuck,
    PenaltyCall::RequestForPickUp,
    PenaltyCall::PlayingWithArmsHands,
    PenaltyCall::Caution,
    PenaltyCall::SendOff,
];

impl Call {
    pub fn action(self, game: &Game) -> VAction {
        let player = |id: RobotId| PlayerNumber::new(id.number);
        match self {
            Self::StopPlay => VAction::StopPlay(StopPlay { resume: false }),
            Self::ResumePlay => VAction::StopPlay(StopPlay { resume: true }),
            Self::Kickoff => VAction::StartSetPlay(StartSetPlay {
                side: game.kicking_side,
                set_play: SetPlay::KickOff,
            }),
            Self::Set => VAction::WaitForSetPlay(WaitForSetPlay),
            Self::Whistle => VAction::FreeSetPlay(FreeSetPlay),
            Self::BallFree => VAction::FinishSetPlay(FinishSetPlay),
            Self::DroppedBall => VAction::GlobalGameStuck(GlobalGameStuck),
            Self::FinishHalf => VAction::FinishHalf(FinishHalf),
            Self::SecondHalf => VAction::SwitchHalf(SwitchHalf),
            Self::RefereeTimeout => VAction::Timeout(Timeout { side: None }),
            Self::AddMinute => VAction::AddAdditionalTime(AddAdditionalTime),
            Self::Goal(team) => VAction::Goal(Goal { side: side(team) }),
            Self::Restart(team, set_play) => VAction::StartSetPlay(StartSetPlay {
                side: Some(side(team)),
                set_play,
            }),
            Self::Timeout(team) => VAction::Timeout(Timeout {
                side: Some(side(team)),
            }),
            Self::Penalize(id, call) => VAction::Penalize(Penalize {
                side: side(id.team),
                player: player(id),
                call,
            }),
            Self::Release(id) => VAction::Unpenalize(Unpenalize {
                side: side(id.team),
                player: player(id),
                force: true,
            }),
        }
    }

    /// The operator's next step through the match sequence, if the state has one.
    pub fn next(game: &Game) -> Option<Self> {
        if game.phase == Phase::PenaltyShootout {
            return None;
        }
        match game.state {
            State::Initial | State::Timeout => Some(Self::Kickoff),
            State::Ready => Some(Self::Set),
            State::Set => Some(Self::Whistle),
            State::Playing if game.set_play != SetPlay::NoSetPlay => Some(Self::BallFree),
            State::Finished if game.phase == Phase::FirstHalf => Some(Self::SecondHalf),
            _ => None,
        }
    }

    /// Why the core refuses this call in the current state, phrased for the operator.
    pub fn refusal(self, game: &Game) -> String {
        let playing = "Only while the ball is in play";
        match self {
            Self::StopPlay => "Play is already stopped".into(),
            Self::ResumePlay => "Play is not stopped".into(),
            Self::Kickoff => "Only before a kickoff or after a timeout".into(),
            Self::Set => "Only while robots walk to their positions in Ready".into(),
            Self::Whistle => "Only in Set".into(),
            Self::BallFree => "Only during a set play".into(),
            Self::DroppedBall | Self::Goal(_) => playing.into(),
            Self::FinishHalf => "Only during a half".into(),
            Self::SecondHalf => "Only during halftime".into(),
            Self::RefereeTimeout => "Not while a half is finished".into(),
            Self::AddMinute => "Only during a half that has used at least a minute".into(),
            Self::Restart(team, _) => {
                if game.state != State::Playing {
                    playing.into()
                } else {
                    format!("{} already has this restart", name(team))
                }
            }
            Self::Timeout(team) => {
                if game.teams[side(team)].timeout_budget == 0 {
                    format!("{} has no timeout left", name(team))
                } else {
                    "Only while the ball is out of play".into()
                }
            }
            Self::Penalize(id, call) => {
                let penalty = player_penalty(game, id);
                if matches!(penalty, Penalty::SentOff | Penalty::Substitute) {
                    format!("{} is {}", short(id), penalty_label(penalty).to_lowercase())
                } else if penalty != Penalty::NoPenalty {
                    format!("{} is already penalized", short(id))
                } else {
                    match call {
                        PenaltyCall::MotionInSet => "Only in Set".into(),
                        PenaltyCall::MotionInStop => "Only while play is stopped".into(),
                        PenaltyCall::LocalGameStuck => playing.into(),
                        PenaltyCall::BallHolding | PenaltyCall::PlayingWithArmsHands => {
                            "Only in Ready or while the ball is in play".into()
                        }
                        _ => "Only in Ready, Set or while the ball is in play".into(),
                    }
                }
            }
            Self::Release(id) => format!("{} is not serving a penalty", short(id)),
        }
    }

    pub fn label(self) -> String {
        match self {
            Self::StopPlay => "Stop play".into(),
            Self::ResumePlay => "Resume play".into(),
            Self::Kickoff => "Ready".into(),
            Self::Set => "Set".into(),
            Self::Whistle => "Whistle".into(),
            Self::BallFree => "Ball free".into(),
            Self::DroppedBall => "Dropped ball".into(),
            Self::FinishHalf => "End half".into(),
            Self::SecondHalf => "Second half".into(),
            Self::RefereeTimeout => "Referee timeout".into(),
            Self::AddMinute => "Add a minute".into(),
            Self::Goal(_) => "Goal".into(),
            Self::Restart(_, set_play) => set_play_label(set_play).into(),
            Self::Timeout(_) => "Timeout".into(),
            Self::Penalize(_, call) => call_label(call).into(),
            Self::Release(_) => "Release".into(),
        }
    }

    /// What the call does, for tooltips.
    pub fn hint(self, game: &Game) -> String {
        match self {
            Self::StopPlay => "Halt the match; robots must stand still".into(),
            Self::ResumePlay => "Let the stopped match continue".into(),
            Self::Kickoff => format!(
                "Robots walk to their kickoff positions; {} kicks off",
                game.kicking_side
                    .map_or("nobody".into(), |side| name(team(side)).to_owned())
            ),
            Self::Set => "End Ready early; the ball is placed for the restart".into(),
            Self::Whistle => "Start play with a whistle every robot hears".into(),
            Self::BallFree => "End the set play now; both teams may play the ball".into(),
            Self::DroppedBall => "Neutral kickoff for a stuck game".into(),
            Self::FinishHalf => "End this half now".into(),
            Self::SecondHalf => "Skip the rest of halftime; teams change ends".into(),
            Self::RefereeTimeout => "Interrupt the match; the clock stops".into(),
            Self::AddMinute => "Add a minute to the match clock".into(),
            Self::Goal(team) => format!("Award a goal to {}", name(team)),
            Self::Restart(team, set_play) => {
                format!(
                    "Award a {} to {}",
                    set_play_label(set_play).to_lowercase(),
                    name(team)
                )
            }
            Self::Timeout(team) => format!(
                "{} takes a timeout; {} left",
                name(team),
                game.teams[side(team)].timeout_budget
            ),
            Self::Penalize(id, call) => format!(
                "Penalize {} for {}",
                short(id),
                call_label(call).to_lowercase()
            ),
            Self::Release(id) => format!("Let {} return now; it walks back itself", short(id)),
        }
    }
}

/// Describes an applied or refused core action in the referee log.
pub(crate) fn describe(action: &VAction, game: &Game) -> String {
    let robot = |side: Side, player: PlayerNumber| {
        short(RobotId {
            team: team(side),
            number: u8::from(player),
        })
    };
    match action {
        VAction::Goal(goal) => format!("Goal for {}", name(team(goal.side))),
        VAction::StartSetPlay(start) => match start.side {
            Some(side) => format!(
                "{} for {}",
                set_play_label(start.set_play),
                name(team(side))
            ),
            None => format!("Neutral {}", set_play_label(start.set_play).to_lowercase()),
        },
        VAction::GlobalGameStuck(_) => "Dropped ball".into(),
        VAction::StopPlay(stop) => if stop.resume {
            "Play resumed"
        } else {
            "Play stopped"
        }
        .into(),
        VAction::WaitForSetPlay(_) => "Set".into(),
        VAction::FreeSetPlay(_) => "Play started with a whistle".into(),
        VAction::FinishSetPlay(_) => "Ball free".into(),
        VAction::FinishHalf(_) => "Half finished".into(),
        VAction::SwitchHalf(_) => "Second half".into(),
        VAction::Timeout(timeout) => match timeout.side {
            Some(side) => format!("Timeout for {}", name(team(side))),
            None => "Referee timeout".into(),
        },
        VAction::AddAdditionalTime(_) => "A minute added".into(),
        VAction::Penalize(penalize) => format!(
            "{} penalized for {}",
            robot(penalize.side, penalize.player),
            call_label(penalize.call).to_lowercase()
        ),
        VAction::Unpenalize(release) => {
            let id = robot(release.side, release.player);
            if game.teams[release.side][release.player].penalty == Penalty::NoPenalty {
                format!("{id} back in play")
            } else {
                format!("{id} placed; penalty time started")
            }
        }
        other => format!("{other:?}"),
    }
}

pub(crate) fn player_penalty(game: &Game, id: RobotId) -> Penalty {
    game.teams[side(id.team)][PlayerNumber::new(id.number)].penalty
}

/// The scene badge, H1–H5 or O1–O5.
pub(crate) fn short(id: RobotId) -> String {
    format!(
        "{}{}",
        match id.team {
            TeamId::Hulks => "H",
            TeamId::Opponents => "O",
        },
        id.number
    )
}

pub(crate) fn name(team: TeamId) -> &'static str {
    match team {
        TeamId::Hulks => "HULKs",
        TeamId::Opponents => "Opponents",
    }
}

pub(crate) fn set_play_label(set_play: SetPlay) -> &'static str {
    match set_play {
        SetPlay::NoSetPlay => "Open play",
        SetPlay::KickOff => "Kickoff",
        SetPlay::DirectFreeKick => "Direct free kick",
        SetPlay::IndirectFreeKick => "Indirect free kick",
        SetPlay::PenaltyKick => "Penalty kick",
        SetPlay::ThrowIn => "Kick-in",
        SetPlay::GoalKick => "Goal kick",
        SetPlay::CornerKick => "Corner kick",
    }
}

pub(crate) fn call_label(call: PenaltyCall) -> &'static str {
    match call {
        PenaltyCall::Pushing => "Pushing",
        PenaltyCall::IncapableRobot => "Incapable robot",
        PenaltyCall::LeavingTheField => "Leaving the field",
        PenaltyCall::IllegalPosition => "Illegal position",
        PenaltyCall::MotionInSet => "Motion in Set",
        PenaltyCall::MotionInStop => "Motion in Stop",
        PenaltyCall::BallHolding => "Ball holding",
        PenaltyCall::LocalGameStuck => "Local game stuck",
        PenaltyCall::RequestForPickUp => "Pick-up",
        PenaltyCall::PlayingWithArmsHands => "Arms or hands",
        PenaltyCall::Caution => "Yellow card",
        PenaltyCall::SendOff => "Red card",
    }
}

pub(crate) fn penalty_label(penalty: Penalty) -> &'static str {
    match penalty {
        Penalty::NoPenalty => "Playing",
        Penalty::Substitute => "Substitute",
        Penalty::PickedUp => "Picked up",
        Penalty::IllegalPositioning => "Illegal position",
        Penalty::MotionInSet => "Motion in Set",
        Penalty::MotionInStop => "Motion in Stop",
        Penalty::LocalGameStuck => "Local game stuck",
        Penalty::IncapableRobot => "Incapable robot",
        Penalty::BallHolding => "Ball holding",
        Penalty::LeavingTheField => "Leaving the field",
        Penalty::PlayingWithArmsHands => "Arms or hands",
        Penalty::Pushing => "Pushing",
        Penalty::Cautioned => "Yellow card",
        Penalty::SentOff => "Sent off",
    }
}

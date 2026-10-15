use std::{f32::consts::FRAC_1_SQRT_2, time::Duration};

use coordinate_systems::Field;
use hsl_network_messages::{GamePhase, Team};
use linear_algebra::{Point2, point, vector};
use ros_z::{Message, time::Time};
use serde::{Deserialize, Serialize};
use types::{
    behavior_tree::Status,
    filtered_game_state::FilteredGameState,
    motion_command::{BodyMotion, OrientationMode},
    motion_type::MotionType,
    primary_state::PrimaryState,
};

use crate::{
    kick::{
        KickStrength, allow_strong_kicks, apply_kick_target, kick, use_kick, use_kick_strength,
    },
    node::Blackboard,
    switch_motion_type::is_allowed_to_switch,
    walk::walk_to_ball_target,
};

/// Retained across the GC's transition from whistle-in-Set to ordinary Playing.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, Message)]
pub enum KickoffState {
    #[default]
    Idle,
    Armed,
    Touch {
        origin: Point2<Field>,
        target: Point2<Field>,
        started: Time,
        moved_at: Option<Time>,
    },
    FollowUp {
        touch_target: Point2<Field>,
        started: Time,
    },
    Settle {
        started: Time,
        since: Time,
    },
    Shot {
        origin: Point2<Field>,
        started: Time,
        moved_at: Option<Time>,
    },
    Done,
}

/// Runs even when a higher-priority behavior handles motion or the ball is lost.
pub fn update_kickoff(blackboard: &mut Blackboard) -> Status {
    let eligible = blackboard
        .world_state
        .filtered_game_controller_state
        .as_ref()
        .is_some_and(|game| {
            game.kicking_team == Some(Team::Hulks)
                && game.sub_state.is_none()
                && !matches!(game.game_phase, GamePhase::PenaltyShootout { .. })
        });
    if blackboard.world_state.robot.primary_state != PrimaryState::Playing {
        blackboard.kickoff =
            if eligible && blackboard.world_state.robot.primary_state == PrimaryState::Set {
                KickoffState::Armed
            } else {
                KickoffState::Idle
            };
    } else if !eligible {
        blackboard.kickoff = KickoffState::Done;
    } else if let KickoffState::Touch { started, .. }
    | KickoffState::FollowUp { started, .. }
    | KickoffState::Settle { started, .. }
    | KickoffState::Shot { started, .. } = blackboard.kickoff
        && blackboard.world_state.now.duration_since(started) > Duration::from_secs(20)
    {
        blackboard.kickoff = KickoffState::Done;
    }
    Status::Success
}

/// A single striker touches diagonally left, recovers behind the ball, then shoots.
pub fn kickoff(blackboard: &mut Blackboard) -> Status {
    let Some(ball) = blackboard.ball.as_ref() else {
        return Status::Failure;
    };
    let ball_position = ball.position;
    let now = blackboard.world_state.now;
    if matches!(blackboard.kickoff, KickoffState::Idle | KickoffState::Armed) {
        let kick_off = blackboard
            .world_state
            .filtered_game_controller_state
            .as_ref()
            .is_some_and(|game| {
                matches!(
                    game.game_state,
                    FilteredGameState::Playing { kick_off: true, .. }
                ) && game.kicking_team == Some(Team::Hulks)
            });
        if !matches!(blackboard.kickoff, KickoffState::Armed) && !kick_off {
            return Status::Failure;
        }
        // Reject a late join to Playing after the ball has already left the centre.
        if ball_position.coords().norm() > 0.4 {
            blackboard.kickoff = KickoffState::Done;
            return Status::Failure;
        }
        blackboard.kickoff = KickoffState::Touch {
            origin: ball_position,
            target: ball_position + vector![FRAC_1_SQRT_2, FRAC_1_SQRT_2] * 2.0,
            started: now,
            moved_at: None,
        };
    }
    let goal = point![blackboard.field_dimensions.length / 2.0, 0.0];
    match blackboard.kickoff {
        KickoffState::Touch {
            origin,
            target,
            started,
            mut moved_at,
        } => {
            if blackboard.last_motion_type == Some(MotionType::Kick)
                && (ball_position - origin).norm() >= 0.25
            {
                moved_at.get_or_insert(now);
            }
            if moved_at.is_some_and(|time| now.duration_since(time) >= Duration::from_millis(350)) {
                blackboard.kickoff = KickoffState::FollowUp {
                    touch_target: target,
                    started,
                };
                return approach(blackboard, goal, target);
            }
            blackboard.kickoff = KickoffState::Touch {
                origin,
                target,
                started,
                moved_at,
            };
            if blackboard.last_motion_type == Some(MotionType::Kick)
                || ready_to_kick(blackboard, target)
                || moved_at.is_some()
            {
                command_kick(blackboard, target, KickStrength::Soft)
            } else {
                approach(blackboard, target, target)
            }
        }
        KickoffState::FollowUp {
            touch_target,
            started,
        } => {
            // Explicitly leave the soft policy before starting a fresh goal shot.
            if blackboard.last_motion_type == Some(MotionType::Walk)
                && is_allowed_to_switch(blackboard)
                && ready_to_kick(blackboard, goal)
                && blackboard
                    .ball
                    .as_ref()
                    .is_some_and(|ball| ball.velocity.norm() < 0.2)
            {
                blackboard.kickoff = KickoffState::Settle {
                    started,
                    since: now,
                };
                blackboard.body_motion = Some(BodyMotion::Stand);
                Status::Success
            } else {
                approach(blackboard, goal, touch_target)
            }
        }
        KickoffState::Settle { started, since } => {
            if now.duration_since(since) >= Duration::from_millis(600) {
                if blackboard
                    .ball
                    .as_ref()
                    .is_some_and(|ball| ball.velocity.norm() >= 0.2)
                {
                    blackboard.kickoff = KickoffState::FollowUp {
                        touch_target: goal,
                        started,
                    };
                    return approach(blackboard, goal, goal);
                }
                blackboard.kickoff = KickoffState::Shot {
                    origin: ball_position,
                    started,
                    moved_at: None,
                };
                command_kick(blackboard, goal, KickStrength::Strong)
            } else {
                blackboard.body_motion = Some(BodyMotion::Stand);
                Status::Success
            }
        }
        KickoffState::Shot {
            origin,
            started,
            mut moved_at,
        } => {
            if (ball_position - origin).norm() >= 0.5 {
                moved_at.get_or_insert(now);
            }
            blackboard.kickoff = if moved_at
                .is_some_and(|time| now.duration_since(time) >= Duration::from_millis(500))
            {
                KickoffState::Done
            } else {
                KickoffState::Shot {
                    origin,
                    started,
                    moved_at,
                }
            };
            command_kick(blackboard, goal, KickStrength::Strong)
        }
        _ => Status::Failure,
    }
}

fn ready_to_kick(blackboard: &Blackboard, target: Point2<Field>) -> bool {
    let Some(pose) = blackboard.world_state.robot.ground_to_field else {
        return false;
    };
    let Some(ball) = &blackboard.ball else {
        return false;
    };
    let ball_in_ground = pose.inverse() * ball.position;
    let direction = pose.inverse() * (target - ball.position);
    let position_ready = if matches!(
        blackboard.kickoff,
        KickoffState::FollowUp { .. } | KickoffState::Shot { .. }
    ) {
        ball_in_ground.x() > 0.75
            && ball_in_ground.coords().norm() < 1.05
            && (ball_in_ground.y() - 0.08).abs() < 0.1
    } else {
        ball_in_ground.coords().norm() < 0.75
            && ball_in_ground.x() > 0.0
            && ball_in_ground.y().abs() < 0.1
    };
    position_ready && direction.y().atan2(direction.x()).abs() < 0.15
}

fn approach(
    blackboard: &mut Blackboard,
    target: Point2<Field>,
    previous_target: Point2<Field>,
) -> Status {
    if blackboard.last_motion_type == Some(MotionType::Walk) || is_allowed_to_switch(blackboard) {
        walk_approach(blackboard, target)
    } else if blackboard.last_motion_type == Some(MotionType::Kick) {
        command_kick(blackboard, previous_target, KickStrength::Soft)
    } else {
        blackboard.body_motion = Some(BodyMotion::Stand);
        Status::Success
    }
}

fn command_kick(
    blackboard: &mut Blackboard,
    target: Point2<Field>,
    strength: KickStrength,
) -> Status {
    if blackboard.last_motion_type != Some(MotionType::Kick) && !is_allowed_to_switch(blackboard) {
        if blackboard.last_motion_type == Some(MotionType::Walk) {
            return walk_approach(blackboard, target);
        }
        blackboard.body_motion = Some(BodyMotion::Stand);
        return Status::Success;
    }
    if kick(blackboard) == Status::Failure
        || apply_kick_target(blackboard, target) == Status::Failure
    {
        return Status::Failure;
    }
    // The stage owns strength, including the transition from the latched soft touch.
    if strength == KickStrength::Strong && !allow_strong_kicks(blackboard) {
        use_kick(blackboard, false)
    } else {
        use_kick_strength(blackboard, strength)
    }
}

fn walk_approach(blackboard: &mut Blackboard, target: Point2<Field>) -> Status {
    let (standoff, lateral_offset) = if matches!(
        blackboard.kickoff,
        KickoffState::FollowUp { .. } | KickoffState::Shot { .. }
    ) {
        (0.85, 0.08)
    } else {
        (0.6, 0.0)
    };
    let status = walk_to_ball_target(blackboard, target, standoff, lateral_offset);
    if let Some(BodyMotion::Walk {
        orientation_mode,
        target_orientation,
        ..
    }) = &mut blackboard.body_motion
    {
        // Align before getting close enough for the walking foot to touch the ball.
        *orientation_mode = OrientationMode::LookTowards {
            direction: *target_orientation,
            tolerance: 0.1,
        };
    }
    status
}

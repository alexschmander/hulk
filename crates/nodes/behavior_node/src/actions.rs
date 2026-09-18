use linear_algebra::vector;
use types::{
    behavior_tree::Status,
    motion_command::{BodyMotion, MotionCommand},
};

use crate::node::Blackboard;

pub fn damping(blackboard: &mut Blackboard) -> Status {
    blackboard.body_motion = Some(BodyMotion::Damping);
    Status::Success
}

pub fn injected_motion_command(blackboard: &mut Blackboard) -> Status {
    if blackboard
        .parameters
        .control
        .injected_motion_command
        .is_some()
    {
        blackboard.is_injected_motion_command = true;
        Status::Success
    } else {
        Status::Failure
    }
}

pub fn prepare(blackboard: &mut Blackboard) -> Status {
    blackboard.body_motion = Some(BodyMotion::Prepare);
    Status::Success
}

pub fn remote_control(blackboard: &mut Blackboard) -> Status {
    let parameters = &blackboard.parameters.control.remote_control;
    let remote_control_body_motion = BodyMotion::WalkWithVelocity {
        velocity: vector![parameters.walk.forward, parameters.walk.left,],
        angular_velocity: parameters.walk.turn,
    };
    blackboard.body_motion = Some(remote_control_body_motion);
    Status::Success
}

pub fn stand(blackboard: &mut Blackboard) -> Status {
    blackboard.body_motion = Some(BodyMotion::Stand);
    Status::Success
}

pub fn stand_up(blackboard: &mut Blackboard) -> Status {
    let fast = match blackboard.last_motion_command {
        MotionCommand::StandUp { fast } => fast,
        _ => blackboard.parameters.stand_up.fast,
    };
    blackboard.body_motion = Some(BodyMotion::StandUp { fast });
    Status::Success
}

/// Safety precedes remote control and injected commands. Motion independently enforces the same gate.
pub fn safety_motion(blackboard: &mut Blackboard) -> Status {
    use types::{
        fall_detection::{MAXIMUM_FALL_DETECTION_AGE, Posture},
        motion_execution::MotionPhase,
    };
    let now = blackboard.world_state.now;
    let Some(fall) = blackboard
        .world_state
        .fall_detection
        .filter(|s| s.is_fresh(now, MAXIMUM_FALL_DETECTION_AGE))
    else {
        return damping(blackboard);
    };
    if let Some(execution) = blackboard
        .world_state
        .motion_execution
        .as_ref()
        .filter(|s| s.is_fresh(now))
    {
        match execution.phase {
            MotionPhase::Fault => return damping(blackboard),
            MotionPhase::Recovering { fast } => {
                blackboard.body_motion = Some(BodyMotion::StandUp { fast });
                return Status::Success;
            }
            MotionPhase::Settling => return stand(blackboard),
            _ => {}
        }
    }
    match fall.posture {
        Posture::Fallen => stand_up(blackboard),
        Posture::Falling | Posture::Unknown => damping(blackboard),
        Posture::Upright => Status::Failure,
    }
}

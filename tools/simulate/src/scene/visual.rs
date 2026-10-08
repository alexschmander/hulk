use bevy::prelude::*;

use super::{ball::BallAssets, goal::GoalAssets, robot::RobotAssets};

#[derive(Resource)]
pub struct ObjectVisualAssets {
    pub ball: BallAssets,
    pub goal: GoalAssets,
    pub robot: RobotAssets,
}

impl FromWorld for ObjectVisualAssets {
    fn from_world(world: &mut World) -> Self {
        Self {
            ball: BallAssets::load(world),
            goal: GoalAssets::load(world),
            robot: RobotAssets::from_world(world),
        }
    }
}

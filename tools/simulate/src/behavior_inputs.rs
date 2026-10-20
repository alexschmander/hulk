//! Ground-truth substitutes for perception/localization, with real downstream composers.
use std::f32::consts::PI;

use color_eyre::Result;
use coordinate_systems::{Field, Ground};
use linear_algebra::{Isometry2, Point2, point, vector};
use ros_z::{Message, prelude::*, time::Time};
use serde::{Deserialize, Serialize};
use types::{ball_position::BallPosition, field_dimensions::GlobalFieldSide, obstacles::Obstacle};

use crate::{
    profiles::Profile,
    reference::Reference,
    team::{RobotId, TeamId},
};

/// Physical truth of another robot, as seen from the robot publishing it.
#[derive(Clone, Debug, Deserialize, Serialize, Message)]
pub struct HumanoidTruth {
    pub team: TruthTeam,
    pub player_number: u8,
    /// Whether the robot is on the observer's team.
    pub is_teammate: bool,
    pub position_in_ground: Point2<Ground>,
    pub position_in_field: Point2<Field>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, Message)]
pub enum TruthTeam {
    Hulks,
    Opponents,
}

pub struct BehaviorInputs {
    pose: Reference<Isometry2<Ground, Field>>,
    ball: Reference<Option<BallPosition<Ground>>>,
    visual_ball: Reference<Option<BallPosition<Ground>>>,
    obstacles: Reference<Vec<Obstacle>>,
    humanoid_obstacles: Reference<Vec<Obstacle>>,
    stationary_obstacles: Reference<Vec<Obstacle>>,
    humanoids: Reference<Vec<HumanoidTruth>>,
    interest: Publisher<Point2<Ground>>,
    selected_ball: ros_z::cache::Cache<Option<BallPosition<Ground>>>,
}

impl BehaviorInputs {
    pub async fn new(node: &Node, profile: Profile) -> Result<Self> {
        Ok(Self {
            pose: Reference::new(node, "ground_to_field", profile).await?,
            ball: Reference::new(node, "ball_filter/ball_position", profile).await?,
            visual_ball: Reference::new(node, "visual_kick/ball_position", profile).await?,
            obstacles: Reference::new(node, "obstacles", profile).await?,
            humanoid_obstacles: Reference::new(node, "humanoid_filter/obstacles", profile).await?,
            stationary_obstacles: Reference::new(
                node,
                "stationary_object_filter/obstacles",
                profile,
            )
            .await?,
            humanoids: Reference::new(node, "humanoids", profile).await?,
            selected_ball: node
                .subscriber("ball_filter/ball_position")
                .cache(1)
                .build()
                .await?,
            interest: node.publisher("position_of_interest").build().await?,
        })
    }

    pub async fn publish(
        &self,
        ground_to_world: nalgebra::Isometry3<f32>,
        ball: Option<([f64; 3], [f64; 3])>,
        field: &types::field_dimensions::FieldDimensions,
        side: GlobalFieldSide,
        robots: &[[f32; 3]],
        robot_ids: &[RobotId],
        observer: RobotId,
        time: Time,
    ) -> Result<()> {
        let pose = ground_to_field(ground_to_world, side);
        let ball = ball
            .map(|(position, velocity)| ball_in_ground(ground_to_world, position, velocity, time));
        let inverse = ground_to_world.inverse();
        let goal_post_obstacles = || goal_posts_in_ground(field, side, ground_to_world);
        let robot_obstacles = || {
            robots
                .iter()
                .map(|position| {
                    let p = inverse * nalgebra::Point3::from(*position);
                    Obstacle::robot(point![p.x, p.y], 0.2, 0.25)
                })
                .collect::<Vec<_>>()
        };
        let obstacles = self.obstacles.needed().then(|| {
            let mut obstacles = goal_post_obstacles();
            obstacles.extend(robot_obstacles());
            obstacles
        });
        if self.humanoid_obstacles.needed() {
            self.humanoid_obstacles
                .publish(&robot_obstacles(), time)
                .await?;
        }
        if self.stationary_obstacles.needed() {
            self.stationary_obstacles
                .publish(&goal_post_obstacles(), time)
                .await?;
        }
        if self.humanoids.needed() {
            let world_to_field = world_to_field(side);
            let humanoids: Vec<_> = robots
                .iter()
                .zip(robot_ids)
                .map(|(position, id)| {
                    let ground = inverse * nalgebra::Point3::from(*position);
                    let field = world_to_field * nalgebra::point![position[0], position[1]];
                    HumanoidTruth {
                        team: match id.team {
                            TeamId::Hulks => TruthTeam::Hulks,
                            TeamId::Opponents => TruthTeam::Opponents,
                        },
                        player_number: id.number,
                        is_teammate: id.team == observer.team,
                        position_in_ground: point![ground.x, ground.y],
                        position_in_field: point![field.x, field.y],
                    }
                })
                .collect();
            self.humanoids.publish(&humanoids, time).await?;
        }
        self.pose.publish(&pose, time).await?;
        self.ball.publish(&ball, time).await?;
        self.visual_ball.publish(&ball, time).await?;
        if let Some(obstacles) = obstacles {
            self.obstacles.publish(&obstacles, time).await?;
        }
        self.interest
            .publish_with_source_time(
                &self
                    .selected_ball
                    .get_latest()
                    .and_then(|ball| *ball)
                    .map_or(point![1.0, 0.0], |b| b.position),
                time,
            )
            .await?;
        Ok(())
    }
}

fn goal_posts_in_ground(
    field: &types::field_dimensions::FieldDimensions,
    side: GlobalFieldSide,
    ground_to_world: nalgebra::Isometry3<f32>,
) -> Vec<Obstacle> {
    use types::field_dimensions::{Half, Side};
    let field_to_world = world_to_field(side).inverse();
    let world_to_ground = ground_to_world.inverse();
    [Half::Own, Half::Opponent]
        .into_iter()
        .flat_map(|half| {
            [Side::Left, Side::Right].map(|field_side| {
                let position = field.goal_post(half, field_side);
                let world = field_to_world * nalgebra::point![position.x(), position.y()];
                let ground = world_to_ground * nalgebra::point![world.x, world.y, 0.0];
                Obstacle::goal_post(point![ground.x, ground.y], field.goal_post_diameter / 2.0)
            })
        })
        .collect()
}

fn world_to_field(side: GlobalFieldSide) -> nalgebra::Isometry2<f32> {
    let rotation = if side == GlobalFieldSide::Home {
        0.0
    } else {
        PI
    };
    nalgebra::Isometry2::new(nalgebra::Vector2::zeros(), rotation)
}

fn ground_to_field(
    pose: nalgebra::Isometry3<f32>,
    side: GlobalFieldSide,
) -> Isometry2<Ground, Field> {
    let yaw = pose.rotation.euler_angles().2;
    let ground = nalgebra::Isometry2::new(
        nalgebra::vector![pose.translation.x, pose.translation.y],
        yaw,
    );
    let rotation = if side == GlobalFieldSide::Home {
        0.0
    } else {
        PI
    };
    Isometry2::wrap(nalgebra::Isometry2::new(nalgebra::Vector2::zeros(), rotation) * ground)
}

fn ball_in_ground(
    pose: nalgebra::Isometry3<f32>,
    position: [f64; 3],
    velocity: [f64; 3],
    time: Time,
) -> BallPosition<Ground> {
    let inverse = pose.inverse();
    let position = inverse * nalgebra::Point3::from(position.map(|v| v as f32));
    let velocity = inverse.rotation * nalgebra::Vector3::from(velocity.map(|v| v as f32));
    BallPosition {
        position: point![position.x, position.y],
        velocity: vector![velocity.x, velocity.y],
        last_seen: time,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ground_truth_rotates_positions_velocities_and_field_side() {
        let pose = nalgebra::Isometry3::new(
            nalgebra::vector![2.0, 3.0, 0.0],
            nalgebra::vector![0.0, 0.0, PI / 2.0],
        );
        let time = Time::from_nanos(42);
        let ball = ball_in_ground(pose, [2.0, 4.0, 0.1], [0.0, 2.0, 0.0], time);
        assert!((ball.position - point![1.0, 0.0]).norm() < 1e-5);
        assert!((ball.velocity - vector![2.0, 0.0]).norm() < 1e-5);
        assert_eq!(ball.last_seen, time);
        for (side, sign) in [(GlobalFieldSide::Home, 1.0), (GlobalFieldSide::Away, -1.0)] {
            let field = ground_to_field(pose, side) * ball.position;
            assert!((field - point![2.0 * sign, 4.0 * sign]).norm() < 1e-5);
        }
    }

    #[test]
    fn goal_post_truth_agrees_with_the_field_pose_on_both_sides() {
        let field = types::field_dimensions::FieldDimensions::SPL_2025;
        let pose = nalgebra::Isometry3::new(
            nalgebra::vector![2.0, 3.0, 0.0],
            nalgebra::vector![0.0, 0.0, PI / 2.0],
        );
        for side in [GlobalFieldSide::Home, GlobalFieldSide::Away] {
            let ground_to_field = ground_to_field(pose, side);
            let mut in_field: Vec<_> = goal_posts_in_ground(&field, side, pose)
                .into_iter()
                .map(|post| ground_to_field * post.position)
                .collect();
            in_field.sort_by(|a, b| (a.x(), a.y()).partial_cmp(&(b.x(), b.y())).unwrap());
            let x = field.length / 2.0;
            let y = (field.goal_inner_width + field.goal_post_diameter) / 2.0;
            for (post, expected) in
                in_field
                    .iter()
                    .zip([point![-x, -y], point![-x, y], point![x, -y], point![x, y]])
            {
                assert!((*post - expected).norm() < 1e-4, "{side:?}: {post:?}");
            }
        }
    }
}

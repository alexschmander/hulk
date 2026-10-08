//! Ground-truth substitutes for perception/localization, with real downstream composers.
use std::f32::consts::PI;

use color_eyre::Result;
use coordinate_systems::{Field, Ground};
use linear_algebra::{Isometry2, Point2, point, vector};
use ros_z::{prelude::*, time::Time};
use types::{ball_position::BallPosition, field_dimensions::GlobalFieldSide, obstacles::Obstacle};

use crate::{profiles::Profile, reference::Reference};

pub struct BehaviorInputs {
    pose: Reference<Isometry2<Ground, Field>>,
    ball: Reference<Option<BallPosition<Ground>>>,
    visual_ball: Reference<Option<BallPosition<Ground>>>,
    obstacles: Reference<Vec<Obstacle>>,
    interest: Publisher<Point2<Ground>>,
    selected_ball: ros_z::cache::Cache<Option<BallPosition<Ground>>>,
}

impl BehaviorInputs {
    pub async fn new(node: &Node, profile: Profile) -> Result<Self> {
        Ok(Self {
            pose: Reference::new(node, "ground_to_field", !profile.localization()).await?,
            ball: Reference::new(node, "ball_filter/ball_position", !profile.filtering()).await?,
            visual_ball: Reference::new(node, "visual_kick/ball_position", !profile.filtering())
                .await?,
            obstacles: Reference::new(node, "obstacles", !profile.filtering()).await?,
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
        time: Time,
    ) -> Result<()> {
        let pose = ground_to_field(ground_to_world, side);
        let ball = ball
            .map(|(position, velocity)| ball_in_ground(ground_to_world, position, velocity, time));
        let inverse = ground_to_world.inverse();
        use types::field_dimensions::{Half, Side};
        let mut obstacles: Vec<_> = [Half::Own, Half::Opponent]
            .into_iter()
            .flat_map(|half| {
                [Side::Left, Side::Right].map(|side| {
                    let position = field.goal_post(half, side);
                    let p = inverse * nalgebra::point![position.x(), position.y(), 0.0];
                    Obstacle::goal_post(point![p.x, p.y], field.goal_post_diameter / 2.0)
                })
            })
            .collect();
        obstacles.extend(robots.iter().map(|position| {
            let p = inverse * nalgebra::Point3::from(*position);
            Obstacle::robot(point![p.x, p.y], 0.2, 0.25)
        }));
        self.pose.publish(&pose, time).await?;
        self.ball.publish(&ball, time).await?;
        self.visual_ball.publish(&ball, time).await?;
        self.obstacles.publish(&obstacles, time).await?;
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
}

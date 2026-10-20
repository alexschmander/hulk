use std::sync::Arc;

use color_eyre::Result;
use eframe::epaint::Stroke;

use coordinate_systems::{Field, Ground};
use linear_algebra::Isometry2;
use ros_z_debug::{SampleRecord, TopicObservation};
use types::{
    field_dimensions::FieldDimensions,
    obstacles::Obstacle,
    tracked_objects::{GoalPostTrack, HumanoidTrack, TrackedObjects},
};

use crate::{backend::RobotBackend, panels::map::layer::Layer};
use twix_visualization::twix_painter::TwixPainter;

use super::tactics_board::{CHALK, INK, dashed_circle, pixels};

/// Hip and foot radii closer than this are drawn as one outline.
const SAME_RADIUS: f32 = 0.01;

/// Obstacles of the humanoid and goal post filters as dashed outlines of the shapes the path
/// planner avoids, so that they never look like the filled uncertainty of the filter layers.
pub struct Obstacles {
    humanoid_obstacles: TopicObservation<Vec<Obstacle>>,
    goal_post_obstacles: TopicObservation<Vec<Obstacle>>,
    humanoids: TopicObservation<TrackedObjects<HumanoidTrack>>,
    goal_posts: TopicObservation<TrackedObjects<GoalPostTrack>>,
}

impl Layer<Ground> for Obstacles {
    const NAME: &'static str = "Obstacles";

    fn new(backend: Arc<RobotBackend>) -> Self {
        let _runtime_handle = backend.runtime_handle().enter();
        let observer = backend.observer();

        Self {
            humanoid_obstacles: observer
                .observe_typed("humanoid_filter/obstacles")
                .expect("failed to construct humanoid obstacles observer")
                .spawn(),
            goal_post_obstacles: observer
                .observe_typed("goal_post_filter/obstacles")
                .expect("failed to construct goal post obstacles observer")
                .spawn(),
            humanoids: observer
                .observe_typed("humanoid_filter/tracks")
                .expect("failed to construct humanoid tracks observer")
                .spawn(),
            goal_posts: observer
                .observe_typed("goal_post_filter/tracks")
                .expect("failed to construct goal post tracks observer")
                .spawn(),
        }
    }

    fn paint(
        &self,
        painter: &TwixPainter<Ground>,
        _field_dimensions: &FieldDimensions,
    ) -> Result<()> {
        for observation in [&self.humanoid_obstacles, &self.goal_post_obstacles] {
            if let Some(SampleRecord {
                value: obstacles, ..
            }) = observation.latest().as_deref()
            {
                paint_obstacles(painter, obstacles, Isometry2::identity());
            }
        }
        Ok(())
    }

    /// Places the obstacles with the ground-to-field transform of their filter's tracks, which
    /// the filter publishes in the same step, so that they line up with the filter layers.
    fn paint_in_field(
        &self,
        painter: &TwixPainter<Field>,
        _field_dimensions: &FieldDimensions,
    ) -> Option<Result<()>> {
        let humanoids_to_field = self
            .humanoids
            .latest()
            .and_then(|record| record.value.ground_to_field);
        let goal_posts_to_field = self
            .goal_posts
            .latest()
            .and_then(|record| record.value.ground_to_field);
        for (observation, ground_to_field) in [
            (&self.humanoid_obstacles, humanoids_to_field),
            (&self.goal_post_obstacles, goal_posts_to_field),
        ] {
            if let Some(SampleRecord {
                value: obstacles, ..
            }) = observation.latest().as_deref()
                && let Some(ground_to_field) = ground_to_field
            {
                paint_obstacles(painter, obstacles, ground_to_field);
            }
        }
        Some(Ok(()))
    }
}

/// Long dashes at foot height and short dots at hip height, in chalk on an ink casing so that they
/// read on the field and on top of the jersey-colored pucks.
fn paint_obstacles<Frame>(
    painter: &TwixPainter<Frame>,
    obstacles: &[Obstacle],
    ground_to_frame: Isometry2<Ground, Frame>,
) {
    for obstacle in obstacles {
        let center = ground_to_frame * obstacle.position;
        paint_outline(painter, center, obstacle.radius_at_foot_height, 6.0);
        if (obstacle.radius_at_hip_height - obstacle.radius_at_foot_height).abs() > SAME_RADIUS {
            paint_outline(painter, center, obstacle.radius_at_hip_height, 2.0);
        }
    }
}

fn paint_outline<Frame>(
    painter: &TwixPainter<Frame>,
    center: linear_algebra::Point2<Frame>,
    radius: f32,
    dash: f32,
) {
    dashed_circle(
        painter,
        center,
        radius,
        dash,
        Stroke::new(pixels(painter, 3.5), INK.gamma_multiply(0.85)),
    );
    dashed_circle(
        painter,
        center,
        radius,
        dash,
        Stroke::new(pixels(painter, 1.5), CHALK),
    );
}

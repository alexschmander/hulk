use std::sync::Arc;

use color_eyre::Result;
use eframe::epaint::Stroke;

use coordinate_systems::{Field, Ground};
use linear_algebra::Isometry2;
use ros_z_debug::{SampleRecord, TopicObservation};
use types::{
    field_dimensions::FieldDimensions,
    tracked_objects::{GoalPostTrack, TrackStatus, TrackedObjects},
};

use crate::{backend::RobotBackend, panels::map::layer::Layer};
use twix_visualization::twix_painter::TwixPainter;

use super::tactics_board::{
    CHALK, INK, MAXIMUM_DRAWN_STANDARD_DEVIATION, odometry_to_ground_rotation, pixels,
};

/// Goal posts of the goal post filter as chalk discs.
pub struct GoalPostFilter {
    goal_posts: TopicObservation<TrackedObjects<GoalPostTrack>>,
}

impl Layer<Ground> for GoalPostFilter {
    const NAME: &'static str = "Goal Post Filter";

    fn new(backend: Arc<RobotBackend>) -> Self {
        let _runtime_handle = backend.runtime_handle().enter();

        Self {
            goal_posts: backend
                .observer()
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
        if let Some(SampleRecord {
            value: goal_posts, ..
        }) = self.goal_posts.latest().as_deref()
        {
            paint_goal_post_tracks(painter, goal_posts, Isometry2::identity());
        }
        Ok(())
    }

    /// Places each message with the ground-to-field transform of its own time, which the filter
    /// publishes with its tracks; messages without localization are left out.
    fn paint_in_field(
        &self,
        painter: &TwixPainter<Field>,
        _field_dimensions: &FieldDimensions,
    ) -> Option<Result<()>> {
        if let Some(SampleRecord {
            value: goal_posts, ..
        }) = self.goal_posts.latest().as_deref()
            && let Some(ground_to_field) = goal_posts.ground_to_field
        {
            paint_goal_post_tracks(painter, goal_posts, ground_to_field);
        }
        Some(Ok(()))
    }
}

/// Goal posts as chalk discs, which they are on the field, at least a few pixels wide.
fn paint_goal_post_tracks<Frame>(
    painter: &TwixPainter<Frame>,
    objects: &TrackedObjects<GoalPostTrack>,
    ground_to_frame: Isometry2<Ground, Frame>,
) {
    let rotation = ground_to_frame.inner.rotation.to_rotation_matrix().matrix()
        * odometry_to_ground_rotation(objects);
    for track in &objects.tracks {
        let covariance = rotation * track.state.covariance * rotation.transpose();
        // Goal post positions are precise when they are well tracked; only drift is worth seeing,
        // as a hairline so that goal posts never outweigh robots.
        let standard_deviation = covariance.symmetric_eigenvalues().max().sqrt();
        if standard_deviation > 0.1 && standard_deviation <= MAXIMUM_DRAWN_STANDARD_DEVIATION {
            painter.covariance(
                ground_to_frame * track.position_in_ground,
                covariance,
                Stroke::new(pixels(painter, 1.0), CHALK.gamma_multiply(0.25)),
                CHALK.gamma_multiply(0.06),
            );
        }
    }
    for track in &objects.tracks {
        let opacity = match track.status {
            TrackStatus::Confirmed => 1.0,
            TrackStatus::Dormant => 0.55,
            TrackStatus::Tentative => 0.25,
        };
        painter.circle(
            ground_to_frame * track.position_in_ground,
            0.06_f32.max(pixels(painter, 5.0)),
            CHALK.gamma_multiply(opacity),
            Stroke::new(pixels(painter, 1.5), INK.gamma_multiply(opacity)),
        );
    }
}

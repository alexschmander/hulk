use std::sync::Arc;

use color_eyre::Result;
use eframe::epaint::{Color32, Stroke};

use coordinate_systems::Ground;
use linear_algebra::Point2;
use ros_z_debug::TopicObservation;
use types::{
    field_dimensions::FieldDimensions, obstacle_filter::Hypothesis, obstacles::ObstacleKind,
    tracked_objects::StationaryClass,
};

use crate::{
    backend::RobotBackend,
    panels::map::{
        layer::Layer,
        layers::object_tracks::{
            Appearance, CHALK, PERSON, Shape, paint_field_feature, paint_marker, paint_uncertainty,
            pixels,
        },
    },
};
use twix_visualization::twix_painter::TwixPainter;

/// The legacy filter has no team identity, so its robots share one neutral color.
const ROBOT: Color32 = Color32::from_rgb(0x9a, 0xa0, 0x92);
/// Mirrors `obstacle_filter.measurement_count_threshold`: hypotheses with more measurements are
/// published as obstacles.
const MEASUREMENT_COUNT_THRESHOLD: usize = 50;

/// Hypotheses of the legacy obstacle filter, drawn like the object tracks for comparison.
///
/// Robots are neutral pucks, people squares, goal posts discs and obstacles of unknown kind small
/// rings. Hypotheses that are published as obstacles are solid; the others are faint.
pub struct ObstacleFilter {
    hypotheses: TopicObservation<Vec<Hypothesis>>,
}

impl Layer<Ground> for ObstacleFilter {
    const NAME: &'static str = "Obstacle Filter";

    fn new(backend: Arc<RobotBackend>) -> Self {
        let _runtime_handle = backend.runtime_handle().enter();

        let hypotheses = backend
            .observer()
            .observe_typed("obstacle_filter_hypotheses")
            .expect("failed to construct obstacle filter hypotheses observer")
            .spawn();

        Self { hypotheses }
    }

    fn paint(
        &self,
        painter: &TwixPainter<Ground>,
        _field_dimensions: &FieldDimensions,
    ) -> Result<()> {
        let Some(record) = self.hypotheses.latest() else {
            return Ok(());
        };
        let hypotheses = &record.value;
        let appearance = |hypothesis: &Hypothesis| {
            if hypothesis.measurement_count > MEASUREMENT_COUNT_THRESHOLD {
                Appearance::Seen
            } else {
                Appearance::Unconfirmed
            }
        };
        for hypothesis in hypotheses {
            if appearance(hypothesis) == Appearance::Seen {
                paint_uncertainty(
                    painter,
                    Point2::from(hypothesis.state.mean),
                    hypothesis.state.covariance,
                    ROBOT,
                );
            }
        }
        for hypothesis in hypotheses {
            let position = Point2::from(hypothesis.state.mean);
            let appearance = appearance(hypothesis);
            match hypothesis.obstacle_kind {
                ObstacleKind::Robot => {
                    paint_marker(painter, position, Shape::Puck, ROBOT, appearance)
                }
                ObstacleKind::Person => {
                    paint_marker(painter, position, Shape::Square, PERSON, appearance)
                }
                ObstacleKind::GoalPost => {
                    let opacity = if appearance == Appearance::Seen {
                        1.0
                    } else {
                        0.25
                    };
                    paint_field_feature(painter, position, StationaryClass::GoalPost, opacity)
                }
                ObstacleKind::Unknown | ObstacleKind::Ball => painter.circle_stroke(
                    position,
                    0.1,
                    Stroke::new(pixels(painter, 1.5), CHALK.gamma_multiply(0.6)),
                ),
            }
        }
        Ok(())
    }
}

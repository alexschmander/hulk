use std::sync::Arc;

use color_eyre::Result;
use eframe::epaint::{Color32, Stroke};

use coordinate_systems::Ground;
use ros_z_debug::{SampleRecord, TopicObservation};
use types::{field_dimensions::FieldDimensions, obstacles::Obstacle};

use crate::{backend::RobotBackend, panels::map::layer::Layer};
use twix_visualization::twix_painter::TwixPainter;

pub struct Obstacles {
    obstacles: TopicObservation<Vec<Obstacle>>,
    humanoid_obstacles: TopicObservation<Vec<Obstacle>>,
    stationary_obstacles: TopicObservation<Vec<Obstacle>>,
}

impl Layer<Ground> for Obstacles {
    const NAME: &'static str = "Obstacles";

    fn new(backend: Arc<RobotBackend>) -> Self {
        let _runtime_handle = backend.runtime_handle().enter();

        let observe = |topic: &str| {
            backend
                .observer()
                .observe_typed(topic)
                .expect("failed to construct obstacles observer")
                .spawn()
        };

        Self {
            obstacles: observe("obstacles"),
            humanoid_obstacles: observe("humanoid_filter/obstacles"),
            stationary_obstacles: observe("stationary_object_filter/obstacles"),
        }
    }

    fn paint(
        &self,
        painter: &TwixPainter<Ground>,
        _field_dimensions: &FieldDimensions,
    ) -> Result<()> {
        for observation in [
            &self.obstacles,
            &self.humanoid_obstacles,
            &self.stationary_obstacles,
        ] {
            let latest = observation.latest();
            let Some(SampleRecord {
                value: obstacles, ..
            }) = latest.as_deref()
            else {
                continue;
            };
            let hip_height_stroke = Stroke {
                width: 0.025,
                color: Color32::RED,
            };
            let foot_height_stroke = Stroke {
                width: 0.025,
                color: Color32::BLUE,
            };
            for obstacle in obstacles {
                painter.circle_stroke(
                    obstacle.position,
                    obstacle.radius_at_hip_height,
                    hip_height_stroke,
                );
                painter.circle_stroke(
                    obstacle.position,
                    obstacle.radius_at_foot_height,
                    foot_height_stroke,
                );
            }
        }
        Ok(())
    }
}

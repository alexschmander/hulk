use std::time::SystemTime;

use crate::{
    factors::visual_reprojection::VisualReprojectionFactor,
    measurements::VisualReprojectionMeasurement,
    symbols::{CameraIntrinsics, State},
};
use factrs::{containers::FactorBuilder, core::Huber, traits::Optimizer};

use super::VinsBackend;

const GLOBAL_VISUAL_HUBER_THRESHOLD: f64 = 2.0;

impl VinsBackend {
    pub(super) fn ingest_visual(&mut self, mut visuals: Vec<Vec<VisualReprojectionMeasurement>>) {
        visuals.retain(|visual| !visual.is_empty());
        let Some(last) = visuals.last() else {
            return;
        };

        let last_time = visual_frame_time(last);
        self.update_last_knot_time(last_time);

        let interval_groups = self.interval_groups(visuals, |visual| visual_frame_time(visual));

        for group in interval_groups {
            if !self.prepare_interval_for_measurements(group.start_index, "visual") {
                continue;
            }
            let latest_group_time = group
                .measurements
                .last()
                .map(|visual| visual_frame_time(visual))
                .expect("visual groups are non-empty");
            self.latest_visual_measurement_time = Some(
                self.latest_visual_measurement_time
                    .map_or(latest_group_time, |current| current.max(latest_group_time)),
            );

            let keys = (
                State(group.start_index),
                State(group.start_index + 1),
                CameraIntrinsics(0),
            );
            let graph = self.optimizer.graph_mut();
            for measurement in group.measurements.into_iter().flatten() {
                let residual = VisualReprojectionFactor::new(
                    group.start_time,
                    group.end_time,
                    [measurement],
                    self.config.visual_feature_noise,
                );
                let factor = FactorBuilder::new(residual, keys)
                    .robust(Huber::new(GLOBAL_VISUAL_HUBER_THRESHOLD))
                    .build();

                graph.add_factor(factor);
            }
        }
    }
}

fn visual_frame_time(visual: &[VisualReprojectionMeasurement]) -> SystemTime {
    visual
        .first()
        .expect("visual frames must contain at least one measurement")
        .time
}

use std::time::SystemTime;

use crate::{
    factors::visual_reprojection::VisualReprojectionFactor,
    measurements::VisualFrameMeasurement,
    symbols::{CameraIntrinsics, LocalToField, State},
};
use factrs::{containers::FactorBuilder, core::Huber, traits::Optimizer};

use super::VinsBackend;

const GLOBAL_VISUAL_HUBER_THRESHOLD: f64 = 2.0;

impl VinsBackend {
    pub(super) fn ingest_visual(&mut self, mut visuals: Vec<VisualFrameMeasurement>) {
        visuals.retain(|visual| !visual.measurements.is_empty());
        let Some(last) = visuals.last() else {
            return;
        };

        let last_time = visual_frame_time(last);
        self.update_last_knot_time(last_time);

        let interval_groups = self.interval_groups(visuals, visual_frame_time);

        for group in interval_groups {
            if !self.prepare_interval_for_measurements(group.start_index, "visual") {
                continue;
            }
            if self.values.get(LocalToField(0)).is_none() {
                self.values.insert(
                    LocalToField(0),
                    group.measurements[0].local_to_field_candidate.clone(),
                );
                let last_state = self
                    .highest_initialized_interval
                    .map_or(0, |index| index + 1);
                for index in 0..=last_state {
                    if self.values.get(State(index)).is_some() {
                        super::graph::add_field_containment_factor(
                            self.optimizer.graph_mut(),
                            State(index),
                            &self.config,
                        );
                    }
                }
            }
            let latest_group_time = group
                .measurements
                .last()
                .map(visual_frame_time)
                .expect("visual groups are non-empty");
            self.latest_visual_measurement_time = Some(
                self.latest_visual_measurement_time
                    .map_or(latest_group_time, |current| current.max(latest_group_time)),
            );

            let keys = (
                State(group.start_index),
                State(group.start_index + 1),
                LocalToField(0),
                CameraIntrinsics(0),
            );
            let graph = self.optimizer.graph_mut();
            for measurement in group
                .measurements
                .into_iter()
                .flat_map(|frame| frame.measurements)
            {
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

fn visual_frame_time(visual: &VisualFrameMeasurement) -> SystemTime {
    visual
        .measurements
        .first()
        .expect("visual frames must contain at least one measurement")
        .time
}

//! Node logic independent of the transport, so that recordings can be replayed through it.

use coordinate_systems::{Field, Ground, Odometry};
use linear_algebra::{Isometry2, Pose2};
use std::time::Duration;

use object_tracking::{multi_bernoulli::ScanReport, odometry_history::OdometryHistory};
use projection::camera_matrix::CameraMatrix;
use ros_z::time::Time;
use types::{
    object_detection::{Object, RobocupObjectLabel},
    obstacles::Obstacle,
    primary_state::PrimaryState,
    time_wrapper::TimeWrapper,
    tracked_objects::{StationaryTrack, TrackedObjects},
};

use crate::{Parameters, filter::StationaryObjectFilter};

/// Inputs the node reads from caches.
pub trait Inputs {
    fn primary_state(&self) -> Option<PrimaryState>;
    fn camera_matrix_nearest(&self, time: Time) -> Option<TimeWrapper<CameraMatrix>>;
}

/// How long odometry poses are kept to pair them with localization estimates of their time.
const ODOMETRY_RETENTION: Duration = Duration::from_secs(2);

pub struct Outputs {
    pub reports: Vec<ScanReport>,
    pub tracks: Option<TrackedObjects<StationaryTrack>>,
    pub obstacles: Vec<Obstacle>,
}

pub struct Processor {
    filter: StationaryObjectFilter,
    odometry_history: OdometryHistory,
    last_primary_state: Option<PrimaryState>,
}

impl Default for Processor {
    fn default() -> Self {
        Self {
            filter: StationaryObjectFilter::default(),
            odometry_history: OdometryHistory::new(ODOMETRY_RETENTION),
            last_primary_state: None,
        }
    }
}

impl Processor {
    pub fn filter(&self) -> &StationaryObjectFilter {
        &self.filter
    }

    /// Records a ground-to-field estimate referring to `time`, the time of its image.
    pub fn receive_ground_to_field(
        &mut self,
        time: Time,
        ground_to_field: Isometry2<Ground, Field>,
    ) {
        self.odometry_history
            .push_ground_to_field(time, ground_to_field);
    }

    /// Processes time-ordered odometry and detections; returns outputs if any image was used.
    pub fn process(
        &mut self,
        items: impl IntoIterator<
            Item = (
                Time,
                (
                    Option<Pose2<Odometry>>,
                    Option<TimeWrapper<Vec<Object<RobocupObjectLabel>>>>,
                ),
            ),
        >,
        inputs: &impl Inputs,
        parameters: &Parameters,
    ) -> Option<Outputs> {
        let filter = &mut self.filter;
        let primary_state = inputs.primary_state();
        if self.last_primary_state == Some(PrimaryState::Penalized)
            && primary_state != Some(PrimaryState::Penalized)
        {
            // The robot was carried; the local map no longer matches its odometry.
            filter.clear();
        }
        self.last_primary_state = primary_state;

        let mut reports = Vec::new();
        let mut last_time = None;
        for (time, (odometry_pose, detected_objects)) in items {
            if let Some(odometry_pose) = odometry_pose {
                self.odometry_history.push_pose(time, odometry_pose);
                filter.predict_to_odometry(time, odometry_pose, parameters);
            }
            if let Some(detected_objects) = detected_objects
                && let Some(camera_matrix) =
                    inputs.camera_matrix_nearest(time).filter(|camera_matrix| {
                        time.abs_diff(camera_matrix.time) <= parameters.maximum_camera_matrix_age
                    })
                && let Some(report) = filter.update_with_detections(
                    time,
                    &detected_objects.inner,
                    &camera_matrix.inner,
                    parameters,
                )
            {
                reports.push(report);
                last_time = Some(time);
            }
        }
        let last_time = last_time?;
        Some(Outputs {
            reports,
            tracks: filter.tracked_objects(
                last_time,
                filter
                    .odometry_pose()
                    .and_then(|pose| self.odometry_history.ground_to_field_at(pose)),
                parameters,
            ),
            obstacles: filter.obstacles(parameters),
        })
    }
}

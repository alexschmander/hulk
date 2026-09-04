use std::collections::VecDeque;

use color_eyre::Result;
use coordinate_systems::{Camera, Robot};
use linear_algebra::Isometry3;
use localization_factrs::{
    OptimizationResult, VinsFrontend, VinsFrontendError, VisualReprojectionAssociation,
    backend::BackendOptimizerStatus,
};
use ros_z::time::Time;
use types::{
    time_wrapper::TimeWrapper,
    visual_localization::{FieldMarkAssociation, VisualLocalizationFrame},
};

const MIN_REPROJECTION_DEPTH: f64 = 0.01;
const MAX_VISUAL_LOCK_REPROJECTION_RMS_PX: f64 = 10.0;
const MIN_CERTIFIED_DETECTION_SEPARATION_PX: f32 = 1.0;
const MIN_CERTIFIED_LANDMARK_SEPARATION_M: f32 = 1.0e-4;
const MAX_PENDING_VISUAL_FRAMES: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlobalVisualLock {
    /// No certified association frame has been accepted.
    Unlocked,
    /// A certified association frame was ingested and awaits a backend result.
    WaitingForBackend,
    /// A backend result established global localization.
    Locked,
}

impl GlobalVisualLock {
    pub(crate) fn has_backend_result(self) -> bool {
        matches!(self, Self::Locked)
    }
}

pub(crate) struct GlobalVisualLockTracker {
    status: GlobalVisualLock,
    pending: VecDeque<PendingVisualFrame>,
}

struct PendingVisualFrame {
    time: Time,
    robot_to_camera: Isometry3<Robot, Camera>,
    associations: Vec<FieldMarkAssociation>,
}

impl Default for GlobalVisualLockTracker {
    fn default() -> Self {
        Self {
            status: GlobalVisualLock::Unlocked,
            pending: VecDeque::new(),
        }
    }
}

impl GlobalVisualLockTracker {
    pub(crate) fn status(&self) -> GlobalVisualLock {
        self.status
    }

    pub(crate) fn has_backend_result(&self) -> bool {
        self.status.has_backend_result()
    }

    pub(crate) fn track_associations(
        &mut self,
        time: Time,
        robot_to_camera: Isometry3<Robot, Camera>,
        associations: &[FieldMarkAssociation],
    ) {
        if !matches!(self.status, GlobalVisualLock::Locked) {
            self.status = GlobalVisualLock::WaitingForBackend;
            if self.pending.len() == MAX_PENDING_VISUAL_FRAMES {
                self.pending.pop_front();
            }
            self.pending.push_back(PendingVisualFrame {
                time,
                robot_to_camera,
                associations: associations.to_vec(),
            });
        }
    }

    pub(crate) fn handle_backend_result(&mut self, result: &OptimizationResult) -> bool {
        if !matches!(result.optimizer_status, BackendOptimizerStatus::Converged) {
            return false;
        }
        if self.status.has_backend_result() {
            return true;
        }
        let Some(acknowledged_time) = result.latest_visual_measurement_time else {
            return false;
        };
        let Some(pending) = self
            .pending
            .iter()
            .find(|pending| pending.time.to_wallclock() == acknowledged_time)
        else {
            return false;
        };
        let reprojection_rms = pending_reprojection_rms(pending, result);
        if reprojection_rms.is_none_or(|rms| rms > MAX_VISUAL_LOCK_REPROJECTION_RMS_PX) {
            return false;
        }

        self.status = GlobalVisualLock::Locked;
        self.pending.clear();
        true
    }

    pub(crate) fn reset_for_damping(&mut self) {
        self.status = GlobalVisualLock::Unlocked;
        self.pending.clear();
    }
}

fn pending_reprojection_rms(
    pending: &PendingVisualFrame,
    result: &OptimizationResult,
) -> Option<f64> {
    let robot_to_field = result.latest_visual_robot_to_field.as_ref()?;
    let field_to_camera =
        pending.robot_to_camera.inner.cast::<f64>() * robot_to_field.inner.inverse();
    let total_squared_error = pending
        .associations
        .iter()
        .map(|association| {
            let point_camera = field_to_camera * association.field_point.inner.cast::<f64>();
            let projected = result
                .camera_intrinsics
                .project_checked(point_camera.coords.as_view(), MIN_REPROJECTION_DEPTH)?;
            let observed = association.detection.inner.coords.cast::<f64>();
            let error = (projected - observed).norm_squared();
            error.is_finite().then_some(error)
        })
        .sum::<Option<f64>>()?;
    Some((total_squared_error / pending.associations.len() as f64).sqrt())
}

pub(crate) fn handle_visual_localization_frame(
    frontend: &mut VinsFrontend,
    global_visual_lock: &mut GlobalVisualLockTracker,
    epoch: u64,
    frame: TimeWrapper<VisualLocalizationFrame>,
) -> Result<(), VinsFrontendError> {
    let TimeWrapper { time, inner } = frame;
    let VisualLocalizationFrame {
        epoch: frame_epoch,
        robot_to_camera,
        local_to_field,
        associations,
    } = inner;
    if frame_epoch != epoch {
        return Ok(());
    }
    if associations.len() < types::visual_localization::MIN_CERTIFIED_VISUAL_ASSOCIATIONS
        || associations.len() > types::visual_localization::MAX_CERTIFIED_VISUAL_ASSOCIATIONS
    {
        return Ok(());
    }
    let valid_extrinsic = robot_to_camera
        .inner
        .to_homogeneous()
        .iter()
        .all(|value| value.is_finite());
    let valid_associations = associations.iter().all(|association| {
        association
            .detection
            .inner
            .iter()
            .chain(association.field_point.inner.iter())
            .all(|value| value.is_finite())
    });
    let distinct_associations = associations.iter().enumerate().all(|(index, association)| {
        associations[index + 1..].iter().all(|other| {
            (association.detection - other.detection).inner.norm()
                > MIN_CERTIFIED_DETECTION_SEPARATION_PX
                && (association.field_point - other.field_point).inner.norm()
                    > MIN_CERTIFIED_LANDMARK_SEPARATION_M
        })
    });

    if !valid_extrinsic || !valid_associations || !distinct_associations {
        return Ok(());
    }

    ingest_visual_localization_associations(
        frontend,
        time,
        robot_to_camera,
        local_to_field,
        associations.iter().cloned(),
    )?;
    global_visual_lock.track_associations(time, robot_to_camera, &associations);
    Ok(())
}

fn ingest_visual_localization_associations(
    frontend: &mut VinsFrontend,
    time: Time,
    robot_to_camera: Isometry3<Robot, Camera>,
    local_to_field: linear_algebra::Isometry2<coordinate_systems::Local, coordinate_systems::Field>,
    associations: impl IntoIterator<Item = FieldMarkAssociation>,
) -> Result<(), VinsFrontendError> {
    let associations = associations
        .into_iter()
        .map(|association| VisualReprojectionAssociation {
            detection: association.detection,
            field_point: association.field_point,
        });
    frontend.ingest_visual_reprojection_associations(
        time.to_wallclock(),
        associations,
        robot_to_camera,
        local_to_field,
    )
}

#[cfg(test)]
mod tests {
    use coordinate_systems::Field;
    use linear_algebra::IntoTransform;
    use localization_factrs::CameraIntrinsics;

    use super::*;

    fn result(time: Time, acknowledged_visual: Option<Time>) -> OptimizationResult {
        OptimizationResult {
            time: time.to_wallclock(),
            generation: 0,
            robot_to_local: nalgebra::Isometry3::identity().framed_transform(),
            local_to_field: Some(nalgebra::Isometry2::identity().framed_transform()),
            robot_to_field: Some(nalgebra::Isometry3::identity().framed_transform()),
            robot_to_field_covariance: Some(nalgebra::SMatrix::identity()),
            velocity: nalgebra::Vector3::zeros(),
            camera_intrinsics: CameraIntrinsics::new(
                nalgebra::vector![200.0, 200.0],
                nalgebra::vector![320.0, 240.0],
            ),
            latest_visual_measurement_time: acknowledged_visual.map(Time::to_wallclock),
            latest_visual_robot_to_local: acknowledged_visual
                .map(|_| nalgebra::Isometry3::identity().framed_transform()),
            latest_visual_robot_to_field: acknowledged_visual
                .map(|_| nalgebra::Isometry3::identity().framed_transform()),
            optimizer_status: BackendOptimizerStatus::Converged,
        }
    }

    fn associations() -> Vec<FieldMarkAssociation> {
        [
            ([320.0, 240.0], [0.0, 0.0, 1.0]),
            ([520.0, 240.0], [1.0, 0.0, 1.0]),
            ([320.0, 440.0], [0.0, 1.0, 1.0]),
        ]
        .into_iter()
        .map(|(pixel, field_point)| FieldMarkAssociation {
            detection: linear_algebra::point![<coordinate_systems::Pixel>, pixel[0], pixel[1]],
            field_point: linear_algebra::point![<Field>,
                field_point[0], field_point[1], field_point[2]
            ],
        })
        .collect()
    }

    fn waiting_tracker(time: Time) -> GlobalVisualLockTracker {
        let mut tracker = GlobalVisualLockTracker::default();
        tracker.track_associations(time, Isometry3::identity(), &associations());
        tracker
    }

    #[test]
    fn result_that_acknowledges_pending_associations_locks() {
        let time = Time::from_nanos(1_000_000_000);
        let mut tracker = waiting_tracker(time);

        assert!(tracker.handle_backend_result(&result(time, Some(time))));
        assert_eq!(tracker.status(), GlobalVisualLock::Locked);
    }

    #[test]
    fn unacknowledged_backend_result_does_not_consume_pending_associations() {
        let pending_time = Time::from_nanos(2_000_000_000);
        let mut tracker = waiting_tracker(pending_time);

        assert!(!tracker.handle_backend_result(&result(pending_time, None)));
        assert_eq!(tracker.status(), GlobalVisualLock::WaitingForBackend);
        assert!(tracker.handle_backend_result(&result(pending_time, Some(pending_time))));
    }

    #[test]
    fn stale_visual_acknowledgement_remains_waiting() {
        let pending_time = Time::from_nanos(3_000_000_000);
        let mut tracker = waiting_tracker(pending_time);

        assert!(
            !tracker.handle_backend_result(&result(
                pending_time,
                Some(Time::from_nanos(2_000_000_000)),
            ))
        );
        assert_eq!(tracker.status(), GlobalVisualLock::WaitingForBackend);
    }

    #[test]
    fn in_flight_frame_can_lock_after_a_newer_frame_arrives() {
        let pending_time = Time::from_nanos(4_000_000_000);
        let mut tracker = waiting_tracker(pending_time);
        tracker.track_associations(
            Time::from_nanos(5_000_000_000),
            Isometry3::identity(),
            &associations(),
        );

        assert!(
            tracker.handle_backend_result(&result(
                Time::from_nanos(5_000_000_000),
                Some(pending_time),
            ))
        );
    }

    #[test]
    fn failed_optimizer_result_does_not_lock() {
        let time = Time::from_nanos(6_000_000_000);
        let mut tracker = waiting_tracker(time);
        let mut failed = result(time, Some(time));
        failed.optimizer_status = BackendOptimizerStatus::FailedToStep;

        assert!(!tracker.handle_backend_result(&failed));
        assert_eq!(tracker.status(), GlobalVisualLock::WaitingForBackend);
    }

    #[test]
    fn max_iterations_with_large_reprojection_error_does_not_lock() {
        let time = Time::from_nanos(7_000_000_000);
        let mut tracker = GlobalVisualLockTracker::default();
        let mut associations = associations();
        associations[0].detection.inner.x += (MAX_VISUAL_LOCK_REPROJECTION_RMS_PX as f32) * 3.0;
        tracker.track_associations(time, Isometry3::identity(), &associations);
        let mut result = result(time, Some(time));
        result.optimizer_status = BackendOptimizerStatus::MaxIterations;

        assert!(!tracker.handle_backend_result(&result));
        assert_eq!(tracker.status(), GlobalVisualLock::WaitingForBackend);
    }

    #[test]
    fn max_iterations_with_valid_reprojection_does_not_lock() {
        let time = Time::from_nanos(7_250_000_000);
        let mut tracker = waiting_tracker(time);
        let mut result = result(time, Some(time));
        result.optimizer_status = BackendOptimizerStatus::MaxIterations;

        assert!(!tracker.handle_backend_result(&result));
        assert_eq!(tracker.status(), GlobalVisualLock::WaitingForBackend);
    }

    #[test]
    fn invalid_reprojection_depth_does_not_lock() {
        let time = Time::from_nanos(7_500_000_000);
        let mut tracker = GlobalVisualLockTracker::default();
        let mut associations = associations();
        for association in &mut associations {
            association.field_point.inner.z = -1.0;
        }
        tracker.track_associations(time, Isometry3::identity(), &associations);

        assert!(!tracker.handle_backend_result(&result(time, Some(time))));
        assert_eq!(tracker.status(), GlobalVisualLock::WaitingForBackend);
    }

    #[test]
    fn failed_result_is_not_accepted_after_lock() {
        let time = Time::from_nanos(8_000_000_000);
        let mut tracker = waiting_tracker(time);
        assert!(tracker.handle_backend_result(&result(time, Some(time))));
        let mut failed = result(time, Some(time));
        failed.optimizer_status = BackendOptimizerStatus::InvalidSystem;

        assert!(!tracker.handle_backend_result(&failed));
        assert_eq!(tracker.status(), GlobalVisualLock::Locked);
    }
}

use std::collections::VecDeque;

use color_eyre::Result;
use coordinate_systems::{Camera, Field, Local, Robot};
use linear_algebra::{IntoTransform, Isometry2, Isometry3};
use localization_factrs::{
    MIN_REPROJECTION_DEPTH, OptimizationResult, VinsFrontend, VinsFrontendError,
    backend::BackendOptimizerStatus,
};
use ros_z::time::Time;
use types::{
    localization::LocalizationState3D,
    time_wrapper::TimeWrapper,
    visual_localization::{FieldMarkAssociation, VisualLocalizationFrame},
};

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

#[derive(Default)]
pub(crate) struct GlobalVisualLockTracker {
    pending: VecDeque<PendingVisualFrame>,
    reject_through: Option<Time>,
    last_frame_time: Option<Time>,
    last_accepted_time: Option<Time>,
}

struct PendingVisualFrame {
    time: Time,
    robot_to_camera: Isometry3<Robot, Camera>,
    associations: Vec<FieldMarkAssociation>,
}

impl GlobalVisualLockTracker {
    pub(crate) fn status(&self) -> GlobalVisualLock {
        if self.has_backend_result() {
            GlobalVisualLock::Locked
        } else if self.last_frame_time.is_some() {
            GlobalVisualLock::WaitingForBackend
        } else {
            GlobalVisualLock::Unlocked
        }
    }

    pub(crate) fn has_backend_result(&self) -> bool {
        self.last_accepted_time.is_some()
    }

    pub(crate) fn last_accepted_time(&self) -> Option<Time> {
        self.last_accepted_time
    }

    pub(crate) fn accepts_frame(&self, time: Time, state: LocalizationState3D) -> bool {
        let last_solve = match state {
            LocalizationState3D::Startup | LocalizationState3D::Tracking { .. } => None,
            LocalizationState3D::LostTrack {
                last_successful_solve,
                ..
            } => Some(last_successful_solve),
        };
        ![self.reject_through, self.last_frame_time, last_solve]
            .into_iter()
            .flatten()
            .any(|boundary| time <= boundary)
    }

    /// Invalidates in-flight associations as well as the lock at a lifecycle boundary.
    pub(crate) fn invalidate(&mut self, time: Time) {
        self.reset_for_damping();
        self.reject_new_frames_through(time);
    }

    /// Reject late arrivals from the previous lifecycle, not acknowledgements already in flight.
    pub(crate) fn reject_new_frames_through(&mut self, time: Time) {
        self.reject_through = Some(self.reject_through.map_or(time, |old| old.max(time)));
    }

    pub(crate) fn track_associations(
        &mut self,
        time: Time,
        robot_to_camera: Isometry3<Robot, Camera>,
        associations: Vec<FieldMarkAssociation>,
    ) {
        if self.reject_through.is_some_and(|boundary| time <= boundary)
            || self
                .last_frame_time
                .is_some_and(|previous| time <= previous)
        {
            return;
        }
        self.last_frame_time = Some(time);
        if self.pending.len() == MAX_PENDING_VISUAL_FRAMES {
            let oldest_unaccepted = usize::from(
                self.pending
                    .front()
                    .is_some_and(|frame| Some(frame.time) == self.last_accepted_time),
            );
            self.pending.remove(oldest_unaccepted);
        }
        self.pending.push_back(PendingVisualFrame {
            time,
            robot_to_camera,
            associations,
        });
    }

    pub(crate) fn handle_backend_result(&mut self, result: &OptimizationResult) -> bool {
        if !matches!(result.optimizer_status, BackendOptimizerStatus::Converged) {
            return false;
        }
        let Some(acknowledged_time) = result.latest_visual_measurement_time else {
            return self.has_backend_result();
        };
        let time = Time::from_wallclock(acknowledged_time);
        if acknowledged_time > result.time {
            return false;
        }
        if self
            .last_accepted_time
            .is_some_and(|previous| time < previous)
        {
            return false;
        }
        let Some(pending) = self
            .pending
            .iter()
            .find(|pending| pending.time.to_wallclock() == acknowledged_time)
        else {
            return false;
        };
        let reprojection_rms = pending_reprojection_rms(pending, result);
        if reprojection_rms
            .is_none_or(|rms| !rms.is_finite() || rms > MAX_VISUAL_LOCK_REPROJECTION_RMS_PX)
        {
            return false;
        }

        self.last_accepted_time = Some(time);
        self.pending.retain(|pending| pending.time >= time);
        true
    }

    pub(crate) fn reset_for_damping(&mut self) {
        self.pending.clear();
        self.last_frame_time = None;
        self.last_accepted_time = None;
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
    state: LocalizationState3D,
    now: Time,
    timeout: std::time::Duration,
    frame: TimeWrapper<VisualLocalizationFrame>,
) -> Result<(), VinsFrontendError> {
    let TimeWrapper { time, inner } = frame;
    if time > now
        || time.saturating_add(timeout) <= now
        || inner.epoch != epoch
        || !global_visual_lock.accepts_frame(time, state)
        || !valid_frame(&inner)
    {
        return Ok(());
    }
    let VisualLocalizationFrame {
        epoch: _,
        robot_to_camera,
        robot_to_local,
        camera_intrinsic,
        mut associations,
    } = inner;
    let local_to_field_candidate = if matches!(state, LocalizationState3D::Startup) {
        let Some(alignment) = seed_alignment(
            robot_to_local,
            robot_to_camera,
            camera_intrinsic,
            &mut associations,
        ) else {
            return Ok(());
        };
        Some(alignment)
    } else {
        None
    };
    frontend.ingest_visual_reprojection_associations(
        time.to_wallclock(),
        associations.iter().copied(),
        robot_to_camera,
        local_to_field_candidate,
    )?;
    global_visual_lock.track_associations(time, robot_to_camera, associations);
    Ok(())
}

fn valid_frame(frame: &VisualLocalizationFrame) -> bool {
    let associations = &frame.associations;
    (types::visual_localization::MIN_CERTIFIED_VISUAL_ASSOCIATIONS
        ..=types::visual_localization::MAX_CERTIFIED_VISUAL_ASSOCIATIONS)
        .contains(&associations.len())
        && frame
            .robot_to_camera
            .inner
            .to_homogeneous()
            .iter()
            .all(|v| v.is_finite())
        && associations.iter().all(|association| {
            association
                .detection
                .inner
                .iter()
                .chain(association.field_point.inner.iter())
                .all(|v| v.is_finite())
        })
        && associations.iter().enumerate().all(|(index, association)| {
            associations[index + 1..].iter().all(|other| {
                (association.detection - other.detection).inner.norm()
                    > MIN_CERTIFIED_DETECTION_SEPARATION_PX
                    && (association.field_point - other.field_point).inner.norm()
                        > MIN_CERTIFIED_LANDMARK_SEPARATION_M
            })
        })
}

/// Startup alone chooses the own-half branch and mirrors its correspondences together.
pub(crate) fn seed_alignment(
    robot_to_local: Isometry3<Robot, Local>,
    robot_to_camera: Isometry3<Robot, Camera>,
    intrinsic: projection::intrinsic::Intrinsic,
    associations: &mut [FieldMarkAssociation],
) -> Option<Isometry2<Local, Field>> {
    let mut alignment = fit_alignment(robot_to_local, robot_to_camera, intrinsic, associations)?;
    let robot_xy = robot_to_local.inner.translation.vector.xy().cast::<f64>();
    if (alignment * nalgebra::Point2::from(robot_xy)).x > 0.0 {
        alignment =
            nalgebra::Isometry2::new(nalgebra::Vector2::zeros(), std::f64::consts::PI) * alignment;
        for association in associations {
            association.field_point.inner.x = -association.field_point.inner.x;
            association.field_point.inner.y = -association.field_point.inner.y;
        }
    }
    let alignment = alignment.cast::<f32>();
    alignment
        .to_homogeneous()
        .iter()
        .all(|v| v.is_finite())
        .then(|| alignment.framed_transform())
}

/// Closed-form planar rigid fit of camera rays intersected with Local z=0.
fn fit_alignment(
    robot_to_local: Isometry3<Robot, Local>,
    robot_to_camera: Isometry3<Robot, Camera>,
    intrinsic: projection::intrinsic::Intrinsic,
    associations: &[FieldMarkAssociation],
) -> Option<nalgebra::Isometry2<f64>> {
    if associations.len() < types::visual_localization::MIN_CERTIFIED_VISUAL_ASSOCIATIONS
        || associations.len() > types::visual_localization::MAX_CERTIFIED_VISUAL_ASSOCIATIONS
        || !intrinsic.is_valid()
    {
        return None;
    }
    let camera_to_local = (robot_to_local * robot_to_camera.inverse())
        .inner
        .cast::<f64>();
    if !camera_to_local
        .to_homogeneous()
        .iter()
        .all(|v| v.is_finite())
    {
        return None;
    }
    let origin = camera_to_local.translation.vector;
    let mut projected = [nalgebra::Vector2::<f64>::zeros();
        types::visual_localization::MAX_CERTIFIED_VISUAL_ASSOCIATIONS];
    let mut local_mean = nalgebra::Vector2::<f64>::zeros();
    let mut field_mean = nalgebra::Vector2::<f64>::zeros();
    for (association, local) in associations.iter().zip(&mut projected) {
        let ray =
            camera_to_local.rotation * intrinsic.bearing(association.detection).inner.cast::<f64>();
        let distance = -origin.z / ray.z;
        if !distance.is_finite() || distance <= MIN_REPROJECTION_DEPTH || ray.z.abs() < 1.0e-6 {
            return None;
        }
        *local = (origin + ray * distance).xy();
        local_mean += *local;
        field_mean += association.field_point.inner.coords.xy().cast::<f64>();
    }
    local_mean /= associations.len() as f64;
    field_mean /= associations.len() as f64;
    let mut dot = 0.0;
    let mut cross = 0.0;
    for (association, local) in associations.iter().zip(projected) {
        let local = local - local_mean;
        let field = association.field_point.inner.coords.xy().cast::<f64>() - field_mean;
        dot += local.dot(&field);
        cross += local.x * field.y - local.y * field.x;
    }
    if !dot.is_finite() || !cross.is_finite() || dot.hypot(cross) < 1.0e-8 {
        return None;
    }
    let rotation = nalgebra::UnitComplex::new(cross.atan2(dot));
    Some(nalgebra::Isometry2::from_parts(
        nalgebra::Translation2::from(field_mean - rotation * local_mean),
        rotation,
    ))
}

#[cfg(test)]
mod tests {
    use coordinate_systems::Field;
    use linear_algebra::IntoTransform;

    use super::*;

    fn result(time: Time, acknowledged_visual: Option<Time>) -> OptimizationResult {
        OptimizationResult {
            latest_visual_measurement_time: acknowledged_visual.map(Time::to_wallclock),
            latest_visual_robot_to_local: acknowledged_visual
                .map(|_| nalgebra::Isometry3::identity().framed_transform()),
            latest_visual_robot_to_field: acknowledged_visual
                .map(|_| nalgebra::Isometry3::identity().framed_transform()),
            ..crate::test_result(time)
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
        tracker.track_associations(time, Isometry3::identity(), associations());
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
            associations(),
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
        tracker.track_associations(time, Isometry3::identity(), associations);
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
        tracker.track_associations(time, Isometry3::identity(), associations);

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

    #[test]
    fn loss_requires_a_new_post_loss_acknowledgement_and_valid_reprojection() {
        let old_time = Time::from_nanos(10);
        let loss_time = Time::from_nanos(20);
        let new_time = Time::from_nanos(30);
        let mut tracker = waiting_tracker(old_time);
        assert!(tracker.handle_backend_result(&result(old_time, Some(old_time))));
        tracker.invalidate(loss_time);
        assert_eq!(tracker.status(), GlobalVisualLock::Unlocked);
        assert!(!tracker.accepts_frame(loss_time, LocalizationState3D::Startup));
        tracker.track_associations(old_time, Isometry3::identity(), associations());
        assert!(!tracker.handle_backend_result(&result(new_time, Some(old_time))));
        tracker.track_associations(new_time, Isometry3::identity(), associations());
        assert!(!tracker.handle_backend_result(&result(new_time, Some(old_time))));
        let mut bad = result(new_time, Some(new_time));
        bad.latest_visual_robot_to_field
            .as_mut()
            .unwrap()
            .inner
            .translation
            .vector
            .x = 5.0;
        assert!(!tracker.handle_backend_result(&bad));
        assert_eq!(tracker.status(), GlobalVisualLock::WaitingForBackend);
        assert!(tracker.handle_backend_result(&result(new_time, Some(new_time))));
        assert_eq!(tracker.last_accepted_time(), Some(new_time));
    }

    #[test]
    fn repeated_acknowledgements_do_not_refresh_visual_freshness() {
        let time = Time::from_nanos(10);
        let mut tracker = waiting_tracker(time);
        assert!(tracker.handle_backend_result(&result(time, Some(time))));
        let later = Time::from_nanos(20);
        assert!(tracker.handle_backend_result(&result(later, Some(time))));
        assert_eq!(tracker.last_accepted_time(), Some(time));
        tracker.track_associations(later, Isometry3::identity(), associations());
        assert!(tracker.handle_backend_result(&result(later, Some(later))));
        assert_eq!(tracker.last_accepted_time(), Some(later));
        let mut drifted = result(Time::from_nanos(30), Some(later));
        drifted
            .latest_visual_robot_to_field
            .as_mut()
            .unwrap()
            .inner
            .translation
            .vector
            .x = 5.0;
        assert!(!tracker.handle_backend_result(&drifted));
        assert_eq!(tracker.last_accepted_time(), Some(later));
    }

    #[test]
    fn acquisition_rejects_new_old_state_frames_but_retains_in_flight_acknowledgements() {
        let time = Time::from_nanos(10);
        let mut tracker = waiting_tracker(time);
        let second = Time::from_nanos(15);
        tracker.track_associations(second, Isometry3::identity(), associations());
        assert!(tracker.handle_backend_result(&result(time, Some(time))));
        tracker.reject_new_frames_through(Time::from_nanos(20));
        assert!(!tracker.accepts_frame(Time::from_nanos(15), LocalizationState3D::Startup));
        assert!(tracker.handle_backend_result(&result(Time::from_nanos(30), Some(time))));
        assert!(tracker.handle_backend_result(&result(Time::from_nanos(30), Some(second))));
        assert_eq!(tracker.last_accepted_time(), Some(second));
        assert!(tracker.accepts_frame(Time::from_nanos(30), LocalizationState3D::Startup));
        tracker.invalidate(Time::from_nanos(40));
        assert!(!tracker.handle_backend_result(&result(Time::from_nanos(50), Some(second))));
    }

    #[test]
    fn read_only_fit_projects_onto_local_plane_and_startup_seed_mirrors_matches() {
        let robot_to_local: Isometry3<Robot, Local> = nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(1.5, -0.7, 0.48),
            nalgebra::UnitQuaternion::from_euler_angles(0.12, -0.18, 0.3),
        )
        .framed_transform();
        let robot_to_camera: Isometry3<Robot, Camera> = nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.1, 0.02, 0.12),
            nalgebra::UnitQuaternion::from_euler_angles(std::f32::consts::PI, 0.0, 0.0),
        )
        .inverse()
        .framed_transform();
        let intrinsic = projection::intrinsic::Intrinsic::new(
            nalgebra::vector![220.0, 230.0],
            linear_algebra::point![320.0, 240.0],
        );
        let expected = nalgebra::Isometry2::new(nalgebra::vector![2.0, 1.0], 0.4);
        let local_to_camera = robot_to_camera * robot_to_local.inverse();
        let original: Vec<_> = [[1.0, -1.0], [2.0, -1.0], [1.5, 0.0]]
            .into_iter()
            .map(|[x, y]| {
                let local = linear_algebra::point![<Local>, x, y, 0.0];
                let camera = local_to_camera * local;
                assert!(camera.z() > 0.0);
                let field = expected * nalgebra::point![x, y];
                FieldMarkAssociation {
                    detection: intrinsic
                        .project(linear_algebra::Vector3::wrap(camera.inner.coords)),
                    field_point: linear_algebra::point![<Field>, field.x, field.y, 0.0],
                }
            })
            .collect();
        for startup in [false, true] {
            let mut matches = original.clone();
            let actual = if startup {
                seed_alignment(robot_to_local, robot_to_camera, intrinsic, &mut matches)
                    .unwrap()
                    .inner
            } else {
                fit_alignment(robot_to_local, robot_to_camera, intrinsic, &matches)
                    .unwrap()
                    .cast::<f32>()
            };
            let expected = if startup {
                nalgebra::Isometry2::new(nalgebra::Vector2::zeros(), std::f32::consts::PI)
                    * expected
            } else {
                expected
            };
            assert!((actual.to_homogeneous() - expected.to_homogeneous()).norm() < 1.0e-5);
            for (actual, original) in matches.iter().zip(&original) {
                assert_eq!(actual.detection, original.detection);
                let sign = if startup { -1.0 } else { 1.0 };
                assert_eq!(
                    actual.field_point.inner.coords,
                    original.field_point.inner.coords * sign
                );
            }
        }
    }

    #[test]
    fn tracking_and_lost_frames_accept_horizon_correspondences_without_a_bootstrap_seed() {
        use crate::parameters::{
            Localization3dParameters, backend_configuration_from_parameters_and_field_dimensions,
        };
        use std::time::Duration;

        let time = Time::from_nanos(10);
        let robot_to_local = nalgebra::Isometry3::translation(0.0, 0.0, 0.5).framed_transform();
        let robot_to_camera: Isometry3<Robot, Camera> =
            nalgebra::Isometry3::rotation(nalgebra::vector![
                0.0,
                -std::f32::consts::FRAC_PI_2,
                0.0
            ])
            .framed_transform();
        let camera_intrinsic = projection::intrinsic::Intrinsic::new(
            nalgebra::vector![200.0, 200.0],
            linear_algebra::point![320.0, 240.0],
        );
        let associations: Vec<_> = [-0.5, 0.0, 0.5]
            .into_iter()
            .map(|y| FieldMarkAssociation {
                detection: linear_algebra::point![320.0, 240.0 + 200.0 * y],
                field_point: linear_algebra::point![1.0, y, 0.5],
            })
            .collect();
        assert!(
            seed_alignment(
                robot_to_local,
                robot_to_camera,
                camera_intrinsic,
                &mut associations.clone(),
            )
            .is_none()
        );
        let estimate = types::localization::LocalizationEstimate3D {
            robot_to_field: robot_to_local.inner.framed_transform(),
            covariance: nalgebra::SMatrix::identity(),
        };
        for state in [
            LocalizationState3D::Startup,
            LocalizationState3D::Tracking {
                estimate,
                last_successful_solve: Time::from_nanos(1),
            },
            LocalizationState3D::LostTrack {
                last_known_estimate: estimate,
                last_successful_solve: Time::from_nanos(1),
            },
        ] {
            let config = backend_configuration_from_parameters_and_field_dimensions(
                &Localization3dParameters {
                    accelerometer_process_noise_variance: 10.0,
                    visual_feature_noise_variance: 1.0,
                    field_containment_sigma: 0.1,
                    tracking_timeout: Duration::from_secs(2),
                    visual_tracking_timeout: Duration::from_secs(2),
                },
                &types::field_dimensions::FieldDimensions::SPL_2025,
            );
            let (mut frontend, mut backend) = localization_factrs::initialize(
                config,
                localization_factrs::InitialState::default(),
            );
            let mut tracker = GlobalVisualLockTracker::default();
            handle_visual_localization_frame(
                &mut frontend,
                &mut tracker,
                0,
                state,
                time,
                Duration::from_secs(2),
                TimeWrapper {
                    time,
                    inner: VisualLocalizationFrame {
                        epoch: 0,
                        robot_to_camera,
                        robot_to_local,
                        camera_intrinsic,
                        associations: associations.clone(),
                    },
                },
            )
            .unwrap();
            assert_eq!(
                tracker.status(),
                if matches!(state, LocalizationState3D::Startup) {
                    GlobalVisualLock::Unlocked
                } else {
                    GlobalVisualLock::WaitingForBackend
                }
            );
            // Tracking correspondences are forwarded, but must never initialize alignment.
            assert!(backend.solve_once().unwrap().is_none());
            assert!(
                backend
                    .values()
                    .get(localization_factrs::LocalToField(0))
                    .is_none()
            );
            if !matches!(state, LocalizationState3D::Startup) {
                let pending = tracker.pending.front().unwrap();
                for (sent, original) in pending.associations.iter().zip(&associations) {
                    assert_eq!(sent.detection, original.detection);
                    assert_eq!(sent.field_point, original.field_point);
                }
            }
        }
    }

    #[test]
    fn seed_rejects_horizon_behind_camera_and_invalid_intrinsics() {
        let pose = nalgebra::Isometry3::translation(0.0, 0.0, 0.5).framed_transform();
        let mut matches = associations();
        let intrinsic = projection::intrinsic::Intrinsic::default();
        assert!(seed_alignment(pose, Isometry3::identity(), intrinsic, &mut matches,).is_none());
        let invalid = projection::intrinsic::Intrinsic {
            focals: nalgebra::vector![0.0, 1.0],
            ..intrinsic
        };
        assert!(seed_alignment(pose, Isometry3::identity(), invalid, &mut matches,).is_none());
    }
}

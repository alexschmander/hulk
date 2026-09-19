use coordinate_systems::{Camera, Field, Local, Robot};
use linear_algebra::{IntoTransform, Isometry2, Isometry3};
use projection::camera_matrix::CameraMatrix;
use ros_z::{cache::Cache, time::Time};
use types::{
    localization::LocalizationState3D, time_wrapper::TimeWrapper,
    visual_localization::AssociationGeometry, visual_odometry::VisualOdometer,
};

use crate::camera::{fresh_camera_matrix, robot_to_camera};

pub(crate) type VisualOdometerCache = Cache<VisualOdometer>;
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LocalPose {
    pub robot_to_local: Isometry3<Robot, Local>,
    pub local_to_field: Option<Isometry2<Local, Field>>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PoseCorrection {
    pub time: Time,
    pub robot_to_local: Isometry3<Robot, Local, f64>,
    pub local_to_field: Option<Isometry2<Local, Field, f64>>,
}

impl PoseCorrection {
    fn pose(self) -> TimeWrapper<LocalPose> {
        TimeWrapper {
            time: self.time,
            inner: LocalPose {
                robot_to_local: self.robot_to_local.inner.cast().framed_transform(),
                local_to_field: self
                    .local_to_field
                    .map(|pose| pose.inner.cast().framed_transform()),
            },
        }
    }
}

#[derive(Default)]
pub(crate) struct LiveVisualOdometryLocalization {
    anchor: Option<LiveVisualOdometryAnchor>,
    pending_correction: Option<PoseCorrection>,
    latest: Option<TimeWrapper<LocalPose>>,
    latest_sample: Option<(VisualOdometer, Isometry3<Robot, Camera>)>,
}

struct LiveVisualOdometryAnchor {
    time: Time,
    odometer_epoch: u64,
    robot_to_local: nalgebra::Isometry3<f64>,
    local_to_field: Option<Isometry2<Local, Field>>,
    left_camera_to_visual_odometer: nalgebra::Isometry3<f32>,
    robot_to_camera: nalgebra::Isometry3<f32>,
}

impl LiveVisualOdometryLocalization {
    pub(crate) fn clear(&mut self) {
        self.anchor = None;
        self.pending_correction = None;
        self.latest = None;
        self.latest_sample = None;
    }

    pub(crate) fn set_initial(&mut self, time: Time, robot_to_local: Isometry3<Robot, Local>) {
        self.clear();
        self.latest = Some(TimeWrapper {
            time,
            inner: LocalPose {
                robot_to_local,
                local_to_field: None,
            },
        });
    }

    pub(crate) fn latest(&self) -> Option<LocalPose> {
        self.latest.as_ref().map(|pose| pose.inner)
    }

    pub(crate) fn latest_time(&self) -> Option<Time> {
        self.latest.as_ref().map(|pose| pose.time)
    }

    /// Until exact anchoring succeeds, only the accepted backend geometry is authoritative.
    pub(crate) fn output(&self) -> Option<TimeWrapper<LocalPose>> {
        self.pending_correction
            .map(PoseCorrection::pose)
            .or_else(|| self.latest.clone())
    }

    pub(crate) fn association_geometry(
        &self,
        state: LocalizationState3D,
        epoch: u64,
    ) -> Option<TimeWrapper<AssociationGeometry>> {
        let pose = self.output()?;
        Some(TimeWrapper {
            time: pose.time,
            inner: AssociationGeometry {
                epoch,
                state,
                robot_to_local: pose.inner.robot_to_local,
                local_to_field: pose.inner.local_to_field,
            },
        })
    }

    pub(crate) fn odometer_epoch_changed(&self, current: &VisualOdometer) -> bool {
        self.latest_sample.as_ref().is_some_and(|(previous, _)| {
            current.time >= previous.time && current.epoch != previous.epoch
        })
    }

    pub(crate) fn defer_reset(&mut self, correction: PoseCorrection) {
        self.pending_correction = Some(correction);
        if let Some((sample, camera)) = self.latest_sample.clone()
            && sample.time == correction.time
        {
            self.reset_with_extrinsic(correction, &sample, camera);
        }
    }

    pub(crate) fn reset(
        &mut self,
        correction: PoseCorrection,
        visual_odometer_cache: &VisualOdometerCache,
        camera_matrix_cache: &Cache<TimeWrapper<CameraMatrix>>,
    ) {
        self.defer_reset(correction);
        self.try_reset_pending(visual_odometer_cache, camera_matrix_cache);
    }

    pub(crate) fn reset_with_exact_samples(
        &mut self,
        correction: PoseCorrection,
        visual_odometer: &VisualOdometer,
        camera_matrix: &CameraMatrix,
    ) -> bool {
        self.reset_with_extrinsic(correction, visual_odometer, robot_to_camera(camera_matrix))
    }

    fn reset_with_extrinsic(
        &mut self,
        correction: PoseCorrection,
        visual_odometer: &VisualOdometer,
        robot_to_camera: Isometry3<Robot, Camera>,
    ) -> bool {
        let time = correction.time;
        if visual_odometer.time != time {
            return false;
        }
        // Epoch changes must first be observed by ingestion, which also invalidates tracking.
        if self
            .latest_sample
            .as_ref()
            .is_some_and(|(sample, _)| sample.epoch != visual_odometer.epoch)
        {
            return false;
        }
        self.latest = Some(correction.pose());
        self.anchor = Some(LiveVisualOdometryAnchor {
            time,
            odometer_epoch: visual_odometer.epoch,
            robot_to_local: correction.robot_to_local.inner,
            local_to_field: self.latest().and_then(|pose| pose.local_to_field),
            left_camera_to_visual_odometer: visual_odometer.current_left_camera_to_visual_odometer,
            robot_to_camera: robot_to_camera.inner,
        });
        self.pending_correction = None;
        if let Some((latest, camera)) = self.latest_sample.clone()
            && latest.time >= time
        {
            self.update_with_extrinsic(&latest, camera);
        } else {
            self.latest_sample = Some((visual_odometer.clone(), robot_to_camera));
        }
        true
    }

    pub(crate) fn try_reset_pending(
        &mut self,
        visual_odometer_cache: &VisualOdometerCache,
        camera_matrix_cache: &Cache<TimeWrapper<CameraMatrix>>,
    ) {
        let Some(correction) = self.pending_correction else {
            return;
        };
        let time = correction.time;
        let Some(odometer) = odometer_at(visual_odometer_cache, time) else {
            return;
        };
        let Some(camera) = fresh_camera_matrix(camera_matrix_cache, time) else {
            return;
        };
        self.reset_with_exact_samples(correction, &odometer, &camera.inner);
    }

    pub(crate) fn update_from_odometer(
        &mut self,
        current: &VisualOdometer,
        camera_matrix_cache: &Cache<TimeWrapper<CameraMatrix>>,
    ) -> Option<LocalPose> {
        let camera = fresh_camera_matrix(camera_matrix_cache, current.time)?;
        self.update_with_exact_sample(current, &camera.inner)
    }

    pub(crate) fn update_with_exact_sample(
        &mut self,
        current: &VisualOdometer,
        camera: &CameraMatrix,
    ) -> Option<LocalPose> {
        if let Some(correction) = self.pending_correction
            && correction.time == current.time
            && self.reset_with_exact_samples(correction, current, camera)
        {
            return self.latest();
        }
        self.update_with_extrinsic(current, robot_to_camera(camera))
    }

    fn update_with_extrinsic(
        &mut self,
        current: &VisualOdometer,
        robot_to_camera: Isometry3<Robot, Camera>,
    ) -> Option<LocalPose> {
        if self.latest_time().is_some_and(|time| current.time < time) {
            return None;
        }
        if !current
            .current_left_camera_to_visual_odometer
            .to_homogeneous()
            .iter()
            .chain(robot_to_camera.inner.to_homogeneous().iter())
            .all(|v| v.is_finite())
        {
            return None;
        }
        if self.odometer_epoch_changed(current)
            && let Some(correction) = self
                .pending_correction
                .take_if(|correction| correction.time <= current.time)
        {
            // Retire the unreachable checkpoint, preserving its accepted pose/alignment.
            // The new anchor below uses only subsequent motion in the new VO epoch.
            self.latest = Some(correction.pose());
        }
        if self
            .anchor
            .as_ref()
            .is_none_or(|anchor| current.epoch != anchor.odometer_epoch)
        {
            let LocalPose {
                robot_to_local,
                local_to_field,
            } = self.latest()?;
            self.anchor = Some(LiveVisualOdometryAnchor {
                time: current.time,
                odometer_epoch: current.epoch,
                robot_to_local: robot_to_local.inner.cast(),
                local_to_field,
                left_camera_to_visual_odometer: current.current_left_camera_to_visual_odometer,
                robot_to_camera: robot_to_camera.inner,
            });
        }
        let pose = self.anchor.as_ref()?.propagate(current, robot_to_camera)?;
        self.latest = Some(TimeWrapper {
            time: current.time,
            inner: pose,
        });
        self.latest_sample = Some((current.clone(), robot_to_camera));
        Some(pose)
    }
}

impl LiveVisualOdometryAnchor {
    fn propagate(
        &self,
        current: &VisualOdometer,
        robot_to_camera: Isometry3<Robot, Camera>,
    ) -> Option<LocalPose> {
        if current.time < self.time {
            return None;
        }
        if current.time == self.time {
            return Some(LocalPose {
                robot_to_local: self.robot_to_local.cast().framed_transform(),
                local_to_field: self.local_to_field,
            });
        }
        let current_robot_to_camera = robot_to_camera.inner;
        let current_camera_to_anchor_camera = self.left_camera_to_visual_odometer.inverse()
            * current.current_left_camera_to_visual_odometer;
        let current_robot_to_anchor_robot = self.robot_to_camera.inverse()
            * current_camera_to_anchor_camera
            * current_robot_to_camera;
        let pose = LocalPose {
            robot_to_local: (self.robot_to_local * current_robot_to_anchor_robot.cast())
                .cast()
                .framed_transform(),
            local_to_field: self.local_to_field,
        };
        pose.robot_to_local
            .inner
            .to_homogeneous()
            .iter()
            .all(|v| v.is_finite())
            .then_some(pose)
    }
}

fn odometer_at(cache: &VisualOdometerCache, time: Time) -> Option<VisualOdometer> {
    if let Some((stamp, exact)) = cache.get_nearest_with_stamp(time)
        && stamp == time
    {
        return Some(exact.as_ref().clone());
    }
    let before = cache.get_before(time)?;
    let after = cache.get_after(time)?;
    if before.epoch != after.epoch || after.time <= before.time {
        return None;
    }
    let fraction = (time.duration_since(before.time).as_secs_f64()
        / after.time.duration_since(before.time).as_secs_f64())
    .clamp(0.0, 1.0) as f32;
    Some(VisualOdometer {
        time,
        epoch: before.epoch,
        current_left_camera_to_visual_odometer: before
            .current_left_camera_to_visual_odometer
            .lerp_slerp(&after.current_left_camera_to_visual_odometer, fraction),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn odometer(time: Time) -> VisualOdometer {
        VisualOdometer {
            time,
            epoch: 3,
            current_left_camera_to_visual_odometer: nalgebra::Isometry3::identity(),
        }
    }

    #[test]
    fn samples_before_anchor_are_rejected_but_equal_sample_returns_anchor() {
        let anchor_time = Time::from_nanos(2);
        let mut live = LiveVisualOdometryLocalization {
            anchor: Some(LiveVisualOdometryAnchor {
                time: anchor_time,
                odometer_epoch: 3,
                robot_to_local: nalgebra::Isometry3::identity(),
                local_to_field: None,
                left_camera_to_visual_odometer: nalgebra::Isometry3::identity(),
                robot_to_camera: nalgebra::Isometry3::identity(),
            }),
            ..Default::default()
        };

        assert!(
            live.update_with_exact_sample(&odometer(Time::from_nanos(1)), &CameraMatrix::default())
                .is_none()
        );
        assert!(
            live.update_with_exact_sample(&odometer(anchor_time), &CameraMatrix::default())
                .is_some()
        );
    }

    #[test]
    fn delayed_solve_replays_latest_motion_but_cannot_cross_a_vo_reset() {
        let time = Time::from_nanos(10);
        let camera = CameraMatrix::default();
        let mut live = LiveVisualOdometryLocalization::default();
        live.set_initial(
            time,
            nalgebra::Isometry3::translation(0.0, 0.0, 0.42).framed_transform(),
        );
        live.update_with_exact_sample(&odometer(time), &camera)
            .unwrap();
        let mut moved = odometer(Time::from_nanos(30));
        moved
            .current_left_camera_to_visual_odometer
            .translation
            .vector
            .x = 0.5;
        live.update_with_exact_sample(&moved, &camera).unwrap();
        let mut correction = PoseCorrection {
            time,
            robot_to_local: nalgebra::Isometry3::translation(1.0, 0.0, 0.45).framed_transform(),
            local_to_field: Some(nalgebra::Isometry2::identity().framed_transform()),
        };
        assert!(live.reset_with_exact_samples(correction, &odometer(time), &camera));
        assert_eq!(live.latest_time(), Some(moved.time));
        let corrected = live.latest().unwrap();
        assert!((corrected.robot_to_local.translation().x() - 1.5).abs() < 1.0e-6);
        assert!((corrected.robot_to_local.translation().z() - 0.45).abs() < 1.0e-6);
        assert!(
            live.update_with_exact_sample(&odometer(Time::from_nanos(20)), &camera)
                .is_none()
        );

        moved.time = Time::from_nanos(40);
        moved.epoch += 1;
        moved.current_left_camera_to_visual_odometer = nalgebra::Isometry3::identity();
        assert_eq!(
            live.update_with_exact_sample(&moved, &camera),
            Some(corrected)
        );
        correction.robot_to_local.inner.translation.vector.x = 100.0;
        assert!(!live.reset_with_exact_samples(correction, &odometer(time), &camera));
        assert_eq!(live.latest(), Some(corrected));
        moved.time = Time::from_nanos(50);
        moved
            .current_left_camera_to_visual_odometer
            .translation
            .vector
            .x = 0.2;
        assert!(
            (live
                .update_with_exact_sample(&moved, &camera)
                .unwrap()
                .robot_to_local
                .translation()
                .x()
                - 1.7)
                .abs()
                < 1.0e-6
        );
    }

    #[test]
    fn live_pose_keeps_propagating_with_trusted_alignment_after_loss() {
        use types::localization::{LocalizationEstimate3D, LocalizationState3D};
        let time = Time::from_nanos(10);
        let initial = nalgebra::Isometry3::translation(0.0, 0.0, 0.42).framed_transform();
        let alignment =
            nalgebra::Isometry2::new(nalgebra::vector![-2.0, 1.0], 0.3).framed_transform();
        let mut live = LiveVisualOdometryLocalization::default();
        live.set_initial(time, initial);
        live.latest = Some(TimeWrapper {
            time,
            inner: LocalPose {
                robot_to_local: initial,
                local_to_field: Some(alignment),
            },
        });
        live.update_with_exact_sample(&odometer(time), &CameraMatrix::default())
            .unwrap();
        let state = LocalizationState3D::Tracking {
            estimate: LocalizationEstimate3D {
                robot_to_field: crate::pose::compose_robot_to_field(initial, alignment),
                covariance: nalgebra::SMatrix::identity(),
            },
            last_successful_solve: time,
        };
        let lost = crate::event_handlers::lose_track(
            state,
            &mut crate::visual_localization::GlobalVisualLockTracker::default(),
            Time::from_nanos(20),
        );
        let mut moved = odometer(Time::from_nanos(30));
        moved
            .current_left_camera_to_visual_odometer
            .translation
            .vector
            .x = 0.5;
        let pose = live
            .update_with_exact_sample(&moved, &CameraMatrix::default())
            .unwrap();
        assert_eq!(pose.local_to_field, Some(alignment));
        assert!((pose.robot_to_local.translation().x() - 0.5).abs() < 1.0e-6);
        assert!((pose.robot_to_local.translation().z() - 0.42).abs() < 1.0e-6);
        assert!(
            matches!(lost, LocalizationState3D::LostTrack { last_successful_solve, .. } if last_successful_solve == time)
        );
        assert!(
            live.update_with_exact_sample(
                &odometer(Time::from_nanos(25)),
                &CameraMatrix::default()
            )
            .is_none()
        );
        assert_eq!(live.latest(), Some(pose));
        moved.epoch += 1;
        moved.time = Time::from_nanos(40);
        moved.current_left_camera_to_visual_odometer = nalgebra::Isometry3::identity();
        assert_eq!(
            live.update_with_exact_sample(&moved, &CameraMatrix::default()),
            Some(pose)
        );
    }
}

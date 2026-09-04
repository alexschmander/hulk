use coordinate_systems::{Field, Local, Robot};
use linear_algebra::{IntoTransform, Isometry2, Isometry3};
use localization_factrs::OptimizationResult;
use projection::camera_matrix::CameraMatrix;
use ros_z::{cache::Cache, time::Time};
use types::{time_wrapper::TimeWrapper, visual_odometry::VisualOdometer};

use crate::camera::{fresh_camera_matrix, robot_to_camera};

pub(crate) type VisualOdometerCache = Cache<VisualOdometer>;
pub(crate) type LocalPose = (Isometry3<Robot, Local>, Option<Isometry2<Local, Field>>);

#[derive(Default)]
pub(crate) struct LiveVisualOdometryLocalization {
    anchor: Option<LiveVisualOdometryAnchor>,
    pending_result: Option<OptimizationResult>,
    latest: Option<LocalPose>,
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
        self.pending_result = None;
        self.latest = None;
    }

    pub(crate) fn set_initial(&mut self, robot_to_local: Isometry3<Robot, Local>) {
        self.clear();
        self.latest = Some((robot_to_local, None));
    }

    pub(crate) fn latest(&self) -> Option<LocalPose> {
        self.latest
    }

    pub(crate) fn reset(
        &mut self,
        result: &OptimizationResult,
        visual_odometer_cache: &VisualOdometerCache,
        camera_matrix_cache: &Cache<TimeWrapper<CameraMatrix>>,
    ) {
        self.latest = Some(local_pose_from_result(result));
        self.pending_result = Some(result.clone());
        self.anchor = None;
        self.try_reset_pending(visual_odometer_cache, camera_matrix_cache);
    }

    pub(crate) fn reset_with_exact_samples(
        &mut self,
        result: &OptimizationResult,
        visual_odometer: &VisualOdometer,
        camera_matrix: &CameraMatrix,
    ) -> bool {
        let time = Time::from_wallclock(result.time);
        if visual_odometer.time != time {
            return false;
        }
        self.latest = Some(local_pose_from_result(result));
        self.anchor = Some(anchor_from_exact_samples(
            result,
            visual_odometer,
            camera_matrix,
        ));
        self.pending_result = None;
        true
    }

    pub(crate) fn try_reset_pending(
        &mut self,
        visual_odometer_cache: &VisualOdometerCache,
        camera_matrix_cache: &Cache<TimeWrapper<CameraMatrix>>,
    ) {
        let Some(result) = self.pending_result.as_ref() else {
            return;
        };
        let time = Time::from_wallclock(result.time);
        let Some(odometer) = odometer_at(visual_odometer_cache, time) else {
            return;
        };
        let Some(camera) = fresh_camera_matrix(camera_matrix_cache, time) else {
            return;
        };
        self.anchor = Some(anchor_from_exact_samples(result, &odometer, &camera.inner));
        self.pending_result = None;
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
        if current.epoch != self.anchor.as_ref()?.odometer_epoch {
            self.anchor = None;
            return None;
        }
        let anchor = self.anchor.as_ref()?;
        if current.time < anchor.time {
            return None;
        }
        if current.time == anchor.time {
            let pose = (
                anchor.robot_to_local.cast().framed_transform(),
                anchor.local_to_field,
            );
            self.latest = Some(pose);
            return Some(pose);
        }
        let current_robot_to_camera = robot_to_camera(camera).inner;
        let current_camera_to_anchor_camera = anchor.left_camera_to_visual_odometer.inverse()
            * current.current_left_camera_to_visual_odometer;
        let current_robot_to_anchor_robot = anchor.robot_to_camera.inverse()
            * current_camera_to_anchor_camera
            * current_robot_to_camera;
        let pose = (
            (anchor.robot_to_local * current_robot_to_anchor_robot.cast())
                .cast()
                .framed_transform(),
            anchor.local_to_field,
        );
        self.latest = Some(pose);
        Some(pose)
    }
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
            pending_result: None,
            latest: None,
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
}

fn anchor_from_exact_samples(
    result: &OptimizationResult,
    odometer: &VisualOdometer,
    camera: &CameraMatrix,
) -> LiveVisualOdometryAnchor {
    LiveVisualOdometryAnchor {
        time: Time::from_wallclock(result.time),
        odometer_epoch: odometer.epoch,
        robot_to_local: result.robot_to_local.inner,
        local_to_field: result
            .local_to_field
            .map(|pose| pose.inner.cast().framed_transform()),
        left_camera_to_visual_odometer: odometer.current_left_camera_to_visual_odometer,
        robot_to_camera: robot_to_camera(camera).inner,
    }
}

fn local_pose_from_result(result: &OptimizationResult) -> LocalPose {
    (
        result.robot_to_local.inner.cast().framed_transform(),
        result
            .local_to_field
            .map(|pose| pose.inner.cast().framed_transform()),
    )
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
        current_left_camera_to_visual_odometer: nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::from(
                before
                    .current_left_camera_to_visual_odometer
                    .translation
                    .vector
                    * (1.0 - fraction)
                    + after
                        .current_left_camera_to_visual_odometer
                        .translation
                        .vector
                        * fraction,
            ),
            before
                .current_left_camera_to_visual_odometer
                .rotation
                .slerp(
                    &after.current_left_camera_to_visual_odometer.rotation,
                    fraction,
                ),
        ),
    })
}

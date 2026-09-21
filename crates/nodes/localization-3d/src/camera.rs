use std::{sync::Arc, time::Duration};

use coordinate_systems::{Camera, Robot};
use linear_algebra::Isometry3;
use projection::camera_matrix::CameraMatrix;
use ros_z::{cache::Cache, time::Time};
use types::time_wrapper::TimeWrapper;

pub(crate) fn fresh_camera_matrix(
    cameras: &Cache<TimeWrapper<CameraMatrix>>,
    time: Time,
) -> Option<Arc<TimeWrapper<CameraMatrix>>> {
    cameras
        .get_nearest(time)
        .filter(|camera| camera.time.abs_diff(time) <= Duration::from_millis(100))
}

pub(crate) fn robot_to_camera(camera: &CameraMatrix) -> Isometry3<Robot, Camera> {
    camera.head_to_camera * camera.robot_to_head
}

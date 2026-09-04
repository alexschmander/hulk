use booster::ImuState;
use coordinate_systems::{Local, Robot};
use factrs::{core::SO3, traits::Variable, variables::SE23};
use linear_algebra::Isometry3;
use nalgebra::{Vector2, Vector3, vector};

use crate::camera_intrinsics::CameraIntrinsics;
use crate::conversions::robot_to_local_to_se23;

#[derive(Clone, Debug)]
pub struct InitialState {
    pub robot_to_local: SE23<f64>,
    pub camera_intrinsics: CameraIntrinsics<f64>,
}

impl InitialState {
    pub fn new(robot_to_local: SE23<f64>, camera_intrinsics: CameraIntrinsics<f64>) -> Self {
        Self {
            robot_to_local,
            camera_intrinsics,
        }
    }

    pub fn from_robot_to_local_and_intrinsics_components(
        robot_to_local: SE23<f64>,
        focal_lengths: Vector2<f64>,
        optical_center: Vector2<f64>,
    ) -> Self {
        Self::new(
            robot_to_local,
            CameraIntrinsics::new(focal_lengths, optical_center),
        )
    }

    pub fn from_initial_height_and_intrinsics(
        initial_height: f64,
        camera_intrinsics: CameraIntrinsics<f64>,
    ) -> Self {
        let robot_to_local = SE23::from_rot_vel_trans(
            SO3::identity(),
            Vector3::zeros(),
            Vector3::new(0.0, 0.0, initial_height),
        );

        Self::new(robot_to_local, camera_intrinsics)
    }

    pub fn from_robot_to_local_and_intrinsics(
        robot_to_local: Isometry3<Robot, Local>,
        camera_intrinsics: CameraIntrinsics<f64>,
    ) -> Self {
        Self::new(robot_to_local_to_se23(robot_to_local), camera_intrinsics)
    }

    pub fn with_imu_orientation(mut self, imu: &ImuState) -> Self {
        let rpy = imu.roll_pitch_yaw.inner.cast::<f64>();
        let rotation = nalgebra::UnitQuaternion::from_euler_angles(rpy.x, rpy.y, 0.0);
        let quaternion = rotation.quaternion();
        self.robot_to_local = SE23::from_rot_vel_trans(
            SO3::from_xyzw(quaternion.i, quaternion.j, quaternion.k, quaternion.w),
            self.robot_to_local.uvw().into_owned(),
            self.robot_to_local.xyz().into_owned(),
        );
        self
    }
}

impl Default for InitialState {
    fn default() -> Self {
        Self {
            robot_to_local: SE23::identity(),
            // Current default callers do not use visual factors yet. Use a valid
            // normalized pinhole calibration instead of a zeroed variable state.
            camera_intrinsics: CameraIntrinsics::new(vector![1.0, 1.0], vector![0.0, 0.0]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_uses_non_degenerate_camera_intrinsics() {
        let initial_state = InitialState::default();

        assert_eq!(initial_state.camera_intrinsics.focals(), vector![1.0, 1.0]);
        assert_eq!(
            initial_state.camera_intrinsics.optical_center(),
            vector![0.0, 0.0]
        );
    }

    #[test]
    fn initial_height_constructor_sets_only_translation_z() {
        let initial_state = InitialState::from_initial_height_and_intrinsics(
            0.52,
            CameraIntrinsics::new(vector![1.0, 1.0], vector![0.0, 0.0]),
        );

        assert_eq!(initial_state.robot_to_local.xyz(), vector![0.0, 0.0, 0.52]);
        assert!(initial_state.robot_to_local.uvw().norm() < 1.0e-9);
    }

    #[test]
    fn robot_to_local_constructor_preserves_pose_and_intrinsics() {
        let rotation = linear_algebra::Orientation3::from_euler_angles(0.1, -0.2, 0.3);
        let initial_pose =
            Isometry3::from_parts(linear_algebra::vector![<Local>, 1.0, 2.0, 0.4], rotation);
        let camera_intrinsics = CameraIntrinsics::new(vector![200.0, 210.0], vector![250.0, 240.0]);

        let initial_state =
            InitialState::from_robot_to_local_and_intrinsics(initial_pose, camera_intrinsics);

        assert!((initial_state.robot_to_local.xyz() - vector![1.0, 2.0, 0.4]).norm() < 1.0e-6);
        assert!((initial_state.robot_to_local.uvw() - vector![0.0, 0.0, 0.0]).norm() < 1.0e-9);
        assert_eq!(
            initial_state.camera_intrinsics.focals(),
            vector![200.0, 210.0]
        );
        assert_eq!(
            initial_state.camera_intrinsics.optical_center(),
            vector![250.0, 240.0]
        );
        assert!((initial_state.robot_to_local.rot().w() - rotation.inner.w as f64).abs() < 1.0e-9);
        assert!((initial_state.robot_to_local.rot().x() - rotation.inner.i as f64).abs() < 1.0e-9);
        assert!((initial_state.robot_to_local.rot().y() - rotation.inner.j as f64).abs() < 1.0e-9);
        assert!((initial_state.robot_to_local.rot().z() - rotation.inner.k as f64).abs() < 1.0e-9);
    }

    #[test]
    fn imu_orientation_preserves_translation_and_velocity() {
        use linear_algebra::IntoFramed;

        let initial_state = InitialState::new(
            SE23::from_rot_vel_trans(
                SO3::identity(),
                vector![1.0, 2.0, 3.0],
                vector![4.0, 5.0, 6.0],
            ),
            CameraIntrinsics::new(vector![1.0, 1.0], vector![0.0, 0.0]),
        );
        let imu = ImuState {
            roll_pitch_yaw: nalgebra::vector![0.1, -0.2, 0.3].framed(),
            ..Default::default()
        };

        let oriented = initial_state.with_imu_orientation(&imu);

        assert_eq!(oriented.robot_to_local.uvw(), vector![1.0, 2.0, 3.0]);
        assert_eq!(oriented.robot_to_local.xyz(), vector![4.0, 5.0, 6.0]);
        assert!(oriented.robot_to_local.rot().log().norm() > 0.1);
    }
}

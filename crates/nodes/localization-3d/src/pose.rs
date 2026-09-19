use booster::ImuState;
use coordinate_systems::{Field, Local, Robot};
use kinematics::robot_kinematics::RobotKinematics;
use linear_algebra::{IntoTransform, Isometry3, Orientation3};
use localization_factrs::{InitialState, OptimizationResult};
use projection::camera_matrix::CameraMatrix;

use crate::camera::camera_intrinsics_from_matrix;

/// Constructs a Local-frame backend state from the first camera and IMU samples.
pub fn initial_state_from_camera_matrix_and_imu(
    camera_matrix: &CameraMatrix,
    imu: &ImuState,
    kinematics: &RobotKinematics,
) -> InitialState {
    InitialState::from_initial_height_and_intrinsics(
        initial_robot_to_local_from_imu(imu, kinematics)
            .translation()
            .z() as f64,
        camera_intrinsics_from_matrix(camera_matrix),
    )
    .with_imu_orientation(imu)
}

pub(crate) fn initial_robot_to_local_from_imu(
    imu: &ImuState,
    kinematics: &RobotKinematics,
) -> Isometry3<Robot, Local> {
    let rpy = imu.roll_pitch_yaw.inner;
    let orientation = Orientation3::from_euler_angles(rpy.x, rpy.y, 0.0);
    let height = support_height(kinematics, orientation.inner);
    Isometry3::from_parts(
        linear_algebra::vector![<Local>, 0.0, 0.0, height],
        orientation,
    )
}

pub(crate) fn support_height(
    kinematics: &RobotKinematics,
    robot_to_local_rotation: nalgebra::UnitQuaternion<f32>,
) -> f32 {
    let left = robot_to_local_rotation * kinematics.left_leg.sole_to_robot.inner.translation.vector;
    let right =
        robot_to_local_rotation * kinematics.right_leg.sole_to_robot.inner.translation.vector;
    -left.z.min(right.z)
}

pub(crate) fn backend_localization_for_result(
    result: &OptimizationResult,
) -> Option<Isometry3<Field, Robot>> {
    let robot_to_field = result.robot_to_field.as_ref()?;
    Some(localization_transform_from_backend_pose(
        &robot_to_field.inner,
    ))
}

pub(crate) fn compose_robot_to_field(
    robot_to_local: Isometry3<Robot, Local>,
    local_to_field: linear_algebra::Isometry2<Local, Field>,
) -> Isometry3<Robot, Field> {
    local_to_field.to_3d() * robot_to_local
}

pub(crate) fn localization_transform_from_backend_pose(
    robot_to_field: &nalgebra::Isometry3<f64>,
) -> Isometry3<Field, Robot> {
    robot_to_field.cast::<f32>().inverse().framed_transform()
}

#[cfg(test)]
mod tests {
    use linear_algebra::point;
    use projection::intrinsic::Intrinsic;

    use super::*;

    #[test]
    fn initial_state_uses_imu_roll_pitch_zero_yaw_height_and_live_intrinsics() {
        let camera_matrix = CameraMatrix {
            intrinsics: Intrinsic::new(nalgebra::vector![216.0, 217.0], point![251.0, 235.0]),
            ..Default::default()
        };

        let imu = ImuState {
            roll_pitch_yaw: linear_algebra::vector![<Robot>, 0.1, -0.2, 1.2],
            ..Default::default()
        };
        let mut kinematics = RobotKinematics::default();
        kinematics.left_leg.sole_to_robot.inner.translation.vector.z = -0.42;
        kinematics
            .right_leg
            .sole_to_robot
            .inner
            .translation
            .vector
            .z = -0.35;
        let expected_height = 0.42_f64 * 0.1_f64.cos() * 0.2_f64.cos();
        let initial_state =
            initial_state_from_camera_matrix_and_imu(&camera_matrix, &imu, &kinematics);

        assert_eq!(
            initial_state.robot_to_local.xyz().xy(),
            nalgebra::vector![0.0, 0.0]
        );
        assert!((initial_state.robot_to_local.xyz().z - expected_height).abs() < 1.0e-6);
        assert!(initial_state.robot_to_local.uvw().norm() < 1.0e-9);
        assert_eq!(
            initial_state.camera_intrinsics.focals(),
            nalgebra::vector![216.0, 217.0]
        );
        assert_eq!(
            initial_state.camera_intrinsics.optical_center(),
            nalgebra::vector![251.0, 235.0]
        );
        let rotation = initial_state.robot_to_local.rot();
        let yaw = (2.0 * (rotation.w() * rotation.z() + rotation.x() * rotation.y()))
            .atan2(1.0 - 2.0 * (rotation.y().powi(2) + rotation.z().powi(2)));
        assert!(yaw.abs() < 1.0e-6);
    }

    #[test]
    fn localization_publisher_outputs_field_to_robot() {
        let robot_to_field = nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(-3.0, 0.25, 0.4),
            nalgebra::UnitQuaternion::from_euler_angles(0.2, -0.3, 0.3),
        );

        let field_to_robot = localization_transform_from_backend_pose(&robot_to_field);
        let roundtrip_robot_to_field = field_to_robot.inverse().inner.cast::<f64>();

        assert!(
            (roundtrip_robot_to_field.translation.vector - robot_to_field.translation.vector)
                .norm()
                < 1.0e-6
        );
        assert!(
            roundtrip_robot_to_field
                .rotation
                .angle_to(&robot_to_field.rotation)
                < 1.0e-6
        );
    }

    #[test]
    fn planar_alignment_preserves_local_height_roll_and_pitch() {
        let robot_to_field: Isometry3<Robot, Field> = nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(4.0, -3.0, 0.6),
            nalgebra::UnitQuaternion::from_euler_angles(0.2, -0.3, 0.7),
        )
        .framed_transform();
        let robot_to_local = robot_to_field.inner.framed_transform();
        let alignment =
            nalgebra::Isometry2::new(nalgebra::vector![1.0, 2.0], 0.5).framed_transform();
        let composed = compose_robot_to_field(robot_to_local, alignment);
        let (roll, pitch, yaw) = composed.inner.rotation.euler_angles();
        assert!((composed.translation().z() - 0.6).abs() < 1.0e-6);
        assert!((roll - 0.2).abs() < 1.0e-6);
        assert!((pitch + 0.3).abs() < 1.0e-6);
        assert!((yaw - 1.2).abs() < 1.0e-6);
    }
}

use std::time::SystemTime;

use factrs::{
    core::SE3,
    linalg::{ForwardProp, Numeric, VectorX},
    traits::{Residual, Variable},
    variables::{MatrixLieGroup, SE2, SE23},
};
use nalgebra::Matrix2;

use crate::{
    SE23Spline,
    camera_intrinsics::CameraIntrinsics,
    measurements::VisualReprojectionMeasurement,
    utils::{interval_dt, tau},
};

pub const MIN_REPROJECTION_DEPTH: f64 = 0.01;

#[derive(Debug, Clone)]
pub struct VisualReprojectionFactor {
    measurement: VisualReprojectionMeasurement,
    measurement_tau: f64,
    robot_to_camera: SE3,
    pixel_information_root: Matrix2<f64>,
    duration: f64,
}

#[factrs::mark]
impl Residual for VisualReprojectionFactor {
    type Input = (SE23, SE23, SE2, CameraIntrinsics);
    type Differ = ForwardProp;

    fn dim_out(&self) -> usize {
        2
    }

    fn residual<T: Numeric>(
        &self,
        (start, end, local_to_field, camera_intrinsics): (
            SE23<T>,
            SE23<T>,
            SE2<T>,
            CameraIntrinsics<T>,
        ),
    ) -> VectorX<T> {
        let spline = SE23Spline::new(&start, &end, T::from(self.duration));
        let robot_to_local = spline.evaluate(T::from(self.measurement_tau));
        let field_to_local = local_to_field.inverse();
        let robot_to_camera = self.robot_to_camera.cast::<T>();
        let field_point = self.measurement.field_point.coords.cast::<T>();
        let field_point_xy = field_point.fixed_rows::<2>(0).into_owned();
        let point_local_xy = field_to_local.apply(field_point_xy.as_view());
        let point_local = nalgebra::Vector3::new(point_local_xy.x, point_local_xy.y, field_point.z);
        let point_robot = robot_to_local.inverse().apply(point_local.as_view());
        let point_camera = robot_to_camera.apply(point_robot.as_view());
        let Some(projected) = camera_intrinsics
            .project_checked(point_camera.as_view(), T::from(MIN_REPROJECTION_DEPTH))
        else {
            return VectorX::zeros(2);
        };
        let reprojection = projected - self.measurement.detection.coords.cast::<T>();
        let whitened = self.pixel_information_root.cast::<T>() * reprojection;
        VectorX::from_column_slice(whitened.as_slice())
    }
}

impl VisualReprojectionFactor {
    pub fn new(
        start_time: SystemTime,
        end_time: SystemTime,
        time: SystemTime,
        robot_to_camera: &SE3,
        measurement: VisualReprojectionMeasurement,
        visual_feature_noise: Matrix2<f64>,
    ) -> Self {
        let pixel_information_root = visual_feature_noise
            .cholesky()
            .expect("visual feature covariance must be positive definite")
            .l()
            .try_inverse()
            .expect("visual feature covariance Cholesky factor must be invertible");
        let duration = interval_dt::<f64>(start_time, end_time);
        Self {
            measurement,
            measurement_tau: tau(start_time, end_time, time),
            robot_to_camera: robot_to_camera.clone(),
            pixel_information_root,
            duration,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime};

    use factrs::{
        core::{SE3, SO3, Vector3},
        linalg::VectorX,
        traits::{Residual, Variable},
        variables::{SE2, SE23},
    };
    use nalgebra::{Matrix2, Point2, Point3, vector};

    use super::*;

    fn state(position: Vector3) -> SE23 {
        SE23::from_rot_vel_trans(SO3::identity(), Vector3::zeros(), position)
    }

    #[test]
    fn residual_is_zero_for_exact_reprojection() {
        let time = SystemTime::UNIX_EPOCH;
        let factor = VisualReprojectionFactor::new(
            time,
            time + Duration::from_secs(1),
            time,
            &SE3::identity(),
            VisualReprojectionMeasurement {
                detection: Point2::new(0.0, 0.0),
                field_point: Point3::new(0.0, 0.0, 2.0),
            },
            Matrix2::identity(),
        );
        let intrinsics = CameraIntrinsics::new(vector![100.0, 100.0], vector![0.0, 0.0]);

        let residual = factor.residual((
            state(Vector3::zeros()),
            state(Vector3::zeros()),
            SE2::identity(),
            intrinsics,
        ));

        assert!(residual.norm() < 1.0e-9);
    }

    #[test]
    fn residual_changes_with_pose_error() {
        let time = SystemTime::UNIX_EPOCH;
        let factor = VisualReprojectionFactor::new(
            time,
            time + Duration::from_secs(1),
            time,
            &SE3::identity(),
            VisualReprojectionMeasurement {
                detection: Point2::new(0.0, 0.0),
                field_point: Point3::new(0.0, 0.0, 2.0),
            },
            Matrix2::identity(),
        );
        let intrinsics = CameraIntrinsics::new(vector![100.0, 100.0], vector![0.0, 0.0]);

        let residual = factor.residual((
            state(vector![0.1, 0.0, 0.0]),
            state(vector![0.1, 0.0, 0.0]),
            SE2::identity(),
            intrinsics,
        ));

        assert!(residual.norm() > 1.0);
    }

    #[test]
    fn reprojection_composes_local_alignment_with_robot_pose() {
        let time = SystemTime::UNIX_EPOCH;
        let factor = VisualReprojectionFactor::new(
            time,
            time + Duration::from_secs(1),
            time,
            &SE3::identity(),
            VisualReprojectionMeasurement {
                detection: Point2::new(0.0, 0.0),
                field_point: Point3::new(1.0, 0.0, 2.0),
            },
            Matrix2::identity(),
        );
        let intrinsics = CameraIntrinsics::new(vector![100.0, 100.0], vector![0.0, 0.0]);

        let residual = factor.residual((
            state(Vector3::zeros()),
            state(Vector3::zeros()),
            SE2::new(0.0, 1.0, 0.0),
            intrinsics,
        ));

        assert!(residual.norm() < 1.0e-9);
    }

    #[test]
    fn invalid_depth_reprojection_is_ignored() {
        let time = SystemTime::UNIX_EPOCH;
        for depth in [-1.0, 1.0e-4] {
            let factor = VisualReprojectionFactor::new(
                time,
                time + Duration::from_secs(1),
                time,
                &SE3::identity(),
                VisualReprojectionMeasurement {
                    detection: Point2::new(10.0, 20.0),
                    field_point: Point3::new(0.0, 0.0, depth),
                },
                Matrix2::identity(),
            );
            let intrinsics = CameraIntrinsics::new(vector![100.0, 100.0], vector![0.0, 0.0]);
            let pose = state(Vector3::zeros());

            let linearized =
                factor.residual_jacobian((pose.clone(), pose, SE2::identity(), intrinsics));

            assert!(linearized.value.iter().all(|value| value.is_finite()));
            assert!(linearized.diff.iter().all(|value| value.is_finite()));
            assert!(linearized.value.norm() < 1.0e-12);
            assert!(linearized.diff.norm() < 1.0e-12);
        }
    }

    #[test]
    fn translation_jacobian_pulls_pose_toward_observation() {
        let time = SystemTime::UNIX_EPOCH;
        let factor = VisualReprojectionFactor::new(
            time,
            time + Duration::from_secs(1),
            time,
            &SE3::identity(),
            VisualReprojectionMeasurement {
                detection: Point2::new(0.0, 0.0),
                field_point: Point3::new(0.0, 0.0, 2.0),
            },
            Matrix2::identity(),
        );
        let intrinsics = CameraIntrinsics::new(vector![100.0, 100.0], vector![0.0, 0.0]);
        let pose = state(vector![0.1, -0.2, 0.0]);

        let linearized = factor.residual_jacobian((
            pose.clone(),
            pose.clone(),
            SE2::identity(),
            intrinsics.clone(),
        ));

        assert_close(linearized.value[0], -5.0, 1.0e-9);
        assert_close(linearized.value[1], 10.0, 1.0e-9);

        // SE23 tangent order is [rot_x, rot_y, rot_z, vel_x, vel_y, vel_z, x, y, z].
        let start_x_column = 6;
        let start_y_column = 7;
        assert_close(linearized.diff[(0, start_x_column)], -50.0, 1.0e-9);
        assert_close(linearized.diff[(1, start_y_column)], -50.0, 1.0e-9);
        assert_close(linearized.diff[(0, start_y_column)], 0.0, 1.0e-9);
        assert_close(linearized.diff[(1, start_x_column)], 0.0, 1.0e-9);

        let mut translation_step = VectorX::zeros(9);
        translation_step[start_x_column] =
            -linearized.value[0] / linearized.diff[(0, start_x_column)];
        translation_step[start_y_column] =
            -linearized.value[1] / linearized.diff[(1, start_y_column)];

        assert_close(translation_step[start_x_column], -0.1, 1.0e-9);
        assert_close(translation_step[start_y_column], 0.2, 1.0e-9);

        let corrected_pose = pose.oplus(translation_step.as_view());
        let corrected_residual =
            factor.residual((corrected_pose, pose, SE2::identity(), intrinsics));

        assert!(corrected_residual.norm() < 1.0e-9);
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {actual} to be within {tolerance} of {expected}"
        );
    }
}

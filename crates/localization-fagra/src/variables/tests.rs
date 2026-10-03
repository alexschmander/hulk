use fagra::{
    Tangent, Variable,
    testing::{TestScalar, TestVariable, Tolerance, proptest::prelude::*},
};
use linear_algebra::Framed;
use nalgebra::{Isometry3, SVector, UnitQuaternion, Vector3};

use super::{CameraIntrinsics, ImuBias, PoseControl, TrajectoryState};

impl<R: TestScalar> TestVariable for PoseControl<R> {
    type Dual = PoseControl<R::Dual>;

    fn states() -> impl Strategy<Value = Self> {
        TrajectoryState::<R>::states().prop_map(|state| Self { pose: state.pose })
    }

    fn increments() -> impl Strategy<Value = Tangent<Self>> {
        prop::array::uniform6(-1.5..1.5)
            .prop_map(|v| SVector::from_column_slice(&v.map(R::from_test_value)))
    }

    fn to_dual(&self) -> Self::Dual {
        Self::Dual {
            pose: Framed::wrap(Isometry3::from_parts(
                self.pose
                    .inner
                    .translation
                    .vector
                    .map(|x| x.dual(0.0))
                    .into(),
                UnitQuaternion::new_unchecked(nalgebra::Quaternion::from_vector(
                    self.pose.inner.rotation.coords.map(|x| x.dual(0.0)),
                )),
            )),
        }
    }

    fn equivalent(&self, other: &Self, tolerance: Tolerance) -> bool {
        close(
            self.pose.inner.to_homogeneous().as_slice(),
            other.pose.inner.to_homogeneous().as_slice(),
            tolerance,
        )
    }

    fn log_is_smooth(&self) -> bool {
        self.pose.inner.rotation.w.test_value().abs() > 1e-3
    }
}

fn close<R: TestScalar>(a: &[R], b: &[R], tolerance: Tolerance) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(&a, &b)| tolerance.close(a.test_value(), b.test_value()))
}

impl<R: TestScalar> TestVariable for TrajectoryState<R> {
    type Dual = TrajectoryState<R::Dual>;

    fn states() -> impl Strategy<Value = Self> {
        prop::array::uniform9(-2.0..2.0).prop_map(|values| {
            let v = values.map(R::from_test_value);
            Self {
                pose: Framed::wrap(Isometry3::from_parts(
                    Vector3::new(v[6], v[7], v[8]).into(),
                    UnitQuaternion::from_euler_angles(v[0], v[1], v[2]),
                )),
                velocity: Framed::wrap(Vector3::new(v[3], v[4], v[5])),
            }
        })
    }

    fn increments() -> impl Strategy<Value = Tangent<Self>> {
        prop::array::uniform9(-1.5..1.5)
            .prop_map(|v| SVector::from_column_slice(&v.map(R::from_test_value)))
    }

    fn to_dual(&self) -> Self::Dual {
        Self::Dual {
            pose: PoseControl { pose: self.pose }.to_dual().pose,
            velocity: Framed::wrap(self.velocity.inner.map(|x| x.dual(0.0))),
        }
    }

    fn equivalent(&self, other: &Self, tolerance: Tolerance) -> bool {
        close(
            self.pose.inner.to_homogeneous().as_slice(),
            other.pose.inner.to_homogeneous().as_slice(),
            tolerance,
        ) && close(
            self.velocity.inner.as_slice(),
            other.velocity.inner.as_slice(),
            tolerance,
        )
    }

    fn log_is_smooth(&self) -> bool {
        self.pose.inner.rotation.quaternion().w.test_value().abs() > 1e-3
    }
}

impl<R: TestScalar> TestVariable for CameraIntrinsics<R> {
    type Dual = CameraIntrinsics<R::Dual>;

    fn tolerance() -> Tolerance {
        let mut tolerance = Tolerance::for_scalar::<R>();
        // Round trips subtract pixel coordinates up to 1000: f32 cancellation
        // is governed by the stored calibration scale, not the small increment.
        tolerance.absolute = tolerance.absolute.max(4.0 * 1000.0 * R::EPSILON);
        tolerance
    }

    fn states() -> impl Strategy<Value = Self> {
        prop::array::uniform4(-1000.0..1000.0).prop_map(|v| {
            let v = v.map(R::from_test_value);
            Self {
                focal_lengths: Framed::wrap(nalgebra::Vector2::new(v[0], v[1])),
                optical_center: Framed::wrap(nalgebra::Point2::new(v[2], v[3])),
            }
        })
    }

    fn increments() -> impl Strategy<Value = Tangent<Self>> {
        prop::array::uniform4(-10.0..10.0)
            .prop_map(|v| SVector::from_column_slice(&v.map(R::from_test_value)))
    }

    fn to_dual(&self) -> Self::Dual {
        Self::Dual {
            focal_lengths: Framed::wrap(self.focal_lengths.inner.map(|x| x.dual(0.0))),
            optical_center: Framed::wrap(self.optical_center.inner.map(|x| x.dual(0.0))),
        }
    }

    fn equivalent(&self, other: &Self, tolerance: Tolerance) -> bool {
        close(
            self.focal_lengths.inner.as_slice(),
            other.focal_lengths.inner.as_slice(),
            tolerance,
        ) && close(
            self.optical_center.inner.coords.as_slice(),
            other.optical_center.inner.coords.as_slice(),
            tolerance,
        )
    }
}

impl<R: TestScalar> TestVariable for ImuBias<R> {
    type Dual = ImuBias<R::Dual>;
    fn states() -> impl Strategy<Value = Self> {
        Self::increments().prop_map(|v| Self::exp(&v))
    }
    fn increments() -> impl Strategy<Value = Tangent<Self>> {
        prop::array::uniform6(-0.2..0.2)
            .prop_map(|v| SVector::from_column_slice(&v.map(R::from_test_value)))
    }
    fn to_dual(&self) -> Self::Dual {
        Self::Dual {
            gyroscope: Framed::wrap(self.gyroscope.inner.map(|x| x.dual(0.0))),
            accelerometer: Framed::wrap(self.accelerometer.inner.map(|x| x.dual(0.0))),
        }
    }
    fn equivalent(&self, other: &Self, tolerance: Tolerance) -> bool {
        close(self.log().as_slice(), other.log().as_slice(), tolerance)
    }
}
fagra::variable_tests!(bias_f64, ImuBias<f64>);
fagra::variable_tests!(bias_f32, ImuBias<f32>);
fagra::variable_tests!(trajectory_f64, TrajectoryState<f64>);
fagra::variable_tests!(pose_control_f64, PoseControl<f64>);
fagra::variable_tests!(pose_control_f32, PoseControl<f32>);
fagra::variable_tests!(trajectory_f32, TrajectoryState<f32>);

fagra::variable_tests!(intrinsics_f64, CameraIntrinsics<f64>);
fagra::variable_tests!(intrinsics_f32, CameraIntrinsics<f32>);

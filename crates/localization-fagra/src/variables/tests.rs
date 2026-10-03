use fagra::{
    Tangent,
    testing::{TestScalar, TestVariable, Tolerance, proptest::prelude::*},
};
use linear_algebra::Framed;
use nalgebra::{Isometry3, SVector, UnitQuaternion, Vector3};

use super::{PoseControl, TrajectoryState};

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

fagra::variable_tests!(trajectory_f64, TrajectoryState<f64>);
fagra::variable_tests!(pose_control_f64, PoseControl<f64>);
fagra::variable_tests!(pose_control_f32, PoseControl<f32>);
fagra::variable_tests!(trajectory_f32, TrajectoryState<f32>);

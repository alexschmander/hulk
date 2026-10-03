use fagra::{
    StateKey, Variable,
    testing::{TestFactor, TestStates, proptest::prelude::*},
};
use linear_algebra::Framed;
use nalgebra::{Isometry3, RealField, SMatrix, UnitQuaternion, Vector3};

use super::*;
use crate::variables::{CameraIntrinsics, ImuBias, PoseControl, TrajectoryState};

fn c<R: RealField>(x: f64) -> R {
    R::from_f64(x).unwrap()
}

fn root<R: RealField + Copy, const N: usize>() -> SMatrix<R, N, N> {
    // Correlated coordinates catch transposed or incorrect whitening.
    SMatrix::from_fn(|i, j| {
        if i == j {
            c(1.0 + i as f64 * 0.1)
        } else if i > j {
            c(0.07)
        } else {
            R::zero()
        }
    })
}

#[derive(Clone, Debug)]
struct Inputs {
    values: [f64; 9],
    bridge: bool,
    stationary: bool,
}

impl Inputs {
    fn cases() -> impl Strategy<Value = Self> {
        prop_oneof![
            Just(Self {
                values: [0.0; 9],
                bridge: false,
                stationary: true
            }),
            (prop::array::uniform9(-0.3..0.3), any::<bool>()).prop_map(|(values, bridge)| Self {
                values,
                bridge,
                stationary: false
            }),
        ]
    }
    fn control<R: RealField + Copy>(&self, i: usize) -> PoseControl<R> {
        if self.stationary {
            return PoseControl::identity();
        }
        let x = i as f64;
        let v = self.values;
        PoseControl {
            pose: Framed::wrap(Isometry3::from_parts(
                Vector3::new(c(v[3] + x * 0.15), c(v[4] - x * x * 0.03), c(0.6 + v[5])).into(),
                UnitQuaternion::from_euler_angles(
                    c(v[0] + x * 0.04),
                    c(v[1] - x * 0.03),
                    c(v[2] + x * 0.06),
                ),
            )),
        }
    }
    fn controls<R: RealField + Copy, const N: usize>(
        &self,
        states: &mut TestStates<R>,
    ) -> [StateKey<PoseControl<R>>; N] {
        std::array::from_fn(|i| states.insert(self.control(i)))
    }
}

macro_rules! ordinary_case {
    ($name:ident, $model:ident, |$input:ident, $states:ident| $build:block) => {
        #[derive(Debug)]
        struct $name(Inputs);
        impl TestFactor for $name {
            type Factor<R: RealField + Copy> = $model<R>;
            fn cases() -> impl Strategy<Value = Self> {
                Inputs::cases().prop_map(Self)
            }
            fn build<R: RealField + Copy>(&self, $states: &mut TestStates<R>) -> $model<R> {
                let $input = &self.0;
                $build
            }
        }
    };
}

ordinary_case!(TrajectoryCase, TrajectoryPrior, |input, states| {
    TrajectoryPrior {
        controls: input.controls(states),
        duration: c(0.25),
        tau: c(0.4),
        reference: TrajectoryState {
            pose: input.control::<R>(1).pose,
            velocity: Framed::wrap(Vector3::new(
                c(input.values[6]),
                c(input.values[7]),
                c(input.values[8]),
            )),
        },
        information_root: root(),
    }
});
ordinary_case!(IntrinsicsCase, CameraIntrinsicsPrior, |input, states| {
    let current = CameraIntrinsics {
        focal_lengths: Framed::wrap(nalgebra::Vector2::new(
            c(300.0 + input.values[0]),
            c(310.0 + input.values[1]),
        )),
        optical_center: Framed::wrap(nalgebra::Point2::new(
            c(160.0 + input.values[2]),
            c(120.0 + input.values[3]),
        )),
    };
    CameraIntrinsicsPrior {
        intrinsics: states.insert(current),
        reference: calibration(),
        information_root: root(),
    }
});
ordinary_case!(MotionCase, MotionPrior, |input, states| {
    MotionPrior {
        controls: input.controls(states),
        duration: c(0.25),
        information_root: root(),
        use_start_velocity: !input.bridge,
    }
});

fn calibration<R: RealField + Copy>() -> CameraIntrinsics<R> {
    CameraIntrinsics {
        focal_lengths: Framed::wrap(nalgebra::Vector2::new(c(300.0), c(310.0))),
        optical_center: Framed::wrap(nalgebra::Point2::new(c(160.0), c(120.0))),
    }
}

ordinary_case!(BiasPriorCase, ImuBiasPrior, |input, states| {
    ImuBiasPrior {
        bias: states.insert(ImuBias::exp(&nalgebra::SVector::<R, 6>::from_fn(|i, _| {
            c(input.values[i])
        }))),
        reference: ImuBias::identity(),
        information_root: root(),
    }
});
ordinary_case!(BiasWalkCase, ImuBiasWalk, |input, states| {
    ImuBiasWalk {
        biases: [
            states.insert(ImuBias::identity()),
            states.insert(ImuBias::exp(&nalgebra::SVector::<R, 6>::from_fn(|i, _| {
                c(input.values[i])
            }))),
        ],
        information_root: root(),
    }
});

fagra::factor_tests!(bias_prior_f64, BiasPriorCase, f64);
fagra::factor_tests!(bias_prior_f32, BiasPriorCase, f32);
fagra::factor_tests!(bias_walk_f64, BiasWalkCase, f64);
fagra::factor_tests!(bias_walk_f32, BiasWalkCase, f32);
fagra::factor_tests!(trajectory_f64, TrajectoryCase, f64);
fagra::factor_tests!(trajectory_f32, TrajectoryCase, f32);
fagra::factor_tests!(intrinsics_f64, IntrinsicsCase, f64);
fagra::factor_tests!(intrinsics_f32, IntrinsicsCase, f32);
fagra::factor_tests!(motion_f64, MotionCase, f64);
fagra::factor_tests!(motion_f32, MotionCase, f32);

use fagra::{EvaluationError, JacobianBlock, LinearizationSink, StateKey, StateStore};
use nalgebra::{RealField, SMatrix, SVector, UnitQuaternion, Vector3};

use crate::{
    spline::PoseSpline,
    variables::{
        PoseControl,
        rotation::{self, scalar},
    },
};

pub(super) use crate::finite;

pub(super) fn positive<R: RealField + Copy>(value: R) -> Result<R, EvaluationError> {
    let inverse = value.recip();
    if value.is_finite() && value > R::zero() && inverse.is_finite() && inverse > R::zero() {
        Ok(inverse)
    } else {
        Err(EvaluationError::InvalidEvaluation)
    }
}

pub(super) fn unique<R: RealField + Copy, const N: usize>(
    keys: &[StateKey<PoseControl<R>>; N],
) -> Result<(), EvaluationError> {
    for (index, key) in keys.iter().enumerate() {
        if keys[index + 1..]
            .iter()
            .any(|other| other.block_id() == key.block_id())
        {
            return Err(EvaluationError::InvalidEvaluation);
        }
    }
    Ok(())
}

pub(super) fn spline<R: RealField + Copy, S: StateStore<PoseControl<R>>>(
    states: &S,
    keys: &[StateKey<PoseControl<R>>; 4],
    duration: R,
) -> Result<PoseSpline<R>, EvaluationError> {
    unique(keys)?;
    let spline = PoseSpline::new(
        [
            states.get(keys[0])?,
            states.get(keys[1])?,
            states.get(keys[2])?,
            states.get(keys[3])?,
        ],
        duration,
    )?;
    spline.check_smooth_rotation()?;
    Ok(spline)
}

pub(super) fn cost<R: RealField + Copy, const N: usize>(
    residual: &SVector<R, N>,
) -> Result<R, EvaluationError> {
    checked_cost(residual.norm_squared() * scalar::<R>(0.5))
}

pub(super) fn checked_cost<R: RealField + Copy>(cost: R) -> Result<R, EvaluationError> {
    if cost.is_finite() && cost >= R::zero() {
        Ok(cost)
    } else {
        Err(EvaluationError::InvalidEvaluation)
    }
}

pub(super) fn emit<
    R: RealField + Copy,
    L: LinearizationSink<Scalar = R>,
    const M: usize,
    const N: usize,
>(
    sink: &mut L,
    keys: &[StateKey<PoseControl<R>>; N],
    r: &SVector<R, M>,
    jacobians: &[SMatrix<R, M, 6>; N],
) -> Result<(), EvaluationError> {
    let blocks: [_; N] = std::array::from_fn(|i| JacobianBlock::new(keys[i], &jacobians[i]));
    emit_blocks(sink, r, blocks)
}

/// Omit exactly-zero blocks without allocating or dropping constant residual cost.
pub(super) fn emit_blocks<
    R: RealField + Copy,
    L: LinearizationSink<Scalar = R>,
    const M: usize,
    const N: usize,
>(
    sink: &mut L,
    residual: &SVector<R, M>,
    mut blocks: [JacobianBlock<'_, R>; N],
) -> Result<(), EvaluationError> {
    let mut count = 0;
    for i in 0..N {
        if blocks[i].jacobian().iter().any(|v| *v != R::zero()) {
            blocks.swap(count, i);
            count += 1;
        }
    }
    sink.residual(residual, &blocks[..count])
}

pub(super) fn rotation_log<R: RealField + Copy>(
    r: &UnitQuaternion<R>,
) -> Result<Vector3<R>, EvaluationError> {
    finite(r.coords.iter())?;
    if r.w.abs() <= scalar(1e-6) {
        return Err(EvaluationError::InvalidEvaluation);
    }
    Ok(rotation::log(r))
}

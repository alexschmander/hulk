use fagra::{EvaluationError, JacobianBlock, LinearizationSink, StateKey, StateStore};
use nalgebra::{Matrix3, RealField, SMatrix, SVector, UnitQuaternion, Vector3};

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
    let mut remaining = keys.as_slice();

    while let Some((current_key, tail_keys)) = remaining.split_first() {
        let current_block_identifier = current_key.block_id();
        let is_duplicate = tail_keys
            .iter()
            .any(|key| key.block_id() == current_block_identifier);

        if is_duplicate {
            return Err(EvaluationError::InvalidEvaluation);
        }

        remaining = tail_keys;
    }

    Ok(())
}

pub(super) fn spline<R: RealField + Copy, S: StateStore<PoseControl<R>>>(
    states: &S,
    keys: &[StateKey<PoseControl<R>>; 4],
    duration: R,
) -> Result<PoseSpline<R>, EvaluationError> {
    unique(keys)?;
    PoseSpline::new(
        [
            states.get(keys[0])?,
            states.get(keys[1])?,
            states.get(keys[2])?,
            states.get(keys[3])?,
        ],
        duration,
    )
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

/// True blockwise Huber cost and square-root frozen IRLS weight. Avoid sqrt(0)
/// so dual-number derivatives remain defined at exact fits.
pub(super) fn huber<R: RealField + Copy, const N: usize>(
    r: &SVector<R, N>,
    threshold: R,
) -> Result<(R, R), EvaluationError> {
    positive(threshold)?;
    let squared = r.norm_squared();
    checked_cost(squared)?;
    if squared <= threshold * threshold {
        Ok((squared * scalar::<R>(0.5), R::one()))
    } else {
        let norm = squared.sqrt();
        let cost = checked_cost(threshold * (norm - threshold * scalar::<R>(0.5)))?;
        Ok((cost, (threshold / norm).sqrt()))
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
    sink.residual(r, &blocks)
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

pub(super) fn validate_up<R: RealField + Copy>(up: &Vector3<R>) -> Result<(), EvaluationError> {
    finite(up.iter())?;
    if (up.norm_squared() - R::one()).abs() <= scalar(1e-5) {
        Ok(())
    } else {
        Err(EvaluationError::InvalidEvaluation)
    }
}

pub(super) fn up<R: RealField + Copy>(r: &UnitQuaternion<R>) -> Vector3<R> {
    r.inverse() * Vector3::z()
}

pub(super) fn heading<R: RealField + Copy>(r: &UnitQuaternion<R>) -> Result<R, EvaluationError> {
    let forward = r * Vector3::x();
    finite(forward.iter())?;
    if forward.x * forward.x + forward.y * forward.y <= scalar(1e-12) {
        return Err(EvaluationError::InvalidEvaluation);
    }
    Ok(forward.y.atan2(forward.x))
}

pub(super) fn heading_jacobian<R: RealField + Copy>(
    r: &UnitQuaternion<R>,
) -> Result<SMatrix<R, 1, 3>, EvaluationError> {
    heading(r)?;
    let rotation = r.to_rotation_matrix().into_inner();
    let h = rotation.column(0);
    let gradient = SMatrix::<R, 1, 3>::new(-h.y, h.x, R::zero()) / (h.x * h.x + h.y * h.y);
    Ok(-gradient * rotation * Vector3::<R>::x().cross_matrix())
}

pub(super) fn yaw_error<R: RealField + Copy>(
    start: &UnitQuaternion<R>,
    end: &UnitQuaternion<R>,
    measured: R,
) -> Result<R, EvaluationError> {
    if !measured.is_finite() {
        return Err(EvaluationError::InvalidEvaluation);
    }
    let difference = heading(end)? - heading(start)? - measured;
    let result = difference.sin().atan2(difference.cos());
    if result.abs() >= R::pi() - scalar::<R>(1e-6) {
        return Err(EvaluationError::InvalidEvaluation);
    }
    Ok(result)
}

/// Right-local point action: d(R q + p)/d[theta,rho].
pub(super) fn point_jacobian<R: RealField + Copy>(
    rotation: Matrix3<R>,
    point: Vector3<R>,
) -> SMatrix<R, 3, 6> {
    let mut j = SMatrix::<R, 3, 6>::zeros();
    j.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(-rotation * point.cross_matrix()));
    j.fixed_view_mut::<3, 3>(0, 3).copy_from(&rotation);
    j
}

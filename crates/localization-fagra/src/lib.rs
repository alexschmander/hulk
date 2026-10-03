//! Framed variables for continuous-time 3D localization.

pub mod variables;

pub mod alignment;

pub mod factors;
pub mod preintegration;
pub mod spline;

use fagra::EvaluationError;
use nalgebra::RealField;

pub(crate) fn finite<'a, R: RealField + 'a>(
    mut values: impl Iterator<Item = &'a R>,
) -> Result<(), EvaluationError> {
    if values.all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(EvaluationError::InvalidEvaluation)
    }
}

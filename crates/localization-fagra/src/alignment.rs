//! Planar alignment from camera bearings and known ground landmarks.

use coordinate_systems::{Camera, Field, Local};
use fagra::EvaluationError;
use linear_algebra::{IntoTransform, Isometry2, Isometry3, Point2, Vector3};
use nalgebra::RealField;

use crate::{finite, variables::rotation::scalar};

/// Fit `field = translation + height * rotation * unit_ground_point`.
/// Unit-ground points are intersections of leveled rays with a plane one metre below
/// the camera. Translation is the camera's field XY; scale is camera height.
pub fn fit_ground_similarity<R: RealField + Copy>(
    correspondences: impl ExactSizeIterator<Item = (nalgebra::Vector2<R>, Point2<Field, R>)> + Clone,
) -> Result<(Isometry2<Local, Field, R>, R), EvaluationError> {
    if correspondences.len() < 3 {
        return Err(EvaluationError::InvalidEvaluation);
    }
    let count = scalar::<R>(correspondences.len() as f64);
    let (mut q_mean, mut p_mean) = (
        nalgebra::Vector2::<R>::zeros(),
        nalgebra::Vector2::<R>::zeros(),
    );
    for (q, p) in correspondences.clone() {
        finite(q.iter().chain(p.inner.coords.iter()))?;
        q_mean += q;
        p_mean += p.inner.coords;
    }
    q_mean /= count;
    p_mean /= count;
    let (mut dot, mut cross, mut variance) = (R::zero(), R::zero(), R::zero());
    let mut scatter = nalgebra::Matrix2::<R>::zeros();
    for (q, p) in correspondences {
        let q = q - q_mean;
        let p = p.inner.coords - p_mean;
        dot += q.dot(&p);
        cross += q.x * p.y - q.y * p.x;
        variance += q.norm_squared();
        scatter += q * q.transpose();
    }
    if variance <= scalar(1.0e-12)
        || scatter.determinant() <= scalar::<R>(1.0e-10) * variance * variance
    {
        return Err(EvaluationError::InvalidEvaluation);
    }
    let height = dot.hypot(cross) / variance;
    if !height.is_finite() || height <= scalar(1.0e-6) {
        return Err(EvaluationError::InvalidEvaluation);
    }
    let rotation = nalgebra::UnitComplex::new(cross.atan2(dot));
    let translation = p_mean - rotation * q_mean * height;
    finite(translation.iter())?;
    Ok((
        nalgebra::Isometry2::from_parts(translation.into(), rotation).framed_transform(),
        height,
    ))
}

/// Intersect camera rays with Local z=0 and fit a rigid Local-to-Field alignment.
/// Height and tilt stay fixed. At least two nondegenerate correspondences are
/// required; certification counts and field-side selection belong to the caller.
/// `min_depth` is the positive minimum optical-camera z at the intersection.
/// Two passes over the correspondences avoid allocating projection storage.
pub fn fit_ground_alignment<R: RealField + Copy>(
    camera_to_local: &Isometry3<Camera, Local, R>,
    correspondences: impl ExactSizeIterator<Item = (Vector3<Camera, R>, Point2<Field, R>)> + Clone,
    min_depth: R,
) -> Result<Isometry2<Local, Field, R>, EvaluationError> {
    if correspondences.len() < 2 || !min_depth.is_finite() || min_depth <= R::zero() {
        return Err(EvaluationError::InvalidEvaluation);
    }
    let origin = camera_to_local.inner.translation.vector;
    finite(
        origin
            .iter()
            .chain(camera_to_local.inner.rotation.coords.iter()),
    )?;
    let project = |(bearing, field): (Vector3<Camera, R>, Point2<Field, R>)| {
        finite(bearing.inner.iter().chain(field.inner.coords.iter()))?;
        let ray = camera_to_local.inner.rotation * bearing.inner;
        let distance = -origin.z / ray.z;
        let depth = distance * bearing.inner.z;
        if distance <= R::zero()
            || !depth.is_finite()
            || depth <= min_depth
            || ray.z.abs() < scalar(1.0e-6)
        {
            return Err(EvaluationError::InvalidEvaluation);
        }
        let local = (origin + ray * distance).xy();
        finite(local.iter())?;
        Ok((local, field.inner.coords))
    };
    let count = scalar::<R>(correspondences.len() as f64);
    let mut local_mean = nalgebra::Vector2::<R>::zeros();
    let mut field_mean = nalgebra::Vector2::<R>::zeros();
    for pair in correspondences.clone() {
        let (local, field) = project(pair)?;
        local_mean += local;
        field_mean += field;
    }
    local_mean /= count;
    field_mean /= count;
    let mut dot = R::zero();
    let mut cross = R::zero();
    for pair in correspondences {
        let (local, field) = project(pair)?;
        let local = local - local_mean;
        let field = field - field_mean;
        dot += local.dot(&field);
        cross += local.x * field.y - local.y * field.x;
    }
    if !dot.is_finite() || !cross.is_finite() || dot.hypot(cross) < scalar(1.0e-8) {
        return Err(EvaluationError::InvalidEvaluation);
    }
    let rotation = nalgebra::UnitComplex::new(cross.atan2(dot));
    let translation = field_mean - rotation * local_mean;
    finite(translation.iter())?;
    Ok(nalgebra::Isometry2::from_parts(translation.into(), rotation).framed_transform())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_recovers_height_yaw_and_position_and_rejects_degeneracy() {
        let expected = nalgebra::Isometry2::new(nalgebra::Vector2::new(-2.0, 1.0), 0.7);
        for height in [0.4_f64, 0.9, 2.0] {
            let points = [(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0), (0.3, 0.7)].map(|(x, y)| {
                let q = nalgebra::Vector2::new(x, y);
                (
                    q,
                    Point2::wrap(expected * nalgebra::Point2::from(q * height)),
                )
            });
            for count in [3, 4] {
                let (pose, fitted_height) =
                    fit_ground_similarity(points[..count].iter().copied()).unwrap();
                assert!((fitted_height - height).abs() < 1e-10);
                assert!((pose.inner.to_homogeneous() - expected.to_homogeneous()).norm() < 1e-10);
            }
            assert!(fit_ground_similarity([points[0]; 3].into_iter()).is_err());
        }
        let line = [0.0, 1.0, 2.0].map(|x| {
            (
                nalgebra::Vector2::new(x, 0.0),
                Point2::wrap(nalgebra::Point2::new(x, 0.0)),
            )
        });
        assert!(fit_ground_similarity(line.into_iter()).is_err());
    }

    #[test]
    fn ground_fit_recovers_rigid_transform_and_rejects_undefined_geometry() {
        let camera: Isometry3<Camera, Local, f64> = nalgebra::Isometry3::new(
            nalgebra::Vector3::new(0.2, -0.3, 1.0),
            nalgebra::Vector3::new(std::f64::consts::PI, 0.0, 0.0),
        )
        .framed_transform();
        let expected = nalgebra::Isometry2::new(nalgebra::Vector2::new(3.0, 2.0), 0.4);
        let observations = [(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)].map(|(x, y)| {
            let p = camera.inner.inverse() * nalgebra::Point3::new(x, y, 0.0);
            (
                Vector3::wrap(p.coords / p.z),
                Point2::wrap(expected * nalgebra::Point2::new(x, y)),
            )
        });
        let fitted = fit_ground_alignment(&camera, observations.into_iter(), 0.01).unwrap();
        assert!((fitted.inner.to_homogeneous() - expected.to_homogeneous()).norm() < 1.0e-10);
        assert!(fit_ground_alignment(&camera, [observations[0]; 3].into_iter(), 0.01).is_err());
        let horizon = [(Vector3::wrap(nalgebra::Vector3::x()), observations[0].1); 2];
        assert!(fit_ground_alignment(&camera, horizon.into_iter(), 0.01).is_err());
        let backwards = observations.map(|(ray, field)| (Vector3::wrap(-ray.inner), field));
        assert!(fit_ground_alignment(&camera, backwards.into_iter(), 0.01).is_err());
        assert!(fit_ground_alignment(&camera, observations.into_iter(), 0.0).is_err());
    }
}

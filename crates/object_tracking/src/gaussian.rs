//! Numerically careful Gaussian operations shared by all trackers.
//!
//! Updates solve through a Cholesky factorization and use the Joseph form, so a single badly
//! conditioned innovation can neither panic nor produce a covariance that is not symmetric.

use std::f32::consts::PI;

use nalgebra::{SMatrix, SVector};
use types::multivariate_normal_distribution::MultivariateNormalDistribution;

pub type Gaussian<const N: usize> = MultivariateNormalDistribution<N>;

/// Prediction of a measurement from a state, linearized where necessary.
#[derive(Clone, Copy, Debug)]
pub struct Innovation<const N: usize, const M: usize> {
    pub residual: SVector<f32, M>,
    pub measurement_jacobian: SMatrix<f32, M, N>,
    pub covariance: SMatrix<f32, M, M>,
    pub squared_mahalanobis_distance: f32,
    pub log_likelihood: f32,
}

pub fn predict<const N: usize>(
    gaussian: &mut Gaussian<N>,
    transition: &SMatrix<f32, N, N>,
    process_noise: &SMatrix<f32, N, N>,
) {
    gaussian.mean = transition * gaussian.mean;
    gaussian.covariance =
        symmetrize(transition * gaussian.covariance * transition.transpose() + process_noise);
}

/// Computes the innovation of a measurement `z = h(x) + v` with `v ~ N(0, R)`.
///
/// `residual` must already be `z - h(mean)`, wrapped for angular measurements.
/// Returns `None` if the innovation covariance is not positive definite.
pub fn innovation<const N: usize, const M: usize>(
    gaussian: &Gaussian<N>,
    residual: SVector<f32, M>,
    measurement_jacobian: SMatrix<f32, M, N>,
    measurement_noise: &SMatrix<f32, M, M>,
) -> Option<Innovation<N, M>> {
    let covariance = symmetrize(
        measurement_jacobian * gaussian.covariance * measurement_jacobian.transpose()
            + measurement_noise,
    );
    let cholesky = covariance.cholesky()?;
    let whitened = cholesky.l().solve_lower_triangular(&residual)?;
    let squared_mahalanobis_distance = whitened.norm_squared();
    let log_determinant = 2.0 * cholesky.l().diagonal().map(f32::ln).sum();
    let log_likelihood =
        -0.5 * (squared_mahalanobis_distance + log_determinant + M as f32 * (2.0 * PI).ln());
    if !log_likelihood.is_finite() {
        return None;
    }
    Some(Innovation {
        residual,
        measurement_jacobian,
        covariance,
        squared_mahalanobis_distance,
        log_likelihood,
    })
}

/// Kalman update with the Joseph-form covariance.
///
/// Returns `None` if the innovation covariance cannot be factorized.
pub fn update<const N: usize, const M: usize>(
    gaussian: &Gaussian<N>,
    innovation: &Innovation<N, M>,
    measurement_noise: &SMatrix<f32, M, M>,
) -> Option<Gaussian<N>> {
    let cholesky = innovation.covariance.cholesky()?;
    let cross_covariance = gaussian.covariance * innovation.measurement_jacobian.transpose();
    let kalman_gain = cholesky.solve(&cross_covariance.transpose()).transpose();
    let mean = gaussian.mean + kalman_gain * innovation.residual;
    let identity_minus_gain =
        SMatrix::<f32, N, N>::identity() - kalman_gain * innovation.measurement_jacobian;
    let covariance = symmetrize(
        identity_minus_gain * gaussian.covariance * identity_minus_gain.transpose()
            + kalman_gain * measurement_noise * kalman_gain.transpose(),
    );
    Some(Gaussian { mean, covariance })
}

/// Moment-matched merge of weighted Gaussians, including the spread between their means.
///
/// Weights need not be normalized. Returns `None` for an empty or zero-weight input.
pub fn merge<'a, const N: usize>(
    components: impl IntoIterator<Item = (f32, &'a Gaussian<N>)> + Clone,
) -> Option<Gaussian<N>> {
    let total_weight: f32 = components
        .clone()
        .into_iter()
        .map(|(weight, _)| weight)
        .sum();
    if total_weight <= 0.0 || !total_weight.is_finite() {
        return None;
    }
    let mean = components
        .clone()
        .into_iter()
        .fold(SVector::<f32, N>::zeros(), |sum, (weight, gaussian)| {
            sum + gaussian.mean * (weight / total_weight)
        });
    let covariance =
        components
            .into_iter()
            .fold(SMatrix::<f32, N, N>::zeros(), |sum, (weight, gaussian)| {
                let spread = gaussian.mean - mean;
                sum + (gaussian.covariance + spread * spread.transpose()) * (weight / total_weight)
            });
    Some(Gaussian {
        mean,
        covariance: symmetrize(covariance),
    })
}

/// Squared Mahalanobis distance between the means of two Gaussians under their summed covariance.
pub fn squared_mahalanobis_distance_between<const N: usize>(
    first: &Gaussian<N>,
    second: &Gaussian<N>,
) -> Option<f32> {
    let difference = first.mean - second.mean;
    let cholesky = symmetrize(first.covariance + second.covariance).cholesky()?;
    let whitened = cholesky.l().solve_lower_triangular(&difference)?;
    Some(whitened.norm_squared())
}

/// Symmetric sigma points of the leading `D` dimensions (usually the position).
///
/// Returns the mean followed by `2 D` points at `±sqrt(D)` standard deviations along the
/// Cholesky columns, each with weight `1 / (2 D + 1)`. Falls back to the mean alone if the
/// marginal covariance is not positive definite.
pub fn marginal_sigma_points<const N: usize, const D: usize>(
    gaussian: &Gaussian<N>,
) -> Vec<SVector<f32, D>> {
    let mean = gaussian.mean.fixed_rows::<D>(0).into_owned();
    let covariance = gaussian.covariance.fixed_view::<D, D>(0, 0).into_owned();
    let Some(cholesky) = covariance.cholesky() else {
        return vec![mean];
    };
    let scale = (D as f32).sqrt();
    let lower = cholesky.l();
    let mut points = Vec::with_capacity(2 * D + 1);
    points.push(mean);
    for column in lower.column_iter() {
        points.push(mean + column * scale);
        points.push(mean - column * scale);
    }
    points
}

pub fn symmetrize<const N: usize>(matrix: SMatrix<f32, N, N>) -> SMatrix<f32, N, N> {
    (matrix + matrix.transpose()) * 0.5
}

/// Wraps an angle difference into `[-pi, pi)`.
pub fn wrap_angle(angle: f32) -> f32 {
    (angle + PI).rem_euclid(2.0 * PI) - PI
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{Matrix1, Matrix2, Matrix2x4, Matrix4, Vector1, Vector2, Vector4};

    use super::*;

    #[test]
    fn update_matches_textbook_scalar_case() {
        let prior = Gaussian::<1> {
            mean: Vector1::new(0.0),
            covariance: Matrix1::new(4.0),
        };
        let noise = Matrix1::new(1.0);
        let innovation = innovation(&prior, Vector1::new(5.0), Matrix1::new(1.0), &noise).unwrap();
        let posterior = update(&prior, &innovation, &noise).unwrap();

        assert_relative_eq!(posterior.mean[0], 4.0, epsilon = 1e-6);
        assert_relative_eq!(posterior.covariance[(0, 0)], 0.8, epsilon = 1e-6);
        assert_relative_eq!(innovation.squared_mahalanobis_distance, 5.0, epsilon = 1e-6);
        let expected_log_likelihood = -0.5 * (5.0 + 5.0_f32.ln() + (2.0 * PI).ln());
        assert_relative_eq!(
            innovation.log_likelihood,
            expected_log_likelihood,
            epsilon = 1e-5
        );
    }

    #[test]
    fn position_update_leaves_unobserved_velocity_consistent() {
        let prior = Gaussian::<4> {
            mean: Vector4::new(1.0, 2.0, 0.5, 0.0),
            covariance: Matrix4::from_diagonal(&Vector4::new(1.0, 1.0, 0.25, 0.25)),
        };
        let jacobian = Matrix2x4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        let noise = Matrix2::identity() * 0.01;
        let residual = Vector2::new(1.5, 2.0) - jacobian * prior.mean;
        let innovation = innovation(&prior, residual, jacobian, &noise).unwrap();
        let posterior = update(&prior, &innovation, &noise).unwrap();

        assert_relative_eq!(posterior.mean[2], 0.5, epsilon = 1e-6);
        assert_relative_eq!(posterior.covariance[(2, 2)], 0.25, epsilon = 1e-6);
        assert!(posterior.covariance[(0, 0)] < 0.01);
        assert_relative_eq!(posterior.covariance, posterior.covariance.transpose());
    }

    #[test]
    fn singular_innovation_is_rejected_instead_of_panicking() {
        let prior = Gaussian::<2> {
            mean: Vector2::zeros(),
            covariance: Matrix2::zeros(),
        };
        assert!(
            innovation(
                &prior,
                Vector2::zeros(),
                Matrix2::identity(),
                &Matrix2::zeros()
            )
            .is_none()
        );
    }

    #[test]
    fn merge_includes_spread_of_means() {
        let left = Gaussian::<1> {
            mean: Vector1::new(-1.0),
            covariance: Matrix1::new(0.1),
        };
        let right = Gaussian::<1> {
            mean: Vector1::new(1.0),
            covariance: Matrix1::new(0.1),
        };
        let merged = merge([(1.0, &left), (1.0, &right)]).unwrap();

        assert_relative_eq!(merged.mean[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(merged.covariance[(0, 0)], 1.1, epsilon = 1e-6);
    }

    #[test]
    fn sigma_points_reproduce_mean_and_covariance() {
        let gaussian = Gaussian::<4> {
            mean: Vector4::new(1.0, -2.0, 3.0, 4.0),
            covariance: Matrix4::new(
                2.0, 0.5, 0.0, 0.0, //
                0.5, 1.0, 0.0, 0.0, //
                0.0, 0.0, 9.0, 0.0, //
                0.0, 0.0, 0.0, 9.0,
            ),
        };
        let points = marginal_sigma_points::<4, 2>(&gaussian);
        let tails = &points[1..];
        let mean = tails.iter().sum::<Vector2<f32>>() / tails.len() as f32;
        let covariance = tails
            .iter()
            .map(|point| (point - mean) * (point - mean).transpose())
            .sum::<Matrix2<f32>>()
            / tails.len() as f32;

        assert_eq!(points.len(), 5);
        assert_relative_eq!(mean, Vector2::new(1.0, -2.0), epsilon = 1e-5);
        assert_relative_eq!(covariance, Matrix2::new(2.0, 0.5, 0.5, 1.0), epsilon = 1e-5);
    }

    #[test]
    fn angles_wrap_into_half_open_interval() {
        assert_relative_eq!(wrap_angle(3.0 * PI), -PI, epsilon = 1e-5);
        assert_relative_eq!(wrap_angle(-0.5), -0.5, epsilon = 1e-6);
        assert_relative_eq!(wrap_angle(2.0 * PI + 0.25), 0.25, epsilon = 1e-5);
    }
}

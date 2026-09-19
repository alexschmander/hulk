use factrs::{core::Vector3, linalg::Matrix3};
use nalgebra::{Matrix2, SMatrix};
use types::field_dimensions::FieldDimensions;

#[derive(Debug, Clone)]
pub struct BackendConfiguration {
    /// The spacing between control knots on the Gaussian Process
    /// Each control knot represents 9 DoFs for the optimizer.
    pub knot_spacing: std::time::Duration,
    /// The maximum optimization window size.
    /// Factors before the optimization window are marginalized.
    pub max_optimization_window: std::time::Duration,
    /// Maximum optimizer iterations per solve call.
    /// Slow solve cadences can spend more iterations on each larger batch.
    pub optimizer_max_iterations: usize,

    pub gyroscope_noise: Matrix3<f64>,
    pub accelerometer_noise: Matrix3<f64>,
    pub use_accelerometer_measurements: bool,
    pub gyroscope_process_noise: Matrix3<f64>,
    pub accelerometer_process_noise: Matrix3<f64>,
    pub gravity: Vector3<f64>,

    /// Block-independent orientation noise: the roll/pitch 2x2 block and yaw variance
    /// are consumed separately. Yaw/tilt cross terms do not affect residuals.
    pub roll_pitch_yaw_noise: Matrix3<f64>,
    pub visual_feature_noise: Matrix2<f64>,
    pub visual_odometry_noise: SMatrix<f64, 6, 6>,
    pub foot_ground_sigma: f64,
    pub field_containment: FieldContainmentConfiguration,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BackendConfigurationError {
    #[error("{0} must be finite and positive")]
    InvalidPositiveValue(&'static str),
    #[error("{0} must be a finite, symmetric positive-definite covariance with finite whitening")]
    InvalidCovariance(&'static str),
    #[error("{0} must produce finite, nonzero whitening information")]
    InvalidWhitening(&'static str),
    #[error("gravity must be finite and nonzero")]
    InvalidGravity,
    #[error("live changes to {0} are unsupported; recreate the backend to change structure")]
    UnsupportedStructuralChange(&'static str),
}

impl BackendConfiguration {
    /// Validates startup or live tuning before it reaches factor constructors.
    pub fn validate(&self) -> Result<(), BackendConfigurationError> {
        for (value, field) in [
            (self.knot_spacing.as_secs_f64(), "knot_spacing"),
            (
                self.max_optimization_window.as_secs_f64(),
                "max_optimization_window",
            ),
            (
                self.optimizer_max_iterations as f64,
                "optimizer_max_iterations",
            ),
            (self.foot_ground_sigma, "foot_ground_sigma"),
            (self.field_containment.x_limit, "field_containment.x_limit"),
            (self.field_containment.y_limit, "field_containment.y_limit"),
            (self.field_containment.sigma, "field_containment.sigma"),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(BackendConfigurationError::InvalidPositiveValue(field));
            }
        }
        if !self.gravity.iter().all(|value| value.is_finite())
            || !self.gravity.norm().is_finite()
            || self.gravity.norm() == 0.0
        {
            return Err(BackendConfigurationError::InvalidGravity);
        }
        for (covariance, field) in [
            (&self.gyroscope_noise, "gyroscope_noise"),
            (&self.accelerometer_noise, "accelerometer_noise"),
            (&self.gyroscope_process_noise, "gyroscope_process_noise"),
            (
                &self.accelerometer_process_noise,
                "accelerometer_process_noise",
            ),
            (&self.roll_pitch_yaw_noise, "roll_pitch_yaw_noise"),
        ] {
            validate_covariance(covariance, field)?;
        }
        for (sigma, field) in [
            (self.foot_ground_sigma, "foot_ground_sigma"),
            (self.field_containment.sigma, "field_containment.sigma"),
        ] {
            let information = sigma.recip().powi(2);
            if !information.is_finite() || information == 0.0 {
                return Err(BackendConfigurationError::InvalidWhitening(field));
            }
        }
        validate_covariance(&self.visual_feature_noise, "visual_feature_noise")?;
        validate_covariance(&self.visual_odometry_noise, "visual_odometry_noise")
    }
}

fn validate_covariance<const N: usize>(
    covariance: &SMatrix<f64, N, N>,
    field: &'static str,
) -> Result<(), BackendConfigurationError> {
    if covariance.iter().all(|value| value.is_finite())
        && *covariance == covariance.transpose()
        && covariance
            .cholesky()
            .and_then(|factor| factor.l().try_inverse())
            .is_some_and(|root| root.iter().all(|value| value.is_finite()))
    {
        Ok(())
    } else {
        Err(BackendConfigurationError::InvalidCovariance(field))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FieldContainmentConfiguration {
    pub x_limit: f64,
    pub y_limit: f64,
    pub sigma: f64,
}

impl FieldContainmentConfiguration {
    pub fn from_field_dimensions(field_dimensions: &FieldDimensions, sigma: f64) -> Self {
        Self {
            x_limit: field_dimensions.length as f64 * 0.5
                + field_dimensions.border_strip_width as f64,
            y_limit: field_dimensions.width as f64 * 0.5
                + field_dimensions.border_strip_width as f64,
            sigma,
        }
    }
}

impl Default for FieldContainmentConfiguration {
    fn default() -> Self {
        Self::from_field_dimensions(&FieldDimensions::SPL_2025, 1.0)
    }
}

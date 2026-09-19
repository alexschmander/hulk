use std::time::Duration;

use localization_factrs::{BackendConfiguration, FieldContainmentConfiguration};
use nalgebra::{Matrix2, Matrix3, SMatrix, Vector3, vector};
use ros_z::Message;
use serde::{Deserialize, Serialize};
use types::field_dimensions::FieldDimensions;

const MAX_TRACKING_TIMEOUT: Duration = Duration::from_secs(24 * 60 * 60);

/// Runtime parameters for the 3D localization node.
#[derive(Clone, Debug, Deserialize, Serialize, Message)]
#[serde(deny_unknown_fields)]
pub struct Localization3dParameters {
    /// Translational white-noise-on-acceleration spectral density.
    pub accelerometer_process_noise_variance: f64,
    /// Pixel residual variance for accepted visual feature associations.
    pub visual_feature_noise_variance: f64,
    /// Soft field containment sigma in meters outside field plus border strip.
    pub field_containment_sigma: f64,
    /// Time without a converged aligned backend result before localization is declared lost.
    #[serde(default = "default_tracking_timeout")]
    pub tracking_timeout: Duration,
    /// Time without a newly acknowledged, valid visual reprojection before tracking is lost.
    #[serde(default = "default_tracking_timeout")]
    pub visual_tracking_timeout: Duration,
}

fn default_tracking_timeout() -> Duration {
    Duration::from_secs(2)
}

impl Localization3dParameters {
    pub(crate) fn validate(&self) -> std::result::Result<(), String> {
        if !self.accelerometer_process_noise_variance.is_finite()
            || self.accelerometer_process_noise_variance <= 0.0
        {
            return Err("accelerometer_process_noise_variance must be finite and > 0".to_string());
        }
        if !self.visual_feature_noise_variance.is_finite()
            || self.visual_feature_noise_variance <= 0.0
        {
            return Err("visual_feature_noise_variance must be finite and > 0".to_string());
        }
        if !self.field_containment_sigma.is_finite() || self.field_containment_sigma <= 0.0 {
            return Err("field_containment_sigma must be finite and > 0".to_string());
        }
        for (name, timeout) in [
            ("tracking_timeout", self.tracking_timeout),
            ("visual_tracking_timeout", self.visual_tracking_timeout),
        ] {
            if timeout.is_zero() || timeout > MAX_TRACKING_TIMEOUT {
                return Err(format!("{name} must be > 0 and <= 24 hours"));
            }
        }
        // This hook runs before remote updates are committed. Match backend validation so
        // accepted tuning cannot later fail in the ingestion loop or factor constructors.
        backend_configuration_from_parameters_and_field_dimensions(self, &FieldDimensions::SPL_2025)
            .validate()
            .map_err(|error| error.to_string())
    }
}

pub(crate) fn backend_configuration_from_parameters_and_field_dimensions(
    parameters: &Localization3dParameters,
    field_dimensions: &FieldDimensions,
) -> BackendConfiguration {
    backend_configuration(
        parameters.accelerometer_process_noise_variance,
        parameters.visual_feature_noise_variance,
        parameters.field_containment_sigma,
        field_dimensions,
    )
}

fn backend_configuration(
    accelerometer_process_noise_variance: f64,
    visual_feature_noise_variance: f64,
    field_containment_sigma: f64,
    field_dimensions: &FieldDimensions,
) -> BackendConfiguration {
    let process_noise = Matrix3::identity() * 0.01;
    BackendConfiguration {
        knot_spacing: Duration::from_millis(200),
        max_optimization_window: Duration::from_secs(2),
        optimizer_max_iterations: 5,
        gyroscope_noise: Matrix3::identity() * 0.1_f64.powi(2),
        // TODO: tune accelerometer noise
        accelerometer_noise: Matrix3::from_diagonal(&vector![
            5.0_f64.powi(2),   // x sigma = 5 m/s^2
            5.0_f64.powi(2),   // y sigma = 5 m/s^2
            100.0_f64.powi(2), // z disabled
        ]),
        use_accelerometer_measurements: false,
        gyroscope_process_noise: process_noise,
        roll_pitch_yaw_noise: Matrix3::from_diagonal(&Vector3::new(0.01, 0.01, 0.00001)),
        accelerometer_process_noise: Matrix3::identity() * accelerometer_process_noise_variance,
        visual_feature_noise: Matrix2::identity() * visual_feature_noise_variance,
        // factrs::SE3 tangent order is [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z].
        visual_odometry_noise: SMatrix::<f64, 6, 6>::identity() * 1.0e-2,
        foot_ground_sigma: 1e-2,
        field_containment: FieldContainmentConfiguration::from_field_dimensions(
            field_dimensions,
            field_containment_sigma,
        ),
        gravity: Vector3::new(0.0, 0.0, 9.81),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_tuning_rejects_unusable_whitening_and_timer_ranges_before_commit() {
        let parameters = Localization3dParameters {
            accelerometer_process_noise_variance: 10.0,
            visual_feature_noise_variance: 10000.0,
            field_containment_sigma: 1.0,
            tracking_timeout: Duration::from_secs(2),
            visual_tracking_timeout: Duration::from_secs(2),
        };
        assert!(parameters.validate().is_ok());
        let mut invalid = parameters.clone();
        invalid.accelerometer_process_noise_variance = 1.0e-250;
        assert!(invalid.validate().is_err());
        invalid = parameters.clone();
        invalid.visual_feature_noise_variance = f64::INFINITY;
        assert!(invalid.validate().is_err());
        invalid = parameters.clone();
        invalid.field_containment_sigma = 1.0e-250;
        assert!(invalid.validate().is_err());
        for timeout in [Duration::ZERO, Duration::MAX] {
            invalid = parameters.clone();
            invalid.tracking_timeout = timeout;
            assert!(invalid.validate().is_err());
            invalid = parameters.clone();
            invalid.visual_tracking_timeout = timeout;
            assert!(invalid.validate().is_err());
        }
    }
}

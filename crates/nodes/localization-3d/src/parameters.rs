use std::time::Duration;

use ros_z::Message;
use serde::{Deserialize, Serialize};

const MAX_TRACKING_TIMEOUT: Duration = Duration::from_secs(24 * 60 * 60);

#[derive(Clone, Debug, Deserialize, Serialize, Message)]
#[serde(deny_unknown_fields)]
pub struct KinematicOdometryNoise {
    /// Forward/lateral displacement noise floor, in metres.
    pub position_sigma: nalgebra::Vector2<f64>,
    /// Forward/lateral displacement variance growth, in m²/s.
    pub translation_variance_per_second: nalgebra::Vector2<f64>,
}

impl Default for KinematicOdometryNoise {
    fn default() -> Self {
        Self {
            position_sigma: nalgebra::Vector2::repeat(0.005),
            translation_variance_per_second: nalgebra::Vector2::repeat(0.001),
        }
    }
}

/// Runtime parameters for the 3D localization node.
#[derive(Clone, Debug, Deserialize, Serialize, Message)]
#[serde(deny_unknown_fields)]
pub struct Localization3dParameters {
    #[serde(default)]
    pub kinematic_odometry_noise: KinematicOdometryNoise,
    /// Translational white-noise-on-acceleration spectral density.
    pub accelerometer_process_noise_variance: f64,
    /// Pixel-noise variance used to set isotropic angular noise for accepted visual
    /// associations: sigma_theta = sqrt(variance) / sqrt(fx * fy), using the frame's
    /// fixed calibration. This is a near-axis approximation, not a pixel likelihood.
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
        let noise = &self.kinematic_odometry_noise;
        if !noise
            .position_sigma
            .iter()
            .chain(noise.translation_variance_per_second.iter())
            .all(|v| valid_scale(*v) && (*v * *v).is_finite())
        {
            return Err("kinematic odometry noise must be finite and > 0".into());
        }
        if !valid_scale(self.accelerometer_process_noise_variance) {
            return Err("accelerometer_process_noise_variance must be finite and > 0".to_string());
        }
        if !valid_scale(self.visual_feature_noise_variance) {
            return Err("visual_feature_noise_variance must be finite and > 0".to_string());
        }
        if !valid_scale(self.field_containment_sigma) {
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
        Ok(())
    }
}

fn valid_scale(value: f64) -> bool {
    value.is_finite() && value >= f64::MIN_POSITIVE.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_tuning_rejects_unusable_whitening_and_timer_ranges_before_commit() {
        let parameters = Localization3dParameters {
            kinematic_odometry_noise: Default::default(),
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

use std::time::Duration;

use ros_z::Message;
use serde::{Deserialize, Serialize};

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

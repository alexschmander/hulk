use std::time::Duration;

use ros_z::Message;
use serde::{Deserialize, Serialize};

use crate::global_association::GlobalAssociationConfig as GlobalLocalizerParameters;

#[derive(Clone, Debug, Deserialize, Serialize, Message)]
#[serde(default, deny_unknown_fields)]
pub struct FieldMarkAssociationParameters {
    /// Shared projection uncertainty/search budgets and global geometric gates.
    pub global_localizer: GlobalLocalizerParameters,
    pub tracking: TrackingAssociationParameters,
}

impl Default for FieldMarkAssociationParameters {
    fn default() -> Self {
        Self {
            global_localizer: GlobalLocalizerParameters::default(),
            tracking: TrackingAssociationParameters::default(),
        }
    }
}

impl FieldMarkAssociationParameters {
    pub fn validate(&self) -> std::result::Result<(), String> {
        self.global_localizer.validate()?;
        self.tracking.validate()?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Message)]
#[serde(default, deny_unknown_fields)]
pub struct TrackingAssociationParameters {
    /// Maximum absolute entry of covariance minus its transpose.
    pub covariance_symmetry_tolerance: f32,
    /// Allowed negative eigenvalue magnitude from covariance roundoff.
    pub covariance_psd_tolerance: f32,
    /// Bounded covariance eigensolver iterations; zero (unbounded) is rejected.
    pub covariance_eigen_max_iterations: usize,
    /// Pixel residual ceiling for winning edges. Plausible rivals beyond it still enter scoring.
    pub max_pixel_distance: f32,
    /// Additional position sigma per second without a successful solve.
    pub position_sigma_per_second: f32,
    /// Additional angular sigma per second without a successful solve.
    pub yaw_sigma_per_second: f32,
    /// Validity horizon since the last successful solve; older predictions reject without fallback.
    pub max_age: Duration,
    /// For 3..=5 features, required best/rival joint Gaussian likelihood ratio;
    /// the log-likelihood gap must exceed ln(score_ratio).
    /// For >5 features, marginal-assignment heuristic: removing any winning edge must lose
    /// more than 1 - 1/score_ratio of that edge's normalized Gaussian benefit.
    pub score_ratio: f32,
}

impl Default for TrackingAssociationParameters {
    fn default() -> Self {
        Self {
            covariance_symmetry_tolerance: 1.0e-5,
            covariance_psd_tolerance: 1.0e-6,
            covariance_eigen_max_iterations: 64,
            max_pixel_distance: 80.0,
            position_sigma_per_second: 0.15,
            yaw_sigma_per_second: 0.1,
            max_age: Duration::from_secs(5),
            score_ratio: 1.05,
        }
    }
}

impl TrackingAssociationParameters {
    fn validate(&self) -> Result<(), String> {
        for value in [
            self.covariance_symmetry_tolerance,
            self.covariance_psd_tolerance,
            self.max_pixel_distance,
            self.position_sigma_per_second,
            self.yaw_sigma_per_second,
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(
                    "tracking distance, uncertainty rates and tolerances must be finite and > 0"
                        .into(),
                );
            }
        }
        if self.covariance_eigen_max_iterations == 0
            || self.max_age.is_zero()
            || !self.score_ratio.is_finite()
            || self.score_ratio <= 1.0
        {
            return Err("invalid tracking eigensolver iteration limit, age or score ratio".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_detection_limits_remain_compatible() {
        let parameters: FieldMarkAssociationParameters = serde_json::from_str("{}").unwrap();
        parameters.validate().unwrap();
        assert_eq!(parameters.global_localizer.max_retained_detections, 32);
        let mut config = parameters.global_localizer;
        config.max_retained_detections = 64;
        config.max_work = 2_000_000;
        config.validate().unwrap();
        config.min_inliers = 2;
        assert!(config.validate().is_err());
        config.min_inliers = 3;
        config.max_input_detections = 63;
        assert!(config.validate().is_err());
    }
}

use std::time::Duration;

use ros_z::Message;
use serde::{Deserialize, Serialize};

use crate::global_association::GlobalAssociationConfig as GlobalLocalizerParameters;

#[derive(Clone, Debug, Deserialize, Serialize, Message)]
#[serde(deny_unknown_fields)]
pub struct FieldMarkAssociationParameters {
    /// Shared projection uncertainty/search budgets and global geometric gates.
    pub global_localizer: GlobalLocalizerParameters,
    pub tracking: TrackingAssociationParameters,
    /// Maximum node-side timestamp difference for association geometry.
    ///
    /// The direct API expects geometry already sampled at the detection time.
    pub max_pose_hint_age: Duration,
}

impl Default for FieldMarkAssociationParameters {
    fn default() -> Self {
        Self {
            global_localizer: GlobalLocalizerParameters::default(),
            tracking: TrackingAssociationParameters::default(),
            max_pose_hint_age: Duration::from_millis(250),
        }
    }
}

impl FieldMarkAssociationParameters {
    pub(crate) fn validate(&self) -> std::result::Result<(), String> {
        self.global_localizer.validate()?;
        self.tracking.validate()?;
        if self.max_pose_hint_age.is_zero() {
            return Err("max_pose_hint_age must be > 0".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Message)]
#[serde(deny_unknown_fields)]
pub struct TrackingAssociationParameters {
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
            self.max_pixel_distance,
            self.position_sigma_per_second,
            self.yaw_sigma_per_second,
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(
                    "tracking distance and uncertainty rates must be finite and > 0".into(),
                );
            }
        }
        if self.max_age.is_zero() || !self.score_ratio.is_finite() || self.score_ratio <= 1.0 {
            return Err("invalid tracking age or score ratio".into());
        }
        Ok(())
    }
}

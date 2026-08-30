use ros_z::Message;
use serde::{Deserialize, Serialize};
use types::visual_localization::{
    MAX_CERTIFIED_VISUAL_ASSOCIATIONS, MIN_CERTIFIED_VISUAL_ASSOCIATIONS,
};

use super::map::FIELD_LANDMARK_COUNT;

pub(crate) const GLOBAL_LOCALIZER_MAX_DETECTIONS: usize = MAX_CERTIFIED_VISUAL_ASSOCIATIONS;
pub(crate) const GLOBAL_LOCALIZER_MAX_PROPOSALS: usize = 8_192;
pub(crate) const GLOBAL_LOCALIZER_MAX_INPUT_DETECTIONS: usize = 128;

/// Configuration for stateless global field-feature association.
///
/// These parameters gate detections, candidate associations, and uniqueness certification before
/// any visual associations are exposed to the backend as fixed reprojection factors.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize, Message)]
#[serde(deny_unknown_fields)]
pub struct GlobalAssociationConfig {
    /// Minimum accepted fixed associations for any published result.
    pub min_inliers: usize,
    /// Minimum detector confidence for all supported detections.
    pub confidence_threshold: f32,
    /// Minimum normalized-ray baseline between two detections.
    pub min_detection_baseline: f32,
    /// Minimum metric baseline between two map landmarks.
    pub min_map_baseline: f32,
    /// Lower plausible camera-height scale for global association hypotheses.
    pub height_min: f32,
    /// Upper plausible camera-height scale for global association hypotheses.
    pub height_max: f32,
    /// Maximum metric distance from a predicted landmark to an accepted same-class landmark.
    pub association_gate: f32,
    /// Isotropic detector uncertainty in pixels.
    pub detection_pixel_sigma: f32,
    /// Isotropic roll/pitch uncertainty around the camera-matrix orientation in radians.
    pub imu_tilt_sigma: f32,
    /// Maximum squared Mahalanobis distance for an accepted association.
    pub mahalanobis_gate: f32,
    /// Maximum RMS metric association residual for an accepted candidate.
    pub rms_threshold: f32,
    /// Minimum best-to-second-best non-equivalent score ratio.
    pub score_ratio: f32,
}

impl Default for GlobalAssociationConfig {
    fn default() -> Self {
        Self {
            min_inliers: 5,
            confidence_threshold: 0.35,
            min_detection_baseline: 0.1,
            min_map_baseline: 0.25,
            height_min: 0.2,
            height_max: 1.0,
            association_gate: 0.6,
            detection_pixel_sigma: 2.0,
            imu_tilt_sigma: 0.02,
            mahalanobis_gate: 9.21,
            rms_threshold: 0.45,
            score_ratio: 1.05,
        }
    }
}

impl GlobalAssociationConfig {
    /// Validates that global-localizer parameters are finite, positive where required, and
    /// consistent with the solver's fixed detection cap.
    pub fn validate(&self) -> Result<(), String> {
        if self.min_inliers < MIN_CERTIFIED_VISUAL_ASSOCIATIONS {
            return Err(format!(
                "global_localizer.min_inliers must be at least \
                 {MIN_CERTIFIED_VISUAL_ASSOCIATIONS}"
            ));
        }
        if self.min_inliers > FIELD_LANDMARK_COUNT {
            return Err(format!(
                "global_localizer.min_inliers must be <= {FIELD_LANDMARK_COUNT} because the map \
                 contains only that many unique landmarks"
            ));
        }
        validate_confidence_threshold(
            self.confidence_threshold,
            "global_localizer.confidence_threshold must be finite and in [0, 1]",
        )?;
        validate_positive_f32(
            self.min_detection_baseline,
            "global_localizer.min_detection_baseline must be finite and > 0",
        )?;
        validate_positive_f32(
            self.min_map_baseline,
            "global_localizer.min_map_baseline must be finite and > 0",
        )?;
        validate_positive_f32(
            self.height_min,
            "global_localizer.height_min must be finite and > 0",
        )?;
        validate_positive_f32(
            self.height_max,
            "global_localizer.height_max must be finite and > 0",
        )?;
        if self.height_min > self.height_max {
            return Err("global_localizer.height_min must be <= height_max".to_string());
        }
        validate_positive_f32(
            self.association_gate,
            "global_localizer.association_gate must be finite and > 0",
        )?;
        validate_positive_f32(
            self.detection_pixel_sigma,
            "global_localizer.detection_pixel_sigma must be finite and > 0",
        )?;
        validate_positive_f32(
            self.imu_tilt_sigma,
            "global_localizer.imu_tilt_sigma must be finite and > 0",
        )?;
        validate_positive_f32(
            self.mahalanobis_gate,
            "global_localizer.mahalanobis_gate must be finite and > 0",
        )?;
        validate_positive_f32(
            self.rms_threshold,
            "global_localizer.rms_threshold must be finite and > 0",
        )?;
        if !self.score_ratio.is_finite() || self.score_ratio <= 1.0 {
            return Err("global_localizer.score_ratio must be finite and > 1".to_string());
        }
        Ok(())
    }
}

fn validate_positive_f32(value: f32, message: &str) -> Result<(), String> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(message.to_string())
    }
}

fn validate_confidence_threshold(value: f32, message: &str) -> Result<(), String> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(message.to_string())
    }
}

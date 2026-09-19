use ros_z::Message;
use serde::{Deserialize, Serialize};
use types::visual_localization::{
    MAX_CERTIFIED_VISUAL_ASSOCIATIONS, MIN_CERTIFIED_VISUAL_ASSOCIATIONS,
};

use crate::map::FIELD_LANDMARK_COUNT;

pub(crate) const GLOBAL_LOCALIZER_MAX_DETECTIONS: usize = MAX_CERTIFIED_VISUAL_ASSOCIATIONS;
pub(crate) const GLOBAL_LOCALIZER_MAX_INPUT_DETECTIONS: usize = 128;
pub(crate) const SEED_POOL_SIZE: usize = 8;

/// Gates for local-plane geometric invariants, not pose hypotheses.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize, Message)]
#[serde(deny_unknown_fields)]
pub struct GlobalAssociationConfig {
    pub min_inliers: usize,
    pub confidence_threshold: f32,
    /// Minimum seed edge length in meters.
    pub min_detection_baseline: f32,
    /// Pixel measurement/calibration floor, shared with image-space tracking.
    pub detection_pixel_sigma: f32,
    /// Shared local roll/pitch uncertainty in radians.
    pub imu_tilt_sigma: f32,
    /// Shared measured camera-height uncertainty in meters.
    pub height_sigma: f32,
    /// Squared bound: global invariant gates use its square root; tracking uses it
    /// directly for 2D residuals and preserves its chi-square tail probability in
    /// joint residuals. This is not an assignment-certification probability.
    pub mahalanobis_gate: f32,
    /// Additional metric tolerance for map/detector systematic error.
    pub geometric_tolerance: f32,
    /// Per-call search operations. Global search charges visits, attempts and pair/contrast checks;
    /// joint tracking (3..=5 features) charges every candidate extension, including duplicate IDs.
    /// Each complete joint assignment requires at most a 10D Gaussian evaluation.
    /// Exhaustion rejects the frame, even after finding a candidate; no truncated winner is emitted.
    /// Preprocessing is separately bounded by 128 inputs, 32 retained detections and 56 seed triples.
    /// Post-seed chirality and three contrasts per non-anchor detection are precomputed once.
    pub max_work: usize,
}

impl Default for GlobalAssociationConfig {
    fn default() -> Self {
        Self {
            min_inliers: 3,
            confidence_threshold: 0.35,
            min_detection_baseline: 0.25,
            detection_pixel_sigma: 2.0,
            imu_tilt_sigma: 0.02,
            height_sigma: 0.02,
            mahalanobis_gate: 9.21,
            geometric_tolerance: 0.03,
            max_work: 100_000,
        }
    }
}

impl GlobalAssociationConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(MIN_CERTIFIED_VISUAL_ASSOCIATIONS..=FIELD_LANDMARK_COUNT).contains(&self.min_inliers) {
            return Err("global_localizer.min_inliers must be between 3 and the map size".into());
        }
        if !self.confidence_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.confidence_threshold)
        {
            return Err("global_localizer.confidence_threshold must be in [0, 1]".into());
        }
        for (name, value) in [
            ("min_detection_baseline", self.min_detection_baseline),
            ("detection_pixel_sigma", self.detection_pixel_sigma),
            ("imu_tilt_sigma", self.imu_tilt_sigma),
            ("height_sigma", self.height_sigma),
            ("mahalanobis_gate", self.mahalanobis_gate),
            ("geometric_tolerance", self.geometric_tolerance),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("global_localizer.{name} must be finite and > 0"));
            }
        }
        if !(1..=1_000_000).contains(&self.max_work) {
            return Err("global_localizer.max_work must be in [1, 1000000]".into());
        }
        Ok(())
    }
}

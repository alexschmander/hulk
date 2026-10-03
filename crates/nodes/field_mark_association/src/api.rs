use coordinate_systems::{Camera, Ground, Robot};
use linear_algebra::{Isometry3, Rotation3};
use projection::intrinsic::Intrinsic;
pub use types::localization::HeadingConstraint;
use types::{
    field_dimensions::FieldDimensions,
    visual_localization_next::{
        FieldMarkAssociation, GlobalLocalizationDebug, VisualAssociationSource,
    },
};

use crate::{DetectedVisualFeatures, GlobalLocalizerParameters, global_association::solver};

#[derive(Clone, Copy)]
pub struct GlobalAssociationInput<'a> {
    pub visual_features: &'a DetectedVisualFeatures,
    /// Exposure-time leveling rotation, with robot heading removed.
    pub robot_to_ground: Rotation3<Robot, Ground>,
    pub robot_to_camera: Isometry3<Robot, Camera>,
    pub camera_intrinsic: Intrinsic,
    pub field_dimensions: &'a FieldDimensions,
    pub parameters: &'a GlobalLocalizerParameters,
    pub heading: Option<HeadingConstraint>,
}

/// Fixed correspondences only; pose estimation belongs to localization.
#[derive(Default)]
pub struct AssociationResult {
    pub associations: Vec<FieldMarkAssociation>,
    pub source: VisualAssociationSource,
    pub debug: Option<GlobalLocalizationDebug>,
}

/// Reject ambiguous assignments and exhausted budgets. Heading constrains oriented
/// assignments; without it, uniqueness is modulo field half-turn symmetry.
pub fn associate_global_visual_features(input: GlobalAssociationInput<'_>) -> AssociationResult {
    solver::associate(input).unwrap_or_default()
}

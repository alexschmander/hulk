use coordinate_systems::{Camera, Ground, Robot};
use linear_algebra::{Isometry3, Rotation3};
use projection::intrinsic::Intrinsic;
use ros_z::time::Time;
pub use types::localization::HeadingConstraint;
use types::{
    field_dimensions::FieldDimensions,
    visual_localization_next::{
        AssociationGeometry, FieldMarkAssociation, GlobalLocalizationDebug, VisualAssociationSource,
    },
};

use crate::{
    DetectedVisualFeatures, FieldMarkAssociationParameters, GlobalLocalizerParameters,
    global_association::solver, tracking,
};

/// Pose prediction for an established Tracking state.
#[derive(Clone, Copy)]
pub struct TrackingAssociationInput<'a> {
    pub visual_features: &'a DetectedVisualFeatures,
    pub robot_to_camera: Isometry3<Robot, Camera>,
    pub geometry: &'a AssociationGeometry,
    pub camera_intrinsic: Intrinsic,
    pub field_dimensions: &'a FieldDimensions,
    pub time: Time,
}

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

/// Predict map landmarks with the supplied pose prior, independent of node lifecycle.
/// Sparse frames use the joint pixel Gaussian; larger frames use marginal assignment.
pub fn associate_tracking_visual_features(
    input: TrackingAssociationInput<'_>,
    parameters: &FieldMarkAssociationParameters,
) -> AssociationResult {
    if parameters.validate().is_err() {
        return AssociationResult::default();
    }
    tracking::associate(input, parameters).unwrap_or_default()
}

/// Reject ambiguous assignments and exhausted budgets. Heading constrains oriented
/// assignments; without it, uniqueness is modulo field half-turn symmetry.
pub fn associate_global_visual_features(input: GlobalAssociationInput<'_>) -> AssociationResult {
    solver::associate(input).unwrap_or_default()
}

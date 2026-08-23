use coordinate_systems::{Camera, Field, Robot};
use linear_algebra::{Isometry3, Point2};
use projection::camera_matrix::CameraMatrix;

mod api;
mod debug;
mod features;
mod frame_processing;
mod global_association;
mod node;
mod parameters;
mod tracking;

pub use api::{
    GlobalVisualLocalization, associate_visual_features, localize_global_visual_features,
    localize_global_visual_features_detailed_debug,
};
pub use features::{
    DetectedVisualFeature, DetectedVisualFeatures, find_detected_goalposts,
    find_detected_visual_features,
};
pub use global_association::{
    GlobalAssociationConfig as GlobalLocalizerParameters, GlobalLocalizationDebugAssociation,
    GlobalLocalizationDebugDetection, GlobalLocalizationDebugProjection,
    GlobalLocalizationDetailedDebug, GlobalLocalizationDetailedStatus, GlobalLocalizationScore,
    PoseHintAssociationConfig as PoseHintAssociationParameters, VisualFeatureClass,
};
pub use node::{run, run_boxed};
pub use parameters::FieldMarkAssociationParameters;
pub use tracking::FieldMarkAssociationState;
pub use types::visual_localization::{
    FieldMarkAssociation, FieldMarkAssociationSource,
    FieldMarkAssociationSource as FieldMarkAssociationKind, GLOBAL_LOCALIZATION_DEBUG_TOPIC,
    GlobalLocalizationDebug, GlobalLocalizationDebugStatus, VisualLocalizationFrame,
    VisualLocalizationFrame as FieldMarkAssociations,
};

/// A semantic point landmark used by the production global-localization map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldFeatureLandmark {
    /// Semantic class used by production global association.
    pub class: VisualFeatureClass,
    /// Landmark position on the field plane.
    pub position: Point2<Field>,
}

/// Returns the exact semantic point landmarks used by production field-mark association.
pub fn field_feature_landmarks(
    field_dimensions: &types::field_dimensions::FieldDimensions,
) -> Vec<FieldFeatureLandmark> {
    global_association::candidate_points(field_dimensions)
        .into_iter()
        .map(|(class, position)| FieldFeatureLandmark { class, position })
        .collect()
}

pub(crate) fn robot_to_camera(camera_matrix: &CameraMatrix) -> Isometry3<Robot, Camera> {
    camera_matrix.head_to_camera * camera_matrix.robot_to_head
}

#[cfg(test)]
mod public_map_tests {
    use super::*;

    #[test]
    fn public_landmarks_match_the_production_map() {
        let landmarks =
            field_feature_landmarks(&types::field_dimensions::FieldDimensions::SPL_2025);

        assert_eq!(landmarks.len(), 31);
        assert_eq!(
            landmarks
                .iter()
                .filter(|landmark| landmark.class == VisualFeatureClass::GoalPost)
                .count(),
            4
        );
    }
}

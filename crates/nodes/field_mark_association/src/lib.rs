use coordinate_systems::Field;
use linear_algebra::Point2;

mod api;
mod features;
mod frame_processing;
mod global_association;
mod map;
mod node;
mod parameters;
mod tracking;

pub use api::{
    AssociationInput, AssociationResult, GlobalAssociationInput, associate_global_visual_features,
    associate_visual_features,
};
pub use features::{
    DetectedVisualFeature, DetectedVisualFeatures, VisualFeatureClass, find_detected_goalposts,
    find_detected_visual_features, raw_detections,
};
pub use global_association::GlobalAssociationConfig as GlobalLocalizerParameters;
pub use node::{run, run_boxed};
pub use parameters::{FieldMarkAssociationParameters, TrackingAssociationParameters};
pub use types::visual_localization::{
    AssociationGeometry, FieldMarkAssociation, GLOBAL_LOCALIZATION_DEBUG_TOPIC,
    GlobalLocalizationDebug, VisualLocalizationFrame,
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
    map::candidate_points(field_dimensions)
        .into_iter()
        .map(|(class, position)| FieldFeatureLandmark { class, position })
        .collect()
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

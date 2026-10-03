pub mod legacy;
pub use legacy::{run, run_boxed};

pub mod api;
mod features;
mod frame_processing;
pub mod global_association;
pub mod map;
pub mod node;
mod parameters;
mod tracking;

pub use api::{
    AssociationInput, AssociationResult, GlobalAssociationInput, HeadingConstraint,
    TrackingAssociationInput, associate_global_visual_features, associate_tracking_visual_features,
    associate_visual_features,
};
pub use features::{
    DetectedVisualFeature, DetectedVisualFeatures, VisualFeatureClass,
    find_detected_visual_features, raw_detections,
};
pub use global_association::GlobalAssociationConfig as GlobalLocalizerParameters;
pub use map::candidate_points;
pub use parameters::{
    AssociationCapacities, FieldMarkAssociationParameters, TrackingAssociationParameters,
};

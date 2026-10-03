pub mod legacy;
pub use legacy::{run, run_boxed};

mod features;
pub mod global_association;
pub mod map;

pub use features::{
    DetectedVisualFeature, DetectedVisualFeatures, VisualFeatureClass,
    find_detected_visual_features, raw_detections,
};
pub use global_association::GlobalAssociationConfig as GlobalLocalizerParameters;
pub use map::candidate_points;

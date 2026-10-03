use coordinate_systems::{Camera, Field, Pixel, Robot};
use linear_algebra::{Isometry3, Point2, Point3};
use ros_z::Message;
use serde::{Deserialize, Serialize};

/// How associations were obtained. Global recovery must preserve the existing
/// field-symmetry branch before its correspondences enter localization.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, Message, PartialEq, Eq)]
pub enum VisualAssociationSource {
    #[default]
    Tracking,
    Global,
}

#[derive(Debug, Clone, Serialize, Deserialize, Message)]
pub struct VisualLocalizationFrame {
    pub epoch: u64,
    pub generation: u64,
    pub source: VisualAssociationSource,
    pub robot_to_camera: Isometry3<Robot, Camera>,
    pub camera_intrinsic: projection::intrinsic::Intrinsic,
    pub associations: Vec<FieldMarkAssociation>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Message)]
pub struct FieldMarkAssociation {
    pub detection: Point2<Pixel>,
    pub field_point: Point3<Field>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Message)]
pub struct GlobalLocalizationDebug {
    pub association_count: usize,
    pub pairwise_distance_rms: f32,
}

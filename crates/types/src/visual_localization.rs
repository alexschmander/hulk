use coordinate_systems::{Camera, Field, Local, Pixel, Robot};
use linear_algebra::{Isometry2, Isometry3, Point2, Point3};
use ros_z::Message;
use serde::{Deserialize, Serialize};

pub const ASSOCIATION_GEOMETRY_TOPIC: &str = "localization/association_geometry";
pub const LOCALIZATION_POSE_3D_TOPIC: &str = "localization/pose_3d";
pub const VISUAL_LOCALIZATION_TOPIC: &str = "field_mark_association/visual_localization_local";
pub const GLOBAL_LOCALIZATION_DEBUG_TOPIC: &str = "debug/global_localization";
pub const MIN_CERTIFIED_VISUAL_ASSOCIATIONS: usize = 3;
pub const MAX_CERTIFIED_VISUAL_ASSOCIATIONS: usize = 32;

#[derive(Debug, Clone, Serialize, Deserialize, Message)]
pub struct AssociationGeometry {
    pub epoch: u64,
    /// Authoritative solver selection, sampled together with this geometry.
    /// Its estimate/covariance remain anchored at `last_successful_solve`, while
    /// `robot_to_local` can be propagated to a newer measurement timestamp.
    pub state: crate::localization::LocalizationState3D,
    pub robot_to_local: Isometry3<Robot, Local>,
    pub local_to_field: Option<Isometry2<Local, Field>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Message)]
pub struct VisualLocalizationFrame {
    pub epoch: u64,
    pub robot_to_camera: Isometry3<Robot, Camera>,
    /// Local geometry used for association and localization-owned bootstrap seeding.
    pub robot_to_local: Isometry3<Robot, Local>,
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

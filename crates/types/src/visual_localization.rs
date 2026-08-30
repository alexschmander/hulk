use coordinate_systems::{Camera, Field, Pixel, Robot};
use linear_algebra::{Isometry3, Point2, Point3};
use ros_z::Message;
use serde::{Deserialize, Serialize};

pub const ASSOCIATION_POSE_HINT_TOPIC: &str = "localization/association_pose_hint";
pub const LOCALIZATION_POSE_3D_TOPIC: &str = "localization/pose_3d";
pub const VISUAL_LOCALIZATION_TOPIC: &str = "field_mark_association/visual_localization";
pub const GLOBAL_LOCALIZATION_DEBUG_TOPIC: &str = "debug/global_localization";
pub const MIN_CERTIFIED_VISUAL_ASSOCIATIONS: usize = 3;
pub const MAX_CERTIFIED_VISUAL_ASSOCIATIONS: usize = 32;

#[derive(Debug, Clone, Serialize, Deserialize, Message)]
pub struct AssociationPoseHint {
    pub robot_to_field: Isometry3<Robot, Field>,
    pub source: AssociationPoseHintSource,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize, Message)]
pub enum AssociationPoseHintSource {
    BackendBranch,
    LiveVisualOdometry,
    StartupPrior,
}

#[derive(Debug, Clone, Serialize, Deserialize, Message)]
pub struct VisualLocalizationFrame {
    pub robot_to_camera: Isometry3<Robot, Camera>,
    pub associations: Vec<FieldMarkAssociation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Message)]
pub struct FieldMarkAssociation {
    pub detection: Point2<Pixel>,
    pub field_point: Point3<Field>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Message)]
pub struct GlobalLocalizationDebug {
    pub inliers: usize,
    pub candidate_score: f32,
    pub metric_rms_residual: f32,
}

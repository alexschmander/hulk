use coordinate_systems::{Camera, Local, Robot};
use linear_algebra::Isometry3;
use projection::intrinsic::Intrinsic;
use ros_z::time::Time;
use types::{
    field_dimensions::FieldDimensions,
    localization::LocalizationState3D,
    visual_localization::{AssociationGeometry, FieldMarkAssociation, GlobalLocalizationDebug},
};

use crate::{
    DetectedVisualFeatures, FieldMarkAssociationParameters, GlobalLocalizerParameters,
    global_association::solver, tracking,
};

/// All geometry is sampled at the detection time. Epoch freshness belongs to the caller.
#[derive(Clone, Copy)]
pub struct AssociationInput<'a> {
    pub visual_features: &'a DetectedVisualFeatures,
    pub robot_to_camera: Isometry3<Robot, Camera>,
    pub geometry: &'a AssociationGeometry,
    pub camera_intrinsic: Intrinsic,
    pub field_dimensions: &'a FieldDimensions,
    pub time: Time,
}

/// Stateless global matching inputs; no field pose, lifecycle state or history is consulted.
#[derive(Clone, Copy)]
pub struct GlobalAssociationInput<'a> {
    pub visual_features: &'a DetectedVisualFeatures,
    /// IMU-derived attitude in a leveled frame. Translation is ignored at startup.
    pub robot_to_local: Isometry3<Robot, Local>,
    pub robot_to_camera: Isometry3<Robot, Camera>,
    pub camera_intrinsic: Intrinsic,
    pub field_dimensions: &'a FieldDimensions,
    pub parameters: &'a GlobalLocalizerParameters,
}

/// Fixed correspondences and diagnostics only. Pose estimation belongs to localization.
#[derive(Default)]
pub struct AssociationResult {
    pub associations: Vec<FieldMarkAssociation>,
    /// Global-only metric diagnostics. Image-space tracking leaves this unset.
    pub debug: Option<GlobalLocalizationDebug>,
}

/// Stateless dispatch by `geometry.state`; previous calls never influence association.
/// Tracking and LostTrack require `geometry.local_to_field` and never fall back to global matching.
/// They project map landmarks into the image, carrying the last solve's full right-tangent covariance.
/// Frames with 3..=5 features compare every gated distinct assignment using the joint pixel Gaussian,
/// including shared anchor/process correlations and uncertainty volume. The full residual keeps
/// the configured 2D gate's tail probability; the best/rival likelihood ratio must exceed `score_ratio`.
/// Every retained feature must match; no subset is silently certified.
/// Search uses `global_localizer.max_work`; exhaustion rejects even after finding a candidate.
/// Frames with >5 features retain the marginal-assignment heuristic, not joint certification.
/// Hard pixel limits only certify winners, after comparing all rivals.
pub fn associate_visual_features(
    input: AssociationInput<'_>,
    parameters: &FieldMarkAssociationParameters,
) -> AssociationResult {
    if parameters.validate().is_err() {
        return AssociationResult::default();
    }
    match input.geometry.state {
        LocalizationState3D::Startup => associate_global_visual_features(GlobalAssociationInput {
            visual_features: input.visual_features,
            robot_to_local: input.geometry.robot_to_local,
            robot_to_camera: input.robot_to_camera,
            camera_intrinsic: input.camera_intrinsic,
            field_dimensions: input.field_dimensions,
            parameters: &parameters.global_localizer,
        }),
        LocalizationState3D::Tracking { .. } | LocalizationState3D::LostTrack { .. } => {
            tracking::associate(input, parameters).unwrap_or_default()
        }
    }
}

/// Geometric matching modulo the field's half-turn symmetry, without using a global pose.
///
/// The representative is chosen by landmark coordinates, not the robot's half. Localization
/// seeding must resolve the remaining half-turn. No state, alignment, epoch or time is needed.
/// Every retained detection must match; no subset is silently certified when a retained outlier exists.
/// Startup fits camera height, planar translation and yaw from unit-height rays using IMU tilt.
/// A non-collinear seed is required. Localizer translation and height are never consulted.
/// Work-budget exhaustion rejects rather than returning a candidate with unproven uniqueness.
pub fn associate_global_visual_features(input: GlobalAssociationInput<'_>) -> AssociationResult {
    solver::associate(input).unwrap_or_default()
}

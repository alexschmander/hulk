use coordinate_systems::{Field, Robot};
use linear_algebra::Isometry3;
use projection::camera_matrix::CameraMatrix;
use types::{field_dimensions::FieldDimensions, visual_localization::FieldMarkAssociation};

use crate::{
    features::DetectedVisualFeatures,
    global_association::{
        GlobalAssociationConfig as GlobalLocalizerParameters, GlobalAssociationResult,
        GlobalLocalizationInput, SolverWorkspace,
    },
    robot_to_camera,
};

/// Result of running visual global localization on one object-detection frame.
pub struct GlobalVisualLocalization {
    /// Debug payload for the best visual global localization result, if any.
    pub debug: Option<types::visual_localization::GlobalLocalizationDebug>,
    /// Globally certified fixed associations.
    pub associations: Vec<FieldMarkAssociation>,
}

/// Reusable stateless global localizer with retained scratch storage.
#[derive(Default)]
pub struct GlobalVisualLocalizer {
    workspace: SolverWorkspace,
}

impl GlobalVisualLocalizer {
    /// Localizes one frame. Previous frames do not influence the result.
    pub fn localize(
        &mut self,
        visual_features: &DetectedVisualFeatures,
        camera_matrix: &CameraMatrix,
        field_dimensions: &FieldDimensions,
        pose_hint: Option<Isometry3<Robot, Field>>,
        parameters: &GlobalLocalizerParameters,
    ) -> GlobalVisualLocalization {
        self.localize_with_debug(
            visual_features,
            camera_matrix,
            field_dimensions,
            pose_hint,
            parameters,
            true,
        )
    }

    pub(crate) fn localize_with_debug(
        &mut self,
        visual_features: &DetectedVisualFeatures,
        camera_matrix: &CameraMatrix,
        field_dimensions: &FieldDimensions,
        pose_hint: Option<Isometry3<Robot, Field>>,
        parameters: &GlobalLocalizerParameters,
        include_debug: bool,
    ) -> GlobalVisualLocalization {
        let result = self.workspace.solve(
            GlobalLocalizationInput {
                visual_features,
                field_dimensions,
                ground_to_robot: camera_matrix.ground_to_robot,
                robot_to_camera: robot_to_camera(camera_matrix),
                camera_intrinsic: camera_matrix.intrinsics,
                pose_hint,
            },
            *parameters,
        );
        localization_result(result, include_debug)
    }
}

/// Runs global localization and returns debug data plus backend-safe associations.
///
/// `pose_hint` is used only to select one representative of an already-certified 180-degree field
/// symmetry. It never creates fallback associations or changes uniqueness certification.
pub fn localize_global_visual_features(
    visual_features: &DetectedVisualFeatures,
    camera_matrix: &CameraMatrix,
    field_dimensions: &FieldDimensions,
    pose_hint: Option<Isometry3<Robot, Field>>,
    parameters: &GlobalLocalizerParameters,
) -> GlobalVisualLocalization {
    GlobalVisualLocalizer::default().localize(
        visual_features,
        camera_matrix,
        field_dimensions,
        pose_hint,
        parameters,
    )
}

fn localization_result(
    result: Option<GlobalAssociationResult>,
    include_debug: bool,
) -> GlobalVisualLocalization {
    match result {
        Some(result) => GlobalVisualLocalization {
            debug: include_debug.then_some(result.debug),
            associations: result.associations,
        },
        None => GlobalVisualLocalization {
            debug: None,
            associations: Vec::new(),
        },
    }
}

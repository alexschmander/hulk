use coordinate_systems::{Camera, Field, Local, Robot};
use linear_algebra::{Isometry2, Isometry3};
use projection::intrinsic::Intrinsic;
use types::{field_dimensions::FieldDimensions, visual_localization::FieldMarkAssociation};

use crate::{
    features::DetectedVisualFeatures,
    global_association::{
        GlobalAssociationConfig as GlobalLocalizerParameters, GlobalAssociationResult,
        GlobalLocalizationInput, SolverWorkspace,
    },
};

/// Result of running visual global localization on one object-detection frame.
pub struct GlobalVisualLocalization {
    /// Debug payload for the best visual global localization result, if any.
    pub debug: Option<types::visual_localization::GlobalLocalizationDebug>,
    /// Globally certified fixed associations.
    pub associations: Vec<FieldMarkAssociation>,
    /// Certified planar alignment for the returned associations.
    pub local_to_field: Option<Isometry2<Local, Field>>,
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
        robot_to_camera: Isometry3<Robot, Camera>,
        robot_to_local: Isometry3<Robot, Local>,
        camera_intrinsic: Intrinsic,
        field_dimensions: &FieldDimensions,
        alignment_hint: Option<Isometry2<Local, Field>>,
        parameters: &GlobalLocalizerParameters,
    ) -> GlobalVisualLocalization {
        self.localize_with_debug(
            visual_features,
            robot_to_camera,
            robot_to_local,
            camera_intrinsic,
            field_dimensions,
            alignment_hint,
            parameters,
            true,
        )
    }

    pub(crate) fn localize_with_debug(
        &mut self,
        visual_features: &DetectedVisualFeatures,
        robot_to_camera: Isometry3<Robot, Camera>,
        robot_to_local: Isometry3<Robot, Local>,
        camera_intrinsic: Intrinsic,
        field_dimensions: &FieldDimensions,
        alignment_hint: Option<Isometry2<Local, Field>>,
        parameters: &GlobalLocalizerParameters,
        include_debug: bool,
    ) -> GlobalVisualLocalization {
        let result = self.workspace.solve(
            GlobalLocalizationInput {
                visual_features,
                field_dimensions,
                robot_to_camera,
                robot_to_local,
                camera_intrinsic,
                alignment_hint,
            },
            *parameters,
        );
        localization_result(result, include_debug)
    }
}

/// Runs global localization and returns debug data plus backend-safe associations.
///
/// `alignment_hint` is used only to select one representative of an already-certified 180-degree field
/// symmetry. It never creates fallback associations or changes uniqueness certification.
pub fn localize_global_visual_features(
    visual_features: &DetectedVisualFeatures,
    robot_to_camera: Isometry3<Robot, Camera>,
    robot_to_local: Isometry3<Robot, Local>,
    camera_intrinsic: Intrinsic,
    field_dimensions: &FieldDimensions,
    alignment_hint: Option<Isometry2<Local, Field>>,
    parameters: &GlobalLocalizerParameters,
) -> GlobalVisualLocalization {
    GlobalVisualLocalizer::default().localize(
        visual_features,
        robot_to_camera,
        robot_to_local,
        camera_intrinsic,
        field_dimensions,
        alignment_hint,
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
            local_to_field: Some(result.local_to_field),
        },
        None => GlobalVisualLocalization {
            debug: None,
            associations: Vec::new(),
            local_to_field: None,
        },
    }
}

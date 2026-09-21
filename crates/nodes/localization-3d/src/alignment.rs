use coordinate_systems::{Camera, Field, Local, Robot};
use linear_algebra::{IntoTransform, Isometry2, Isometry3, Point2, Vector3};
use localization_fagra::alignment::fit_ground_alignment;
use projection::intrinsic::Intrinsic;
use types::visual_localization::FieldMarkAssociation;

const MIN_REPROJECTION_DEPTH: f64 = 0.01;
const MIN_DETECTION_SEPARATION_PX: f32 = 1.0;
const MIN_LANDMARK_SEPARATION_M: f32 = 1.0e-4;

pub(crate) fn valid_visual_frame(
    frame: &types::visual_localization::VisualLocalizationFrame,
) -> bool {
    let associations = &frame.associations;
    (types::visual_localization::MIN_CERTIFIED_VISUAL_ASSOCIATIONS
        ..=types::visual_localization::MAX_CERTIFIED_VISUAL_ASSOCIATIONS)
        .contains(&associations.len())
        && frame.camera_intrinsic.is_valid()
        && frame
            .robot_to_camera
            .inner
            .to_homogeneous()
            .iter()
            .all(|value| value.is_finite())
        && associations.iter().all(|association| {
            association
                .detection
                .inner
                .iter()
                .chain(association.field_point.inner.iter())
                .all(|value| value.is_finite())
        })
        && associations.iter().enumerate().all(|(index, association)| {
            associations[index + 1..].iter().all(|other| {
                (association.detection - other.detection).inner.norm() > MIN_DETECTION_SEPARATION_PX
                    && (association.field_point - other.field_point).inner.norm()
                        > MIN_LANDMARK_SEPARATION_M
            })
        })
}

/// Seed planar field alignment by intersecting associated camera rays with local z=0.
pub(crate) fn seed_alignment(
    robot_to_local: Isometry3<Robot, Local>,
    robot_to_camera: Isometry3<Robot, Camera>,
    intrinsic: Intrinsic,
    associations: &mut [FieldMarkAssociation],
) -> Option<Isometry2<Local, Field>> {
    let camera_to_local = (robot_to_local * robot_to_camera.inverse())
        .inner
        .cast::<f64>()
        .framed_transform();
    let mut alignment = fit_ground_alignment(
        &camera_to_local,
        associations.iter().map(|association| {
            (
                Vector3::wrap(intrinsic.bearing(association.detection).inner.cast::<f64>()),
                Point2::wrap(
                    association
                        .field_point
                        .inner
                        .coords
                        .xy()
                        .cast::<f64>()
                        .into(),
                ),
            )
        }),
        MIN_REPROJECTION_DEPTH,
    )
    .ok()?
    .inner;
    let robot_xy = robot_to_local.inner.translation.vector.xy().cast::<f64>();
    if (alignment * nalgebra::Point2::from(robot_xy)).x > 0.0 {
        alignment =
            nalgebra::Isometry2::new(nalgebra::Vector2::zeros(), std::f64::consts::PI) * alignment;
        for association in associations {
            association.field_point.inner.x = -association.field_point.inner.x;
            association.field_point.inner.y = -association.field_point.inner.y;
        }
    }
    let alignment = alignment.cast::<f32>();
    alignment
        .to_homogeneous()
        .iter()
        .all(|value| value.is_finite())
        .then(|| alignment.framed_transform())
}

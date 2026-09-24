use coordinate_systems::{Camera, Field, Local, Robot};
use linear_algebra::{IntoTransform, Isometry2, Isometry3, Point2};
use localization_fagra::alignment::fit_ground_similarity;
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

/// Fit camera height and field alignment from IMU tilt and bearings.
/// The incoming local translation is deliberately ignored.
/// Startup canonicalizes the field half; recovery selects it from trusted heading.
pub(crate) fn seed_alignment(
    robot_to_local: Isometry3<Robot, Local>,
    robot_to_camera: Isometry3<Robot, Camera>,
    intrinsic: Intrinsic,
    associations: &mut [FieldMarkAssociation],
    canonicalize: bool,
) -> Option<(Isometry3<Robot, Local>, Isometry2<Local, Field>)> {
    let rotation = robot_to_local.inner.rotation.cast::<f64>();
    let camera_to_robot = robot_to_camera.inner.cast::<f64>().inverse();
    let camera_rotation = rotation * camera_to_robot.rotation;
    let mut points = Vec::with_capacity(associations.len());
    for a in associations.iter() {
        let ray = camera_rotation * intrinsic.bearing(a.detection).inner.cast::<f64>();
        if !ray.iter().all(|v| v.is_finite()) || ray.z >= -1e-6 {
            return None;
        }
        points.push((
            -ray.xy() / ray.z,
            Point2::wrap(a.field_point.inner.coords.xy().cast::<f64>().into()),
        ));
    }
    let (camera_alignment, height) = fit_ground_similarity(points.into_iter()).ok()?;
    let offset = rotation * camera_to_robot.translation.vector;
    let body_height = height - offset.z;
    if body_height <= 0.0 {
        return None;
    }
    let pose: Isometry3<Robot, Local> = nalgebra::Isometry3::from_parts(
        nalgebra::Translation3::new(0.0, 0.0, body_height),
        rotation,
    )
    .cast()
    .framed_transform();
    let mut alignment = nalgebra::Isometry2::from_parts(
        (camera_alignment.inner.translation.vector - camera_alignment.inner.rotation * offset.xy())
            .into(),
        camera_alignment.inner.rotation,
    );
    let framed_alignment: Isometry2<Local, Field, f64> = alignment.framed_transform();
    let field_to_camera = robot_to_camera.inner.cast::<f64>()
        * pose.inner.cast::<f64>().inverse()
        * framed_alignment.to_3d().inner.inverse();
    let mut squared = 0.0;
    for a in associations.iter() {
        let p = field_to_camera * a.field_point.inner.cast::<f64>();
        if p.z <= MIN_REPROJECTION_DEPTH {
            return None;
        }
        let pixel = nalgebra::Vector2::new(
            intrinsic.focals.x as f64 * p.x / p.z + intrinsic.optical_center.x() as f64,
            intrinsic.focals.y as f64 * p.y / p.z + intrinsic.optical_center.y() as f64,
        );
        squared += (pixel - a.detection.inner.coords.cast::<f64>()).norm_squared();
    }
    let rms = (squared / associations.len() as f64).sqrt();
    if !rms.is_finite() || rms > 10.0 {
        return None;
    }
    if canonicalize && alignment.translation.vector.x > 0.0 {
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
        .then(|| (pose, alignment.framed_transform()))
}

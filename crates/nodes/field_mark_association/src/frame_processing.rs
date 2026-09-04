use std::{future::ready, time::Duration};

use color_eyre::Result;
use coordinate_systems::{Camera, Field, Local, Robot};
use linear_algebra::{Isometry2, Isometry3};
use projection::camera_matrix::CameraMatrix;
use projection::intrinsic::Intrinsic;
use ros_z::{cache::Cache, parameter::NodeParameters, pubsub::Publisher, time::Time};
use ros_z_streams::FutureItem;
use types::{
    field_dimensions::FieldDimensions,
    object_detection::{Object, RobocupObjectLabel},
    primary_state::PrimaryState,
    time_wrapper::TimeWrapper,
    visual_localization::{AssociationGeometry, GlobalLocalizationDebug, VisualLocalizationFrame},
};

use crate::{
    GlobalVisualLocalization, GlobalVisualLocalizer, parameters::FieldMarkAssociationParameters,
    robot_to_camera,
};

const MAX_CAMERA_MATRIX_TIME_DISTANCE: Duration = Duration::from_millis(100);

type DetectedObjects = TimeWrapper<Vec<Object<RobocupObjectLabel>>>;
type DetectedObjectsItem<'a> = FutureItem<'a, (Option<DetectedObjects>,)>;

pub(crate) struct DetectionProcessingContext<'a> {
    pub(crate) association_solver: &'a mut GlobalVisualLocalizer,
    pub(crate) parameters: &'a NodeParameters<FieldMarkAssociationParameters>,
    pub(crate) camera_matrix_cache: &'a Cache<TimeWrapper<CameraMatrix>>,
    pub(crate) field_dimensions_cache: &'a Cache<FieldDimensions>,
    pub(crate) association_geometry_cache: &'a Cache<TimeWrapper<AssociationGeometry>>,
    pub(crate) primary_state_cache: &'a Cache<PrimaryState>,
    pub(crate) associations_publisher: &'a Publisher<TimeWrapper<VisualLocalizationFrame>>,
    pub(crate) global_localization_publisher: &'a Publisher<Option<GlobalLocalizationDebug>>,
}

struct PreparedDetectionFrame {
    image_time: Time,
    objects: Vec<Object<RobocupObjectLabel>>,
    robot_to_camera: Isometry3<Robot, Camera>,
    camera_intrinsic: Intrinsic,
    epoch: u64,
    robot_to_local: Isometry3<Robot, Local>,
    field_dimensions: FieldDimensions,
    alignment_hint: Option<Isometry2<Local, Field>>,
    parameters: FieldMarkAssociationParameters,
    include_debug: bool,
}

struct ProcessedDetectionFrame {
    image_time: Time,
    robot_to_camera: Isometry3<Robot, Camera>,
    epoch: u64,
    localization: GlobalVisualLocalization,
}

pub(crate) async fn process_detected_objects(
    item: DetectedObjectsItem<'_>,
    ctx: DetectionProcessingContext<'_>,
) -> Result<()> {
    for (image_time, (objects,)) in item.persistent {
        let Some(processed_frame) =
            tokio::task::block_in_place(|| -> Result<Option<ProcessedDetectionFrame>> {
                if association_is_damping(ctx.primary_state_cache) {
                    return Ok(None);
                }

                let Some(frame) = prepare_detection_frame(image_time, objects, &ctx) else {
                    return Ok(None);
                };
                let image_time = frame.image_time;
                let robot_to_camera = frame.robot_to_camera;
                let epoch = frame.epoch;

                let localization = associate_detection_frame(frame, ctx.association_solver)?;

                if association_is_damping(ctx.primary_state_cache)
                    || ctx
                        .association_geometry_cache
                        .get_latest()
                        .is_none_or(|geometry| geometry.inner.epoch != epoch)
                {
                    return Ok(None);
                }

                Ok(Some(ProcessedDetectionFrame {
                    image_time,
                    robot_to_camera,
                    epoch,
                    localization,
                }))
            })?
        else {
            continue;
        };

        publish_localization_frame(
            ctx.associations_publisher,
            ctx.global_localization_publisher,
            processed_frame,
        )
        .await?;
    }
    Ok(())
}

fn prepare_detection_frame(
    image_time: Time,
    objects: Option<DetectedObjects>,
    ctx: &DetectionProcessingContext<'_>,
) -> Option<PreparedDetectionFrame> {
    let camera_matrix = ctx.camera_matrix_cache.get_nearest(image_time)?;
    if !camera_matrix_is_fresh(&camera_matrix, image_time) {
        return None;
    }
    let field_dimensions = ctx.field_dimensions_cache.get_nearest(image_time)?;

    let parameters = ctx.parameters.snapshot().typed().clone();
    let camera_matrix = camera_matrix.inner.clone();
    let geometry =
        association_geometry_at(image_time, &parameters, ctx.association_geometry_cache)?;

    Some(PreparedDetectionFrame {
        image_time,
        objects: objects.map(|item| item.inner).unwrap_or_default(),
        robot_to_camera: robot_to_camera(&camera_matrix),
        camera_intrinsic: camera_matrix.intrinsics,
        epoch: geometry.epoch,
        robot_to_local: geometry.robot_to_local,
        field_dimensions: *field_dimensions.as_ref(),
        alignment_hint: geometry.local_to_field,
        parameters,
        include_debug: ctx.global_localization_publisher.has_subscribers(),
    })
}

fn association_geometry_at(
    image_time: Time,
    parameters: &FieldMarkAssociationParameters,
    geometry_cache: &Cache<TimeWrapper<AssociationGeometry>>,
) -> Option<AssociationGeometry> {
    let geometry =
        geometry_cache
            .get_nearest_with_stamp(image_time)
            .and_then(|(stamp, localization)| {
                if time_distance(stamp, image_time) > parameters.max_pose_hint_age {
                    return None;
                }
                Some(localization.inner.clone())
            })?;
    geometry_cache
        .get_latest()
        .is_some_and(|latest| latest.inner.epoch == geometry.epoch)
        .then_some(geometry)
}

fn associate_detection_frame(
    frame: PreparedDetectionFrame,
    solver: &mut GlobalVisualLocalizer,
) -> Result<GlobalVisualLocalization> {
    let visual_features = crate::find_detected_visual_features(&frame.objects);
    if visual_features.supported_feature_count() == 0 {
        return Ok(GlobalVisualLocalization {
            debug: None,
            associations: Vec::new(),
            local_to_field: None,
        });
    }

    Ok(solver.localize_with_debug(
        &visual_features,
        frame.robot_to_camera,
        frame.robot_to_local,
        frame.camera_intrinsic,
        &frame.field_dimensions,
        frame.alignment_hint,
        &frame.parameters.global_localizer,
        frame.include_debug,
    ))
}

async fn publish_localization_frame(
    associations_publisher: &Publisher<TimeWrapper<VisualLocalizationFrame>>,
    global_localization_publisher: &Publisher<Option<GlobalLocalizationDebug>>,
    processed_frame: ProcessedDetectionFrame,
) -> Result<()> {
    let debug = processed_frame.localization.debug.clone();
    global_localization_publisher
        .publish_if_subscribed(|| ready(debug))
        .await?;
    let Some(local_to_field) = processed_frame.localization.local_to_field else {
        return Ok(());
    };
    associations_publisher
        .publish(&TimeWrapper {
            time: processed_frame.image_time,
            inner: VisualLocalizationFrame {
                epoch: processed_frame.epoch,
                robot_to_camera: processed_frame.robot_to_camera,
                local_to_field,
                associations: processed_frame.localization.associations,
            },
        })
        .await?;
    Ok(())
}

pub(crate) fn association_is_damping(primary_state_cache: &Cache<PrimaryState>) -> bool {
    primary_state_cache
        .get_latest()
        .is_none_or(|state| *state == PrimaryState::Damping)
}

fn camera_matrix_is_fresh(camera_matrix: &TimeWrapper<CameraMatrix>, time: Time) -> bool {
    time_distance(camera_matrix.time, time) <= MAX_CAMERA_MATRIX_TIME_DISTANCE
}

fn time_distance(a: Time, b: Time) -> Duration {
    Duration::from_nanos(a.as_nanos().abs_diff(b.as_nanos()))
}

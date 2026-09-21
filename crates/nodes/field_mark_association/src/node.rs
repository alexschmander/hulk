use std::{future::Future, num::NonZeroUsize, pin::Pin, sync::Arc};

use color_eyre::Result;
use projection::camera_matrix::CameraMatrix;
use ros_z::{
    context::Context,
    parameter::NodeParametersExt,
    qos::{QosDurability, QosHistory, QosProfile},
};
use types::{
    field_dimensions::FieldDimensions,
    object_detection::{Object, RobocupObjectLabel},
    time_wrapper::TimeWrapper,
    visual_localization::{
        GLOBAL_LOCALIZATION_DEBUG_TOPIC, GlobalLocalizationDebug, VISUAL_LOCALIZATION_TOPIC,
        VisualLocalizationFrame,
    },
};

use crate::{
    frame_processing::{
        DetectionProcessingContext, keep_latest_detection, process_detected_objects,
    },
    parameters::FieldMarkAssociationParameters,
};

/// Starts the field-mark association node and erases the concrete future type for node runners.
pub fn run_boxed(ctx: Arc<Context>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
    Box::pin(run(ctx))
}

pub async fn run(ctx: Arc<Context>) -> Result<()> {
    let node = ctx.create_node("field_mark_association").build().await?;
    let parameters =
        node.bind_parameter_as::<FieldMarkAssociationParameters>("field_mark_association")?;
    parameters.add_validation_hook(FieldMarkAssociationParameters::validate)?;

    let camera_matrix_cache = node
        .subscriber::<TimeWrapper<CameraMatrix>>("camera_matrix")
        .cache(128)
        .with_stamp(|message| message.time)
        .build()
        .await?;

    let field_dimensions_cache = node
        .subscriber::<FieldDimensions>("field_dimensions")
        .qos(QosProfile {
            durability: QosDurability::TransientLocal,
            ..Default::default()
        })
        .cache(1)
        .build()
        .await?;

    let estimates = node
        .subscriber::<types::localization::LocalizationEstimate>(
            types::localization::LOCALIZATION_ESTIMATE_TOPIC,
        )
        .cache(128)
        .with_stamp(|message| message.time)
        .build()
        .await?;
    let status = node
        .subscriber::<types::localization::LocalizationStatus>(
            types::localization::LOCALIZATION_STATUS_TOPIC,
        )
        .qos(QosProfile {
            durability: QosDurability::TransientLocal,
            ..Default::default()
        })
        .cache(1)
        .build()
        .await?;

    // Consume only payloads, without joining the announcement protocol or its KeepAll queue.
    let detected_objects = node
        .subscriber::<TimeWrapper<Vec<Object<RobocupObjectLabel>>>>("detected_objects")
        .qos(QosProfile {
            history: QosHistory::KeepLast(NonZeroUsize::MIN),
            ..Default::default()
        })
        .build()
        .await?;

    let associations_publisher = node
        .publisher::<TimeWrapper<VisualLocalizationFrame>>(VISUAL_LOCALIZATION_TOPIC)
        .build()
        .await?;
    let global_localization_publisher = node
        .publisher::<Option<GlobalLocalizationDebug>>(GLOBAL_LOCALIZATION_DEBUG_TOPIC)
        .build()
        .await?;
    let processing_context = DetectionProcessingContext {
        parameters: &parameters,
        camera_matrix_cache: &camera_matrix_cache,
        field_dimensions_cache: &field_dimensions_cache,
        estimates: &estimates,
        status: &status,
        tracking_reference: std::sync::Mutex::new(None),
        associations_publisher: Arc::new(associations_publisher),
        global_localization_publisher: Arc::new(global_localization_publisher),
        clock: node.clock(),
    };
    let mut pending_frame = None;
    loop {
        // A completion may win select while a newer payload is already in the subscriber queue.
        // This loop is the sole receiver, so a ready queue cannot be drained by another task.
        if pending_frame.is_some() && detected_objects.is_ready() {
            keep_latest_detection(&mut pending_frame, detected_objects.recv().await?);
        }
        let objects = match pending_frame.take() {
            Some(frame) => frame,
            None => detected_objects.recv().await?,
        };
        let image_time = objects.time;
        let processing = process_detected_objects(objects, &processing_context);
        tokio::pin!(processing);
        // Keep receiving even while the solver or either publisher is waiting.
        loop {
            tokio::select! {
                result = &mut processing => {
                    result?;
                    break;
                }
                objects = detected_objects.recv() => {
                    let objects = objects?;
                    if objects.time > image_time {
                        keep_latest_detection(&mut pending_frame, objects);
                    }
                }
            }
        }
    }
}

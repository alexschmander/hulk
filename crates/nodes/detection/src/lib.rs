use std::{boxed::Box, future::Future, pin::Pin, sync::Arc, time::Duration};

use color_eyre::{
    Result,
    eyre::{bail, eyre},
};
use linear_algebra::point;
use ort::{
    ep::{CUDA, TensorRT},
    inputs,
    session::{Session, SessionOutputs, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use ros_z_streams::CreateAnnouncingPublisher;
use ros2::sensor_msgs::image::Image;

use ros_z::{prelude::*, qos::QosHistory};
use tokio::{task::block_in_place, time::Instant};
use types::{
    object_detection::{Object, ObjectFeature, RobocupObjectLabel},
    parameters::DetectionParameters,
    pose_detection::{FieldLine, Keypoint},
    time_wrapper::TimeWrapper,
};

pub const NUMBER_OF_DETECTIONS: usize = 300;

struct DetectionOutput {
    inference_duration: Duration,
    post_processing_duration: Duration,
    non_maximum_suppression_duration: Duration,
    detected_objects: Vec<Object<RobocupObjectLabel>>,
    detected_features: Vec<ObjectFeature>,
    detected_lines: Vec<FieldLine>,
}

struct ObjectCandidate {
    object: Object<RobocupObjectLabel>,
    anchor: Keypoint,
}

pub fn run_boxed(ctx: Arc<Context>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
    Box::pin(run(ctx))
}

async fn run(ctx: Arc<Context>) -> Result<()> {
    let node = ctx.create_node("detection").build().await?;

    let node_parameters = node.bind_parameter_as::<DetectionParameters>("detection")?;
    let mut parameter_receiver = node_parameters.subscribe();

    let image_sub = node
        .subscriber::<Image>("inputs/left_image")
        .qos(QosProfile {
            history: QosHistory::from_depth(1),
            ..Default::default()
        })
        .build()
        .await?;
    let inference_duration_pub = node
        .publisher::<Duration>("inference_duration")
        .build()
        .await?;
    let post_processing_duration_pub = node
        .publisher::<Duration>("post_processing_duration")
        .build()
        .await?;
    let non_maximum_suppression_duration_pub = node
        .publisher::<Duration>("non_maximum_suppression_duration")
        .build()
        .await?;
    let detected_objects_pub = node
        .announcing_publisher::<TimeWrapper<Vec<Object<RobocupObjectLabel>>>>("detected_objects")
        .await?;

    let detected_features_pub = node
        .announcing_publisher::<TimeWrapper<Vec<ObjectFeature>>>("detected_features")
        .await?;
    let detected_lines_pub = node
        .announcing_publisher::<TimeWrapper<Vec<FieldLine>>>("detected_lines")
        .await?;

    let initial_parameters_snapshot = node_parameters.snapshot();
    let parameters = initial_parameters_snapshot.typed();
    let model_path = parameters
        .neural_networks_folder
        .join(&parameters.model_name);

    let tensor_rt = TensorRT::default()
        .with_device_id(0)
        .with_fp16(true)
        .with_engine_cache(true)
        .with_engine_cache_path(parameters.neural_networks_folder.display())
        .build();
    let cuda = CUDA::default().build();

    let mut session = block_in_place(|| {
        Session::builder()?
            .with_execution_providers([tensor_rt, cuda])?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_inter_op_spinning(false)?
            .with_intra_op_spinning(false)?
            .with_intra_threads(2)?
            .commit_from_file(model_path)
    })?;

    loop {
        parameter_receiver
            .wait_for(|parameters| parameters.typed().enable)
            .await?;

        let image = image_sub.recv().await?;

        let parameter_snapshot = node_parameters.snapshot();
        let parameters = parameter_snapshot.typed();
        if !parameters.enable {
            continue;
        }

        let image_time = image.header.stamp.into();
        let detected_objects_pending = detected_objects_pub.announce(image_time).await?;

        let detected_features_pending = detected_features_pub.announce(image_time).await?;
        let detected_lines_pending = detected_lines_pub.announce(image_time).await?;
        check_image(&image)?;

        let output = block_in_place(|| {
            let inference_start = Instant::now();

            let nv12_data = TensorRef::from_array_view((
                [image.height as usize / 2, image.width as usize / 2, 6],
                &image.data[..],
            ))?;
            let outputs: SessionOutputs = session.run(inputs!["raw_bytes_input" => nv12_data])?;

            let inference_duration = inference_start.elapsed();

            let post_processing_start = Instant::now();

            let threshold = parameters
                .object_detection_parameters
                .minimum_candidate_confidence;
            let candidates = output_rows::<9>(&outputs, "object_output")?
                .iter()
                .filter_map(|row| object_candidate(row, threshold))
                .collect();
            let detected_lines = output_rows::<21>(&outputs, "line_output")?
                .iter()
                .filter_map(|row| field_line(row, threshold, image.width, image.height))
                .collect();
            let post_processing_duration = post_processing_start.elapsed();
            let non_maximum_suppression_start = Instant::now();
            let retained = non_maximum_suppression(
                candidates,
                parameters
                    .object_detection_parameters
                    .maximum_intersection_over_union,
            );
            let non_maximum_suppression_duration = non_maximum_suppression_start.elapsed();
            let detected_features = retained
                .iter()
                .filter(|candidate| candidate.object.label != RobocupObjectLabel::Robot)
                .map(|candidate| ObjectFeature {
                    label: candidate.object.label,
                    confidence: candidate.object.bounding_box.confidence,
                    anchor: candidate.anchor,
                })
                .collect();
            let detected_objects = retained
                .into_iter()
                .map(|candidate| candidate.object)
                .collect();

            Ok::<_, color_eyre::eyre::Error>(DetectionOutput {
                inference_duration,
                post_processing_duration,
                non_maximum_suppression_duration,
                detected_objects,
                detected_features,
                detected_lines,
            })
        })?;

        inference_duration_pub
            .publish(&output.inference_duration)
            .await?;
        post_processing_duration_pub
            .publish(&output.post_processing_duration)
            .await?;
        non_maximum_suppression_duration_pub
            .publish(&output.non_maximum_suppression_duration)
            .await?;

        detected_objects_pending
            .publish(&TimeWrapper {
                time: image_time,
                inner: output.detected_objects,
            })
            .await?;
        detected_features_pending
            .publish(&TimeWrapper {
                time: image_time,
                inner: output.detected_features,
            })
            .await?;
        detected_lines_pending
            .publish(&TimeWrapper {
                time: image_time,
                inner: output.detected_lines,
            })
            .await?;
    }
}

fn check_image(image: &Image) -> Result<()> {
    if image.encoding != "nv12" {
        bail!("unsupported image encoding: {}", image.encoding);
    }

    if !image.width.is_multiple_of(32) || !image.height.is_multiple_of(32) {
        bail!(
            "image dimensions must be multiples of 32 (got {}x{})",
            image.width,
            image.height
        );
    }

    Ok(())
}

fn output_rows<'a, const N: usize>(
    outputs: &'a SessionOutputs<'_>,
    name: &str,
) -> Result<&'a [[f32; N]]> {
    let tensor = outputs
        .get(name)
        .ok_or_else(|| eyre!("model has no {name} output"))?;
    let (shape, data) = tensor.try_extract_tensor::<f32>()?;
    if **shape != [1, NUMBER_OF_DETECTIONS as i64, N as i64] {
        bail!("expected {name} shape [1, {NUMBER_OF_DETECTIONS}, {N}], got {shape:?}");
    }
    Ok(data.as_chunks::<N>().0)
}

fn object_candidate(row: &[f32; 9], threshold: f32) -> Option<ObjectCandidate> {
    let &[left, top, right, bottom, score, class, x, y, presence] = row;
    if score < threshold
        || !row.iter().all(|v| v.is_finite())
        || class.fract() != 0.0
        || !(0.0..=6.0).contains(&class)
    {
        return None;
    }
    Some(ObjectCandidate {
        object: Object::from([left, top, right, bottom, score, class]),
        anchor: Keypoint {
            point: point![x, y],
            confidence: presence,
        },
    })
}

fn field_line(row: &[f32; 21], threshold: f32, width: u32, height: u32) -> Option<FieldLine> {
    if row[4] < threshold || row[5] != 0.0 || !row.iter().all(|v| v.is_finite()) {
        return None;
    }
    // Convert exported edge coordinates to the pixel centers used by the line evaluator.
    let keypoints = std::array::from_fn(|index| {
        let offset = 6 + index * 3;
        Keypoint {
            point: point![
                row[offset].clamp(0.0, width as f32) - 0.5,
                row[offset + 1].clamp(0.0, height as f32) - 0.5
            ],
            confidence: row[offset + 2],
        }
    });
    Some(FieldLine {
        confidence: row[4],
        keypoints,
    })
}

fn non_maximum_suppression(
    mut sorted_candidate_detections: Vec<ObjectCandidate>,
    maximum_intersection_over_union: f32,
) -> Vec<ObjectCandidate> {
    sorted_candidate_detections.sort_by(|detection1, detection2| {
        detection1
            .object
            .bounding_box
            .confidence
            .total_cmp(&detection2.object.bounding_box.confidence)
    });

    let mut remaining_detections = Vec::new();

    while let Some(detection) = sorted_candidate_detections.pop() {
        sorted_candidate_detections.retain(|detection_candidate| {
            detection
                .object
                .bounding_box
                .intersection_over_union(&detection_candidate.object.bounding_box)
                < maximum_intersection_over_union
        });

        remaining_detections.push(detection)
    }

    remaining_detections
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_objects_keep_the_winning_anchor() {
        let high = [1., 2., 20., 30., 0.9, 1., 10., 29., 0.75];
        let mut low = high;
        low[4] = 0.8;
        low[6] = 15.;
        let objects = [low, high]
            .iter()
            .filter_map(|row| object_candidate(row, 0.05))
            .collect();
        let retained = non_maximum_suppression(objects, 0.4);
        assert_eq!(retained.len(), 1);
        assert_eq!(retained[0].object.label, RobocupObjectLabel::GoalPost);
        assert_eq!(retained[0].anchor.point, point![10., 29.]);
        assert_eq!(retained[0].anchor.confidence, 0.75);
    }

    #[test]
    fn line_vertices_keep_order_and_presence_in_native_pixels() {
        let row = [
            0., 0., 1., 1., 0.39, 0., -1., 2., 0.1, 100., 5., 0.2, 300., 9., 0.3, 600., 15., 0.4,
            1285., 16., 0.5,
        ];
        assert!(field_line(&row, 0.4, 1280, 1088).is_none());
        let line = field_line(&row, 0.39, 1280, 1088).unwrap();
        assert_eq!(
            line.keypoints.map(|p| p.point.x()),
            [-0.5, 99.5, 299.5, 599.5, 1279.5]
        );
        assert_eq!(
            line.keypoints.map(|p| p.confidence),
            [0.1, 0.2, 0.3, 0.4, 0.5]
        );
    }
}

use std::sync::Arc;

use color_eyre::Report;
use coordinate_systems::Pixel;
use eframe::egui::{Color32, DragValue, Stroke, Ui};
use ros_z::{Message, time::Time};
use ros_z_debug::{SampleRecord, query::plan::Selection};
use twix_visualization::twix_painter::TwixPainter;
use types::{
    object_detection::{Object, ObjectFeature, RobocupObjectLabel},
    pose_detection::FieldLine,
    time_wrapper::TimeWrapper,
};

use super::super::{
    image_overlay::ImageOverlay,
    inputs::{OverlayInput, OverlayInputs},
};

use super::prediction_colors;

struct DetectionOverlay<T> {
    topic: &'static str,
    default_threshold: f32,
    confidence_threshold: f32,
    sample: Option<Arc<SampleRecord<TimeWrapper<Vec<T>>>>>,
    paint: fn(&TwixPainter<Pixel>, &T, f32),
}

impl<T> DetectionOverlay<T> {
    fn new(topic: &'static str, threshold: f32, paint: fn(&TwixPainter<Pixel>, &T, f32)) -> Self {
        Self {
            topic,
            default_threshold: threshold,
            confidence_threshold: threshold,
            sample: None,
            paint,
        }
    }
}

impl<T> ImageOverlay for DetectionOverlay<T>
where
    TimeWrapper<Vec<T>>: Message,
{
    fn inputs(&self) -> Vec<OverlayInput> {
        vec![OverlayInput::new::<TimeWrapper<Vec<T>>>(self.topic, Selection::Exact).prefer_frame()]
    }

    fn prepare(&mut self, time: Time, inputs: &mut OverlayInputs) -> Result<(), Report> {
        self.sample = None;
        self.sample = inputs
            .decode(0, time)?
            .map(|records| Arc::clone(records.last()));
        Ok(())
    }

    fn dropdown_ui(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Confidence");
            ui.add(
                DragValue::new(&mut self.confidence_threshold)
                    .range(0.0..=1.0)
                    .speed(0.01)
                    .fixed_decimals(2),
            );
        });
    }

    fn reset_settings(&mut self) {
        self.confidence_threshold = self.default_threshold;
    }

    fn image_overlay(&self, painter: &TwixPainter<Pixel>) {
        if let Some(sample) = &self.sample {
            for detection in &sample.value.inner {
                (self.paint)(painter, detection, self.confidence_threshold);
            }
        }
    }
}

pub(in crate::panels::image) fn object_boxes() -> impl ImageOverlay {
    DetectionOverlay::new(
        "detected_objects",
        0.5,
        |painter, detection: &Object<RobocupObjectLabel>, threshold| {
            if detection.bounding_box.confidence >= threshold {
                painter.detection_box(
                    detection.bounding_box,
                    detection.label.into(),
                    prediction_colors::robocup_object(detection.label),
                );
            }
        },
    )
}

pub(in crate::panels::image) fn feature_points() -> impl ImageOverlay {
    DetectionOverlay::new(
        "detected_features",
        0.5,
        |painter, feature: &ObjectFeature, threshold| {
            if feature.confidence >= threshold {
                painter.circle_filled(
                    feature.anchor.point,
                    2.5,
                    prediction_colors::robocup_object(feature.label),
                );
            }
        },
    )
}

pub(in crate::panels::image) fn field_lines() -> impl ImageOverlay {
    DetectionOverlay::new(
        "detected_lines",
        0.39,
        |painter, line: &FieldLine, threshold| {
            if line.confidence < threshold {
                return;
            }
            let color = Color32::LIGHT_GREEN;
            for pair in line.keypoints.windows(2) {
                painter.line_segment(pair[0].point, pair[1].point, Stroke::new(1.5, color));
            }
            for vertex in &line.keypoints {
                painter.circle_filled(vertex.point, 2.0, color);
            }
        },
    )
}

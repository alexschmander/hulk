use std::{fmt::Write, path::Path};

use base64::{Engine, engine::general_purpose::STANDARD};
use color_eyre::{Result, eyre::ContextCompat};
use image::{ImageEncoder, codecs::png::PngEncoder};
use types::object_detection::RobocupObjectLabel;

use detection_replay::Recording;

#[allow(clippy::too_many_arguments)]
pub fn export(
    recording: &Recording,
    run_selector: &str,
    frame: usize,
    output: &Path,
    raw: bool,
    details: bool,
    object_confidence: f32,
    anchor_presence: f32,
    line_confidence: f32,
) -> Result<()> {
    let run = recording
        .load_runs()?
        .into_iter()
        .find(|run| run.key == run_selector || run.label == run_selector)
        .wrap_err_with(|| format!("no cached run matched {run_selector}"))?;
    let chunk = run.prediction_chunk(frame)?;
    let offset = frame - chunk.start_frame;
    let prediction = chunk.predictions[offset]
        .as_ref()
        .wrap_err_with(|| format!("run {} has no prediction for frame {frame}", run.key))?;
    let supplement = chunk.hydra_supplements[offset]
        .as_ref()
        .wrap_err_with(|| format!("run {} has no Hydra output for frame {frame}", run.key))?;
    let image = recording.load_proxy_frame(frame)?;
    let (width, height) = image.dimensions();
    let mut png = Vec::new();
    PngEncoder::new(&mut png).write_image(
        image.as_raw(),
        width,
        height,
        image::ExtendedColorType::Rgb8,
    )?;

    let mut svg = String::new();
    writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">"#
    )?;
    writeln!(
        svg,
        r#"<image width="{width}" height="{height}" href="data:image/png;base64,{}"/>"#,
        STANDARD.encode(png)
    )?;
    writeln!(
        svg,
        r#"<style>text{{font-family:monospace;font-size:9px;fill:white;stroke:black;stroke-width:2px;paint-order:stroke}} .raw{{stroke-dasharray:3 2}}</style>"#
    )?;

    for (object, anchor) in prediction.objects.iter().zip(&supplement.object_anchors) {
        if object.bounding_box.confidence < object_confidence {
            continue;
        }
        let color = label_color(object.label);
        let area = object.bounding_box.area;
        writeln!(
            svg,
            r#"<rect x="{:.2}" y="{:.2}" width="{:.2}" height="{:.2}" fill="none" stroke="{color}" stroke-width="2"/>"#,
            area.min.x(),
            area.min.y(),
            area.max.x() - area.min.x(),
            area.max.y() - area.min.y(),
        )?;
        writeln!(
            svg,
            r#"<text x="{:.2}" y="{:.2}">{} {:.3}</text>"#,
            area.min.x() + 2.0,
            area.min.y() + 10.0,
            object.label.as_str(),
            object.bounding_box.confidence,
        )?;
        if let Some(anchor) = anchor.filter(|anchor| anchor.confidence >= anchor_presence) {
            let center = area.center();
            writeln!(
                svg,
                r#"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="{color}" stroke-width="1"/><circle cx="{:.2}" cy="{:.2}" r="4" fill="{color}" stroke="white"/>"#,
                center.x(),
                center.y(),
                anchor.point.x(),
                anchor.point.y(),
                anchor.point.x(),
                anchor.point.y(),
            )?;
            if details {
                writeln!(
                    svg,
                    r#"<text x="{:.2}" y="{:.2}">anchor x={:.2} y={:.2} p={:.3}</text>"#,
                    anchor.point.x() + 6.0,
                    anchor.point.y() + 10.0,
                    anchor.point.x(),
                    anchor.point.y(),
                    anchor.confidence,
                )?;
            }
        }
    }

    for line in supplement
        .lines
        .iter()
        .filter(|line| line.bounding_box.confidence >= line_confidence)
    {
        draw_line(&mut svg, line, 0.5, "#46e6ff", false, details)?;
    }

    if raw {
        for object in supplement
            .raw_objects
            .iter()
            .filter(|row| row.object.bounding_box.confidence >= object_confidence)
        {
            let area = object.object.bounding_box.area;
            let color = label_color(object.object.label);
            writeln!(
                svg,
                r#"<rect class="raw" x="{:.2}" y="{:.2}" width="{:.2}" height="{:.2}" fill="none" stroke="{color}" stroke-width="1"/><text x="{:.2}" y="{:.2}">raw {} {:.3}</text>"#,
                area.min.x(),
                area.min.y(),
                area.max.x() - area.min.x(),
                area.max.y() - area.min.y(),
                area.min.x() + 2.0,
                area.min.y() + 20.0,
                object.object.label.as_str(),
                object.object.bounding_box.confidence,
            )?;
            if object.anchor_is_supervised {
                writeln!(
                    svg,
                    r#"<circle class="raw" cx="{:.2}" cy="{:.2}" r="5" fill="none" stroke="{color}"/>"#,
                    object.anchor.point.x(),
                    object.anchor.point.y(),
                )?;
            } else {
                let x = object.anchor.point.x();
                let y = object.anchor.point.y();
                writeln!(
                    svg,
                    r##"<path d="M {:.2} {:.2} L {:.2} {:.2} M {:.2} {:.2} L {:.2} {:.2}" stroke="#ff7070"/><text x="{:.2}" y="{:.2}">UNSUPERVISED anchor</text>"##,
                    x - 5.0,
                    y - 5.0,
                    x + 5.0,
                    y + 5.0,
                    x - 5.0,
                    y + 5.0,
                    x + 5.0,
                    y - 5.0,
                    x + 6.0,
                    y - 7.0,
                )?;
            }
            if details {
                writeln!(
                    svg,
                    r#"<text x="{:.2}" y="{:.2}">raw anchor x={:.2} y={:.2} p={:.3} {}</text>"#,
                    object.anchor.point.x() + 6.0,
                    object.anchor.point.y() + 10.0,
                    object.anchor.point.x(),
                    object.anchor.point.y(),
                    object.anchor.confidence,
                    if object.anchor_is_supervised {
                        "supervised"
                    } else {
                        "UNSUPERVISED"
                    },
                )?;
            }
        }
        for line in supplement
            .raw_lines
            .iter()
            .filter(|line| line.bounding_box.confidence >= line_confidence)
        {
            draw_line(&mut svg, line, 0.0, "#ffaa37", true, details)?;
        }
    }
    writeln!(svg, "</svg>")?;
    std::fs::write(output, svg)?;
    Ok(())
}

fn draw_line(
    svg: &mut String,
    line: &types::pose_detection::FieldLineDetection,
    shift: f32,
    color: &str,
    raw: bool,
    details: bool,
) -> Result<()> {
    let points = line
        .keypoints
        .iter()
        .map(|point| {
            format!(
                "{:.2},{:.2}",
                point.point.x() + shift,
                point.point.y() + shift
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    writeln!(
        svg,
        r#"<polyline {} points="{points}" fill="none" stroke="{color}" stroke-width="{}"/>"#,
        if raw { r#"class="raw""# } else { "" },
        if raw { 1 } else { 3 },
    )?;
    for (index, point) in line.keypoints.iter().enumerate() {
        let x = point.point.x() + shift;
        let y = point.point.y() + shift;
        writeln!(
            svg,
            r#"<circle cx="{x:.2}" cy="{y:.2}" r="3" fill="{color}"/>"#
        )?;
        if details {
            writeln!(
                svg,
                r#"<text x="{:.2}" y="{:.2}">v{index} x={:.2} y={:.2} p={:.3}</text>"#,
                x + 4.0,
                y + 10.0,
                point.point.x(),
                point.point.y(),
                point.confidence,
            )?;
        }
    }
    let first = line.keypoints[0].point;
    writeln!(
        svg,
        r#"<text x="{:.2}" y="{:.2}">{}FieldLine class=0 score={:.3}</text>"#,
        first.x() + shift + 4.0,
        first.y() + shift - 6.0,
        if raw { "raw " } else { "" },
        line.bounding_box.confidence,
    )?;
    Ok(())
}

fn label_color(label: RobocupObjectLabel) -> &'static str {
    match label {
        RobocupObjectLabel::Ball => "#ffbf45",
        RobocupObjectLabel::GoalPost => "#fff176",
        RobocupObjectLabel::LSpot => "#7fd8ff",
        RobocupObjectLabel::PenaltySpot => "#d6a4ff",
        RobocupObjectLabel::Robot => "#ff6b6b",
        RobocupObjectLabel::TSpot => "#4ee3a1",
        RobocupObjectLabel::XSpot => "#ff8fd8",
        RobocupObjectLabel::Person => "#a0a0a0",
    }
}

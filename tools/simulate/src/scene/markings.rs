//! Field marking geometry shared by simulator visual backends.
use coordinate_systems::Field;
use geometry::line_segment::LineSegment;
use linear_algebra::{Point2, point};
use types::field_dimensions::FieldDimensions;
pub(crate) enum FieldMark {
    Line { line: LineSegment<Field> },
    Circle { center: Point2<Field>, radius: f32 },
}
pub fn field_marks_from_field_dimensions(field_dimensions: &FieldDimensions) -> Vec<FieldMark> {
    vec![
        FieldMark::Line {
            line: LineSegment(
                point![-field_dimensions.length / 2.0, field_dimensions.width / 2.0],
                point![field_dimensions.length / 2.0, field_dimensions.width / 2.0],
            ),
        },
        FieldMark::Line {
            line: LineSegment(
                point![
                    -field_dimensions.length / 2.0,
                    -field_dimensions.width / 2.0
                ],
                point![field_dimensions.length / 2.0, -field_dimensions.width / 2.0],
            ),
        },
        FieldMark::Line {
            line: LineSegment(
                point![
                    -field_dimensions.length / 2.0,
                    -field_dimensions.width / 2.0
                ],
                point![-field_dimensions.length / 2.0, field_dimensions.width / 2.0],
            ),
        },
        FieldMark::Line {
            line: LineSegment(
                point![field_dimensions.length / 2.0, -field_dimensions.width / 2.0],
                point![field_dimensions.length / 2.0, field_dimensions.width / 2.0],
            ),
        },
        FieldMark::Line {
            line: LineSegment(
                point![
                    -field_dimensions.length / 2.0,
                    field_dimensions.penalty_area_width / 2.0
                ],
                point![
                    -field_dimensions.length / 2.0 + field_dimensions.penalty_area_length,
                    field_dimensions.penalty_area_width / 2.0
                ],
            ),
        },
        FieldMark::Line {
            line: LineSegment(
                point![
                    -field_dimensions.length / 2.0,
                    -field_dimensions.penalty_area_width / 2.0
                ],
                point![
                    -field_dimensions.length / 2.0 + field_dimensions.penalty_area_length,
                    -field_dimensions.penalty_area_width / 2.0
                ],
            ),
        },
        FieldMark::Line {
            line: LineSegment(
                point![
                    -field_dimensions.length / 2.0 + field_dimensions.penalty_area_length,
                    -field_dimensions.penalty_area_width / 2.0
                ],
                point![
                    -field_dimensions.length / 2.0 + field_dimensions.penalty_area_length,
                    field_dimensions.penalty_area_width / 2.0
                ],
            ),
        },
        FieldMark::Line {
            line: LineSegment(
                point![
                    -field_dimensions.length / 2.0,
                    field_dimensions.goal_box_area_width / 2.0
                ],
                point![
                    -field_dimensions.length / 2.0 + field_dimensions.goal_box_area_length,
                    field_dimensions.goal_box_area_width / 2.0
                ],
            ),
        },
        FieldMark::Line {
            line: LineSegment(
                point![
                    -field_dimensions.length / 2.0,
                    -field_dimensions.goal_box_area_width / 2.0
                ],
                point![
                    -field_dimensions.length / 2.0 + field_dimensions.goal_box_area_length,
                    -field_dimensions.goal_box_area_width / 2.0
                ],
            ),
        },
        FieldMark::Line {
            line: LineSegment(
                point![
                    -field_dimensions.length / 2.0 + field_dimensions.goal_box_area_length,
                    -field_dimensions.goal_box_area_width / 2.0
                ],
                point![
                    -field_dimensions.length / 2.0 + field_dimensions.goal_box_area_length,
                    field_dimensions.goal_box_area_width / 2.0
                ],
            ),
        },
        FieldMark::Line {
            line: LineSegment(
                point![
                    field_dimensions.length / 2.0 - field_dimensions.penalty_area_length,
                    field_dimensions.penalty_area_width / 2.0
                ],
                point![
                    field_dimensions.length / 2.0,
                    field_dimensions.penalty_area_width / 2.0
                ],
            ),
        },
        FieldMark::Line {
            line: LineSegment(
                point![
                    field_dimensions.length / 2.0 - field_dimensions.penalty_area_length,
                    -field_dimensions.penalty_area_width / 2.0
                ],
                point![
                    field_dimensions.length / 2.0,
                    -field_dimensions.penalty_area_width / 2.0
                ],
            ),
        },
        FieldMark::Line {
            line: LineSegment(
                point![
                    field_dimensions.length / 2.0 - field_dimensions.penalty_area_length,
                    -field_dimensions.penalty_area_width / 2.0
                ],
                point![
                    field_dimensions.length / 2.0 - field_dimensions.penalty_area_length,
                    field_dimensions.penalty_area_width / 2.0
                ],
            ),
        },
        FieldMark::Line {
            line: LineSegment(
                point![
                    field_dimensions.length / 2.0 - field_dimensions.goal_box_area_length,
                    field_dimensions.goal_box_area_width / 2.0
                ],
                point![
                    field_dimensions.length / 2.0,
                    field_dimensions.goal_box_area_width / 2.0
                ],
            ),
        },
        FieldMark::Line {
            line: LineSegment(
                point![
                    field_dimensions.length / 2.0 - field_dimensions.goal_box_area_length,
                    -field_dimensions.goal_box_area_width / 2.0
                ],
                point![
                    field_dimensions.length / 2.0,
                    -field_dimensions.goal_box_area_width / 2.0
                ],
            ),
        },
        FieldMark::Line {
            line: LineSegment(
                point![
                    field_dimensions.length / 2.0 - field_dimensions.goal_box_area_length,
                    -field_dimensions.goal_box_area_width / 2.0
                ],
                point![
                    field_dimensions.length / 2.0 - field_dimensions.goal_box_area_length,
                    field_dimensions.goal_box_area_width / 2.0
                ],
            ),
        },
        FieldMark::Line {
            line: LineSegment(
                point![0.0, -field_dimensions.width / 2.0],
                point![0.0, field_dimensions.width / 2.0],
            ),
        },
        FieldMark::Circle {
            center: Point2::origin(),
            radius: field_dimensions.center_circle_diameter / 2.0,
        },
        FieldMark::Line {
            line: LineSegment(
                point![
                    -field_dimensions.length / 2.0 + field_dimensions.penalty_marker_distance
                        - field_dimensions.penalty_marker_size / 2.0,
                    0.0
                ],
                point![
                    -field_dimensions.length / 2.0
                        + field_dimensions.penalty_marker_distance
                        + field_dimensions.penalty_marker_size / 2.0,
                    0.0
                ],
            ),
        },
        FieldMark::Line {
            line: LineSegment(
                point![
                    -field_dimensions.length / 2.0 + field_dimensions.penalty_marker_distance,
                    -field_dimensions.penalty_marker_size / 2.0
                ],
                point![
                    -field_dimensions.length / 2.0 + field_dimensions.penalty_marker_distance,
                    field_dimensions.penalty_marker_size / 2.0
                ],
            ),
        },
        FieldMark::Line {
            line: LineSegment(
                point![
                    field_dimensions.length / 2.0
                        - field_dimensions.penalty_marker_distance
                        - field_dimensions.penalty_marker_size / 2.0,
                    0.0
                ],
                point![
                    field_dimensions.length / 2.0 - field_dimensions.penalty_marker_distance
                        + field_dimensions.penalty_marker_size / 2.0,
                    0.0
                ],
            ),
        },
        FieldMark::Line {
            line: LineSegment(
                point![
                    field_dimensions.length / 2.0 - field_dimensions.penalty_marker_distance,
                    -field_dimensions.penalty_marker_size / 2.0
                ],
                point![
                    field_dimensions.length / 2.0 - field_dimensions.penalty_marker_distance,
                    field_dimensions.penalty_marker_size / 2.0
                ],
            ),
        },
    ]
}

/// Non-colliding decoration for the headless renderer, built from the same markings as Twix.
pub(crate) fn mujoco_field(
    dimensions: FieldDimensions,
) -> Result<mujoco_rs::prelude::MjSpec, String> {
    use mujoco_rs::prelude::{MjSpec, MjtGeom, SpecItem};
    let mut spec = MjSpec::new();
    let body = spec.world_body_mut().add_body().with_name("field");
    body.add_geom()
        .with_type(MjtGeom::mjGEOM_BOX)
        .with_size([
            f64::from(dimensions.length / 2.0 + dimensions.border_strip_width),
            f64::from(dimensions.width / 2.0 + dimensions.border_strip_width),
            0.0005,
        ])
        .with_pos([0.0, 0.0, 0.001])
        .with_rgba([0.04, 0.34, 0.13, 1.0])
        .with_contype(0)
        .with_conaffinity(0);
    let mut segment = |a: [f64; 2], b: [f64; 2]| {
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let angle = dy.atan2(dx) / 2.0;
        body.add_geom()
            .with_type(MjtGeom::mjGEOM_BOX)
            .with_size([
                (dx * dx + dy * dy).sqrt() / 2.0,
                f64::from(dimensions.line_width) / 2.0,
                0.0005,
            ])
            .with_pos([(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, 0.002])
            .with_quat([angle.cos(), 0.0, 0.0, angle.sin()])
            .with_rgba([1.0, 1.0, 1.0, 1.0])
            .with_contype(0)
            .with_conaffinity(0);
    };
    for mark in field_marks_from_field_dimensions(&dimensions) {
        match mark {
            FieldMark::Line { line } => segment(
                [line.0.x() as f64, line.0.y() as f64],
                [line.1.x() as f64, line.1.y() as f64],
            ),
            FieldMark::Circle { center, radius } => {
                for i in 0..96 {
                    let point = |i: u32| {
                        let angle = i as f64 * std::f64::consts::TAU / 96.0;
                        [
                            center.x() as f64 + radius as f64 * angle.cos(),
                            center.y() as f64 + radius as f64 * angle.sin(),
                        ]
                    };
                    segment(point(i), point(i + 1));
                }
            }
        }
    }
    Ok(spec)
}

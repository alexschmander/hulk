//! Shared look of the object filter and obstacle layers: chalk and ink on the field's green, like
//! a tactics board.

use eframe::epaint::{Color32, Stroke};
use linear_algebra::{Point2, vector};
use nalgebra::Matrix2;
use twix_visualization::twix_painter::TwixPainter;
use types::tracked_objects::TrackedObjects;

/// Field lines' white, for rings, motion arrows and goal posts: the overlay reads like chalk on a
/// tactics board.
pub(super) const CHALK: Color32 = Color32::from_rgb(0xee, 0xf2, 0xe6);
/// Dark green behind text.
pub(super) const INK: Color32 = Color32::from_rgb(0x0f, 0x2e, 0x12);
/// Uncertainty ellipses wider than this would outweigh the objects and are left out.
pub(super) const MAXIMUM_DRAWN_STANDARD_DEVIATION: f32 = 1.0;

/// A length in screen pixels as a length in the painter's frame.
pub(super) fn pixels<Frame>(painter: &TwixPainter<Frame>, length: f32) -> f32 {
    length / painter.scaling()
}

pub(super) fn odometry_to_ground_rotation<T>(objects: &TrackedObjects<T>) -> Matrix2<f32> {
    *objects
        .ground_to_odometry
        .inner
        .rotation
        .inverse()
        .to_rotation_matrix()
        .matrix()
}

/// A dashed circle with dashes of about `dash` screen pixels and equal gaps.
///
/// Dashes mark shapes, e.g. obstacles, as opposed to the filled, solid-edged washes of uncertainty.
pub(super) fn dashed_circle<Frame>(
    painter: &TwixPainter<Frame>,
    center: Point2<Frame>,
    radius: f32,
    dash: f32,
    stroke: Stroke,
) {
    let circumference = std::f32::consts::TAU * radius;
    let dashes = ((circumference / pixels(painter, 2.0 * dash)).round() as usize).max(6);
    let step = std::f32::consts::TAU / dashes as f32;
    for index in 0..dashes {
        let start = index as f32 * step;
        let points = (0..=4).map(|sample| {
            let angle = start + 0.5 * step * sample as f32 / 4.0;
            center + vector![radius * angle.cos(), radius * angle.sin()]
        });
        painter.polyline(points, stroke);
    }
}

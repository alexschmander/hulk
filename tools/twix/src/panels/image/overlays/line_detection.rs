use color_eyre::{Report, eyre::bail};
use ros_z::time::Time;

use super::super::image_overlay::{ImageOverlay, ImageOverlayPainter};
use crate::repaint::ObservationContext;

pub(in crate::panels::image) struct LineDetectionOverlay;

impl ImageOverlay for LineDetectionOverlay {
    type Sample = ();
    const NAME: &'static str = "Line Detection";
    const STORAGE_KEY: &'static str = "line_detection";

    fn new<C: ObservationContext>(_: &C) -> Result<Self, Report> {
        bail!("omitted: line debug pixels have no image timestamp")
    }
    fn prepare(&self, _: Time) -> Option<()> {
        None
    }
    fn paint(_: &ImageOverlayPainter, _: &()) {}
}

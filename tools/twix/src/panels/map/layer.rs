use std::{marker::PhantomData, sync::Arc};

use color_eyre::Result;
use convert_case::{Case, Casing};
use eframe::egui::Ui;
use log::error;
use serde_json::{Value, json};

use coordinate_systems::Field;
use types::field_dimensions::FieldDimensions;

use crate::backend::RobotBackend;
use twix_visualization::twix_painter::TwixPainter;

pub trait Layer<Frame> {
    const NAME: &'static str;
    fn new(backend: Arc<RobotBackend>) -> Self;
    fn paint(&self, painter: &TwixPainter<Frame>, field_dimensions: &FieldDimensions)
    -> Result<()>;

    /// Paints in the field frame with the layer's own ground-to-field transforms.
    ///
    /// Ground moves with the robot, so data only lines up with the field through the
    /// ground-to-field transform of its own time, not the panel's latest one. Layers whose
    /// messages carry that transform implement this; `None` paints the layer in its frame with the
    /// panel's latest transform.
    fn paint_in_field(
        &self,
        _painter: &TwixPainter<Field>,
        _field_dimensions: &FieldDimensions,
    ) -> Option<Result<()>> {
        None
    }
}

pub struct EnabledLayer<T, Frame>
where
    T: Layer<Frame>,
{
    backend: Arc<RobotBackend>,
    layer: Option<T>,
    frame: PhantomData<Frame>,
}

impl<T, Frame> EnabledLayer<T, Frame>
where
    T: Layer<Frame>,
{
    pub fn new(backend: Arc<RobotBackend>, value: Option<&Value>, active: bool) -> Self {
        let active = value
            .and_then(|value| value.get(T::NAME.to_case(Case::Snake)))
            .and_then(|value| value.get("active"))
            .and_then(|value| value.as_bool())
            .unwrap_or(active);
        let layer = active.then(|| T::new(backend.clone()));
        Self {
            backend,
            layer,
            frame: PhantomData,
        }
    }

    pub fn checkbox(&mut self, ui: &mut Ui) {
        let mut active = self.layer.is_some();
        if ui.checkbox(&mut active, T::NAME).changed() {
            match self.layer.is_some() {
                false => self.layer = Some(T::new(self.backend.clone())),
                true => self.layer = None,
            }
        }
    }

    pub fn paint_or_disable(
        &mut self,
        painter: &TwixPainter<Frame>,
        field_dimensions: &FieldDimensions,
    ) {
        if let Some(layer) = &self.layer
            && let Err(error) = layer.paint(painter, field_dimensions)
        {
            error!(
                "map panel: failed to paint map overlay {}: {:#}",
                T::NAME,
                error
            );
            self.layer = None;
        }
    }

    /// Paints with the layer's own field transforms if it has them; returns whether it painted.
    pub fn paint_in_field_or_disable(
        &mut self,
        painter: &TwixPainter<Field>,
        field_dimensions: &FieldDimensions,
    ) -> bool {
        let Some(layer) = &self.layer else {
            return false;
        };
        match layer.paint_in_field(painter, field_dimensions) {
            None => false,
            Some(Ok(())) => true,
            Some(Err(error)) => {
                error!(
                    "map panel: failed to paint map overlay {}: {:#}",
                    T::NAME,
                    error
                );
                self.layer = None;
                true
            }
        }
    }

    pub fn save(&self) -> Value {
        json!({
            "active": self.layer.is_some(),
        })
    }
}

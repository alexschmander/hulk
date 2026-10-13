use std::sync::Arc;

use eframe::egui::{Context, Ui};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::backend::RobotBackend;

pub struct PanelCreationContext<'a> {
    pub backend: Arc<RobotBackend>,
    pub value: Option<&'a Value>,
    pub egui_context: Context,
    pub render_state: Option<eframe::egui_wgpu::RenderState>,
}

pub struct PanelUiContext<'a> {
    pub backend: &'a Arc<RobotBackend>,
    pub egui_context: &'a Context,
}

pub trait Panel {
    const STORAGE_ID: &'static str;
    const DISPLAY_NAME: &'static str;
    const ICON: &'static str;

    fn new(context: PanelCreationContext<'_>) -> Self;

    fn header_ui(&mut self, _ui: &mut Ui, _context: PanelUiContext<'_>) {}

    /// Focus this panel's topic input on its next render, if it has one.
    fn focus_topic(&mut self) {}

    /// Toggle playback when the pane itself has keyboard focus.
    fn toggle_pause(&mut self) {}

    fn ui(&mut self, ui: &mut Ui, context: PanelUiContext<'_>);

    fn save(&self) -> Value {
        json!({})
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedPanel {
    pub kind: String,
    #[serde(default)]
    pub state: Value,
}

/// A centered placeholder for panels without content, with optional follow-up actions.
pub fn empty_state(
    ui: &mut Ui,
    icon: &str,
    title: &str,
    detail: &str,
    actions: impl FnOnce(&mut Ui),
) {
    use eframe::egui::{Label, RichText};
    ui.vertical_centered(|ui| {
        ui.add_space((ui.available_height() * 0.25).clamp(12.0, 160.0));
        ui.label(RichText::new(icon).size(32.0).weak());
        ui.add_space(6.0);
        ui.label(RichText::new(title).strong());
        if !detail.is_empty() {
            ui.add_space(2.0);
            ui.set_max_width(ui.available_width().min(420.0));
            ui.add(Label::new(RichText::new(detail).weak()).wrap());
        }
        ui.add_space(8.0);
        actions(ui);
    });
}

use std::sync::Arc;

use bevy::{camera_controller::pan_orbit_camera::prelude::PanOrbitCamera, prelude::*};
use eframe::egui::Ui;
use egui_bevy::BevyWidget;
use ros_z::qos::{QosDurability, QosProfile};
use ros_z_debug::{ObservationPolicy, SampleRecord};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use types::field_dimensions::FieldDimensions;

use crate::{
    panel::{Panel, PanelCreationContext, PanelUiContext},
    repaint::ObservationContext,
};
use observation::Observation;

mod field;
#[cfg(test)]
mod gpu_test;
mod observation;

#[derive(Clone, Resource, Serialize, Deserialize)]
#[serde(default)]
struct Settings {
    field: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self { field: true }
    }
}

#[derive(Default, Resource)]
struct ViewerData {
    field_dimensions: Option<Arc<SampleRecord<FieldDimensions>>>,
}

struct Observations {
    dimensions: Observation<FieldDimensions>,
}

impl Observations {
    fn new(context: &impl ObservationContext) -> color_eyre::Result<Self> {
        Ok(Self {
            dimensions: Observation::new(
                context,
                "field_dimensions",
                1,
                ObservationPolicy::default().with_subscriber_qos(QosProfile {
                    durability: QosDurability::TransientLocal,
                    ..Default::default()
                }),
            )?,
        })
    }
}

pub struct Map3DPanel {
    settings: Settings,
    widget: Option<BevyWidget>,
    observations: Result<Observations, String>,
}

impl Panel for Map3DPanel {
    const STORAGE_ID: &'static str = "map_3d";
    const DISPLAY_NAME: &'static str = "3D Map";
    const ICON: &'static str = egui_material_icons::icons::ICON_MAP.codepoint;

    fn new(context: PanelCreationContext<'_>) -> Self {
        let settings: Settings = context
            .value
            .and_then(|value| serde_json::from_value(value.clone()).ok())
            .unwrap_or_default();
        let observations = Observations::new(&context).map_err(|error| error.to_string());
        let widget = context.render_state.clone().map(|render_state| {
            let mut widget = BevyWidget::new(render_state);
            widget
                .bevy_app
                .insert_resource(ViewerData::default())
                .insert_resource(settings.clone())
                .insert_resource(GlobalAmbientLight {
                    color: Color::WHITE,
                    brightness: 600.0,
                    ..default()
                })
                .add_systems(Startup, field::setup)
                .add_systems(
                    Update,
                    (
                        position_camera_once,
                        field::visibility,
                        field::update_field_plane,
                        field::update_field_markings
                            .run_if(|settings: Res<Settings>| settings.field),
                    ),
                );
            widget.bevy_app.finish();
            widget.bevy_app.cleanup();
            widget
        });
        Self {
            settings,
            widget,
            observations,
        }
    }

    fn save(&self) -> Value {
        serde_json::to_value(&self.settings).expect("settings serialize")
    }

    fn ui(&mut self, ui: &mut Ui, context: PanelUiContext<'_>) {
        ui.checkbox(&mut self.settings.field, "Field");
        let Some(widget) = &mut self.widget else {
            ui.label("3D rendering requires the WGPU renderer.");
            return;
        };
        let data = match &self.observations {
            Ok(observations) => ViewerData {
                field_dimensions: observations.dimensions.latest(&context.backend.namespace()),
            },
            Err(error) => {
                ui.colored_label(ui.visuals().error_fg_color, error);
                ViewerData::default()
            }
        };
        widget.bevy_app.world_mut().insert_resource(data);
        widget
            .bevy_app
            .world_mut()
            .insert_resource(self.settings.clone());
        if ui.available_width() >= 1.0 && ui.available_height() >= 1.0 {
            ui.add(widget);
        }
    }
}

fn position_camera_once(
    mut positioned: Local<bool>,
    mut cameras: Query<(&mut Transform, &mut PanOrbitCamera), With<Camera3d>>,
) {
    if *positioned {
        return;
    }
    for (mut transform, mut camera) in &mut cameras {
        *transform = Transform::from_xyz(4.0, 6.0, 7.0).looking_at(Vec3::ZERO, Vec3::Y);
        camera.last_anchor_depth = -(transform.translation.length() as f64);
        *positioned = true;
    }
}

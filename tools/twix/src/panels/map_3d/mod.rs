use std::sync::Arc;

use bevy::{camera_controller::pan_orbit_camera::prelude::PanOrbitCamera, prelude::*};
use coordinate_systems::{Field, Robot};
use eframe::egui::{ComboBox, Ui};
use egui_bevy::BevyWidget;
use kinematics::robot_kinematics::RobotKinematics;
use linear_algebra::Isometry3;
use projection::camera_matrix::CameraMatrix;
use ros_z::qos::{QosDurability, QosProfile};
use ros_z_debug::{ObservationPolicy, SampleRecord};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use types::{
    field_dimensions::FieldDimensions, time_wrapper::TimeWrapper, visual_odometry::VisualOdometer,
};

use crate::{
    panel::{Panel, PanelCreationContext, PanelUiContext},
    repaint::ObservationContext,
};
use observation::Observation;
use transforms::*;

mod field;
#[cfg(test)]
mod gpu_test;
mod observation;
mod robot;
mod transforms;

#[derive(Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
enum PoseSource {
    #[default]
    Localization,
    VisualOdometer,
}

#[derive(Clone, Resource, Serialize, Deserialize)]
#[serde(default)]
struct Settings {
    field: bool,
    robot: bool,
    pose_source: PoseSource,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            field: true,
            robot: true,
            pose_source: PoseSource::Localization,
        }
    }
}

#[derive(Default, Resource)]
struct ViewerData {
    pose_source: PoseSource,
    field_dimensions: Option<Arc<SampleRecord<FieldDimensions>>>,
    localization: Option<Isometry3<Field, Robot>>,
    visual_odometer: Option<nalgebra::Isometry3<f32>>,
    robot_kinematics: Option<Arc<SampleRecord<TimeWrapper<RobotKinematics>>>>,
    camera_matrix: Option<CameraMatrix>,
}

struct Observations {
    dimensions: Observation<FieldDimensions>,
    localization: Observation<types::localization::LocalizationEstimate>,
    localization_status: Observation<types::localization::LocalizationStatus>,
    odometer: Observation<VisualOdometer>,
    kinematics: Observation<TimeWrapper<RobotKinematics>>,
    matrix: Observation<TimeWrapper<CameraMatrix>>,
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
            localization: Observation::new(
                context,
                "localization/estimate",
                1,
                Default::default(),
            )?,
            localization_status: Observation::new(
                context,
                "localization/status",
                1,
                ObservationPolicy::default().with_subscriber_qos(QosProfile {
                    durability: QosDurability::TransientLocal,
                    ..Default::default()
                }),
            )?,
            odometer: Observation::new(
                context,
                "visual_odometry/current_left_camera_to_visual_odometer",
                64,
                Default::default(),
            )?,
            kinematics: Observation::new(context, "robot_kinematics", 1024, Default::default())?,
            matrix: Observation::new(context, "camera_matrix", 1024, Default::default())?,
        })
    }

    fn snapshot(&self, namespace: &str, settings: &Settings) -> ViewerData {
        let frame_id = self
            .localization_status
            .latest(namespace)
            .map(|status| (status.value.epoch, status.value.generation));
        ViewerData {
            pose_source: settings.pose_source,
            field_dimensions: self.dimensions.latest(namespace),
            localization: self
                .localization
                .latest(namespace)
                .filter(|record| frame_id == Some((record.value.epoch, record.value.generation)))
                .and_then(|record| record.value.robot_to_field)
                .map(|field| Isometry3::wrap(field.pose.inner.cast::<f32>().inverse())),
            visual_odometer: self
                .odometer
                .all(namespace)
                .into_iter()
                .max_by_key(|record| record.value.time)
                .map(|record| record.value.current_left_camera_to_visual_odometer),
            robot_kinematics: self.kinematics.aligned(namespace, None),
            camera_matrix: self
                .matrix
                .aligned(namespace, None)
                .map(|record| record.value.inner.clone()),
        }
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
                .add_systems(Startup, (field::setup, robot::setup))
                .add_systems(
                    Update,
                    (
                        position_camera_once,
                        field::visibility,
                        field::update_field_plane,
                        robot::update,
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
        ui.checkbox(&mut self.settings.robot, "Robot");
        ComboBox::from_id_salt(ui.id().with("pose_source"))
            .selected_text(match self.settings.pose_source {
                PoseSource::Localization => "Localization",
                PoseSource::VisualOdometer => "Visual odometry",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut self.settings.pose_source,
                    PoseSource::Localization,
                    "Localization",
                );
                ui.selectable_value(
                    &mut self.settings.pose_source,
                    PoseSource::VisualOdometer,
                    "Visual odometry",
                );
            });
        let Some(widget) = &mut self.widget else {
            ui.label("3D rendering requires the WGPU renderer.");
            return;
        };
        let data = match &self.observations {
            Ok(observations) => observations.snapshot(&context.backend.namespace(), &self.settings),
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

pub const RENDER_STATE_ID: &str = "twix_render_state";

#[cfg(feature = "simulator")]
pub use enabled::SimulatorPanel;

#[cfg(feature = "simulator")]
mod enabled {
    use super::RENDER_STATE_ID;
    use std::{path::PathBuf, sync::mpsc};

    use eframe::{
        egui::{self, Ui},
        egui_wgpu::RenderState,
    };
    use serde::{Deserialize, Serialize};
    use serde_json::Value;
    use simulate::{Configuration, ControllerSource, PreparedSimulator, Profile, Simulator};
    use tokio_util::task::AbortOnDropHandle;

    use crate::panel::{Panel, PanelCreationContext, PanelUiContext};

    #[derive(Serialize, Deserialize)]
    #[serde(default)]
    struct Settings {
        parameter_root: PathBuf,
        model_directory: PathBuf,
        namespace: String,
        profile: Profile,
        controller: ControllerSource,
    }
    impl Default for Settings {
        fn default() -> Self {
            Self {
                parameter_root: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../etc/parameters"),
                model_directory: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../etc/neural_networks"),
                namespace: "/simulator/robot".into(),
                profile: Profile::default(),
                controller: ControllerSource::default(),
            }
        }
    }

    pub struct SimulatorPanel {
        settings: Settings,
        simulator: Option<Simulator>,
        pending: Option<mpsc::Receiver<Result<PreparedSimulator, String>>>,
        task: Option<AbortOnDropHandle<()>>,
        error: Option<String>,
    }

    impl Panel for SimulatorPanel {
        const STORAGE_ID: &'static str = "motion_simulator";
        const DISPLAY_NAME: &'static str = "Simulator";
        const ICON: &'static str = egui_material_icons::icons::ICON_SPORTS_SOCCER.codepoint;

        fn new(context: PanelCreationContext<'_>) -> Self {
            Self {
                settings: context
                    .value
                    .and_then(|value| serde_json::from_value(value.clone()).ok())
                    .unwrap_or_default(),
                simulator: None,
                pending: None,
                task: None,
                error: None,
            }
        }

        fn header_ui(&mut self, ui: &mut Ui, _: PanelUiContext<'_>) {
            if self.simulator.is_some() && ui.button("Stop simulator").clicked() {
                self.simulator = None;
            }
            ui.label(&self.settings.namespace);
        }

        fn toggle_pause(&mut self) {
            if let Some(simulator) = &mut self.simulator {
                simulator.toggle_pause();
            }
        }

        fn ui(&mut self, ui: &mut Ui, context: PanelUiContext<'_>) {
            if context.backend.transport_scope() != Some(simulate::ZENOH_NAMESPACE) {
                ui.label("Launch ./twix --simulator to use the simulator's isolated transport.");
                return;
            }
            if let Some(pending) = &self.pending {
                let result = match pending.try_recv() {
                    Ok(result) => Some(result),
                    Err(mpsc::TryRecvError::Empty) => None,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        Some(Err("Simulator startup stopped unexpectedly".into()))
                    }
                };
                if let Some(result) = result {
                    self.pending = None;
                    self.task = None;
                    match result {
                        Ok(prepared) => {
                            let renderer = ui.ctx().data(|data| {
                                data.get_temp::<RenderState>(egui::Id::new(RENDER_STATE_ID))
                            });
                            if let Some(renderer) = renderer {
                                if let Err(error) = context
                                    .backend
                                    .set_namespace(self.settings.namespace.clone())
                                {
                                    self.error = Some(format!("{error:#}"));
                                }
                                self.simulator = Some(Simulator::new(prepared, renderer));
                            } else {
                                self.error =
                                    Some("The simulator requires Twix's wgpu renderer.".into());
                            }
                        }
                        Err(error) => self.error = Some(error),
                    }
                }
            }
            if let Some(simulator) = &mut self.simulator {
                simulator.ui(ui);
                return;
            }
            ui.label("Run one simulated K1 with the real behavior, motion, and HSL message nodes.");
            ui.label(
                "Start the HSL Game Controller separately. Its UDP messages drive game state.",
            );
            ui.add_enabled_ui(self.pending.is_none(), |ui| {
                egui::ComboBox::from_label("Profile")
                    .selected_text(self.settings.profile.label())
                    .show_ui(ui, |ui| {
                        for profile in Profile::ALL {
                            ui.selectable_value(&mut self.settings.profile, profile, profile.label());
                        }
                    });
                ui.label(self.settings.profile.description());
                egui::ComboBox::from_label("Gamepad source")
                    .selected_text(match self.settings.controller {
                        ControllerSource::Local => "Local gamepad",
                        ControllerSource::External => "External controller",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.settings.controller, ControllerSource::Local, "Local gamepad");
                        ui.selectable_value(&mut self.settings.controller, ControllerSource::External, "External controller");
                    });
                if self.settings.controller == ControllerSource::External {
                    ui.label("Publish inputs/controller_input in this robot namespace and simulator transport scope.");
                }
                ui.horizontal(|ui| {
                    ui.label("Namespace");
                    ui.text_edit_singleline(&mut self.settings.namespace);
                });
                ui.horizontal(|ui| {
                    ui.label("Robotics parameters");
                    let mut path = self.settings.parameter_root.to_string_lossy().into_owned();
                    if ui.text_edit_singleline(&mut path).changed() {
                        self.settings.parameter_root = path.into();
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Motion models");
                    let mut path = self.settings.model_directory.to_string_lossy().into_owned();
                    if ui.text_edit_singleline(&mut path).changed() {
                        self.settings.model_directory = path.into();
                    }
                });
            });
            if let Some(error) = &self.error {
                ui.colored_label(egui::Color32::RED, error);
            }
            if self.pending.is_some() {
                ui.spinner();
                ui.label("Starting simulator…");
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_millis(50));
            } else if ui.button("Start simulator").clicked() {
                self.error = None;
                let configuration = Configuration {
                    parameter_root: self.settings.parameter_root.clone(),
                    model_directory: self.settings.model_directory.clone(),
                    namespace: self.settings.namespace.clone(),
                    router: context.backend.router().map(str::to_owned),
                    profile: self.settings.profile,
                    controller: self.settings.controller,
                };
                let runtime = context.backend.runtime_handle().clone();
                let (sender, receiver) = mpsc::channel();
                self.pending = Some(receiver);
                let repaint = ui.ctx().clone();
                self.task = Some(AbortOnDropHandle::new(runtime.clone().spawn_blocking(
                    move || {
                        let result = runtime
                            .block_on(PreparedSimulator::new(runtime.clone(), configuration))
                            .map_err(|error| format!("{error:#}"));
                        let _ = sender.send(result);
                        repaint.request_repaint();
                    },
                )));
            }
        }

        fn save(&self) -> Value {
            serde_json::to_value(&self.settings).unwrap()
        }
    }
    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn legacy_layout_defaults_and_profile_choices_survive_serialization() {
            let legacy: Settings =
                serde_json::from_value(serde_json::json!({"namespace":"/saved/robot"})).unwrap();
            assert_eq!(legacy.profile, Profile::MotionBehavior);
            assert_eq!(legacy.controller, ControllerSource::Local);
            for profile in Profile::ALL {
                let settings = Settings {
                    profile,
                    controller: ControllerSource::External,
                    ..Settings::default()
                };
                let saved: Settings =
                    serde_json::from_value(serde_json::to_value(settings).unwrap()).unwrap();
                assert_eq!(saved.profile, profile);
                assert_eq!(saved.controller, ControllerSource::External);
            }
        }
    }
}
#[cfg(not(feature = "simulator"))]
pub struct SimulatorPanel(serde_json::Value);

#[cfg(not(feature = "simulator"))]
impl crate::panel::Panel for SimulatorPanel {
    const STORAGE_ID: &'static str = "motion_simulator";
    const DISPLAY_NAME: &'static str = "Simulator";
    const ICON: &'static str = egui_material_icons::icons::ICON_SPORTS_SOCCER.codepoint;
    fn new(context: crate::panel::PanelCreationContext<'_>) -> Self {
        Self(context.value.cloned().unwrap_or_default())
    }
    fn ui(&mut self, ui: &mut eframe::egui::Ui, _: crate::panel::PanelUiContext<'_>) {
        ui.label("Launch ./twix --simulator to enable the simulator.");
    }
    fn save(&self) -> serde_json::Value {
        self.0.clone()
    }
}

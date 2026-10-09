pub const RENDER_STATE_ID: &str = "twix_render_state";

#[cfg(feature = "simulator")]
pub use enabled::SimulatorPanel;

#[cfg(feature = "simulator")]
mod enabled {
    use super::RENDER_STATE_ID;
    use std::{path::PathBuf, sync::mpsc};

    use eframe::{
        egui::{self, RichText, Ui},
        egui_wgpu::RenderState,
    };
    use egui_material_icons::icons;
    use serde::{Deserialize, Serialize};
    use serde_json::Value;
    use simulate::{
        Configuration, ControllerSource, FieldConfiguration, PreparedSimulator, Profile, Simulator,
        widgets,
    };
    use tokio_util::task::AbortOnDropHandle;

    use crate::panel::{Panel, PanelCreationContext, PanelUiContext};

    #[derive(Serialize, Deserialize)]
    #[serde(default)]
    struct Settings {
        parameter_root: PathBuf,
        model_directory: PathBuf,
        robot_count: u8,
        opponent_count: u8,
        location: String,
        profile: Profile,
        controller: ControllerSource,
        referee: simulate::RefereeMode,
        competition: simulate::Competition,
        ball: simulate::BallSize,
    }
    impl Default for Settings {
        fn default() -> Self {
            Self {
                parameter_root: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../etc/parameters"),
                model_directory: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../etc/neural_networks"),
                robot_count: 1,
                opponent_count: 0,
                location: "incheon_small".into(),
                profile: Profile::default(),
                controller: ControllerSource::default(),
                referee: simulate::RefereeMode::default(),
                competition: simulate::Competition::default(),
                ball: simulate::BallSize::default(),
            }
        }
    }

    pub struct SimulatorPanel {
        settings: Settings,
        cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
        simulator: Option<Simulator>,
        pending: Option<mpsc::Receiver<Result<PreparedSimulator, String>>>,
        task: Option<AbortOnDropHandle<()>>,
        error: Option<String>,
        preview: Option<Preview>,
    }

    /// The field preview follows the selected location without reloading it every frame.
    struct Preview {
        parameter_root: PathBuf,
        location: String,
        field: Result<FieldConfiguration, String>,
    }

    impl Drop for SimulatorPanel {
        fn drop(&mut self) {
            self.cancelled
                .store(true, std::sync::atomic::Ordering::Release);
        }
    }
    impl SimulatorPanel {
        pub fn select_robot(&mut self, number: Option<simulate::RobotId>) {
            if let Some(simulator) = &mut self.simulator {
                simulator.select_robot(number);
            }
        }
    }
    impl Panel for SimulatorPanel {
        const STORAGE_ID: &'static str = "motion_simulator";
        const DISPLAY_NAME: &'static str = "Simulator";
        const ICON: &'static str = egui_material_icons::icons::ICON_SPORTS_SOCCER.codepoint;

        fn new(context: PanelCreationContext<'_>) -> Self {
            Self {
                cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                settings: context
                    .value
                    .and_then(|value| serde_json::from_value(value.clone()).ok())
                    .unwrap_or_default(),
                simulator: None,
                pending: None,
                task: None,
                error: None,
                preview: None,
            }
        }

        fn header_ui(&mut self, ui: &mut Ui, context: PanelUiContext<'_>) {
            let Some(simulator) = &mut self.simulator else {
                ui.weak("Simulate robot teams with the real robotics nodes.");
                return;
            };
            if let Some(number) = simulate::robot_id(&context.backend.namespace()) {
                simulator.select_robot(Some(number));
            }
            if simulator.header(ui) {
                self.simulator = None;
            }
        }

        fn toggle_pause(&mut self) {
            if let Some(simulator) = &mut self.simulator {
                simulator.toggle_pause();
            }
        }

        fn ui(&mut self, ui: &mut Ui, context: PanelUiContext<'_>) {
            if !context.backend.transport_scope().is_some_and(|scope| {
                scope == simulate::ZENOH_NAMESPACE
                    || scope.starts_with(&format!("{}/", simulate::ZENOH_NAMESPACE))
            }) {
                super::launch_hint(ui, "The simulator needs its own transport scope.");
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
                                if let Err(error) = context.backend.set_namespace(
                                    simulate::robot_namespace(simulate::RobotId::FIRST),
                                ) {
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
                if let Some(id) = simulator.take_selection()
                    && let Err(error) = context.backend.set_namespace(simulate::robot_namespace(id))
                {
                    self.error = Some(format!("{error:#}"));
                }
                return;
            }
            egui::ScrollArea::vertical()
                .auto_shrink(false)
                .show(ui, |ui| {
                    egui::Frame::new()
                        .inner_margin(egui::Margin::symmetric(12, 10))
                        .show(ui, |ui| {
                            if ui.available_width() >= 760.0 {
                                ui.horizontal_top(|ui| {
                                    ui.vertical(|ui| {
                                        ui.set_width(340.0);
                                        self.setup_ui(ui, &context);
                                    });
                                    ui.add_space(28.0);
                                    ui.vertical(|ui| self.preview_ui(ui, 560.0));
                                });
                            } else {
                                self.preview_ui(ui, 260.0);
                                ui.add_space(16.0);
                                self.setup_ui(ui, &context);
                            }
                        });
                });
        }

        fn save(&self) -> Value {
            serde_json::to_value(&self.settings).unwrap()
        }
    }
    impl SimulatorPanel {
        /// The HSL division and its Foundation or Advanced configuration.
        fn competition_ui(&mut self, ui: &mut Ui) {
            let competition = self.settings.competition;
            ui.horizontal(|ui| {
                let mut division = competition.division();
                egui::ComboBox::from_id_salt("referee_division")
                    .selected_text(format!("{} division", division.label()))
                    .show_ui(ui, |ui| {
                        for option in simulate::Division::ALL {
                            ui.selectable_value(
                                &mut division,
                                option,
                                format!("{} division", option.label()),
                            );
                        }
                    });
                ui.spacing_mut().item_spacing.x = 2.0;
                let mut advanced = competition.advanced();
                for (value, label) in [(false, "Foundation"), (true, "Advanced")] {
                    let players = simulate::Competition::new(division, value).players();
                    if widgets::segment(ui, advanced == value, label)
                        .on_hover_text(format!("Up to {players} players per team"))
                        .clicked()
                    {
                        advanced = value;
                    }
                }
                self.settings.competition = simulate::Competition::new(division, advanced);
            });
            let limit = self.settings.competition.players();
            if self.settings.robot_count.max(self.settings.opponent_count) > limit {
                ui.add(
                    egui::Label::new(
                        RichText::new(format!(
                            "{} allows {limit} players per team; higher numbers start as substitutes beside the field.",
                            self.settings.competition.label()
                        ))
                        .size(11.5)
                        .color(ui.visuals().warn_fg_color),
                    )
                    .wrap(),
                );
            }
        }

        /// Ball size is independent of the field, because divisions share field sizes.
        fn ball_ui(&mut self, ui: &mut Ui) {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                for ball in simulate::BallSize::ALL {
                    let hint = match (ball.radius(), ball.division()) {
                        (Some(radius), Some(division)) => format!(
                            "FIFA {}, {} cm across; {division}",
                            ball.label().to_lowercase(),
                            centimeters(radius)
                        ),
                        _ => "The ball defined by the field location".to_owned(),
                    };
                    if widgets::segment(ui, self.settings.ball == ball, ball.label())
                        .on_hover_text(hint)
                        .clicked()
                    {
                        self.settings.ball = ball;
                    }
                }
            });
            let radius = self.settings.ball.radius().or_else(|| {
                self.preview
                    .as_ref()
                    .filter(|preview| preview.location == self.settings.location)
                    .and_then(|preview| preview.field.as_ref().ok())
                    .map(|field| field.dimensions.ball_radius)
            });
            if let Some(radius) = radius {
                ui.label(
                    RichText::new(format!("{} cm across", centimeters(radius)))
                        .size(11.5)
                        .weak(),
                );
            }
        }

        fn setup_ui(&mut self, ui: &mut Ui, context: &PanelUiContext<'_>) {
            ui.add_enabled_ui(self.pending.is_none(), |ui| {
                ui.spacing_mut().item_spacing.y = 6.0;
                ui.label(RichText::new("HULKs · team 24").strong());
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 2.0;
                    for count in 1..=5 {
                        let response = widgets::segment(
                            ui,
                            self.settings.robot_count == count,
                            &count.to_string(),
                        )
                        .on_hover_text(format!(
                                "Start with {count} {}",
                                if count == 1 { "robot" } else { "robots" }
                            ));
                        if response.clicked() {
                            self.settings.robot_count = count;
                        }
                    }
                });
                ui.add_space(10.0);
                ui.label(RichText::new("Opponents · team 5").strong());
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 2.0;
                    for count in 0..=5 {
                        if widgets::segment(ui, self.settings.opponent_count == count, &count.to_string()).clicked() {
                            self.settings.opponent_count = count;
                        }
                    }
                });
                ui.add_space(10.0);
                ui.label(RichText::new("Referee").strong());
                ui.horizontal(|ui| {
                    for (mode,label) in [(simulate::RefereeMode::External,"External GameController"),(simulate::RefereeMode::Automatic,"Auto referee")] {
                        if widgets::segment(ui,self.settings.referee == mode,label).clicked() { self.settings.referee = mode; }
                    }
                });
                if self.settings.referee == simulate::RefereeMode::Automatic {
                    self.competition_ui(ui);
                }
                ui.add_space(10.0);
                ui.label(RichText::new("Field").strong());
                match FieldConfiguration::locations(&self.settings.parameter_root) {
                    Ok(locations) => {
                        egui::ComboBox::from_id_salt("simulator_location")
                            .selected_text(&self.settings.location)
                            .width(220.0)
                            .show_ui(ui, |ui| {
                                for location in locations {
                                    ui.selectable_value(
                                        &mut self.settings.location,
                                        location.clone(),
                                        &location,
                                    );
                                }
                            });
                    }
                    Err(error) => widgets::error_banner(ui, &format!("Locations: {error:#}")),
                }
                ui.add_space(10.0);
                ui.label(RichText::new("Ball").strong());
                self.ball_ui(ui);
                ui.add_space(10.0);
                ui.label(RichText::new("Profile").strong());
                widgets::profile_ladder(ui, &mut self.settings.profile);
                ui.add_space(10.0);
                ui.label(RichText::new("Gamepad").strong());
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 2.0;
                    for (source, label) in [
                        (ControllerSource::Local, "Local gamepad"),
                        (ControllerSource::External, "External controller"),
                    ] {
                        if widgets::segment(ui, self.settings.controller == source, label)
                            .clicked()
                        {
                            self.settings.controller = source;
                        }
                    }
                });
                if self.settings.controller == ControllerSource::External {
                    ui.add(
                        egui::Label::new(
                            RichText::new(
                                "Publish inputs/controller_input in this robot namespace and simulator transport scope.",
                            )
                            .size(11.5)
                            .weak(),
                        )
                        .wrap(),
                    );
                }
                ui.add_space(6.0);
                egui::CollapsingHeader::new("Parameter and model paths")
                    .id_salt("simulator_paths")
                    .show(ui, |ui| {
                        for (label, path) in [
                            ("Robotics parameters", &mut self.settings.parameter_root),
                            ("Motion models", &mut self.settings.model_directory),
                        ] {
                            ui.weak(label);
                            let mut text = path.to_string_lossy().into_owned();
                            if ui
                                .add(
                                    egui::TextEdit::singleline(&mut text)
                                        .desired_width(f32::INFINITY),
                                )
                                .changed()
                            {
                                *path = text.into();
                            }
                        }
                    });
            });
            ui.add_space(12.0);
            if let Some(error) = &self.error {
                widgets::error_banner(ui, error);
                ui.add_space(8.0);
            }
            if self.pending.is_some() {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Starting simulator…");
                });
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_millis(50));
            } else if widgets::primary(ui, icons::ICON_PLAY_ARROW.codepoint, "Start simulator")
                .clicked()
            {
                self.start(ui, context);
            }
            ui.add_space(6.0);
            ui.add(
                egui::Label::new(
                    RichText::new(
                        if self.settings.referee == simulate::RefereeMode::External {
                            "Game state comes from an external HSL GameController."
                        } else {
                            "HSL rules with automatic restarts and penalty placement."
                        },
                    )
                    .size(11.5)
                    .weak(),
                )
                .wrap(),
            );
        }

        fn preview_ui(&mut self, ui: &mut Ui, max_height: f32) {
            let settings = &self.settings;
            if self.preview.as_ref().is_none_or(|preview| {
                preview.parameter_root != settings.parameter_root
                    || preview.location != settings.location
            }) {
                self.preview = Some(Preview {
                    parameter_root: settings.parameter_root.clone(),
                    location: settings.location.clone(),
                    field: FieldConfiguration::load(&settings.parameter_root, &settings.location)
                        .map_err(|error| format!("{error:#}")),
                });
            }
            let preview = self.preview.as_ref().unwrap();
            match &preview.field {
                Ok(field) => {
                    widgets::field_preview(
                        ui,
                        field,
                        self.settings.robot_count,
                        self.settings.opponent_count,
                        max_height,
                    );
                    ui.add_space(4.0);
                    ui.weak(format!(
                        "{}, {} × {} m",
                        preview.location,
                        widgets::meters(field.dimensions.length),
                        widgets::meters(field.dimensions.width)
                    ));
                }
                Err(error) => {
                    ui.weak(format!("No field preview for {}", preview.location));
                    ui.add(egui::Label::new(RichText::new(error).small().weak()).wrap());
                }
            }
        }

        fn start(&mut self, ui: &Ui, context: &PanelUiContext<'_>) {
            self.cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let cancelled = self.cancelled.clone();
            self.error = None;
            let field = match FieldConfiguration::load(
                &self.settings.parameter_root,
                &self.settings.location,
            ) {
                Ok(field) => field.with_ball(self.settings.ball),
                Err(error) => {
                    self.error = Some(format!("{error:#}"));
                    return;
                }
            };
            let configuration = Configuration {
                parameter_root: self.settings.parameter_root.clone(),
                model_directory: self.settings.model_directory.clone(),
                namespace: simulate::robot_namespace(simulate::RobotId::FIRST),
                robot_count: self.settings.robot_count,
                opponent_count: self.settings.opponent_count,
                location: Some(self.settings.location.clone()),
                field_configuration: Some(field),
                router: context.backend.router().map(str::to_owned),
                profile: self.settings.profile,
                controller: self.settings.controller,
                referee: self.settings.referee,
                competition: self.settings.competition,
            };
            let runtime = context.backend.runtime_handle().clone();
            let (sender, receiver) = mpsc::channel();
            self.pending = Some(receiver);
            let repaint = ui.ctx().clone();
            self.task = Some(AbortOnDropHandle::new(runtime.clone().spawn_blocking(
                move || {
                    let result = runtime
                        .block_on(PreparedSimulator::new(
                            runtime.clone(),
                            configuration,
                            cancelled,
                        ))
                        .map_err(|error| format!("{error:#}"));
                    let _ = sender.send(result);
                    repaint.request_repaint();
                },
            )));
        }
    }
    fn centimeters(radius: f32) -> String {
        format!("{:.1}", radius * 200.0)
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
            assert_eq!(legacy.ball, simulate::BallSize::Location);
            assert_eq!(legacy.competition, simulate::Competition::Middle);
            for profile in Profile::ALL {
                let settings = Settings {
                    profile,
                    controller: ControllerSource::External,
                    competition: simulate::Competition::MiddleFoundation,
                    ball: simulate::BallSize::Size4,
                    ..Settings::default()
                };
                let saved: Settings =
                    serde_json::from_value(serde_json::to_value(settings).unwrap()).unwrap();
                assert_eq!(saved.profile, profile);
                assert_eq!(saved.controller, ControllerSource::External);
                assert_eq!(saved.competition, simulate::Competition::MiddleFoundation);
                assert_eq!(saved.ball, simulate::BallSize::Size4);
            }
        }
    }
}
fn launch_hint(ui: &mut eframe::egui::Ui, reason: &str) {
    use eframe::egui::RichText;
    ui.vertical_centered(|ui| {
        ui.add_space((ui.available_height() * 0.3).max(12.0));
        ui.label(
            RichText::new(egui_material_icons::icons::ICON_SPORTS_SOCCER.codepoint)
                .size(36.0)
                .weak(),
        );
        ui.add_space(6.0);
        ui.label(RichText::new(reason).strong());
        ui.add_space(4.0);
        ui.weak("Restart Twix with");
        ui.code("./twix --simulator /1");
    });
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
        launch_hint(ui, "This Twix build has no simulator.");
    }
    fn save(&self) -> serde_json::Value {
        self.0.clone()
    }
}

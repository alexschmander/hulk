//! Twix's simulator. All hardware substitutes and node wiring live in this tool.
use std::sync::atomic::{AtomicBool, Ordering};

use bevy::prelude::*;
use color_eyre::{Result, eyre::ensure};
use eframe::{
    egui::{self, RichText, Ui, Widget},
    egui_wgpu::RenderState,
};
use egui_bevy::BevyWidget;
use egui_material_icons::icons;
use tokio::runtime::Handle;

mod controller;
mod field;
mod network;
mod team;
pub use field::FieldConfiguration;
pub use team::{RobotId, TeamId, robot_id, robot_namespace, transport_scope};
mod autoref;
mod behavior_inputs;
mod bevy_mujoco;
mod buttons;
mod observations;
mod parameters;
pub use autoref::{Competition, RefereeMode};
mod profiles;
mod readiness;
mod reference;
mod robot_io;
mod robotics;
mod scene;
mod sensors;
mod simulated_sdk;
mod simulation;
mod viewport;
mod whistle;
pub mod widgets;

use bevy_mujoco::{SharedPhysics, SimulationMode};
pub use profiles::{ControllerSource, Profile};
pub use robotics::Configuration;

/// Global transport scope, including the otherwise unnamespaced Booster SDK.
pub const ZENOH_NAMESPACE: &str = "hulk_simulator";

static RUNNING: AtomicBool = AtomicBool::new(false);
struct Lease;
impl Drop for Lease {
    fn drop(&mut self) {
        RUNNING.store(false, Ordering::Release);
    }
}

pub struct PreparedSimulator {
    team: team::Team,
}
impl PreparedSimulator {
    pub async fn new(
        runtime: Handle,
        configuration: Configuration,
        cancelled: std::sync::Arc<AtomicBool>,
    ) -> Result<Self> {
        ensure!(
            !RUNNING.swap(true, Ordering::AcqRel),
            "A simulator is already running. Close it before starting another."
        );
        let lease = Lease;
        {
            use mujoco_rs::prelude::MjSpec;
            let mut spec =
                MjSpec::from_xml(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/k1_robot.xml"))?;
            spec.compile()?;
            for name in ["football_base_color.png", "football_normal.png"] {
                let bytes = std::fs::read(
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("assets/textures")
                        .join(name),
                )?;
                ensure!(
                    bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
                    "Missing simulator texture {name}; run git lfs pull"
                );
            }
        }
        let team =
            team::Team::new_cancellable(runtime, configuration, cancelled, Some(lease)).await?;
        Ok(Self { team })
    }
}

pub struct Simulator {
    // Stop and join physics before the Bevy world drops its node resources.
    physics: simulation::PhysicsWorker,
    widget: BevyWidget,
    buttons: buttons::BodyButtons,
    viewport: viewport::Viewport,
    error: Option<String>,
    pending_robot: Option<tokio_util::task::AbortOnDropHandle<()>>,
    added_robot: Option<std::sync::mpsc::Receiver<Result<RobotId, String>>>,
    selected: Option<RobotId>,
    started: ros_z::time::Time,
}

impl Drop for Simulator {
    fn drop(&mut self) {
        self.widget
            .bevy_app
            .world()
            .resource::<team::Team>()
            .cancel();
        if let Some(task) = self.pending_robot.take() {
            let runtime = self
                .widget
                .bevy_app
                .world()
                .resource::<team::Team>()
                .runtime()
                .clone();
            tokio::task::block_in_place(|| {
                let _ = runtime.block_on(task);
            });
        }
        if let Some(member) = self
            .widget
            .bevy_app
            .world()
            .resource::<team::Team>()
            .selected()
        {
            let _ = self.buttons.cancel(&member.io);
        }
    }
}

impl Simulator {
    pub fn new(prepared: PreparedSimulator, renderer: RenderState) -> Self {
        let started = prepared.team.now();
        let mut widget = BevyWidget::new(renderer);
        let app = &mut widget.bevy_app;
        let first = prepared.team.members()[0].io.clone();
        app.add_plugins(parameters::SimulatorParametersPlugin::new(
            first.parameters.clone(),
            first.field_dimensions,
            prepared
                .team
                .configuration()
                .field_configuration
                .as_ref()
                .unwrap()
                .goal_height,
        ))
        .insert_resource(first)
        .insert_resource(prepared.team.clone())
        .add_plugins((
            bevy_mujoco::MujocoWorldPlugin,
            scene::field::FieldPlugin,
            scene::ObjectsPlugin,
            simulation::MotionSimulationPlugin,
        ))
        .add_systems(PreUpdate, viewport::setup_scene);
        app.finish();
        app.cleanup();
        app.world().resource::<SharedPhysics>().lock().mode = SimulationMode::Paused;
        let physics = simulation::PhysicsWorker::start_team(
            app.world().resource::<SharedPhysics>().clone(),
            prepared.team,
        );
        Self {
            physics,
            widget,
            buttons: buttons::BodyButtons::default(),
            viewport: viewport::Viewport::default(),
            error: None,
            pending_robot: None,
            added_robot: None,
            selected: Some(RobotId::FIRST),
            started,
        }
    }

    pub fn select_robot(&mut self, number: Option<RobotId>) {
        let team = self.widget.bevy_app.world().resource::<team::Team>();
        if number != self.selected {
            if let Some(member) = team.selected()
                && let Err(error) = self.buttons.cancel(&member.io)
            {
                self.error = Some(format!("{error:#}"));
            }
            self.selected = number;
        }
        team.select(number);
    }

    pub fn toggle_pause(&mut self) {
        if self.error.is_some() || !self.physics.ready() {
            return;
        }
        let mut physics = self
            .widget
            .bevy_app
            .world()
            .resource::<SharedPhysics>()
            .lock();
        physics.mode = match physics.mode {
            SimulationMode::Paused => SimulationMode::Running,
            SimulationMode::Running => SimulationMode::Paused,
        };
    }

    /// Controls for Twix's panel header: the whole simulation first, then the selected player.
    /// Returns whether Stop simulator was pressed.
    pub fn header(&mut self, ui: &mut Ui) -> bool {
        let compact = ui.available_width() < 640.0;
        // Do not touch physics after a worker panic may have poisoned its mutex.
        let failed = self.physics.poll().is_some();
        let mut stop = false;
        ui.vertical(|ui| {
            if !failed
                && let Some(referee) = self
                    .widget
                    .bevy_app
                    .world()
                    .resource::<team::Team>()
                    .referee()
            {
                referee.scoreboard(ui);
            }
            ui.horizontal_wrapped(|ui| {
                if !failed {
                    self.simulation_controls(ui, compact);
                }
                widgets::trailing(ui, 150.0, |ui| {
                    stop = widgets::action(ui, icons::ICON_STOP.codepoint, "Stop simulator", false)
                        .clicked();
                });
            });
            if !failed {
                ui.horizontal_wrapped(|ui| {
                    self.player_controls(ui);
                    widgets::trailing(ui, 130.0, |ui| self.viewport.toolbar(ui, compact));
                });
            }
        });
        stop
    }

    fn simulation_controls(&mut self, ui: &mut Ui, compact: bool) {
        let world = self.widget.bevy_app.world_mut();
        let team = world.resource::<team::Team>().clone();
        let ready = self.physics.ready();
        let paused = world.resource::<SharedPhysics>().lock().mode == SimulationMode::Paused;
        let (icon, label) = if paused {
            (icons::ICON_PLAY_ARROW.codepoint, "Run")
        } else {
            (icons::ICON_PAUSE.codepoint, "Pause")
        };
        if ui
            .add_enabled_ui(self.error.is_none() && ready, |ui| {
                widgets::primary(ui, icon, label)
            })
            .inner
            .on_hover_text("Space also toggles while the Simulator is focused")
            .clicked()
        {
            world.resource::<SharedPhysics>().lock().mode = if paused {
                SimulationMode::Running
            } else {
                SimulationMode::Paused
            };
        }
        ui.label(RichText::new(clock(team.now().duration_since(self.started))).monospace())
            .on_hover_text("Simulation time since start; it stops while paused");
        if !ready {
            ui.spinner();
            ui.weak("Initializing robot…");
        }
        ui.separator();
        if let Some(referee) = team.referee() {
            referee.ui(ui, self.selected);
        }
        if widgets::action(ui, icons::ICON_SPORTS.codepoint, "Whistle", compact)
            .on_hover_text("Send a whistle detection to every robot")
            .clicked()
        {
            if let Some(referee) = team.referee() {
                referee.engine.lock().unwrap().manual_whistle = true;
            } else {
                for member in team.members() {
                    member.io.whistle();
                }
            }
        }
        if ui
            .add_enabled_ui(ready && team.referee().is_none(), |ui| {
                widgets::action(ui, icons::ICON_SPORTS_SOCCER.codepoint, "Add ball", compact)
            })
            .inner
            .on_hover_text("Place a ball one meter from the center")
            .clicked()
        {
            world.resource_scope(|world, assets: Mut<scene::visual::ObjectVisualAssets>| {
                scene::ball::spawn(
                    &mut world.commands(),
                    &assets.ball,
                    Transform::from_xyz(1.0, 0.0, 0.0),
                );
            });
        }
        let mut add_team = None;
        ui.add_enabled_ui(self.pending_robot.is_none() && ready, |ui| {
            ui.menu_button(
                format!("{} Add robot", icons::ICON_PERSON_ADD.codepoint),
                |ui| {
                    for side in TeamId::ALL {
                        let count = team
                            .members()
                            .iter()
                            .filter(|member| member.id.team == side)
                            .count();
                        if ui
                            .add_enabled(
                                count < 5,
                                egui::Button::new(format!("{} ({count}/5)", side.name())),
                            )
                            .clicked()
                        {
                            add_team = Some(side);
                            ui.close();
                        }
                    }
                },
            );
        });
        if let Some(side) = add_team {
            let occupied = {
                let physics = world.resource::<SharedPhysics>().lock();
                physics
                    .robots
                    .iter()
                    .filter_map(|entity| physics.object_pose(*entity))
                    .collect::<Vec<_>>()
            };
            let field = world
                .resource::<parameters::CurrentSimulatorParameters>()
                .field_dimensions;
            if let Some(pose) = team::vacant_spawn(&field, &occupied, team.away(side)) {
                let (sender, receiver) = std::sync::mpsc::channel();
                self.added_robot = Some(receiver);
                let team = team.clone();
                let runtime = team.runtime().clone();
                let repaint = ui.ctx().clone();
                self.pending_robot = Some(tokio_util::task::AbortOnDropHandle::new(
                    runtime.clone().spawn_blocking(move || {
                        let result = runtime
                            .block_on(team.add(side, pose))
                            .map_err(|error| format!("{error:#}"));
                        let _ = sender.send(result);
                        repaint.request_repaint();
                    }),
                ));
            } else {
                self.error = Some("No free sideline spawn position".into());
            }
        }
        if self.pending_robot.is_some() {
            ui.spinner();
        }
    }

    fn player_controls(&mut self, ui: &mut Ui) {
        let world = self.widget.bevy_app.world_mut();
        let selected = world
            .resource::<team::Team>()
            .selected()
            .expect("team always has a selected player");
        let exists = Some(selected.id) == self.selected;
        let name = RichText::new(format!(
            "Player {}",
            self.selected
                .map_or_else(|| "No robot".into(), |id| id.to_string())
        ));
        if exists {
            ui.label(name.strong());
        } else {
            ui.label(name.weak()).on_hover_text(format!(
                "No robot runs as player {}. Add robot starts the next free player.",
                self.selected
                    .map_or_else(|| "No robot".into(), |id| id.to_string())
            ));
        }
        let led = exists.then(|| selected.io.led_color()).flatten();
        ui.weak("LED");
        let (rect, response) = ui.allocate_exact_size(egui::vec2(28.0, 16.0), egui::Sense::hover());
        let color = led.map_or(ui.visuals().faint_bg_color, |color| {
            egui::Color32::from_rgb(color.r, color.g, color.b)
        });
        ui.painter().rect_filled(rect, 2.0, color);
        ui.painter().rect_stroke(
            rect,
            2.0,
            ui.visuals().widgets.noninteractive.bg_stroke,
            egui::StrokeKind::Inside,
        );
        response.on_hover_text(led.map_or_else(
            || "LED: no active command".to_owned(),
            |color| format!("LED: RGB ({}, {}, {})", color.r, color.g, color.b),
        ));
        ui.separator();
        ui.add_enabled_ui(self.physics.ready() && exists, |ui| {
            if let Err(error) = self.buttons.ui(ui, &selected.io) {
                self.error = Some(format!("{error:#}"));
            }
            if widgets::action(ui, icons::ICON_RESTART_ALT.codepoint, "Reset pose", false)
                .on_hover_text("Pause and return this robot to its spawn pose")
                .clicked()
            {
                simulation::reset(world);
            }
        });
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        if let Some(error) = self.physics.poll() {
            widgets::error_banner(ui, &error);
            self.error = Some(error);
            // Do not touch physics after a worker panic may have poisoned its mutex.
            return;
        }
        if let Some(receiver) = &self.added_robot {
            match receiver.try_recv() {
                Ok(result) => {
                    if let Err(error) = result {
                        self.error = Some(error);
                    }
                    self.added_robot = None;
                    self.pending_robot = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.error = Some("Robot startup stopped unexpectedly".into());
                    self.added_robot = None;
                    self.pending_robot = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        let world = self.widget.bevy_app.world_mut();
        let selected = world
            .resource::<team::Team>()
            .selected()
            .expect("team always has a selected player");
        if let Some(error) = selected.io.poll() {
            self.error = Some(error.to_owned());
            world.resource::<SharedPhysics>().lock().mode = SimulationMode::Paused;
        }
        for error in [self.error.as_deref(), self.viewport.error()]
            .into_iter()
            .flatten()
        {
            widgets::error_banner(ui, error);
            ui.add_space(4.0);
        }
        let rect = ui.available_rect_before_wrap();
        self.viewport
            .input(ui, rect, self.widget.bevy_app.world_mut());
        self.widget.ui(ui);
        self.viewport
            .paint(ui, rect, self.widget.bevy_app.world_mut(), self.selected);
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(16));
    }
}

fn clock(elapsed: std::time::Duration) -> String {
    let seconds = elapsed.as_secs_f64();
    format!("{}:{:04.1}", (seconds / 60.0).floor(), seconds % 60.0)
}

/// The same storage-buffer limit used by Twix's legacy Bevy panels.
pub fn configure_renderer(options: &mut eframe::NativeOptions) {
    options.renderer = eframe::Renderer::Wgpu;
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut options.wgpu_options.wgpu_setup {
        let descriptor = setup.device_descriptor.clone();
        setup.device_descriptor = std::sync::Arc::new(move |adapter| {
            let mut descriptor = descriptor(adapter);
            descriptor
                .required_limits
                .max_storage_buffers_per_shader_stage = 9;
            descriptor
        });
    }
}

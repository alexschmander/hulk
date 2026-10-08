//! Twix's simulator. All hardware substitutes and node wiring live in this tool.
use std::sync::atomic::{AtomicBool, Ordering};

use bevy::prelude::*;
use color_eyre::{Result, eyre::ensure};
use eframe::{
    egui::{self, Ui, Widget},
    egui_wgpu::RenderState,
};
use egui_bevy::BevyWidget;
use tokio::runtime::Handle;

mod controller;
mod field;
mod network;
mod team;
pub use field::FieldConfiguration;
mod behavior_inputs;
mod bevy_mujoco;
mod buttons;
mod observations;
mod parameters;
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
    added_robot: Option<std::sync::mpsc::Receiver<Result<u8, String>>>,
    selected: u8,
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
            selected: 1,
        }
    }

    pub fn select_robot(&mut self, number: u8) {
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

    pub fn ui(&mut self, ui: &mut Ui) {
        if let Some(error) = self.physics.poll() {
            ui.colored_label(egui::Color32::RED, &error);
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
        let team = world.resource::<team::Team>().clone();
        let selected = team.selected().expect("team always has a selected player");
        let selection_exists = selected.number == self.selected;
        if let Some(error) = selected.io.poll() {
            self.error = Some(error.to_owned());
            world.resource::<SharedPhysics>().lock().mode = SimulationMode::Paused;
        }
        ui.horizontal_wrapped(|ui| {
            let paused = world.resource::<SharedPhysics>().lock().mode == SimulationMode::Paused;
            if ui
                .add_enabled(
                    self.error.is_none() && self.physics.ready(),
                    egui::Button::new(if paused { "Run" } else { "Pause" }),
                )
                .clicked()
            {
                world.resource::<SharedPhysics>().lock().mode = if paused {
                    SimulationMode::Running
                } else {
                    SimulationMode::Paused
                };
            }
            if ui
                .add_enabled(
                    self.physics.ready() && selection_exists,
                    egui::Button::new("Reset pose"),
                )
                .clicked()
            {
                simulation::reset(world);
            }
            ui.add_enabled_ui(self.physics.ready() && selection_exists, |ui| {
                if let Err(error) = self.buttons.ui(ui, &selected.io) {
                    self.error = Some(format!("{error:#}"));
                }
            });
            if ui
                .button("Whistle")
                .on_hover_text("Inject a whistle detection pulse")
                .clicked()
            {
                for member in team.members() {
                    member.io.whistle();
                }
            }
            self.viewport.toolbar(ui);
            if ui
                .add_enabled(self.physics.ready(), egui::Button::new("Add ball"))
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
            if ui
                .add_enabled(
                    team.members().len() < 5
                        && self.pending_robot.is_none()
                        && self.physics.ready(),
                    egui::Button::new("Add robot"),
                )
                .clicked()
            {
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
                if let Some(pose) = team::vacant_spawn(&field, &occupied, selected.io.away()) {
                    let (sender, receiver) = std::sync::mpsc::channel();
                    self.added_robot = Some(receiver);
                    let team = team.clone();
                    let runtime = team.runtime().clone();
                    let repaint = ui.ctx().clone();
                    self.pending_robot = Some(tokio_util::task::AbortOnDropHandle::new(
                        runtime.clone().spawn_blocking(move || {
                            let result = runtime
                                .block_on(team.add(pose))
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
            let led = selection_exists.then(|| selected.io.led_color()).flatten();
            ui.label("LED");
            let (rect, response) =
                ui.allocate_exact_size(egui::vec2(28.0, 16.0), egui::Sense::hover());
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
            if !self.physics.ready() {
                ui.spinner();
                ui.label("Initializing robot…");
            }
        });
        if let Some(error) = &self.error {
            ui.colored_label(egui::Color32::RED, error);
        }
        let rect = ui.available_rect_before_wrap();
        self.viewport
            .input(ui, rect, self.widget.bevy_app.world_mut());
        self.widget.ui(ui);
        self.viewport
            .paint(ui, rect, self.widget.bevy_app.world_mut());
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(16));
    }
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

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
    robotics: robotics::Robotics,
    lease: Lease,
}
impl PreparedSimulator {
    pub async fn new(runtime: Handle, configuration: Configuration) -> Result<Self> {
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
        let robotics = robotics::Robotics::new(runtime, configuration).await?;
        Ok(Self { robotics, lease })
    }
}

pub struct Simulator {
    // Stop and join physics before the Bevy world drops its node resources.
    physics: simulation::PhysicsWorker,
    widget: BevyWidget,
    _lease: Lease,
    buttons: buttons::BodyButtons,
    viewport: viewport::Viewport,
    error: Option<String>,
}

impl Simulator {
    pub fn new(prepared: PreparedSimulator, renderer: RenderState) -> Self {
        let mut widget = BevyWidget::new(renderer);
        let app = &mut widget.bevy_app;
        app.add_plugins(parameters::SimulatorParametersPlugin::new(
            prepared.robotics.parameters.clone(),
            prepared.robotics.field_dimensions,
        ))
        .insert_resource(prepared.robotics)
        .add_plugins((
            bevy_mujoco::MujocoWorldPlugin,
            scene::field::FieldPlugin,
            scene::ObjectsPlugin,
            simulation::MotionSimulationPlugin,
        ))
        .add_systems(PreUpdate, viewport::setup_scene);
        app.finish();
        app.cleanup();
        let physics = simulation::PhysicsWorker::start(
            app.world().resource::<SharedPhysics>().clone(),
            app.world().resource::<robotics::Robotics>().clone(),
        );
        Self {
            physics,
            widget,
            _lease: prepared.lease,
            buttons: buttons::BodyButtons::default(),
            viewport: viewport::Viewport::default(),
            error: None,
        }
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
        let world = self.widget.bevy_app.world_mut();
        if let Some(error) = world.resource_mut::<robotics::Robotics>().poll() {
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
                .add_enabled(self.physics.ready(), egui::Button::new("Reset pose"))
                .clicked()
            {
                simulation::reset(world);
            }
            ui.add_enabled_ui(self.physics.ready(), |ui| {
                if let Err(error) = self.buttons.ui(ui, world.resource::<robotics::Robotics>()) {
                    self.error = Some(format!("{error:#}"));
                }
            });
            if ui
                .button("Whistle")
                .on_hover_text("Inject a whistle detection pulse")
                .clicked()
            {
                world.resource::<robotics::Robotics>().whistle();
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
            let led = world.resource::<robotics::Robotics>().led_color();
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

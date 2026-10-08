//! Twix's motion simulator. All hardware substitutes and node wiring live in this tool.
use std::sync::atomic::{AtomicBool, Ordering};

use bevy::{camera_controller::pan_orbit_camera::prelude::PanOrbitCamera, prelude::*};
use color_eyre::{Result, eyre::ensure};
use eframe::{
    egui::{self, Ui, Widget},
    egui_wgpu::RenderState,
};
use egui_bevy::BevyWidget;
use tokio::runtime::Handle;

mod behavior_inputs;
mod bevy_mujoco;
mod parameters;
mod robot_io;
mod robotics;
mod scene;
mod simulated_sdk;
mod simulation;

use bevy_mujoco::{SharedPhysics, SimulationMode};
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
    placement: [f32; 6],
    ball_position: [f32; 2],
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
        .add_systems(Update, setup_camera);
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
            placement: [0.0; 6],
            ball_position: [1.0, 0.0],
            error: None,
        }
    }

    pub fn toggle_pause(&mut self) {
        if self.error.is_some() {
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
                    self.error.is_none(),
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
            if ui.button("Reset pose").clicked() {
                simulation::reset(world);
            }
            for (name, long) in [("Stand: prepare", false), ("Stand: enable", true)] {
                if ui.button(name).clicked()
                    && let Err(error) = world.resource::<robotics::Robotics>().press_stand(long)
                {
                    self.error = Some(format!("{error:#}"));
                }
            }
            if ui.button("Add ball").clicked() {
                let position = self.ball_position;
                world.resource_scope(|world, assets: Mut<scene::visual::ObjectVisualAssets>| {
                    scene::ball::spawn(
                        &mut world.commands(),
                        &assets.ball,
                        Transform::from_xyz(position[0], 0.0, -position[1]),
                    );
                });
            }
            ui.label(world.resource::<robotics::Robotics>().status());
        });
        ui.collapsing("Placement", |ui| {
            ui.horizontal(|ui| {
                ui.label("New ball x/y (m)");
                for value in &mut self.ball_position {
                    ui.add(egui::DragValue::new(value).speed(0.05));
                }
                if ui.button("Remove balls").clicked() {
                    let balls = world.resource::<scene::ball::SpawnedBalls>().0.clone();
                    for ball in balls {
                        world.despawn(ball);
                    }
                }
            });
            let paused = world.resource::<SharedPhysics>().lock().mode == SimulationMode::Paused;
            ui.add_enabled_ui(paused, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label("Robot x/y/z (m), roll/pitch/yaw (degrees)");
                    for value in &mut self.placement {
                        ui.add(egui::DragValue::new(value).speed(0.05));
                    }
                    if ui.button("Place robot").clicked() {
                        let robot = world
                            .query_filtered::<Entity, With<simulation::ControlledRobot>>()
                            .single(world);
                        if let Ok(robot) = robot {
                            let [x, y, z, roll, pitch, yaw] = self.placement;
                            let height = world
                                .resource::<scene::visual::ObjectVisualAssets>()
                                .robot
                                .ground_offset();
                            let rotation = Quat::from_rotation_y(yaw.to_radians())
                                * Quat::from_rotation_z(-pitch.to_radians())
                                * Quat::from_rotation_x(roll.to_radians());
                            if let Err(error) =
                                world.resource::<SharedPhysics>().lock().set_object_pose(
                                    robot,
                                    Transform::from_xyz(x, z + height, -y).with_rotation(rotation),
                                )
                            {
                                self.error = Some(error);
                            }
                        }
                    }
                });
            });
        });
        if let Some(error) = &self.error {
            ui.colored_label(egui::Color32::RED, error);
        }
        ui.label("Drag to orbit; right-drag to pan; scroll to zoom. Edit motion overrides and parameters in Twix's Parameter panel.");
        self.widget.ui(ui);
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(16));
    }
}

fn setup_camera(mut cameras: Query<(&mut Transform, &mut PanOrbitCamera), Added<PanOrbitCamera>>) {
    for (mut transform, mut orbit) in &mut cameras {
        *transform = Transform::from_xyz(4.0, 4.0, 7.0).looking_at(Vec3::ZERO, Vec3::Y);
        *orbit = PanOrbitCamera {
            last_anchor_depth: -transform.translation.length() as f64,
            ..Default::default()
        };
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

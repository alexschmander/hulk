//! Physics and sensor publication run independently of Twix's repaint rate and active tab.
use crate::{
    bevy_mujoco::{SharedPhysics, SimulationMode},
    parameters::CurrentSimulatorParameters,
    robot_io::RobotBinding,
    robotics::Robotics,
    scene::{robot, visual::ObjectVisualAssets},
};
use bevy::prelude::*;
use std::{
    f32::consts::FRAC_PI_2,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
use types::field_dimensions::FieldDimensions;

#[derive(Component)]
pub struct ControlledRobot;

pub struct MotionSimulationPlugin;
impl Plugin for MotionSimulationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_robot);
    }
}
fn initial_pose(field: &FieldDimensions) -> Transform {
    // Localization's prior is (-length/2, -width/2, yaw +90°) in field coordinates.
    // Bevy uses (field x, height, -field y); physics sets the standing height.
    Transform::from_xyz(-field.length / 2.0, 0.0, field.width / 2.0)
        .with_rotation(Quat::from_rotation_y(FRAC_PI_2))
}

fn spawn_robot(
    mut commands: Commands,
    assets: Res<ObjectVisualAssets>,
    parameters: Res<CurrentSimulatorParameters>,
) {
    let robot = robot::spawn(
        &mut commands,
        &assets.robot,
        initial_pose(&parameters.field_dimensions),
    );
    commands.entity(robot).insert(ControlledRobot);
}

pub struct PhysicsWorker {
    stop: Arc<AtomicBool>,
    failure: Option<String>,
    ready: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<Result<(), String>>>,
}
impl PhysicsWorker {
    pub fn ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    pub fn poll(&mut self) -> Option<String> {
        if self.failure.is_none()
            && self
                .thread
                .as_ref()
                .is_some_and(|thread| thread.is_finished())
        {
            self.failure = Some(match self.thread.take().unwrap().join() {
                Ok(Err(error)) => error,
                Ok(Ok(())) => "Physics worker exited unexpectedly".into(),
                Err(_) => "Physics worker panicked".into(),
            });
        }
        self.failure.clone()
    }
    pub fn start(physics: SharedPhysics, io: Robotics) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let ready = Arc::new(AtomicBool::new(false));
        let initialized = ready.clone();
        let thread = thread::spawn(move || {
            let mut startup = crate::buttons::Startup::default();
            let mut placed = false;
            let mut generation = None;
            let mut binding = None;
            let mut controller = crate::simulated_sdk::Controller::default();
            let mut last_frame = None;
            let mut sequence = 0;
            let mut startup_complete = false;
            let startup_deadline = Instant::now();
            let mut period = Duration::from_millis(2);
            while !stopped.load(Ordering::Acquire) {
                let started = Instant::now();
                let failed = io.poll().is_some();
                let mut sample = None;
                let mut frame = None;
                // Both physics and robotics time stop during a scene rebuild.
                // Never restamp a frozen observation or catch up elapsed wall time.
                if let Some(mut world) = physics.try_lock() {
                    if failed {
                        world.mode = SimulationMode::Paused;
                    }
                    if generation != Some(world.generation) {
                        binding = world
                            .robot
                            .map(|robot| {
                                RobotBinding::new(
                                    world.data(),
                                    &format!("object_{}_", robot.to_bits()),
                                )
                                .map_err(|error| format!("Robot binding failed: {error:#}"))
                            })
                            .transpose()?;
                        generation = Some(world.generation);
                    }
                    period = Duration::from_secs_f64(world.data().model_opt().timestep);
                    if let Some(robot) = &binding {
                        if !placed {
                            robot.reset_joints(world.data_mut());
                            let entity = world.robot.expect("bound robot");
                            world.ground_object(entity, initial_pose(&io.field_dimensions))?;
                            placed = true;
                        }
                        if !startup_complete
                            && io.actuator_control().mode == booster::RobotMode::Prepare
                        {
                            world.mode = SimulationMode::Running;
                        }
                        // Startup must run the real button/mode pipeline and fill
                        // sensor caches before exposing the initially paused scene.
                        if !failed
                            && (world.mode == SimulationMode::Running
                                || !initialized.load(Ordering::Acquire))
                        {
                            let time = io.now() + period;
                            if world.mode == SimulationMode::Running {
                                controller.apply(robot, world.data_mut(), &io.actuator_control());
                                world.data_mut().step();
                            }
                            world.data_mut().forward();
                            let mut observation = robot.observe(world.data());
                            if world.mode == SimulationMode::Paused {
                                observation.stationary();
                            }
                            if last_frame.is_none_or(|last| {
                                time.duration_since(last) >= Duration::from_millis(33)
                            }) {
                                sequence += 1;
                                last_frame = Some(time);
                                let mut balls = crate::observations::balls(&world);
                                if world.mode == SimulationMode::Paused {
                                    for ball in &mut balls {
                                        ball.velocity = [0.0; 3];
                                    }
                                }
                                let field = io
                                    .field
                                    .get_latest()
                                    .map(|f| *f)
                                    .unwrap_or(io.field_dimensions);
                                let detections = crate::observations::detect(
                                    world.data_mut(),
                                    &observation,
                                    &balls,
                                    &field,
                                );
                                frame = Some(crate::observations::Frame {
                                    time,
                                    sequence,
                                    epoch: world.motion_epoch,
                                    sample: observation.clone(),
                                    balls,
                                    detections,
                                    field,
                                });
                            }
                            sample = Some((observation, time));
                        }
                    }
                }
                if let Some((sample, time)) = sample {
                    // Make the sensors available before waking robotics timers.
                    io.publish_observation(&sample, time)
                        .map_err(|error| format!("Simulator sensor publication: {error:#}"))?;
                    io.advance_time(period)
                        .map_err(|error| format!("Simulator clock: {error:#}"))?;
                    if let Some(frame) = frame {
                        io.publish_frame(frame);
                    }
                    if !startup_complete && startup.advance(&io)? {
                        physics.lock().mode = SimulationMode::Paused;
                        startup_complete = true;
                    }
                    if startup_complete && io.ready() {
                        initialized.store(true, Ordering::Release);
                    }
                }
                if !initialized.load(Ordering::Acquire)
                    && startup_deadline.elapsed() > Duration::from_secs(20)
                {
                    return Err(format!("Simulator startup timed out: {}", io.status()));
                }
                // No burst of stale catch-up samples after rendering or model recompilation.
                thread::sleep(period.saturating_sub(started.elapsed()));
            }
            Ok(())
        });
        Self {
            stop,
            failure: None,
            ready,
            thread: Some(thread),
        }
    }
}
impl Drop for PhysicsWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

pub fn reset(world: &mut World) {
    let pose = initial_pose(
        &world
            .resource::<CurrentSimulatorParameters>()
            .field_dimensions,
    );
    let mut physics = world.resource::<SharedPhysics>().lock();
    physics.mode = SimulationMode::Paused;
    if let Some(robot) = physics.robot {
        let binding = RobotBinding::new(physics.data(), &format!("object_{}_", robot.to_bits()))
            .expect("controlled robot joints");
        binding.reset_joints(physics.data_mut());
        physics
            .ground_object(robot, pose)
            .expect("reset controlled robot");
    }
}

#[cfg(test)]
#[path = "profile_tests.rs"]
mod tests;

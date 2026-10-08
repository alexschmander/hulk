//! Physics and sensor publication run independently of Twix's repaint rate and active tab.
use crate::{
    bevy_mujoco::{SharedPhysics, SimulationMode},
    robot_io::RobotBinding,
    robotics::Robotics,
    scene::{robot, visual::ObjectVisualAssets},
};
use bevy::prelude::*;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

#[derive(Component)]
pub struct ControlledRobot;

pub struct MotionSimulationPlugin;
impl Plugin for MotionSimulationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_robot);
    }
}
fn spawn_robot(mut commands: Commands, assets: Res<ObjectVisualAssets>) {
    let robot = robot::spawn(&mut commands, &assets.robot, Transform::default());
    commands.entity(robot).insert(ControlledRobot);
}

pub struct PhysicsWorker {
    stop: Arc<AtomicBool>,
    failure: Option<String>,
    thread: Option<thread::JoinHandle<Result<(), String>>>,
}
impl PhysicsWorker {
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
        let thread = thread::spawn(move || {
            let mut generation = None;
            let mut binding = None;
            let mut controller = crate::simulated_sdk::Controller::default();
            let mut last_world = Instant::now();
            while !stopped.load(Ordering::Acquire) {
                let started = Instant::now();
                let failed = io.poll().is_some();
                let mut world = physics.lock();
                if failed {
                    world.mode = SimulationMode::Paused;
                }
                if generation != Some(world.generation) {
                    binding = world
                        .robot
                        .map(|robot| {
                            RobotBinding::new(world.data(), &format!("object_{}_", robot.to_bits()))
                                .map_err(|error| format!("Robot binding failed: {error:#}"))
                        })
                        .transpose()?;
                    generation = Some(world.generation);
                }
                let period = Duration::from_secs_f64(world.data().model_opt().timestep);
                if let Some(robot) = &binding {
                    if world.mode == SimulationMode::Running {
                        controller.apply(robot, world.data_mut(), &io.actuator_control());
                        world.data_mut().step();
                    }
                    world.data_mut().forward();
                    let observation = robot.observe(world.data());
                    let ground = robot.ground_to_world(world.data());
                    let balls = crate::scene::ball::SpawnedBalls(world.balls.clone());
                    let ball = crate::scene::ball::first_position(&world, &balls)
                        .ok()
                        .zip(crate::scene::ball::first_velocity(&world, &balls).ok());
                    drop(world);
                    io.publish_observation(observation)
                        .map_err(|error| format!("Simulator sensor publication: {error:#}"))?;
                    if last_world.elapsed() >= Duration::from_millis(20) {
                        io.publish_world(ground, ball)
                            .map_err(|error| format!("Simulator world publication: {error:#}"))?;
                        last_world = Instant::now();
                    }
                } else {
                    drop(world);
                }
                // No burst of stale catch-up samples after rendering or model recompilation.
                thread::sleep(period.saturating_sub(started.elapsed()));
            }
            Ok(())
        });
        Self {
            stop,
            failure: None,
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
    let height = world.resource::<ObjectVisualAssets>().robot.ground_offset();
    let mut physics = world.resource::<SharedPhysics>().lock();
    physics.mode = SimulationMode::Paused;
    if let Some(robot) = physics.robot {
        let binding = RobotBinding::new(physics.data(), &format!("object_{}_", robot.to_bits()))
            .expect("controlled robot joints");
        binding.reset_joints(physics.data_mut());
        physics
            .set_object_pose(robot, Transform::from_xyz(0.0, height, 0.0))
            .expect("reset controlled robot");
    }
}

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
            let mut last_world = Instant::now();
            let mut snapshot = None;
            let mut period = Duration::from_millis(2);
            while !stopped.load(Ordering::Acquire) {
                let started = Instant::now();
                let failed = io.poll().is_some();
                // Scene recompilation freezes physics, but must not starve the real
                // nodes' sensor deadlines. Keep publishing the frozen scene snapshot.
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
                        if !initialized.load(Ordering::Acquire)
                            && io.actuator_control().mode == booster::RobotMode::Prepare
                        {
                            world.mode = SimulationMode::Running;
                        }
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
                        snapshot = Some((observation, ground, ball));
                    }
                }
                if let Some((observation, ground, ball)) = &snapshot {
                    io.publish_observation(observation)
                        .map_err(|error| format!("Simulator sensor publication: {error:#}"))?;
                    if !initialized.load(Ordering::Acquire) && startup.advance(&io)? {
                        physics.lock().mode = SimulationMode::Paused;
                        initialized.store(true, Ordering::Release);
                    }
                    if last_world.elapsed() >= Duration::from_millis(20) {
                        io.publish_world(*ground, *ball)
                            .map_err(|error| format!("Simulator world publication: {error:#}"))?;
                        last_world = Instant::now();
                    }
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
mod tests {
    use super::*;
    use crate::{
        bevy_mujoco::{MjcfObject, MujocoWorldPlugin},
        robotics::Configuration,
    };
    use ros_z::prelude::*;
    use types::primary_state::PrimaryState;

    #[test]
    #[ignore = "requires ONNX Runtime, motion models, and free GameController UDP ports"]
    fn startup_reaches_initial_and_scene_edits_keep_sensors_live() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let router = runtime
            .block_on(
                ContextBuilder::default()
                    .with_mode("router")
                    .disable_multicast_scouting()
                    .with_connect_endpoints(std::iter::empty::<&str>())
                    .with_listen_endpoints(["tcp/127.0.0.1:0"])
                    .build(),
            )
            .unwrap();
        let endpoint = runtime
            .block_on(async { router.session().info().locators().await })
            .first()
            .unwrap()
            .to_string();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let io = runtime
            .block_on(Robotics::new(
                runtime.handle().clone(),
                Configuration {
                    parameter_root: root.join("../../etc/parameters"),
                    model_directory: root.join("../../etc/neural_networks"),
                    router: Some(endpoint),
                    namespace: "/simulator/startup_test".into(),
                },
            ))
            .unwrap();
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, MujocoWorldPlugin));
        let robot = app
            .world_mut()
            .spawn((
                MjcfObject::new(root.join("assets/k1_robot.xml"), "Trunk")
                    .with_free_joint("world_joint")
                    .grounded(),
                Transform::default(),
            ))
            .id();
        app.update();
        let physics = app.world().resource::<SharedPhysics>().clone();
        let mut worker = PhysicsWorker::start(physics.clone(), io.clone());
        let started = Instant::now();
        while !worker.ready() {
            assert!(worker.poll().is_none());
            assert!(started.elapsed() < Duration::from_secs(12));
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(*io.primary.get_latest().unwrap(), PrimaryState::Initial);
        assert_eq!(physics.lock().mode, SimulationMode::Paused);
        let pose = physics.lock().object_pose(robot).unwrap();
        assert!(pose.translation.y > 0.4);
        assert!((pose.translation.x + io.field_dimensions.length / 2.0).abs() < 0.1);
        assert!((pose.translation.z - io.field_dimensions.width / 2.0).abs() < 0.1);
        assert!((pose.rotation * Vec3::X).distance(Vec3::NEG_Z) < 0.1);
        // Deliberately hold the same mutex as model recompilation, beyond the real
        // 40/50 ms inference/sensor deadlines. Physics freezes, publications continue.
        let scene_edit = physics.lock();
        thread::sleep(Duration::from_millis(300));
        assert!(
            io.safe_pose
                .latest_stamp()
                .is_some_and(|stamp| io.now().duration_since(stamp) < Duration::from_millis(50))
        );
        assert_eq!(*io.primary.get_latest().unwrap(), PrimaryState::Initial);
        assert!(!io.status().contains("Emergency stop"));
        drop(scene_edit);
        assert!(worker.poll().is_none());
        drop(worker);
        drop(io);
        router.shutdown().unwrap();
    }
}

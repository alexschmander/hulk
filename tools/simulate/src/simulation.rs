//! Physics and sensor publication run independently of Twix's repaint rate and active tab.
use crate::{
    bevy_mujoco::{SharedPhysics, SimulationMode},
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
        app.add_systems(PreUpdate, (spawn_robot, spawn_match_ball));
    }
}
fn spawn_match_ball(
    mut commands: Commands,
    assets: Res<ObjectVisualAssets>,
    team: Res<crate::team::Team>,
    mut done: Local<bool>,
) {
    if !*done && team.referee().is_some() {
        crate::scene::ball::spawn(&mut commands, &assets.ball, Transform::default());
    }
    *done = true;
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
    team: Res<crate::team::Team>,
) {
    for member in team.members() {
        if member.entity.is_some() {
            continue;
        }
        let robot = robot::spawn(&mut commands, &assets.robot, member.pose);
        commands.entity(robot).insert(ControlledRobot);
        team.bind(member.id, robot);
    }
}

/// Fill the new robot's sensors and operate its real body buttons at frozen time.
/// The prepared pose already matches firmware Prepare; no other robot needs to step.
pub(crate) fn bootstrap(io: &Robotics, cancelled: &AtomicBool) -> color_eyre::Result<()> {
    thread::scope(|scope| {
        scope
            .spawn(|| {
                let mut app = App::new();
                app.add_plugins((MinimalPlugins, crate::bevy_mujoco::MujocoWorldPlugin));
                let entity = app
                    .world_mut()
                    .spawn((
                        crate::bevy_mujoco::MjcfObject::new(
                            concat!(env!("CARGO_MANIFEST_DIR"), "/assets/k1_robot.xml"),
                            "Trunk",
                        )
                        .with_free_joint("world_joint")
                        .grounded(),
                        Transform::default(),
                    ))
                    .id();
                app.update();
                let physics = app.world().resource::<SharedPhysics>().clone();
                let observation = {
                    let mut world = physics.lock();
                    let binding =
                        RobotBinding::new(world.data(), &format!("object_{}_", entity.to_bits()))?;
                    binding.reset_joints(world.data_mut());
                    world
                        .ground_object(entity, initial_pose(&io.field_dimensions))
                        .map_err(|error| color_eyre::eyre::eyre!(error))?;
                    let mut observation = binding.observe(world.data());
                    observation.stationary();
                    observation
                };
                let mut startup = crate::buttons::Startup::default();
                let deadline = Instant::now() + Duration::from_secs(20);
                loop {
                    color_eyre::eyre::ensure!(
                        !cancelled.load(Ordering::Acquire),
                        "Robot startup cancelled"
                    );
                    color_eyre::eyre::ensure!(
                        Instant::now() < deadline,
                        "Robot startup timed out: {}",
                        io.status()
                    );
                    if let Some(error) = io.poll() {
                        return Err(color_eyre::eyre::eyre!(error));
                    }
                    io.publish_observation(&observation, io.now())?;
                    if io.inference.get_latest().is_some_and(|status| {
                        matches!(status.state, motion_inference::node::State::Initialized)
                    }) && startup
                        .advance(io)
                        .map_err(|error| color_eyre::eyre::eyre!(error))?
                    {
                        break;
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                Ok(())
            })
            .join()
            .map_err(|_| color_eyre::eyre::eyre!("Robot initialization panicked"))?
    })
}

pub struct PhysicsWorker {
    stop: Arc<AtomicBool>,
    failure: Option<String>,
    ready: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<Result<(), String>>>,
}
impl PhysicsWorker {
    pub(crate) fn start_team(physics: SharedPhysics, team: crate::team::Team) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let ready = Arc::new(AtomicBool::new(false));
        let initialized = ready.clone();
        let thread = thread::spawn(move || {
            struct State {
                binding: RobotBinding,
                controller: crate::simulated_sdk::Controller,
            }
            let mut states = std::collections::BTreeMap::<crate::RobotId, State>::new();
            let mut placed = std::collections::HashSet::new();
            let mut generation = None;
            let mut last_frame = None;
            let mut sequence = 0;
            while !stopped.load(Ordering::Acquire) {
                let started = Instant::now();
                if let Some(error) = team.poll_controller() {
                    return Err(error);
                }
                let members = team.members();
                for member in &members {
                    if let Some(error) = member.io.poll() {
                        return Err(format!("Player {}: {error}", member.id));
                    }
                }
                let mut period = Duration::from_millis(2);
                let mut samples = Vec::new();
                let mut advanced = false;
                if let Some(mut world) = physics.try_lock() {
                    let rebuilt = generation != Some(world.generation);
                    if rebuilt {
                        let mut previous = std::mem::take(&mut states);
                        for member in &members {
                            if let Some(entity) = member.entity
                                && world.contains_object(entity)
                            {
                                let binding = RobotBinding::new(
                                    world.data(),
                                    &format!("object_{}_", entity.to_bits()),
                                )
                                .map_err(|error| {
                                    format!("Player {} binding: {error:#}", member.id)
                                })?;
                                if placed.insert(member.id) {
                                    binding.reset_joints(world.data_mut());
                                    world.ground_object(entity, member.pose)?;
                                }
                                states.insert(
                                    member.id,
                                    State {
                                        binding,
                                        controller: previous
                                            .remove(&member.id)
                                            .map(|state| state.controller)
                                            .unwrap_or_default(),
                                    },
                                );
                            }
                        }
                        generation = Some(world.generation);
                    }
                    period = Duration::from_secs_f64(world.data().model_opt().timestep);
                    let running = world.mode == SimulationMode::Running;
                    let ready = states.len() == members.len();
                    initialized.store(ready, Ordering::Release);
                    if running || !ready || rebuilt {
                        let time = team.now();
                        if running {
                            for member in &members {
                                if let Some(state) = states.get_mut(&member.id) {
                                    state.controller.apply(
                                        &state.binding,
                                        world.data_mut(),
                                        &member.io.actuator_control(),
                                    );
                                }
                            }
                            world.data_mut().step();
                            advanced = true;
                        }
                        world.data_mut().forward();
                        if running && let Some(referee) = team.referee() {
                            referee.update(&mut world, &members, period)?;
                        }
                        let observations: Vec<_> = members
                            .iter()
                            .filter_map(|member| {
                                let mut observation =
                                    states.get(&member.id)?.binding.observe(world.data());
                                if !running {
                                    observation.stationary();
                                }
                                Some((member, observation))
                            })
                            .collect();
                        let frame_due = rebuilt
                            || !ready
                            || last_frame.is_none_or(|last| {
                                time.duration_since(last) >= Duration::from_millis(33)
                            });
                        if frame_due {
                            sequence += 1;
                            last_frame = Some(time);
                        }
                        let balls = crate::observations::balls(&world);
                        for (member, observation) in &observations {
                            let frame = if frame_due {
                                let others = observations
                                    .iter()
                                    .filter(|(other, _)| other.id != member.id)
                                    .map(|(_, other)| {
                                        let p = other.ground_to_world.translation.vector;
                                        [p.x, p.y, p.z]
                                    })
                                    .collect::<Vec<_>>();
                                let field = member.io.field_dimensions;
                                let detections = crate::observations::detect_scene(
                                    world.data_mut(),
                                    observation,
                                    &balls,
                                    &field,
                                    &others,
                                    team.configuration()
                                        .field_configuration
                                        .as_ref()
                                        .unwrap()
                                        .goal_height,
                                );
                                Some(crate::observations::Frame {
                                    time,
                                    sequence,
                                    epoch: world.object_epoch(member.entity.unwrap()),
                                    sample: observation.clone(),
                                    balls: balls.clone(),
                                    robots: others,
                                    detections,
                                    field,
                                })
                            } else {
                                None
                            };
                            samples.push((member.io.clone(), observation.clone(), time, frame));
                        }
                    }
                }
                for (io, observation, time, _) in &samples {
                    io.publish_observation(observation, *time)
                        .map_err(|error| format!("Sensor publication: {error:#}"))?;
                }
                if advanced {
                    team.advance(period)
                        .map_err(|error| format!("Simulation clock: {error:#}"))?;
                }
                for (io, _, _, frame) in samples {
                    if let Some(frame) = frame {
                        io.publish_frame(frame);
                    }
                }
                thread::sleep(period.saturating_sub(started.elapsed()));
            }
            Ok(())
        });
        Self {
            stop,
            ready,
            failure: None,
            thread: Some(thread),
        }
    }

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
    #[cfg(test)]
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
                            .robots
                            .first()
                            .copied()
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
                            let entity = *world.robots.first().expect("bound robot");
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
                                    robots: Vec::new(),
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
    let mut physics = world.resource::<SharedPhysics>().lock();
    physics.mode = SimulationMode::Paused;
    if let Some(member) = world.resource::<crate::team::Team>().selected()
        && let Some(robot) = member.entity
    {
        let binding = RobotBinding::new(physics.data(), &format!("object_{}_", robot.to_bits()))
            .expect("controlled robot joints");
        binding.reset_joints(physics.data_mut());
        physics
            .ground_object(
                robot,
                crate::team::on_field_side(
                    member.pose,
                    (member.pose.translation.x > 0.0) != member.io.away(),
                ),
            )
            .expect("reset controlled robot");
    }
}

#[cfg(test)]
#[path = "profile_tests.rs"]
mod tests;

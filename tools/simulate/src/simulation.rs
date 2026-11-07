//! Physics and sensor publication run independently of Twix's repaint rate and active tab.
use crate::{
    bevy_mujoco::{SharedPhysics, SimulationMode},
    robot_io::{GroundContacts, RobotBinding},
    robotics::Robotics,
    scene::{robot, visual::ObjectVisualAssets},
};
use bevy::prelude::*;
use rayon::prelude::*;
use std::{
    f32::consts::FRAC_PI_2,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, AtomicU32, Ordering},
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

// Reserve control time before the shared timeline so a new robot can initialize
// without advancing an existing match or rewinding its own timers at handover.
pub(crate) const STARTUP_WARMUP: Duration = Duration::from_secs(2);

/// Run the real Damping -> Prepare -> Initial control path against a held pose,
/// then catch up to the shared clock before adding this robot to the scene.
pub(crate) fn bootstrap(
    io: &Robotics,
    clock: &ros_z::time::Clock,
    cancelled: &AtomicBool,
) -> color_eyre::Result<()> {
    thread::scope(|scope| {
        scope
            .spawn(|| {
                let mut app = App::new();
                app.add_plugins((MinimalPlugins, crate::bevy_mujoco::MujocoWorldPlugin));
                let entity = app
                    .world_mut()
                    .spawn((
                        crate::bevy_mujoco::MjcfObject::from_shared_file(
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
                // This observer is only needed during warm-up. A completed
                // command cycle acknowledges each clock step before catch-up.
                let commands = io.startup_commands()?;
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
                    let initialized = io.inference.get_latest().is_some_and(|status| {
                        matches!(status.state, motion_inference::node::State::Initialized)
                    });
                    let complete = initialized
                        && startup
                            .advance(io)
                            .map_err(|error| color_eyre::eyre::eyre!(error))?;
                    let current = io.now();
                    let command_current = commands
                        .latest_stamp()
                        .is_some_and(|stamp| stamp >= current);
                    let target = clock.now();
                    if complete && command_current && current == target {
                        break;
                    }
                    if initialized && current < target && (!complete || command_current) {
                        // After handover, catch up in policy-sized steps, keeping
                        // sensor and command ages below their production limits.
                        let period = Duration::from_millis(if complete { 20 } else { 2 });
                        io.advance_time(period.min(target.duration_since(current)))?;
                    }
                    thread::sleep(Duration::from_millis(if initialized { 1 } else { 10 }));
                }
                Ok(())
            })
            .join()
            .map_err(|_| color_eyre::eyre::eyre!("Robot initialization panicked"))?
    })
}

/// Playback changes wall pacing, never the physics timestep or robotics time per step.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Speed {
    Quarter,
    Half,
    #[default]
    Normal,
    Double,
    Quadruple,
    Octuple,
    Sixteenfold,
}
impl Speed {
    pub const ALL: [Self; 7] = [
        Self::Quarter,
        Self::Half,
        Self::Normal,
        Self::Double,
        Self::Quadruple,
        Self::Octuple,
        Self::Sixteenfold,
    ];
    pub fn factor(self) -> f64 {
        match self {
            Self::Quarter => 0.25,
            Self::Half => 0.5,
            Self::Normal => 1.0,
            Self::Double => 2.0,
            Self::Quadruple => 4.0,
            Self::Octuple => 8.0,
            Self::Sixteenfold => 16.0,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Quarter => "¼×",
            Self::Half => "½×",
            Self::Normal => "1×",
            Self::Double => "2×",
            Self::Quadruple => "4×",
            Self::Octuple => "8×",
            Self::Sixteenfold => "16×",
        }
    }
}
struct Playback {
    speed: AtomicU8,
    actual: AtomicU32,
}
impl Default for Playback {
    fn default() -> Self {
        Self {
            speed: AtomicU8::new(Speed::Normal as u8),
            actual: AtomicU32::new(0),
        }
    }
}
impl Playback {
    fn speed(&self) -> Speed {
        Speed::ALL[self.speed.load(Ordering::Relaxed) as usize]
    }
}
struct Pacing {
    measured_at: Instant,
    simulated: Duration,
}
impl Pacing {
    fn new() -> Self {
        Self {
            measured_at: Instant::now(),
            simulated: Duration::ZERO,
        }
    }
    fn finish(&mut self, started: Instant, step: Duration, advanced: bool, playback: &Playback) {
        if advanced {
            self.simulated += step;
        }
        let elapsed = self.measured_at.elapsed();
        if elapsed >= Duration::from_millis(500) {
            let actual = self.simulated.as_secs_f32() / elapsed.as_secs_f32();
            playback.actual.store(actual.to_bits(), Ordering::Relaxed);
            self.measured_at = Instant::now();
            self.simulated = Duration::ZERO;
        }
        let budget = if advanced {
            step.div_f64(playback.speed().factor())
        } else {
            step
        };
        // Rebase each step on wall time: never catch up a pause, rebuild, or overloaded frame.
        thread::sleep(budget.saturating_sub(started.elapsed()));
    }
}

// At most one sample per robot may be in flight. Publication overlaps the next
// physics tick, with a barrier before camera frames and before the next sample.
// Controls read the latest command available without waiting for a new one.
struct SensorPublication {
    io: Robotics,
    observation: crate::robot_io::Observation,
    time: ros_z::time::Time,
    completed: std::sync::mpsc::SyncSender<Result<(), String>>,
}

struct ObservationWorkers {
    sender: Option<crossbeam_channel::Sender<SensorPublication>>,
    workers: Vec<thread::JoinHandle<()>>,
}
impl ObservationWorkers {
    fn new(count: usize) -> Result<Self, String> {
        // Every robot has at most one sample in flight, with at most ten robots.
        let (sender, receiver) = crossbeam_channel::bounded::<SensorPublication>(10);
        let mut pool = Self {
            sender: Some(sender),
            workers: Vec::new(),
        };
        for index in 0..count {
            let receiver = receiver.clone();
            let worker = thread::Builder::new()
                .name(format!("simulator-pub-{index}"))
                .spawn(move || {
                    crate::cpu_affinity::worker();
                    while let Ok(sample) = receiver.recv() {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            sample
                                .io
                                .publish_observation(&sample.observation, sample.time)
                                .map_err(|e| format!("Sensor publication: {e:#}"))
                        }))
                        .unwrap_or_else(|_| Err("Sensor worker panicked".into()));
                        let _ = sample.completed.send(result);
                    }
                })
                .map_err(|e| format!("Sensor worker: {e}"))?;
            pool.workers.push(worker);
        }
        Ok(pool)
    }
}
impl Drop for ObservationWorkers {
    fn drop(&mut self) {
        self.sender.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}
struct ObservationPublisher {
    sender: crossbeam_channel::Sender<SensorPublication>,
    io: Robotics,
    completed: std::sync::mpsc::SyncSender<Result<(), String>>,
    completion: std::sync::mpsc::Receiver<Result<(), String>>,
    pending: bool,
}
impl ObservationPublisher {
    fn new(io: Robotics, pool: &ObservationWorkers) -> Self {
        let (completed, completion) = std::sync::mpsc::sync_channel(1);
        Self {
            sender: pool.sender.as_ref().unwrap().clone(),
            io,
            completed,
            completion,
            pending: false,
        }
    }
    fn send(
        &mut self,
        observation: crate::robot_io::Observation,
        time: ros_z::time::Time,
    ) -> Result<(), String> {
        if self.pending {
            return Err("Previous sensor publication has not completed".into());
        }
        self.sender
            .send(SensorPublication {
                io: self.io.clone(),
                observation,
                time,
                completed: self.completed.clone(),
            })
            .map_err(|_| "Sensor workers stopped".to_owned())?;
        self.pending = true;
        Ok(())
    }
    fn finish(&mut self) -> Result<(), String> {
        if !self.pending {
            return Ok(());
        }
        let result = self
            .completion
            .recv()
            .map_err(|_| "Sensor worker stopped".to_owned())?;
        self.pending = false;
        result
    }
}
impl Drop for ObservationPublisher {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

pub struct PhysicsWorker {
    playback: Arc<Playback>,
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
        let playback = Arc::new(Playback::default());
        let pacing_settings = playback.clone();
        let run = move || {
            let cameras = rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .thread_name(|i| format!("simulator-cam-{i}"))
                .start_handler(|_| crate::cpu_affinity::worker())
                .build()
                .map_err(|e| format!("Camera workers: {e}"))?;
            let publications = ObservationWorkers::new(4)?;
            crate::cpu_affinity::physics();
            let mut pacing = Pacing::new();
            struct State {
                binding: RobotBinding,
                controller: crate::simulated_sdk::Controller,
                publisher: ObservationPublisher,
            }
            let mut states = std::collections::BTreeMap::<crate::RobotId, State>::new();
            let mut contacts = GroundContacts::default();
            let mut placed = std::collections::HashSet::new();
            let mut generation = None;
            let mut last_frame = None;
            let mut sequence = 0;
            let mut pending_frames = Vec::<(Robotics, crate::observations::Frame)>::new();
            let mut samples = Vec::new();
            let mut observations = Vec::new();
            let mut robot_positions = Vec::new();
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
                let period;
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
                                let previous = previous.remove(&member.id);
                                let (controller, publisher) = previous
                                    .map(|state| (state.controller, state.publisher))
                                    .unwrap_or_else(|| {
                                        (
                                            Default::default(),
                                            ObservationPublisher::new(
                                                member.io.clone(),
                                                &publications,
                                            ),
                                        )
                                    });
                                states.insert(
                                    member.id,
                                    State {
                                        binding,
                                        controller,
                                        publisher,
                                    },
                                );
                            }
                        }
                        generation = Some(world.generation);
                    }
                    period = Duration::from_secs_f64(world.data().model_opt().timestep);
                    let running = world.mode == SimulationMode::Running;
                    if !running {
                        for state in states.values_mut() {
                            state.publisher.finish()?;
                        }
                        for (io, frame) in pending_frames.drain(..) {
                            io.publish_frame(frame);
                        }
                    }
                    let ready = states.len() == members.len();
                    initialized.store(ready, Ordering::Release);
                    if running || !ready || rebuilt {
                        let time = team.now();
                        if running {
                            world.prepare_step(|data| {
                                // Cameras must follow their corresponding body inputs.
                                // Other ticks can finish dynamics while publication runs.
                                if !pending_frames.is_empty() {
                                    for state in states.values_mut() {
                                        state.publisher.finish()?;
                                    }
                                    for (io, frame) in pending_frames.drain(..) {
                                        io.publish_frame(frame);
                                    }
                                }
                                for member in &members {
                                    if let Some(state) = states.get_mut(&member.id) {
                                        state.controller.apply(
                                            &state.binding,
                                            data,
                                            &member.io.actuator_control_ref(),
                                        );
                                    }
                                }
                                Ok(())
                            })?;
                            advanced = true;
                        }
                        if !running {
                            world.forward();
                        }
                        if running && let Some(referee) = team.referee() {
                            referee.update(&mut world, &members, period)?;
                        }
                        contacts.update(world.data());
                        observations.extend(members.iter().enumerate().filter_map(
                            |(index, member)| {
                                let mut observation = states
                                    .get(&member.id)?
                                    .binding
                                    .observe_with_contacts(world.data(), &contacts);
                                if !running {
                                    observation.stationary();
                                }
                                Some((index, observation))
                            },
                        ));
                        let frame_due = rebuilt
                            || !ready
                            || last_frame.is_none_or(|last| {
                                time.duration_since(last) >= Duration::from_millis(33)
                            });
                        if frame_due {
                            sequence += 1;
                            last_frame = Some(time);
                            robot_positions.clear();
                            robot_positions.extend(observations.iter().map(
                                |(index, observation)| {
                                    let p = observation.ground_to_world.translation.vector;
                                    (members[*index].id, [p.x, p.y, p.z])
                                },
                            ));
                        }
                        let balls = if frame_due {
                            crate::observations::balls(&world)
                        } else {
                            Vec::new()
                        };
                        let make_sample =
                            |(index, observation): (usize, crate::robot_io::Observation)| {
                                let member = &members[index];
                                let frame = if frame_due {
                                    let others = robot_positions
                                        .iter()
                                        .filter(|(id, _)| *id != member.id)
                                        .map(|(_, position)| *position)
                                        .collect::<Vec<_>>();
                                    let field = member.io.field_dimensions;
                                    let detections = member.io.needs_detections().then(|| {
                                        crate::observations::detect_scene(
                                            world.data(),
                                            &observation,
                                            &balls,
                                            &field,
                                            &others,
                                            team.configuration()
                                                .field_configuration
                                                .as_ref()
                                                .unwrap()
                                                .goal_height,
                                        )
                                    });
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
                                (member.id, member.io.clone(), observation, time, frame)
                            };
                        if frame_due && observations.len() >= 4 {
                            cameras.install(|| {
                                observations
                                    .par_drain(..)
                                    .map(make_sample)
                                    .collect_into_vec(&mut samples)
                            });
                        } else {
                            samples.extend(observations.drain(..).map(make_sample));
                        }
                    }
                    for state in states.values_mut() {
                        state.publisher.finish()?;
                    }
                    for (id, io, observation, time, frame) in samples.drain(..) {
                        let state = states.get_mut(&id).unwrap();
                        state.publisher.send(observation, time)?;
                        if let Some(frame) = frame {
                            pending_frames.push((io, frame));
                        }
                    }
                    if advanced {
                        world.integrate();
                        team.advance(period)
                            .map_err(|error| format!("Simulation clock: {error:#}"))?;
                        if let Some(remaining) = &mut world.remaining_steps {
                            *remaining = remaining.saturating_sub(1);
                            if *remaining == 0 {
                                world.mode = SimulationMode::Paused;
                            }
                        }
                    }
                } else {
                    // A scene reader only borrowed the world. Retry promptly rather
                    // than spending an entire physics tick idle after it releases it.
                    period = Duration::from_micros(100);
                }
                pacing.finish(started, period, advanced, &pacing_settings);
            }
            for state in states.values_mut() {
                state.publisher.finish()?;
            }
            for (io, frame) in pending_frames {
                io.publish_frame(frame);
            }
            Ok(())
        };
        let thread = thread::Builder::new()
            .name("simulator-physics".into())
            .spawn(run)
            .expect("failed to start physics worker");
        Self {
            playback,
            stop,
            ready,
            failure: None,
            thread: Some(thread),
        }
    }

    pub(crate) fn speed(&self) -> Speed {
        self.playback.speed()
    }
    pub(crate) fn set_speed(&self, speed: Speed) {
        self.playback.speed.store(speed as u8, Ordering::Relaxed);
    }
    pub(crate) fn actual_speed(&self) -> f32 {
        f32::from_bits(self.playback.actual.load(Ordering::Relaxed))
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
        let playback = Arc::new(Playback::default());
        let pacing_settings = playback.clone();
        let thread = thread::spawn(move || {
            let mut pacing = Pacing::new();
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
                            let time = io.now();
                            if world.mode == SimulationMode::Running {
                                world.prepare_step(|data| {
                                    controller.apply(robot, data, &io.actuator_control());
                                    Ok(())
                                })?;
                            }
                            if world.mode != SimulationMode::Running {
                                world.forward();
                            }
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
                                    detections: Some(detections),
                                    field,
                                });
                            }
                            sample = Some((observation, time));
                            if world.mode == SimulationMode::Running {
                                world.integrate();
                            }
                        }
                    }
                }
                let advanced = sample.is_some();
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
                pacing.finish(started, period, advanced, &pacing_settings);
            }
            Ok(())
        });
        Self {
            playback,
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

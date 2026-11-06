//! Headless mixed-match measurements using the same team and worker as Twix.
use std::{
    path::PathBuf,
    sync::{Arc, atomic::AtomicBool},
    thread,
    time::{Duration, Instant},
};

use bevy::prelude::*;
use color_eyre::{
    Result,
    eyre::{ensure, eyre},
};
use ros_z::context::ContextBuilder;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    bevy_mujoco::{MjcfObject, MujocoWorldPlugin, SharedPhysics, SimulationMode},
    simulation::{PhysicsWorker, Speed},
    team::Team,
};

#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
/// Fixed-frame mixed-match workload; startup and recording drain are not timed.
pub struct BenchmarkOptions {
    pub worker: PathBuf,
    pub source: PathBuf,
    pub profile: crate::Profile,
    pub recording: bool,
    pub players: u8,
    pub warmup_frames: u64,
    pub frames: u64,
    pub playing: bool,
    pub location: String,
}

impl Default for BenchmarkOptions {
    fn default() -> Self {
        Self {
            worker: std::env::var_os("BHUMAN_WORKER").unwrap_or_default().into(),
            source: std::env::var_os("BHUMAN_SOURCE").unwrap_or_default().into(),
            profile: crate::Profile::Localization,
            recording: true,
            players: 5,
            warmup_frames: 2500,
            frames: 15_000,
            playing: false,
            location: "hsl_middle".into(),
        }
    }
}

/// Run a fresh isolated match and return throughput, pose and recording evidence.
pub fn run_benchmark(options: BenchmarkOptions) -> Result<Value> {
    ensure!((1..=5).contains(&options.players), "players must be 1..5");
    ensure!(options.frames > 0, "frames must be positive");
    let started = Instant::now();
    let runtime = tokio::runtime::Runtime::new()?;
    let router = runtime.block_on(
        ContextBuilder::default()
            .with_mode("router")
            .disable_multicast_scouting()
            .with_connect_endpoints(std::iter::empty::<&str>())
            .with_listen_endpoints(["tcp/127.0.0.1:0"])
            .build(),
    )?;
    let endpoint =
        runtime.block_on(async { router.session().info().locators().await[0].to_string() });
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let parameter_root = root.join("../../etc/parameters");
    let field = crate::FieldConfiguration::load(&parameter_root, &options.location)?;
    let team = runtime.block_on(Team::new_with_controller(
        runtime.handle().clone(),
        crate::Configuration {
            referee: crate::RefereeMode::Automatic,
            competition: crate::Competition::default(),
            parameter_root,
            model_directory: root.join("../../etc/neural_networks"),
            router: Some(endpoint),
            namespace: crate::robot_namespace(crate::RobotId::FIRST),
            location: Some(options.location.clone()),
            robot_count: options.players,
            opponent_count: options.players,
            opponent: crate::Opponent::BHuman {
                executable: options.worker.clone(),
                source: options.source.clone(),
            },
            field_configuration: Some(field),
            profile: options.profile,
            mcap_enabled: Some(options.recording),
        },
        Arc::new(AtomicBool::new(false)),
        None,
        false,
    ))?;
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, MujocoWorldPlugin));
    for member in team.members() {
        let entity = app
            .world_mut()
            .spawn((
                MjcfObject::new(root.join("assets/k1_robot.xml"), "Trunk")
                    .with_free_joint("world_joint")
                    .grounded(),
                member.pose,
            ))
            .id();
        team.bind(member.id, entity);
    }
    {
        let member = &team.members()[0];
        let radius = member.io.field_dimensions.ball_radius as f64;
        let parameters = member.io.parameters.snapshot().typed().ball.clone();
        app.world_mut().spawn((
            MjcfObject::from_factory(
                move || crate::scene::ball::ball_spec(radius, &parameters),
                "ball",
            )
            .with_free_joint("ball_free_joint"),
            Transform::from_xyz(0.0, radius as f32, 0.0),
            crate::scene::ball::Ball,
        ));
    }
    let dimensions = team.members()[0].io.field_dimensions;
    app.world_mut().spawn((
        MjcfObject::from_factory(
            move || crate::scene::markings::mujoco_field(dimensions),
            "field",
        ),
        Transform::default(),
    ));
    for (x, angle) in [
        (-dimensions.length / 2.0, 0.0),
        (dimensions.length / 2.0, std::f32::consts::PI),
    ] {
        let goal = crate::scene::goal::GoalDimensions::from(&dimensions);
        app.world_mut().spawn((
            MjcfObject::from_factory(move || crate::scene::goal::goal_spec(goal), "goal")
                .with_mocap_body("goal"),
            Transform::from_xyz(x, 0.0, 0.0).with_rotation(Quat::from_rotation_y(angle)),
        ));
    }
    app.update();
    let physics = app.world().resource::<SharedPhysics>().clone();
    physics.lock().mode = SimulationMode::Paused;
    let members = team.members();
    let mut estimates = Vec::new();
    if options.profile.estimators().localization {
        for member in members.iter().filter(|member| !member.io.is_bhuman()) {
            let cache = runtime.block_on(
                member
                    .io
                    .node()
                    .subscriber::<types::localization::LocalizationEstimate>(
                        "localization/estimate",
                    )
                    .cache(1)
                    .build(),
            )?;
            estimates.push((member.id, cache));
        }
    }
    let directories: Vec<_> = members
        .iter()
        .filter_map(|member| {
            member
                .io
                .recording_status()
                .map(|(path, _)| path.to_owned())
        })
        .collect();
    let mut worker = PhysicsWorker::start_team(physics.clone(), team.clone());
    worker.set_speed(Speed::Sixteenfold);
    eprintln!("BENCHMARK startup {:.3}s", started.elapsed().as_secs_f64());
    advance(&mut worker, &physics, options.warmup_frames)?;
    if options.playing {
        // With a ball present the automatic referee has already entered Ready.
        for command in ["set", "whistle", "ball_free"] {
            team.referee()
                .unwrap()
                .command(command, &json!({}))
                .map_err(|error| eyre!("benchmark referee {command}: {error}"))?;
            advance(&mut worker, &physics, 250)?;
        }
    }
    let origin = team.now();
    eprintln!(
        "BENCHMARK measuring pid={} frames={}",
        std::process::id(),
        options.frames
    );
    let measure = Instant::now();
    advance(&mut worker, &physics, options.frames)?;
    let wall_seconds = measure.elapsed().as_secs_f64();
    eprintln!("BENCHMARK measured wall_seconds={wall_seconds:.6}");
    let simulation_seconds = options.frames as f64 * super::STEP_NS as f64 / 1e9;
    ensure!(
        team.now().as_nanos() - origin.as_nanos() == (options.frames * super::STEP_NS) as i64,
        "simulation clock did not advance exactly the requested number of frames"
    );
    let paused = team.now();
    thread::sleep(Duration::from_millis(50));
    ensure!(paused == team.now(), "clock advanced while paused");
    let robots = {
        let mut world = physics.lock();
        world.forward();
        members
            .iter()
            .map(|member| {
                let pose = world.object_pose(member.entity.unwrap()).unwrap();
                let up = (pose.rotation * Vec3::Y).dot(Vec3::Y);
                json!({"id": member.id.to_string(), "position": pose.translation.to_array(),
                "rotation_xyzw": pose.rotation.to_array(),
                "horizontal_displacement_m": (pose.translation.xz() - member.pose.translation.xz()).length(),
                "up": up, "upright": pose.translation.y > 0.4 && up > 0.5,
                "clock_matches": member.io.now() == team.now(),
                "primary": format!("{:?}", member.io.primary.get_latest()),
                "failure": member.io.poll()})
            })
            .collect::<Vec<_>>()
    };
    let localization = estimates.iter().map(|(id, cache)| {
        let estimate = cache.get_latest();
        json!({"id": id.to_string(), "estimate": estimate.as_deref(),
            "age_seconds": estimate.as_ref().map(|estimate| (team.now().as_nanos() - estimate.time.as_nanos()) as f64 / 1e9)})
    }).collect::<Vec<_>>();
    drop(worker);
    drop(estimates);
    drop(members);
    drop(team);
    drop(app);
    let recording_errors = crate::finish_recordings();
    let recordings = directories
        .iter()
        .map(|directory| -> Result<Value> {
            let finished: Value =
                serde_json::from_slice(&std::fs::read(directory.join("finished.json"))?)?;
            let summary = if options.recording {
                let file = std::fs::File::open(directory.join("recording.mcap"))?;
                // Finalized recordings are immutable. Mapping avoids copying large MCAPs.
                let bytes = unsafe { memmap2::Mmap::map(&file)? };
                let summary = mcap::Summary::read(&bytes)?
                    .ok_or_else(|| eyre!("missing MCAP footer: {}", directory.display()))?;
                let stats = summary.stats.ok_or_else(|| eyre!("missing MCAP statistics"))?;
                let mut topics = std::collections::BTreeMap::<String, u64>::new();
                for (id, channel) in &summary.channels {
                    *topics.entry(channel.topic.clone()).or_default() += stats.channel_message_counts.get(id).copied().unwrap_or(0);
                }
                let mut measured_topics = std::collections::BTreeMap::<String, u64>::new();
                for message in mcap::MessageStream::new(&bytes)? {
                    let message = message?;
                    if (origin.as_nanos() as u64..paused.as_nanos() as u64).contains(&message.publish_time) {
                        *measured_topics.entry(message.channel.topic.clone()).or_default() += 1;
                    }
                }
                Some(json!({"messages": stats.message_count, "topics": topics, "measured_topics": measured_topics,
                    "start_ns": stats.message_start_time.to_string(), "end_ns": stats.message_end_time.to_string()}))
            } else { None };
            Ok(json!({"directory": directory, "finished": finished, "summary": summary}))
        })
        .collect::<Result<Vec<_>>>()?;
    router.shutdown()?;
    Ok(json!({"options": options, "wall_seconds": wall_seconds,
        "measurement_time_ns": {"start": origin.as_nanos().to_string(), "end": paused.as_nanos().to_string()},
        "simulation_seconds": simulation_seconds, "speed": simulation_seconds / wall_seconds,
        "robots": robots, "localization": localization,
        "recording_errors": recording_errors, "recordings": recordings}))
}

fn advance(worker: &mut PhysicsWorker, physics: &SharedPhysics, frames: u64) -> Result<()> {
    if frames == 0 {
        return Ok(());
    }
    {
        let mut world = physics.lock();
        world.remaining_steps = Some(frames);
        world.mode = SimulationMode::Running;
    }
    let deadline = Instant::now() + Duration::from_secs(180);
    while physics.lock().remaining_steps != Some(0) {
        if let Some(error) = worker.poll() {
            return Err(eyre!(error));
        }
        ensure!(Instant::now() < deadline, "benchmark timed out");
        thread::sleep(Duration::from_millis(2));
    }
    Ok(())
}

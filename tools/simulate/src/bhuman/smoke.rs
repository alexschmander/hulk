use std::time::{Duration, Instant};

use bevy::prelude::*;
use booster::{CommandType, LowCommand, MotorCommand};
use color_eyre::{Result, eyre::ensure};
use serde::Serialize;

use super::{Ball, Configuration, Frame, JointSample, Motion, Robot, STEP_NS, World};
use crate::{
    bevy_mujoco::{MjcfObject, MujocoWorldPlugin, SharedPhysics},
    robot_io::RobotBinding,
};

#[derive(Debug, Serialize)]
pub struct SmokeReport {
    pub frames: u64,
    pub simulation_seconds: f64,
    pub elapsed_seconds: f64,
    pub minimum_height: f32,
    pub maximum_displacement: f32,
    pub final_phase: String,
    pub requested_motions: std::collections::BTreeSet<String>,
    pub ball_displacement: f32,
}

/// Exercises native B-Human motion in the same K1 model and physics implementation as Twix.
/// Cognition mode starts in Initial, then lets native soccer behavior play the ball.
pub fn run_smoke(configuration: Configuration, frames: u64) -> Result<SmokeReport> {
    ensure!(
        frames >= 1000,
        "Use at least 1000 frames to exercise stand after Prepare"
    );
    let soccer = configuration.cognition;
    let radius = configuration
        .field
        .as_ref()
        .map_or(0.095, |f| f.dimensions.ball_radius) as f64;
    let mut robot = Robot::start(configuration)?;
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, MujocoWorldPlugin));
    let entity = app
        .world_mut()
        .spawn((
            MjcfObject::new(
                concat!(env!("CARGO_MANIFEST_DIR"), "/assets/k1_robot.xml"),
                "Trunk",
            )
            .with_free_joint("world_joint")
            .grounded(),
            Transform::default(),
        ))
        .id();
    let ball = if soccer {
        let parameters: crate::parameters::SimulatorParameters =
            json5::from_str(include_str!("../../parameters/simulator.json5"))?;
        Some(
            app.world_mut()
                .spawn((
                    MjcfObject::from_factory(
                        move || crate::scene::ball::ball_spec(radius, &parameters.ball),
                        "ball",
                    )
                    .with_free_joint("ball_free_joint")
                    .grounded(),
                    Transform::from_xyz(1.5, 0.0, 0.0),
                ))
                .id(),
        )
    } else {
        None
    };
    app.update();
    let physics = app.world().resource::<SharedPhysics>().clone();
    let mut world = physics.lock();
    let binding = RobotBinding::new(world.data(), &format!("object_{}_", entity.to_bits()))?;
    binding.reset_joints(world.data_mut());
    world
        .ground_object(entity, Transform::default())
        .map_err(|error| color_eyre::eyre::eyre!(error))?;
    ensure!(
        (world.data().model_opt().timestep - 0.002).abs() < 1e-12,
        "B-Human requires a 2 ms physics step"
    );
    let initial = binding
        .observe(world.data())
        .robot_to_world
        .translation
        .vector;
    if let Some(ball) = ball {
        world
            .ground_object(ball, Transform::from_xyz(1.5, 0.0, 0.0))
            .map_err(|error| color_eyre::eyre::eyre!(error))?;
    }
    drop(world);
    let started = Instant::now();
    let mut minimum_height = initial.z;
    let mut maximum_displacement = 0.0f32;
    let mut phase = String::new();
    let mut requested_motions = std::collections::BTreeSet::new();
    let mut ball_displacement = 0.0f32;
    for index in 0..frames {
        let (observation, balls) = {
            let world = physics.lock();
            (
                binding.observe(world.data()),
                crate::observations::balls(&world),
            )
        };
        let truth = soccer.then(|| World {
            pose: [
                observation.ground_to_world.translation.x,
                observation.ground_to_world.translation.y,
                observation.ground_to_world.rotation.euler_angles().2,
            ],
            ball: balls.first().map(|ball| Ball {
                position: ball.position.map(|v| v as f32),
                velocity: ball.velocity.map(|v| v as f32),
            }),
            players: Vec::new(),
        });
        if let Some(ball) = balls.first() {
            ball_displacement = ball_displacement
                .max(((ball.position[0] - 1.5).powi(2) + ball.position[1].powi(2)).sqrt() as f32);
        }
        let low = &observation.low_state;
        // Extract values explicitly: the manufacturer's IMU type uses its own coordinate marker.
        let angles = &low.imu_state.roll_pitch_yaw;
        let gyro = &low.imu_state.angular_velocity;
        let acceleration = &low.imu_state.linear_acceleration;
        let result = robot.step(&Frame {
            time_ns: index * STEP_NS,
            joints: std::array::from_fn(|i| {
                let motor = &low.motor_state_serial[i];
                JointSample {
                    position: motor.position,
                    velocity: motor.velocity,
                    torque: motor.torque,
                }
            }),
            angle: [angles.x(), angles.y(), angles.z()],
            gyro: [gyro.x(), gyro.y(), gyro.z()],
            acceleration: [acceleration.x(), acceleration.y(), acceleration.z()],
            motion: if index < 500 {
                Motion::Prepare
            } else {
                Motion::Stand
            },
            velocity: [0.0; 3],
            world: truth,
            game_controller: (soccer && index.is_multiple_of(250))
                .then(|| game_packet(index, if index < 500 { 0 } else { 3 })),
            ..Default::default()
        })?;
        let command = LowCommand {
            command_type: CommandType::Serial,
            motor_commands: result
                .joints
                .iter()
                .map(|joint| MotorCommand {
                    command_type: CommandType::Serial,
                    position: joint.position,
                    velocity: joint.velocity,
                    torque: joint.torque,
                    kp: joint.kp,
                    kd: joint.kd,
                    weight: 1.0,
                })
                .collect(),
        };
        RobotBinding::validate_command(&command)?;
        let limits = result.joints.map(|joint| joint.torque_limit);
        let mut world = physics.lock();
        binding.apply_with_limits(world.data_mut(), Some(&command), Some(&limits));
        world.data_mut().step();
        let pose = binding.observe(world.data()).robot_to_world;
        let p = pose.translation.vector;
        minimum_height = minimum_height.min(p.z);
        maximum_displacement = maximum_displacement.max((p.xy() - initial.xy()).norm());
        ensure!(
            p.z > 0.35 && pose.rotation.to_rotation_matrix()[(2, 2)] > 0.5,
            "B-Human fell at frame {index}, height {}, phase {}",
            p.z,
            result.phase
        );
        requested_motions.insert(result.motion);
        phase = result.phase;
    }
    robot.stop()?;
    if soccer {
        ensure!(
            requested_motions
                .iter()
                .any(|motion| motion.starts_with("walk") || motion == "dribble"),
            "Native soccer behavior never requested movement"
        );
        ensure!(
            maximum_displacement > 0.1,
            "Native soccer behavior did not move the physical robot"
        );
    }
    Ok(SmokeReport {
        frames,
        simulation_seconds: Duration::from_nanos(frames * STEP_NS).as_secs_f64(),
        elapsed_seconds: started.elapsed().as_secs_f64(),
        minimum_height,
        maximum_displacement,
        final_phase: phase,
        requested_motions,
        ball_displacement,
    })
}

pub(super) fn game_packet(frame: u64, state: u8) -> Vec<u8> {
    let mut packet = vec![0; 158];
    packet[..4].copy_from_slice(b"RGme");
    packet[4] = 20;
    packet[5] = (frame / 250) as u8;
    packet[6] = 5;
    packet[7] = 0;
    packet[10] = state;
    packet[12] = 1;
    packet[13] = 255;
    packet[14..16].copy_from_slice(&600i16.to_le_bytes());
    packet[18] = 5;
    packet[19] = 4;
    packet[20] = 7;
    packet[21] = 1;
    packet[26..28].copy_from_slice(&12000u16.to_le_bytes());
    packet[88] = 24;
    packet[89] = 2;
    packet[91] = 1;
    packet[96..98].copy_from_slice(&12000u16.to_le_bytes());
    for number in 3..=20 {
        packet[28 + (number - 1) * 3] = 13;
    }
    for number in 6..=20 {
        packet[98 + (number - 1) * 3] = 13;
    }
    packet
}

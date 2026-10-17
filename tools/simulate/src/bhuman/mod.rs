//! One native B-Human player per child process, with separate cognition and motion graphs.
#[allow(dead_code, unused_imports, mismatched_lifetime_syntaxes, unsafe_op_in_unsafe_fn)]
#[rustfmt::skip]
mod protocol_generated;
mod field;
mod simulator;
mod smoke;
use field::write_field_configuration;
pub(crate) use simulator::SimulatorRobot;
pub use smoke::{SmokeReport, run_smoke};

use std::{
    io::{Read, Write},
    net::Shutdown,
    os::unix::net::{UnixListener, UnixStream},
    path::PathBuf,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use color_eyre::{
    Result,
    eyre::{WrapErr, bail, ensure, eyre},
};
use flatbuffers::FlatBufferBuilder;
use protocol_generated::hulk::bhuman as wire;

pub const REVISION: &str = "2e3374ba896da8288607f9a1881b876642fe18b8";
pub const STEP_NS: u64 = 2_000_000;
pub const JOINT_NAMES: [&str; 22] = [
    "headYaw",
    "headPitch",
    "lShoulderPitch",
    "lShoulderRoll",
    "lElbowPitch",
    "lElbowYaw",
    "rShoulderPitch",
    "rShoulderRoll",
    "rElbowPitch",
    "rElbowYaw",
    "lHipPitch",
    "lHipRoll",
    "lHipYaw",
    "lKneePitch",
    "lAnklePitch",
    "lAnkleRoll",
    "rHipPitch",
    "rHipRoll",
    "rHipYaw",
    "rKneePitch",
    "rAnklePitch",
    "rAnkleRoll",
];
const MAX_MESSAGE: usize = 1024 * 1024;

#[derive(Clone)]
pub struct Configuration {
    pub executable: PathBuf,
    pub source: PathBuf,
    pub player: u8,
    pub team: u8,
    pub cognition: bool,
    /// Activate through the native simulator startup request. False boots in damping.
    pub active: bool,
    pub field: Option<crate::FieldConfiguration>,
    pub timeout: Duration,
}

#[derive(Clone, Copy, Default)]
pub struct JointSample {
    pub position: f32,
    pub velocity: f32,
    pub torque: f32,
}

#[derive(Clone, Copy, Default)]
pub enum Motion {
    #[default]
    Stand,
    Walk,
    Prepare,
    Damping,
}

#[derive(Default)]
pub struct Frame {
    pub time_ns: u64,
    pub joints: [JointSample; 22],
    /// Torso sensor angles in radians, and body-frame angular velocity / acceleration in SI.
    pub angle: [f32; 3],
    pub gyro: [f32; 3],
    pub acceleration: [f32; 3],
    pub motion: Motion,
    /// Forward and lateral velocity in m/s, followed by angular velocity in rad/s.
    pub velocity: [f32; 3],
    pub world: Option<World>,
    pub buttons: Buttons,
    pub whistle: Option<Whistle>,
    /// A new, unmodified GameController v20 datagram, supplied only when received.
    pub game_controller: Option<Vec<u8>>,
    /// Native B-Human packets received from this robot's teammates.
    pub team_messages: Vec<Vec<u8>>,
}

#[derive(Clone, Copy, Default)]
pub struct Buttons {
    pub f1: bool,
    pub stand: bool,
    pub walk: bool,
}
#[derive(Clone, Copy)]
pub struct Whistle {
    pub time_ns: u64,
    pub confidence: f32,
    pub channels: u8,
}
#[derive(Clone, Default)]
pub struct World {
    /// Field coordinates in metres and radians, with the opponent goal at positive x.
    pub pose: [f32; 3],
    pub ball: Option<Ball>,
    pub players: Vec<Player>,
}
#[derive(Clone, Copy)]
pub struct Ball {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
}
#[derive(Clone, Copy)]
pub struct Player {
    pub number: u8,
    pub teammate: bool,
    pub upright: bool,
    pub pose: [f32; 3],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LedState {
    Off,
    On,
    Blinking,
    FastBlinking,
    Half,
    HalfBlinking,
    HalfFastBlinking,
}

#[derive(Clone, Copy, Debug)]
pub struct JointCommand {
    pub position: f32,
    pub velocity: f32,
    pub torque: f32,
    pub kp: f32,
    pub kd: f32,
    pub torque_limit: f32,
}
pub struct Actuation {
    pub joints: [JointCommand; 22],
    pub phase: String,
    pub motion: String,
    pub game_state: String,
    pub player_state: String,
    /// Native red, green and blue LED requests. Blink timing uses simulation time.
    pub leds: [LedState; 3],
    pub team_messages: Vec<Vec<u8>>,
    pub teammates: u8,
    pub cognition_time_ns: u64,
}

pub struct Robot {
    configuration: Configuration,
    child: Child,
    stream: Option<UnixStream>,
    _socket_directory: tempfile::TempDir,
    epoch: u64,
    sequence: u64,
    time_ns: u64,
}

impl Robot {
    pub fn start(configuration: Configuration) -> Result<Self> {
        ensure!(
            (1..=5).contains(&configuration.player),
            "B-Human player must be 1 through 5"
        );
        ensure!(
            !configuration.timeout.is_zero(),
            "B-Human timeout must be positive"
        );
        ensure!(configuration.team > 0, "B-Human team must be positive");
        let directory = tempfile::tempdir()?;
        let socket = directory.path().join("robot");
        let listener = UnixListener::bind(&socket)?;
        listener.set_nonblocking(true)?;
        let executable = configuration.executable.canonicalize()?;
        let child = Command::new(executable)
            .arg("--socket")
            .arg(&socket)
            .current_dir(&configuration.source)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .wrap_err("starting B-Human worker")?;
        let mut robot = Self {
            configuration,
            child,
            stream: None,
            _socket_directory: directory,
            epoch: (uuid::Uuid::new_v4().as_u128() as u64).max(1),
            sequence: 0,
            time_ns: 0,
        };
        let deadline = Instant::now() + robot.configuration.timeout;
        loop {
            match listener.accept() {
                Ok((stream, _)) => {
                    robot.stream = Some(stream);
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    if let Some(status) = robot.child.try_wait()? {
                        bail!("B-Human worker exited during startup: {status}");
                    }
                    ensure!(
                        Instant::now() < deadline,
                        "B-Human worker connection timed out"
                    );
                    thread::sleep(Duration::from_millis(1));
                }
                Err(error) => return Err(error.into()),
            }
        }
        let mut builder = FlatBufferBuilder::new();
        let config_directory = if let Some(field) = &robot.configuration.field {
            field.validate().map_err(|error| eyre!(error))?;
            write_field_configuration(
                &robot.configuration.source,
                robot._socket_directory.path(),
                field,
            )?;
            Some(
                builder.create_string(
                    robot
                        ._socket_directory
                        .path()
                        .to_str()
                        .ok_or_else(|| eyre!("non UTF-8 adapter directory"))?,
                ),
            )
        } else {
            None
        };
        let payload = wire::Initialize::create(
            &mut builder,
            &wire::InitializeArgs {
                player: robot.configuration.player,
                team: robot.configuration.team,
                cognition: robot.configuration.cognition,
                active: robot.configuration.active,
                config_directory,
            },
        );
        let bytes = robot.exchange(
            &mut builder,
            wire::Payload::Initialize,
            payload.as_union_value(),
        )?;
        let ready = decode(&bytes, robot.epoch, 0)?
            .payload_as_ready()
            .ok_or_else(|| eyre!("expected B-Human Ready"))?;
        ensure!(
            ready.revision() == REVISION,
            "unexpected B-Human release {}",
            ready.revision()
        );
        ensure!(
            ready.joints().iter().eq(JOINT_NAMES),
            "B-Human serial joint mapping differs"
        );
        robot.sequence = 1;
        Ok(robot)
    }

    pub fn step(&mut self, frame: &Frame) -> Result<Actuation> {
        ensure!(
            frame.time_ns == self.time_ns,
            "expected simulation time {}, got {}",
            self.time_ns,
            frame.time_ns
        );
        ensure!(
            frame
                .joints
                .iter()
                .all(|j| [j.position, j.velocity, j.torque]
                    .into_iter()
                    .all(f32::is_finite))
                && [frame.angle, frame.gyro, frame.acceleration, frame.velocity]
                    .into_iter()
                    .flatten()
                    .all(f32::is_finite),
            "non-finite B-Human input"
        );
        let result = self.step_inner(frame);
        if result.is_err() {
            self.terminate();
        }
        result
    }

    fn step_inner(&mut self, frame: &Frame) -> Result<Actuation> {
        let mut builder = FlatBufferBuilder::new();
        let samples = frame
            .joints
            .map(|j| wire::JointSample::new(j.position, j.velocity, j.torque));
        let joints = builder.create_vector(&samples);
        let vector = |v: [f32; 3]| wire::Vector3::new(v[0], v[1], v[2]);
        let angle = vector(frame.angle);
        let gyro = vector(frame.gyro);
        let acceleration = vector(frame.acceleration);
        let velocity = vector(frame.velocity);
        let motion = match frame.motion {
            Motion::Stand => wire::Motion::Stand,
            Motion::Walk => wire::Motion::Walk,
            Motion::Prepare => wire::Motion::Prepare,
            Motion::Damping => wire::Motion::Damping,
        };
        let pose = |v: [f32; 3]| wire::Pose2::new(v[0], v[1], v[2]);
        let world = frame.world.as_ref().map(|world| {
            let ball = world.ball.map(|ball| {
                wire::Ball::create(
                    &mut builder,
                    &wire::BallArgs {
                        position: Some(&vector(ball.position)),
                        velocity: Some(&vector(ball.velocity)),
                    },
                )
            });
            let players: Vec<_> = world
                .players
                .iter()
                .map(|player| {
                    wire::Player::create(
                        &mut builder,
                        &wire::PlayerArgs {
                            number: player.number,
                            teammate: player.teammate,
                            upright: player.upright,
                            pose: Some(&pose(player.pose)),
                        },
                    )
                })
                .collect();
            let players = builder.create_vector(&players);
            wire::World::create(
                &mut builder,
                &wire::WorldArgs {
                    pose: Some(&pose(world.pose)),
                    ball,
                    players: Some(players),
                },
            )
        });
        let buttons = wire::Buttons::new(frame.buttons.f1, frame.buttons.stand, frame.buttons.walk);
        let whistle = frame.whistle.map(|whistle| {
            wire::Whistle::create(
                &mut builder,
                &wire::WhistleArgs {
                    time_ns: whistle.time_ns,
                    confidence: whistle.confidence,
                    channels: whistle.channels,
                },
            )
        });
        let game_controller = frame.game_controller.as_ref().map(|packet| {
            let bytes = builder.create_vector(packet);
            wire::Packet::create(&mut builder, &wire::PacketArgs { bytes: Some(bytes) })
        });
        let packets: Vec<_> = frame
            .team_messages
            .iter()
            .map(|packet| {
                let bytes = builder.create_vector(packet);
                wire::Packet::create(&mut builder, &wire::PacketArgs { bytes: Some(bytes) })
            })
            .collect();
        let team_messages = builder.create_vector(&packets);
        let payload = wire::Step::create(
            &mut builder,
            &wire::StepArgs {
                time_ns: frame.time_ns,
                joints: Some(joints),
                angle: Some(&angle),
                gyro: Some(&gyro),
                acceleration: Some(&acceleration),
                motion,
                velocity: Some(&velocity),
                world,
                buttons: Some(&buttons),
                whistle,
                game_controller,
                team_messages: Some(team_messages),
            },
        );
        let bytes = self.exchange(&mut builder, wire::Payload::Step, payload.as_union_value())?;
        let envelope = decode(&bytes, self.epoch, self.sequence)?;
        let commands = envelope
            .payload_as_commands()
            .ok_or_else(|| eyre!("expected B-Human Commands"))?;
        ensure!(
            commands.joints().len() == 22,
            "B-Human returned the wrong joint count"
        );
        let joints = std::array::from_fn(|i| {
            let c = commands.joints().get(i);
            JointCommand {
                position: c.position(),
                velocity: c.velocity(),
                torque: c.torque(),
                kp: c.kp(),
                kd: c.kd(),
                torque_limit: c.torque_limit(),
            }
        });
        ensure!(
            joints.iter().all(
                |c| [c.position, c.velocity, c.torque, c.kp, c.kd, c.torque_limit]
                    .into_iter()
                    .all(f32::is_finite)
                    && c.kp >= 0.0
                    && c.kd >= 0.0
                    && c.torque_limit >= 0.0
            ),
            "invalid B-Human actuator command"
        );
        self.sequence += 1;
        self.time_ns += STEP_NS;
        let leds = commands
            .leds()
            .ok_or_else(|| eyre!("missing B-Human LED state"))?;
        let led = |state| -> Result<LedState> {
            Ok(match state {
                wire::LedState::Off => LedState::Off,
                wire::LedState::On => LedState::On,
                wire::LedState::Blinking => LedState::Blinking,
                wire::LedState::FastBlinking => LedState::FastBlinking,
                wire::LedState::Half => LedState::Half,
                wire::LedState::HalfBlinking => LedState::HalfBlinking,
                wire::LedState::HalfFastBlinking => LedState::HalfFastBlinking,
                _ => bail!("invalid B-Human LED state"),
            })
        };
        Ok(Actuation {
            joints,
            phase: commands.phase().to_owned(),
            motion: commands.motion().to_owned(),
            game_state: commands.game_state().to_owned(),
            player_state: commands.player_state().to_owned(),
            leds: [led(leds.red())?, led(leds.green())?, led(leds.blue())?],
            team_messages: commands
                .team_messages()
                .into_iter()
                .flatten()
                .map(|packet| packet.bytes().bytes().to_vec())
                .collect(),
            teammates: commands.teammates(),
            cognition_time_ns: commands.cognition_time_ns(),
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        let configuration = self.configuration.clone();
        self.terminate();
        *self = Self::start(configuration)?;
        Ok(())
    }

    pub fn stop(mut self) -> Result<()> {
        let mut builder = FlatBufferBuilder::new();
        let payload = wire::Stop::create(&mut builder, &wire::StopArgs {});
        let bytes = self.exchange(&mut builder, wire::Payload::Stop, payload.as_union_value())?;
        ensure!(
            decode(&bytes, self.epoch, self.sequence)?
                .payload_as_stopped()
                .is_some(),
            "expected B-Human Stopped"
        );
        let deadline = Instant::now() + self.configuration.timeout;
        loop {
            if let Some(status) = self.child.try_wait()? {
                ensure!(status.success(), "B-Human worker shutdown failed: {status}");
                return Ok(());
            }
            ensure!(Instant::now() < deadline, "B-Human shutdown timed out");
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn exchange<'a>(
        &mut self,
        builder: &mut FlatBufferBuilder<'a>,
        kind: wire::Payload,
        payload: flatbuffers::WIPOffset<flatbuffers::UnionWIPOffset>,
    ) -> Result<Vec<u8>> {
        let envelope = wire::Envelope::create(
            builder,
            &wire::EnvelopeArgs {
                version: 2,
                epoch: self.epoch,
                sequence: self.sequence,
                payload_type: kind,
                payload: Some(payload),
            },
        );
        wire::finish_size_prefixed_envelope_buffer(builder, envelope);
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| eyre!("B-Human worker is stopped"))?;
        stream.set_write_timeout(Some(self.configuration.timeout))?;
        stream
            .write_all(builder.finished_data())
            .wrap_err("sending B-Human frame")?;
        let deadline = Instant::now() + self.configuration.timeout;
        let mut prefix = [0u8; 4];
        read_exact_until(stream, &mut prefix, deadline)?;
        let length = u32::from_le_bytes(prefix) as usize;
        ensure!(
            (8..=MAX_MESSAGE).contains(&length),
            "invalid B-Human message length {length}"
        );
        let mut bytes = vec![0; length + 4];
        bytes[..4].copy_from_slice(&prefix);
        read_exact_until(stream, &mut bytes[4..], deadline)?;
        Ok(bytes)
    }

    fn terminate(&mut self) {
        if let Some(stream) = self.stream.take() {
            let _ = stream.shutdown(Shutdown::Both);
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
impl Drop for Robot {
    fn drop(&mut self) {
        self.terminate();
    }
}

fn read_exact_until(
    stream: &mut UnixStream,
    mut bytes: &mut [u8],
    deadline: Instant,
) -> Result<()> {
    while !bytes.is_empty() {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .ok_or_else(|| eyre!("B-Human response timed out"))?;
        ensure!(!remaining.is_zero(), "B-Human response timed out");
        stream.set_read_timeout(Some(remaining))?;
        let count = stream.read(bytes).wrap_err("reading B-Human response")?;
        ensure!(count != 0, "B-Human worker disconnected");
        bytes = &mut bytes[count..];
    }
    Ok(())
}

fn decode(bytes: &[u8], epoch: u64, sequence: u64) -> Result<wire::Envelope<'_>> {
    ensure!(bytes.len() >= 12, "truncated B-Human response");
    ensure!(
        wire::envelope_size_prefixed_buffer_has_identifier(bytes),
        "wrong B-Human protocol identifier"
    );
    let envelope = wire::size_prefixed_root_as_envelope(bytes)?;
    ensure!(
        envelope.version() == 2 && envelope.epoch() == epoch && envelope.sequence() == sequence,
        "stale or incompatible B-Human response"
    );
    if let Some(error) = envelope.payload_as_error() {
        bail!("B-Human worker: {}", error.message());
    }
    Ok(envelope)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn response_identity_rejects_stale_epochs_and_sequences() {
        let mut builder = FlatBufferBuilder::new();
        let payload = wire::Stopped::create(&mut builder, &wire::StoppedArgs {});
        let envelope = wire::Envelope::create(
            &mut builder,
            &wire::EnvelopeArgs {
                epoch: 7,
                version: 2,
                sequence: 3,
                payload_type: wire::Payload::Stopped,
                payload: Some(payload.as_union_value()),
                ..Default::default()
            },
        );
        wire::finish_size_prefixed_envelope_buffer(&mut builder, envelope);
        let bytes = builder.finished_data();
        assert!(decode(bytes, 7, 3).is_ok());
        assert!(decode(bytes, 8, 3).is_err());
        assert!(decode(bytes, 7, 4).is_err());
        assert!(decode(&bytes[..8], 7, 3).is_err());
    }
    #[test]
    fn response_deadline_bounds_partial_messages() {
        let (mut input, mut output) = UnixStream::pair().unwrap();
        output.write_all(&[1]).unwrap();
        let mut bytes = [0u8; 4];
        let started = Instant::now();
        assert!(
            read_exact_until(&mut input, &mut bytes, started + Duration::from_millis(30)).is_err()
        );
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    fn native_configuration(player: u8) -> Configuration {
        Configuration {
            executable: std::env::var_os("BHUMAN_WORKER")
                .expect("Set BHUMAN_WORKER")
                .into(),
            source: std::env::var_os("BHUMAN_SOURCE")
                .expect("Set BHUMAN_SOURCE")
                .into(),
            player,
            team: 5,
            cognition: false,
            active: true,
            field: None,
            timeout: Duration::from_secs(30),
        }
    }

    fn stationary_frame(time_ns: u64) -> Frame {
        Frame {
            time_ns,
            joints: crate::simulated_sdk::PREPARE_POSE.map(|position| JointSample {
                position,
                ..Default::default()
            }),
            angle: [0.0; 3],
            gyro: [0.0; 3],
            acceleration: [0.0, 0.0, 9.81],
            motion: Motion::Prepare,
            velocity: [0.0; 3],
            ..Default::default()
        }
    }

    fn soccer_frame(time_ns: u64, state: u8) -> Frame {
        let mut packet = vec![0; 158];
        packet[..4].copy_from_slice(b"RGme");
        packet[4] = 20;
        packet[5] = (time_ns / STEP_NS) as u8;
        packet[6] = 5;
        packet[7] = 1;
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
        for number in 6..=20 {
            packet[28 + (number - 1) * 3] = 13;
            packet[98 + (number - 1) * 3] = 13;
        }
        Frame {
            world: Some(World {
                pose: [-1.5, 0.0, 0.0],
                ball: Some(Ball {
                    position: [0.0, 0.0, 0.1],
                    velocity: [0.0; 3],
                }),
                players: Vec::new(),
            }),
            game_controller: Some(packet),
            ..stationary_frame(time_ns)
        }
    }

    #[test]
    #[ignore = "Requires the pinned B-Human worker and source checkout"]
    fn native_cognition_generates_motion_from_game_state_and_ground_truth() {
        let mut configuration = native_configuration(2);
        configuration.cognition = true;
        let mut robot = Robot::start(configuration).unwrap();
        for index in 0..100 {
            let result = robot.step(&soccer_frame(index * STEP_NS, 0)).unwrap();
            if index == 99 {
                assert_eq!(result.player_state, "active");
                assert_eq!(result.game_state, "beforeHalf");
                assert_eq!(result.motion, "stand");
            }
        }
        let mut motions = std::collections::BTreeSet::new();
        for index in 100..300 {
            let result = robot.step(&soccer_frame(index * STEP_NS, 3)).unwrap();
            motions.insert(result.motion);
        }
        assert!(
            motions
                .iter()
                .any(|motion| motion.starts_with("walk") || motion == "dribble"),
            "{motions:?}"
        );
        robot.stop().unwrap();
    }

    #[test]
    #[ignore = "Requires the pinned B-Human worker and source checkout"]
    fn native_handlers_apply_whistle_confidence_buttons_penalties_and_leds() {
        let mut configuration = native_configuration(2);
        configuration.cognition = true;
        let mut robot = Robot::start(configuration).unwrap();
        let mut last = None;
        for index in 0..1600 {
            let mut frame = soccer_frame(index * STEP_NS, if index < 150 { 2 } else { 3 });
            if index < 150 {
                frame.game_controller.as_mut().unwrap()[13] = 5;
            }
            if index == 30 || index == 80 {
                frame.whistle = Some(Whistle {
                    time_ns: frame.time_ns,
                    confidence: if index == 30 { 0.1 } else { 2.0 },
                    channels: 2,
                });
            }
            if index == 180 || index == 190 {
                // A GC penalty is authoritative even while autonomous behavior is playing.
                frame.game_controller.as_mut().unwrap()[31] = 1;
            }
            frame.buttons.f1 = (220..240).contains(&index);
            frame.buttons.stand = (300..1400).contains(&index);
            let result = robot.step(&frame).unwrap();
            match index {
                20 | 60 => {
                    assert!(
                        result.game_state.starts_with("waitFor"),
                        "{}",
                        result.game_state
                    );
                    assert_eq!(result.leds, [LedState::On, LedState::On, LedState::Off]);
                }
                100 => assert_eq!(result.game_state, "ownKickOff"),
                190 => {
                    assert_ne!(result.player_state, "active");
                    assert_eq!(result.leds, [LedState::On, LedState::Off, LedState::Off]);
                }
                250 | 1000 => {
                    assert_eq!(result.player_state, "unstiff");
                    assert_eq!(result.motion, "playDead");
                    assert_eq!(
                        result.leds,
                        [LedState::Off, LedState::Off, LedState::Blinking]
                    );
                }
                1500 => assert_eq!(result.player_state, "active"),
                _ => {}
            }
            last = Some(result);
        }
        assert!(last.unwrap().cognition_time_ns > 0);
        robot.stop().unwrap();
    }

    #[test]
    #[ignore = "Requires the pinned B-Human worker and source checkout"]
    fn native_team_messages_round_trip_through_host_delivery() {
        let mut robots: Vec<_> = [2, 3]
            .into_iter()
            .map(|player| {
                let mut configuration = native_configuration(player);
                configuration.cognition = true;
                Robot::start(configuration).unwrap()
            })
            .collect();
        let mut pending = [Vec::new(), Vec::new()];
        let mut received = [false; 2];
        let mut sent = 0;
        for index in 0..2000 {
            let mut outgoing = [Vec::new(), Vec::new()];
            for robot in 0..2 {
                let mut frame = soccer_frame(index * STEP_NS, 3);
                frame.game_controller.as_mut().unwrap()[5] = (index / 250) as u8;
                frame.team_messages = std::mem::take(&mut pending[robot]);
                let result = robots[robot].step(&frame).unwrap();
                received[robot] |= result.teammates > 0;
                sent += result.team_messages.len();
                outgoing[robot] = result.team_messages;
            }
            pending[0] = std::mem::take(&mut outgoing[1]);
            pending[1] = std::mem::take(&mut outgoing[0]);
        }
        assert!(sent > 0, "native send policy never produced a packet");
        assert_eq!(
            received, [true; 2],
            "native receivers did not decode teammate packets"
        );
        for robot in robots {
            robot.stop().unwrap();
        }
    }

    #[test]
    #[ignore = "Requires the pinned B-Human worker and source checkout"]
    fn native_workers_keep_independent_clocks_and_reset_state() {
        let mut first = Robot::start(native_configuration(1)).unwrap();
        let mut second = Robot::start(native_configuration(2)).unwrap();
        assert_ne!(first.child.id(), second.child.id());
        let original = first.step(&stationary_frame(0)).unwrap();
        first.step(&stationary_frame(STEP_NS)).unwrap();
        second.step(&stationary_frame(0)).unwrap();
        first.reset().unwrap();
        let reset = first.step(&stationary_frame(0)).unwrap();
        assert_eq!(original.phase, reset.phase);
        for (original, reset) in original.joints.iter().zip(&reset.joints) {
            assert!((original.position - reset.position).abs() < 1e-6);
            assert_eq!(original.kp, reset.kp);
        }
        second.step(&stationary_frame(STEP_NS)).unwrap();
        thread::sleep(Duration::from_millis(30));
        first.step(&stationary_frame(STEP_NS)).unwrap();
        first.stop().unwrap();
        second.stop().unwrap();
    }

    #[test]
    #[ignore = "Requires the pinned B-Human worker and source checkout"]
    fn native_worker_crash_invalidates_the_connection() {
        let mut robot = Robot::start(native_configuration(1)).unwrap();
        robot.child.kill().unwrap();
        assert!(robot.step(&stationary_frame(0)).is_err());
        assert!(robot.stream.is_none());
        assert!(robot.child.try_wait().unwrap().is_some());
    }

    #[test]
    #[ignore = "Requires the pinned B-Human worker and source checkout"]
    fn native_worker_rejects_out_of_order_frames() {
        let mut robot = Robot::start(native_configuration(1)).unwrap();
        robot.sequence += 1;
        let error = robot.step(&stationary_frame(0)).err().unwrap().to_string();
        assert!(error.contains("out-of-order"), "{error}");
        assert!(robot.stream.is_none());
    }
}

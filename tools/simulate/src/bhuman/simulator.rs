use std::{collections::VecDeque, net::UdpSocket, sync::Mutex};

use booster::{ButtonEventType, CommandType, LedColor, LowCommand, MotorCommand};
use color_eyre::{Result, eyre::ensure};
use ros_z::prelude::*;
use types::primary_state::PrimaryState;

use super::{
    Actuation, Buttons, Configuration, Frame, JointSample, LedState, Robot, STEP_NS, Whistle, World,
};
use crate::{network::Ports, robot_io::Observation, simulated_sdk::Control};

#[cfg(test)]
#[path = "simulator_tests.rs"]
mod tests;

pub(crate) struct SimulatorRobot {
    state: Mutex<State>,
    primary: Publisher<PrimaryState>,
    runtime: tokio::runtime::Handle,
}
struct State {
    robot: Robot,
    time_ns: u64,
    buttons: Buttons,
    button_queue: VecDeque<(i32, bool)>,
    whistle: bool,
    incoming: Vec<Vec<u8>>,
    game: UdpSocket,
    sender: UdpSocket,
    ports: Ports,
    away: bool,
    packet: Option<Vec<u8>>,
    output: Option<Actuation>,
    primary: Option<PrimaryState>,
}

impl SimulatorRobot {
    pub async fn start(configuration: Configuration, ports: Ports, node: &Node) -> Result<Self> {
        let robot = tokio::task::spawn_blocking(move || Robot::start(configuration)).await??;
        let game = UdpSocket::bind(("127.0.0.1", ports.state))?;
        game.set_nonblocking(true)?;
        Ok(Self {
            runtime: tokio::runtime::Handle::current(),
            primary: node
                .publisher("primary_state")
                .qos(QosProfile {
                    durability: ros_z::qos::QosDurability::TransientLocal,
                    ..Default::default()
                })
                .build()
                .await?,
            state: Mutex::new(State {
                robot,
                time_ns: 0,
                buttons: Buttons::default(),
                button_queue: VecDeque::new(),
                whistle: false,
                incoming: Vec::new(),
                game,
                sender: UdpSocket::bind("127.0.0.1:0")?,
                ports,
                away: true,
                packet: None,
                output: None,
                primary: None,
            }),
        })
    }

    /// Called once for each committed physics step, outside the physics mutex.
    pub fn step(&self, observation: &Observation, mut world: World) -> Result<Vec<Vec<u8>>> {
        let mut state = self.state.lock().unwrap();
        let mut packet = None;
        for _ in 0..64 {
            let mut buffer = [0u8; 65536];
            match state.game.recv(&mut buffer) {
                Ok(size) => {
                    let mut bytes = buffer[..size].to_vec();
                    // Network presents opponents as team 24 for the HULK endpoints.
                    // Undo that translation before passing data to native team 5.
                    if crate::network::translate_state(&mut bytes) {
                        state.away = bytes[18] != 5;
                        state.packet = Some(bytes.clone());
                        packet = Some(bytes);
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) => return Err(error.into()),
            }
        }
        if state.away {
            let rotate = |pose: &mut [f32; 3]| {
                pose[0] = -pose[0];
                pose[1] = -pose[1];
                pose[2] = (pose[2] + std::f32::consts::PI)
                    .sin()
                    .atan2((pose[2] + std::f32::consts::PI).cos());
            };
            rotate(&mut world.pose);
            for player in &mut world.players {
                rotate(&mut player.pose);
            }
            if let Some(ball) = &mut world.ball {
                for i in 0..2 {
                    ball.position[i] = -ball.position[i];
                    ball.velocity[i] = -ball.velocity[i];
                }
            }
        }
        // At most one edge per cognition frame, so a short tap between physics
        // steps is visible to native KeyStateEnhancer as a complete press/release.
        if state.time_ns.is_multiple_of(20_000_000)
            && let Some((button, pressed)) = state.button_queue.pop_front()
        {
            match button {
                0 => state.buttons.f1 = pressed,
                1 => state.buttons.stand = pressed,
                2 => state.buttons.walk = pressed,
                _ => unreachable!(),
            }
        }
        let low = &observation.low_state;
        let imu = &low.imu_state;
        let frame = Frame {
            time_ns: state.time_ns,
            joints: std::array::from_fn(|index| {
                let sample = &low.motor_state_serial[index];
                JointSample {
                    position: sample.position,
                    velocity: sample.velocity,
                    torque: sample.torque,
                }
            }),
            angle: [
                imu.roll_pitch_yaw.x(),
                imu.roll_pitch_yaw.y(),
                imu.roll_pitch_yaw.z(),
            ],
            gyro: [
                imu.angular_velocity.x(),
                imu.angular_velocity.y(),
                imu.angular_velocity.z(),
            ],
            acceleration: [
                imu.linear_acceleration.x(),
                imu.linear_acceleration.y(),
                imu.linear_acceleration.z(),
            ],
            buttons: state.buttons,
            whistle: std::mem::take(&mut state.whistle).then_some(Whistle {
                time_ns: state.time_ns,
                confidence: 2.0,
                channels: 2,
            }),
            game_controller: packet,
            team_messages: std::mem::take(&mut state.incoming),
            world: Some(world.clone()),
            ..Default::default()
        };
        let result = state.robot.step(&frame)?;
        let primary = primary_state(&result);
        if state.primary != Some(primary) {
            // The caller is the blocking physics worker, never a runtime worker.
            self.runtime.block_on(self.primary.publish(&primary))?;
            state.primary = Some(primary);
        }
        if state.time_ns.is_multiple_of(500_000_000) {
            let bytes = return_packet(&state, &world, observation);
            state
                .sender
                .send_to(&bytes, ("127.0.0.1", state.ports.returns))?;
        }
        for packet in &result.team_messages {
            state
                .sender
                .send_to(packet, ("127.0.0.1", state.ports.team))?;
        }
        let outgoing = result.team_messages.clone();
        state.output = Some(result);
        state.time_ns += STEP_NS;
        Ok(outgoing)
    }

    pub fn enqueue_team(&self, packets: &[Vec<u8>]) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        ensure!(
            state.incoming.len() + packets.len() <= 64,
            "B-Human teammate queue overflow"
        );
        state.incoming.extend_from_slice(packets);
        Ok(())
    }
    pub fn button(&self, button: i32, event: ButtonEventType) -> Result<()> {
        ensure!((0..=2).contains(&button), "unknown body button");
        let pressed = match event {
            ButtonEventType::PressDown => true,
            ButtonEventType::PressUp => false,
            _ => return Ok(()),
        };
        let mut state = self.state.lock().unwrap();
        ensure!(
            state.button_queue.len() < 64,
            "B-Human button queue overflow"
        );
        state.button_queue.push_back((button, pressed));
        Ok(())
    }
    pub fn whistle(&self) {
        self.state.lock().unwrap().whistle = true;
    }
    pub fn away(&self) -> bool {
        self.state.lock().unwrap().away
    }
    pub fn poll(&self) -> Option<String> {
        let mut state = self.state.lock().unwrap();
        match state.robot.child.try_wait() {
            Ok(None) => None,
            Ok(Some(status)) => Some(format!("B-Human worker exited: {status}")),
            Err(error) => Some(format!("B-Human worker status: {error}")),
        }
    }
    pub fn status(&self) -> String {
        let state = self.state.lock().unwrap();
        state.output.as_ref().map_or_else(
            || "B-Human · initialized".into(),
            |output| {
                format!(
                    "B-Human · {} · {} · {}",
                    output.player_state, output.game_state, output.phase
                )
            },
        )
    }
    pub fn control(&self) -> Control {
        let state = self.state.lock().unwrap();
        let Some(output) = &state.output else {
            return Control::default();
        };
        let command = LowCommand {
            command_type: CommandType::Serial,
            motor_commands: output
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
        let mut control = Control::native(command, output.joints.map(|joint| joint.torque_limit));
        let time = 100_000 + state.time_ns.saturating_sub(STEP_NS) / 1_000_000;
        let [r, g, b] = output.leds.map(|led| led_brightness(led, time));
        control.led = Some(LedColor { r, g, b });
        control
    }
    pub fn jersey_color(&self, player: u8) -> Option<hsl_network_messages::TeamColor> {
        let state = self.state.lock().unwrap();
        let mut bytes = state.packet.clone()?;
        if !crate::network::translate_state(&mut bytes) {
            return None;
        }
        let data =
            hsl_network_messages::GameControllerStateMessage::try_from(bytes.as_slice()).ok()?;
        let team = data.hulks_team;
        Some(
            if team
                .goal_keeper_player_number
                .is_some_and(|number| number as u8 + 1 == player)
            {
                team.goal_keeper_color
            } else {
                team.field_player_color
            },
        )
    }
    pub fn reset(&self) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        state.robot.reset()?;
        state.time_ns = 0;
        state.output = None;
        state.buttons = Buttons::default();
        state.button_queue.clear();
        state.incoming.clear();
        state.whistle = false;
        state.primary = None;
        Ok(())
    }
}

fn primary_state(output: &Actuation) -> PrimaryState {
    if output.player_state == "unstiff" {
        return PrimaryState::Damping;
    }
    if output.player_state == "calibration" {
        return PrimaryState::Prepare;
    }
    if output.player_state != "active" {
        return PrimaryState::Penalized;
    }
    match output.game_state.as_str() {
        "beforeHalf" => PrimaryState::Initial,
        "afterHalf" => PrimaryState::Finished,
        state if state.starts_with("setup") => PrimaryState::Ready,
        state if state.starts_with("waitFor") => PrimaryState::Set,
        _ => PrimaryState::Playing,
    }
}

/// Matches native BoosterProvider's RGB conversion, on the logical robot clock.
pub(super) fn led_brightness(state: LedState, time_ms: u64) -> u8 {
    let on = time_ms / 400 & 1 == 1;
    let fast = time_ms / 80 & 1 == 0;
    match state {
        LedState::On => 255,
        LedState::Half => 128,
        LedState::Blinking if on => 255,
        LedState::FastBlinking if fast => 255,
        LedState::HalfBlinking if on => 128,
        LedState::HalfFastBlinking if fast => 128,
        _ => 0,
    }
}

fn return_packet(state: &State, world: &World, observation: &Observation) -> Vec<u8> {
    let pose = world.pose;
    let ball = world.ball.map(|ball| {
        let dx = ball.position[0] - pose[0];
        let dy = ball.position[1] - pose[1];
        [
            pose[2].cos() * dx + pose[2].sin() * dy,
            -pose[2].sin() * dx + pose[2].cos() * dy,
        ]
    });
    let mut packet = vec![0u8; 32];
    packet[..4].copy_from_slice(b"RGrt");
    packet[4] = 4;
    packet[5] = state.robot.configuration.player;
    packet[6] = 24;
    packet[7] = u8::from(
        observation.robot_to_world.translation.z < 0.4
            || observation.robot_to_world.rotation.to_rotation_matrix()[(2, 2)] < 0.5,
    );
    let values = [
        pose[0] * 1000.0,
        pose[1] * 1000.0,
        pose[2],
        if ball.is_some() { 0.0 } else { -1.0 },
        ball.map_or(0.0, |b| b[0] * 1000.0),
        ball.map_or(0.0, |b| b[1] * 1000.0),
    ];
    for (index, value) in values.into_iter().enumerate() {
        packet[8 + index * 4..12 + index * 4].copy_from_slice(&value.to_le_bytes());
    }
    packet
}

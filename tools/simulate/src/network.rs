//! Raw UDP fan-out for the unchanged production message nodes.
use crate::TeamId;
use color_eyre::Result;
use std::collections::BTreeMap;
use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
};
use tokio::{net::UdpSocket, task::JoinHandle};

#[derive(Clone, Copy)]
pub(crate) struct Ports {
    pub state: u16,
    pub returns: u16,
    pub team: u16,
}
pub(crate) struct Connection {
    pub ports: Ports,
    clients: Arc<Mutex<Vec<(u16, TeamId)>>>,
    task: JoinHandle<()>,
    runtime: tokio::runtime::Handle,
}
impl Drop for Connection {
    fn drop(&mut self) {
        self.clients
            .lock()
            .unwrap()
            .retain(|(port, _)| *port != self.ports.state);
        self.task.abort();
        drain(&self.runtime, std::iter::once(&mut self.task));
    }
}
pub(crate) struct Network {
    clients: Arc<Mutex<Vec<(u16, TeamId)>>>,
    source: Arc<Mutex<Option<SocketAddr>>>,
    socket: Arc<UdpSocket>,
    tasks: Vec<JoinHandle<()>>,
    return_port: u16,
    team_ports: BTreeMap<TeamId, u16>,
    runtime: tokio::runtime::Handle,
}
impl Network {
    pub async fn new(port: u16, return_port: u16) -> Result<Self> {
        let socket = Arc::new(UdpSocket::bind(("0.0.0.0", port)).await?);
        let clients = Arc::new(Mutex::new(Vec::<(u16, TeamId)>::new()));
        let source = Arc::new(Mutex::new(None));
        let mut team_ports = BTreeMap::new();
        let mut team_sockets = Vec::new();
        for team in TeamId::ALL {
            // Each side has a private broadcast channel. Production teammate
            // messages carry a player number, but no team identifier.
            let team_socket = socket2::Socket::new(
                socket2::Domain::IPV4,
                socket2::Type::DGRAM,
                Some(socket2::Protocol::UDP),
            )?;
            team_socket.set_reuse_address(true)?;
            #[cfg(target_os = "macos")]
            team_socket.set_reuse_port(true)?;
            team_socket
                .bind(&std::net::SocketAddrV4::new(std::net::Ipv4Addr::UNSPECIFIED, 0).into())?;
            team_socket.set_nonblocking(true)?;
            let team_socket = UdpSocket::from_std(team_socket.into())?;
            team_ports.insert(team, team_socket.local_addr()?.port());
            team_sockets.push((team, team_socket));
        }
        let task = {
            let socket = socket.clone();
            let clients = clients.clone();
            let source = source.clone();
            tokio::spawn(async move {
                let mut buffer = [0; 65536];
                while let Ok((size, from)) = socket.recv_from(&mut buffer).await {
                    *source.lock().unwrap() = Some(from);
                    let ports = clients.lock().unwrap().clone();
                    for (port, team) in ports {
                        let mut packet = buffer[..size].to_vec();
                        if team == TeamId::Opponents && !translate_state(&mut packet) {
                            continue;
                        }
                        if let Err(error) = socket.send_to(&packet, ("127.0.0.1", port)).await {
                            log::warn!("Simulator GameController delivery: {error}");
                        }
                    }
                }
            })
        };
        let mut tasks = vec![task];
        for (team, team_socket) in team_sockets {
            let team_source = source.clone();
            tasks.push(tokio::spawn(async move {
                let mut buffer = [0; 65536];
                while let Ok((size, _)) = team_socket.recv_from(&mut buffer).await {
                    let destination = *team_source.lock().unwrap();
                    if let Some(destination) = destination
                        && let Err(error) = team_socket
                            .send_to(
                                &buffer[..size],
                                (destination.ip(), 10000 + u16::from(team.wire_number())),
                            )
                            .await
                    {
                        log::warn!("Simulator team packet to GameController: {error}");
                    }
                }
            }));
        }
        Ok(Self {
            clients,
            source,
            socket,
            tasks,
            team_ports,
            runtime: tokio::runtime::Handle::current(),
            return_port,
        })
    }
    pub async fn add_robot(&mut self, team: TeamId) -> Result<Connection> {
        // The production endpoint must own the state socket. Reserve its port
        // until all other sockets for this robot have been allocated.
        let reserved = UdpSocket::bind("127.0.0.1:0").await?;
        let state = reserved.local_addr()?.port();
        let returns = UdpSocket::bind("127.0.0.1:0").await?;
        let port = returns.local_addr()?.port();
        let source = self.source.clone();
        let return_port = self.return_port;
        let socket = self.socket.clone();
        let task = tokio::spawn(async move {
            let mut buffer = [0; 65536];
            while let Ok((size, _)) = returns.recv_from(&mut buffer).await {
                if team == TeamId::Opponents && !translate_return(&mut buffer[..size]) {
                    continue;
                }
                let destination = *source.lock().unwrap();
                if let Some(destination) = destination
                    && let Err(error) = socket
                        .send_to(&buffer[..size], (destination.ip(), return_port))
                        .await
                {
                    log::warn!("Simulator GameController return: {error}");
                }
            }
        });
        self.clients.lock().unwrap().push((state, team));
        Ok(Connection {
            ports: Ports {
                state,
                returns: port,
                team: self.team_ports[&team],
            },
            clients: self.clients.clone(),
            task,
            runtime: self.runtime.clone(),
        })
    }
}
impl Drop for Network {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
        drain(&self.runtime, self.tasks.iter_mut());
    }
}

// Production runs on Twix's multithreaded runtime; drain aborted tasks before
// releasing the simulator lease so an immediate restart can bind the sockets.
fn drain<'a>(
    runtime: &tokio::runtime::Handle,
    tasks: impl Iterator<Item = &'a mut JoinHandle<()>>,
) {
    let join = async {
        for task in tasks {
            let _ = task.await;
        }
    };
    if tokio::runtime::Handle::try_current().is_ok() {
        if runtime.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread {
            tokio::task::block_in_place(|| runtime.block_on(join));
        }
    } else {
        runtime.block_on(join);
    }
}

// Wire offsets from RoboCupGameControlData.hpp, version 20 (158 bytes).
// Swap identifiers, never team records: their order defines the physical side.
fn translate_state(packet: &mut [u8]) -> bool {
    if packet.len() != 158 || &packet[..4] != b"RGme" || packet[4] != 20 {
        return false;
    }
    if ![packet[18], packet[88]].contains(&24) || ![packet[18], packet[88]].contains(&5) {
        return false;
    }
    for offset in [13, 18, 88] {
        packet[offset] = match packet[offset] {
            24 => 5,
            5 => 24,
            other => other,
        };
    }
    true
}
// RoboCupGameControlReturnData, version 4. Coordinates are already team-relative.
fn translate_return(packet: &mut [u8]) -> bool {
    if packet.len() != 32 || &packet[..4] != b"RGrt" || packet[4] != 4 || packet[6] != 24 {
        return false;
    }
    packet[6] = 5;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state_packet() -> Vec<u8> {
        let mut packet = vec![0; 158];
        packet[..4].copy_from_slice(b"RGme");
        packet[4] = 20;
        packet[6] = 5;
        packet[7] = 1;
        packet[10] = 1;
        packet[12] = 1;
        packet[13] = 24;
        packet[18] = 24;
        packet[88] = 5;
        packet[22] = 1;
        packet[92] = 2;
        packet[104] = 6; // Away player 3, pick up.
        packet
    }
    #[test]
    fn adapter_preserves_side_penalties_scores_and_kickoff_with_production_parser() {
        use hsl_network_messages::{GameControllerStateMessage, GamePhase, Team};
        for reverse_sides in [false, true] {
            for phase in [0, 1] {
                for kicking in [24, 5, 255] {
                    let mut packet = state_packet();
                    packet[9] = phase;
                    packet[13] = kicking;
                    if reverse_sides {
                        let (first, second) = packet[18..].split_at_mut(70);
                        first.swap_with_slice(second);
                    }
                    let before = packet.clone();
                    assert!(translate_state(&mut packet));
                    let parsed = GameControllerStateMessage::try_from(packet.as_slice()).unwrap();
                    assert_eq!(parsed.hulks_team_is_home_after_coin_toss, reverse_sides);
                    assert_eq!(parsed.hulks_team.score, 2);
                    assert_eq!(parsed.opponent_team.score, 1);
                    assert!(parsed.hulks_team.players[2].penalty.is_some());
                    assert!(parsed.opponent_team.players[2].penalty.is_none());
                    assert_eq!(
                        parsed.kicking_team,
                        match kicking {
                            24 => Some(Team::Opponent),
                            5 => Some(Team::Hulks),
                            _ => None,
                        }
                    );
                    if phase == 1 {
                        assert_eq!(
                            parsed.game_phase,
                            GamePhase::PenaltyShootout {
                                kicking_team: if kicking == 5 {
                                    Team::Hulks
                                } else {
                                    Team::Opponent
                                }
                            }
                        );
                    }
                    for i in 0..packet.len() {
                        if ![13, 18, 88].contains(&i) {
                            assert_eq!(packet[i], before[i]);
                        }
                    }
                }
            }
        }
        for size in [0, 4, 157, 159] {
            assert!(!translate_state(&mut vec![0; size]));
        }
        let mut packet = state_packet();
        packet[4] = 21;
        assert!(!translate_state(&mut packet));
        let mut packet = state_packet();
        packet[88] = 7;
        assert!(!translate_state(&mut packet));
        let mut returns: Vec<_> = (0..32).collect();
        returns[..4].copy_from_slice(b"RGrt");
        returns[4] = 4;
        returns[5] = 3;
        returns[6] = 24;
        let before = returns.clone();
        assert!(translate_return(&mut returns));
        assert_eq!(returns[6], 5);
        for i in 0..32 {
            if i != 6 {
                assert_eq!(returns[i], before[i]);
            }
        }
        returns[4] = 5;
        assert!(!translate_return(&mut returns));
    }
    #[tokio::test]
    async fn both_sides_receive_their_view_and_return_distinct_team_ids() {
        let returns = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let mut network = Network::new(0, returns.local_addr().unwrap().port())
            .await
            .unwrap();
        let mut robots = Vec::new();
        for team in TeamId::ALL {
            let connection = network.add_robot(team).await.unwrap();
            let socket = UdpSocket::bind(("127.0.0.1", connection.ports.state))
                .await
                .unwrap();
            robots.push((team, connection, socket));
        }
        assert_ne!(robots[0].1.ports.team, robots[1].1.ports.team);
        let sender = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        sender
            .send_to(&state_packet(), network.socket.local_addr().unwrap())
            .await
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(3), async {
            let mut buffer = [0; 1024];
            for (team, connection, robot) in &robots {
                let (size, _) = robot.recv_from(&mut buffer).await.unwrap();
                let parsed =
                    hsl_network_messages::GameControllerStateMessage::try_from(&buffer[..size])
                        .unwrap();
                assert_eq!(
                    parsed.hulks_team_is_home_after_coin_toss,
                    *team == TeamId::Hulks
                );
                let mut packet = [0; 32];
                packet[..4].copy_from_slice(b"RGrt");
                packet[4] = 4;
                packet[5] = 1;
                packet[6] = 24;
                robot
                    .send_to(&packet, ("127.0.0.1", connection.ports.returns))
                    .await
                    .unwrap();
                let (size, _) = returns.recv_from(&mut buffer).await.unwrap();
                assert_eq!(size, 32);
                assert_eq!(buffer[5], 1);
                assert_eq!(buffer[6], team.wire_number());
            }
        })
        .await
        .unwrap();
    }
    #[tokio::test]
    async fn fans_out_raw_packets_and_preserves_every_robot_return() {
        let returns = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let mut network = Network::new(0, returns.local_addr().unwrap().port())
            .await
            .unwrap();
        let mut robots = Vec::new();
        for _ in 0..5 {
            let connection = network.add_robot(TeamId::Hulks).await.unwrap();
            let ports = connection.ports;
            let socket = UdpSocket::bind(("127.0.0.1", ports.state)).await.unwrap();
            robots.push((connection, socket));
        }
        let sender = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        sender
            .send_to(
                b"RGme-real-payload",
                ("127.0.0.1", network.socket.local_addr().unwrap().port()),
            )
            .await
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(3), async {
            let mut buffer = [0; 1024];
            for (index, (connection, robot)) in robots.iter().enumerate() {
                let (size, source) = robot.recv_from(&mut buffer).await.unwrap();
                assert_eq!(&buffer[..size], b"RGme-real-payload");
                robot
                    .send_to(&[index as u8], (source.ip(), connection.ports.returns))
                    .await
                    .unwrap();
            }
            let mut received = Vec::new();
            for _ in 0..5 {
                let (size, _) = returns.recv_from(&mut buffer).await.unwrap();
                assert_eq!(size, 1);
                received.push(buffer[0]);
            }
            received.sort();
            assert_eq!(received, [0, 1, 2, 3, 4]);
        })
        .await
        .unwrap();
        drop(robots);
        assert!(network.clients.lock().unwrap().is_empty());
    }
}

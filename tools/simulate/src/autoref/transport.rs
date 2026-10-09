//! Real GameController UDP, on private loopback ports in automatic mode.
use super::{Engine, side};
use crate::TeamId;
use bytes::Bytes;
use game_controller_core::{action::VAction, actions::TeamMessage, types::ActionSource};
use game_controller_msgs::{ControlMessage, StatusMessage};
use std::{
    collections::BTreeMap,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::{net::UdpSocket, task::JoinHandle};

#[derive(Clone)]
pub(crate) struct Endpoints {
    pub source: SocketAddr,
    pub returns: u16,
    pub budgets: BTreeMap<TeamId, u16>,
}
pub(super) struct Transport {
    socket: Arc<UdpSocket>,
    endpoints: Endpoints,
    tasks: Vec<JoinHandle<()>>,
    runtime: tokio::runtime::Handle,
}
impl Transport {
    pub async fn new(engine: Arc<Mutex<Engine>>) -> color_eyre::Result<Self> {
        let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").await?);
        let returns = UdpSocket::bind("127.0.0.1:0").await?;
        let mut endpoints = Endpoints {
            source: socket.local_addr()?,
            returns: returns.local_addr()?.port(),
            budgets: BTreeMap::new(),
        };
        let mut budgets = Vec::new();
        for team in TeamId::ALL {
            let socket = UdpSocket::bind("127.0.0.1:0").await?;
            endpoints.budgets.insert(team, socket.local_addr()?.port());
            budgets.push((team, socket));
        }
        let observed = engine.clone();
        let mut tasks = vec![tokio::spawn(async move {
            let mut buffer = [0; 1024];
            while let Ok((size, _)) = returns.recv_from(&mut buffer).await {
                if let Ok(status) = StatusMessage::try_from(Bytes::copy_from_slice(&buffer[..size]))
                {
                    let mut engine = observed.lock().unwrap();
                    if engine
                        .core
                        .params
                        .game
                        .get_side(status.team_number)
                        .is_some()
                    {
                        let time = engine.core.get_time();
                        engine
                            .returns
                            .insert((status.team_number, status.player_number), time);
                    }
                }
            }
        })];
        for (team, socket) in budgets {
            let engine = engine.clone();
            tasks.push(tokio::spawn(async move {
                let mut buffer = [0; 65536];
                while let Ok((size, _)) = socket.recv_from(&mut buffer).await {
                    engine.lock().unwrap().core.apply(
                        VAction::TeamMessage(TeamMessage {
                            side: side(team),
                            illegal: size > game_controller_msgs::TEAM_MESSAGE_MAX_SIZE,
                        }),
                        ActionSource::Network,
                    );
                }
            }));
        }
        Ok(Self {
            socket,
            endpoints,
            tasks,
            runtime: tokio::runtime::Handle::current(),
        })
    }
    pub fn endpoints(&self) -> Endpoints {
        self.endpoints.clone()
    }
    pub fn connect(
        &mut self,
        destination: SocketAddr,
        engine: Arc<Mutex<Engine>>,
        clock: ros_z::time::Clock,
    ) {
        let socket = self.socket.clone();
        self.tasks.push(tokio::spawn(async move {
            let period = Duration::from_millis(100);
            let mut sequence = 0;
            loop {
                let packet: Bytes = {
                    let engine = engine.lock().unwrap();
                    ControlMessage::new(
                        engine.core.get_game(true),
                        &engine.core.params,
                        sequence,
                        false,
                    )
                    .into()
                };
                if socket.send_to(&packet, destination).await.is_err() {
                    break;
                }
                sequence = sequence.wrapping_add(1);
                // Scale packet cadence with robotics time, retaining a wall-time heartbeat
                // while paused or slow. Rebase after each send instead of bursting on resume.
                tokio::select! {
                    _ = clock.sleep(period) => {},
                    _ = tokio::time::sleep(period) => {},
                }
            }
        }));
    }
    pub fn poll(&self) -> Option<String> {
        self.tasks
            .iter()
            .any(JoinHandle::is_finished)
            .then(|| "Automatic referee UDP task stopped".into())
    }
}
impl Drop for Transport {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
        crate::network::drain(&self.runtime, self.tasks.iter_mut());
    }
}

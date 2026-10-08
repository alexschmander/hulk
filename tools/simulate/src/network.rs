//! Raw UDP fan-out for the unchanged production message nodes.
use color_eyre::Result;
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
    clients: Arc<Mutex<Vec<u16>>>,
    task: JoinHandle<()>,
    runtime: tokio::runtime::Handle,
}
impl Drop for Connection {
    fn drop(&mut self) {
        self.clients
            .lock()
            .unwrap()
            .retain(|port| *port != self.ports.state);
        self.task.abort();
        drain(&self.runtime, std::iter::once(&mut self.task));
    }
}
pub(crate) struct Network {
    clients: Arc<Mutex<Vec<u16>>>,
    source: Arc<Mutex<Option<SocketAddr>>>,
    socket: Arc<UdpSocket>,
    tasks: Vec<JoinHandle<()>>,
    return_port: u16,
    team_port: u16,
    runtime: tokio::runtime::Handle,
}
impl Network {
    pub async fn new(port: u16, return_port: u16) -> Result<Self> {
        let socket = Arc::new(UdpSocket::bind(("0.0.0.0", port)).await?);
        let clients = Arc::new(Mutex::new(Vec::<u16>::new()));
        let source = Arc::new(Mutex::new(None));
        // Keep production team packets on an ephemeral port shared only by this
        // simulation, then forward a copy to the GameController for accounting.
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
        let team_port = team_socket.local_addr()?.port();
        let task = {
            let socket = socket.clone();
            let clients = clients.clone();
            let source = source.clone();
            tokio::spawn(async move {
                let mut buffer = [0; 65536];
                while let Ok((size, from)) = socket.recv_from(&mut buffer).await {
                    *source.lock().unwrap() = Some(from);
                    let ports = clients.lock().unwrap().clone();
                    for port in ports {
                        if let Err(error) =
                            socket.send_to(&buffer[..size], ("127.0.0.1", port)).await
                        {
                            log::warn!("Simulator GameController delivery: {error}");
                        }
                    }
                }
            })
        };
        let team_source = source.clone();
        let team_task = tokio::spawn(async move {
            let mut buffer = [0; 65536];
            while let Ok((size, _)) = team_socket.recv_from(&mut buffer).await {
                let destination = *team_source.lock().unwrap();
                if let Some(destination) = destination
                    && let Err(error) = team_socket
                        .send_to(&buffer[..size], (destination.ip(), 10024))
                        .await
                {
                    log::warn!("Simulator team packet to GameController: {error}");
                }
            }
        });
        Ok(Self {
            clients,
            source,
            socket,
            tasks: vec![task, team_task],
            team_port,
            runtime: tokio::runtime::Handle::current(),
            return_port,
        })
    }
    pub async fn add_robot(&mut self) -> Result<Connection> {
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
        self.clients.lock().unwrap().push(state);
        Ok(Connection {
            ports: Ports {
                state,
                returns: port,
                team: self.team_port,
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

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn fans_out_raw_packets_and_preserves_every_robot_return() {
        let returns = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let mut network = Network::new(0, returns.local_addr().unwrap().port())
            .await
            .unwrap();
        let mut robots = Vec::new();
        for _ in 0..5 {
            let connection = network.add_robot().await.unwrap();
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

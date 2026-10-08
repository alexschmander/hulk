use super::*;
use hsl_network_messages::{HulkMessage, PlayerNumber, StateMessage};
use std::{net::UdpSocket as StdSocket, time::Duration};
use tokio::net::UdpSocket;
use types::{filtered_game_state::FilteredGameState, players::Players, world_state::PlayerState};

fn reserve_port() -> StdSocket {
    StdSocket::bind("127.0.0.1:0").unwrap()
}

// Protocol v20 from crates/hsl_network_messages/headers/RoboCupGameControlData.hpp.
fn game_packet(state: u8, penalized: bool) -> Vec<u8> {
    let mut bytes = vec![0; 158];
    bytes[..4].copy_from_slice(b"RGme");
    bytes[4] = 20;
    bytes[6] = 5;
    bytes[10] = state;
    bytes[12] = 1;
    bytes[13] = 24;
    bytes[14..16].copy_from_slice(&600i16.to_le_bytes());
    for (offset, team) in [(18, 24), (88, 25)] {
        bytes[offset] = team;
        bytes[offset + 3] = 1;
        bytes[offset + 8..offset + 10].copy_from_slice(&1000u16.to_le_bytes());
    }
    if penalized {
        bytes[18 + 10 + 2 * 3] = 6;
    } // HULKs player three, PickUp.
    bytes
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn real_udp_game_controller_returns_penalties_and_team_messages() {
    let gc_port = reserve_port();
    let hsl_port = reserve_port();
    let gc_address = gc_port.local_addr().unwrap();
    let hsl_address = hsl_port.local_addr().unwrap();
    let returns = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let parameters = tempfile::tempdir().unwrap();
    std::fs::write(parameters.path().join("message_receiver.json5"), serde_json::json!({"ports": {
        "game_controller_state": gc_address.port(), "game_controller_return": returns.local_addr().unwrap().port(),
        "hsl": hsl_address.port(), "hsl_broadcast_address": {"octets": [127,0,0,1]}
    }}).to_string()).unwrap();
    // A private peer prevents this test from discovering another running robot.
    let context = Arc::new(
        ContextBuilder::default()
            .with_namespace("/network_test")
            .with_parameter_layers([
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../etc/parameters/base"),
                parameters.path().to_owned(),
            ])
            .disable_multicast_scouting()
            .with_connect_endpoints(std::iter::empty::<&str>())
            .with_listen_endpoints(["tcp/127.0.0.1:0"])
            .build()
            .await
            .unwrap(),
    );
    let node = context.create_node("observer").build().await.unwrap();
    let game = node
        .subscriber::<FilteredGameControllerState>("filtered_game_controller_state")
        .build()
        .await
        .unwrap();
    let players = node
        .subscriber::<Players<Option<TimeWrapper<PlayerState>>>>("player_states")
        .build()
        .await
        .unwrap();
    let mut tasks = JoinSet::new();
    tasks.spawn(global_parameter_provider::run_boxed(context.clone()));
    tasks.spawn(behavior_node::run_boxed(context.clone()));
    drop(gc_port);
    drop(hsl_port);
    spawn_network(&context, &mut tasks);
    let sender = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    tokio::time::timeout(Duration::from_secs(8), async {
        // Repetition models the real GC and allows subscriptions to finish discovery.
        loop {
            sender
                .send_to(&game_packet(1, false), gc_address)
                .await
                .unwrap();
            if let Ok(Ok(state)) =
                tokio::time::timeout(Duration::from_millis(50), game.recv()).await
            {
                assert_eq!(state.game_state, FilteredGameState::Ready);
                break;
            }
            assert!(tasks.try_join_next().is_none(), "a network node exited");
        }
        let mut buffer = [0; 1024];
        let length = returns.recv(&mut buffer).await.unwrap();
        assert_eq!(&buffer[..4], b"RGrt");
        assert_eq!(length, 32);
        assert_eq!(buffer[5], 3);
        assert_eq!(buffer[6], 24);
        loop {
            sender
                .send_to(&game_packet(1, true), gc_address)
                .await
                .unwrap();
            let state = game.recv().await.unwrap();
            if state.penalties[PlayerNumber::Three].is_some() {
                break;
            }
        }
        let teammate = bincode::serialize(&HulkMessage::State(StateMessage {
            player_number: PlayerNumber::Two,
            ..Default::default()
        }))
        .unwrap();
        loop {
            sender.send_to(&teammate, hsl_address).await.unwrap();
            if let Ok(Ok(state)) =
                tokio::time::timeout(Duration::from_millis(50), players.recv()).await
                && state[PlayerNumber::Two].is_some()
            {
                break;
            }
        }
    })
    .await
    .expect("real network pipeline should receive UDP and send a GC return packet");
    tasks.abort_all();
    while tasks.join_next().await.is_some() {}
    context.shutdown().unwrap();
    // The message handler must release both UDP ports when the simulator closes.
    StdSocket::bind(gc_address).unwrap();
    StdSocket::bind(hsl_address).unwrap();
}

// Run through tests/hsl/roundtrip.sh: requires the real upstream runtime in another
// network namespace shared by both applications.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires upstream HSL GameController; run tools/simulate/tests/hsl/roundtrip.sh"]
async fn upstream_hsl_game_controller_roundtrip() {
    let exchange = PathBuf::from(std::env::var("HSL_EXCHANGE").expect("roundtrip runner"));
    let parameters = tempfile::tempdir().unwrap();
    std::fs::write(
        parameters.path().join("message_receiver.json5"),
        r#"{ports: {hsl_broadcast_address: {octets: [10,0,255,255]}}}"#,
    )
    .unwrap();
    let context = Arc::new(
        ContextBuilder::default()
            .with_namespace("/hsl_roundtrip")
            .with_parameter_layers([
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../etc/parameters/base"),
                parameters.path().to_owned(),
            ])
            .disable_multicast_scouting()
            .with_connect_endpoints(std::iter::empty::<&str>())
            .with_listen_endpoints(["tcp/127.0.0.1:0"])
            .build()
            .await
            .unwrap(),
    );
    let node = context.create_node("observer").build().await.unwrap();
    let game = node
        .subscriber::<FilteredGameControllerState>("filtered_game_controller_state")
        .build()
        .await
        .unwrap();
    let mut tasks = JoinSet::new();
    tasks.spawn(global_parameter_provider::run_boxed(context.clone()));
    tasks.spawn(behavior_node::run_boxed(context.clone()));
    spawn_network(&context, &mut tasks);
    tasks.spawn(whistle_filter::run_boxed(context.clone()));
    let (pulse, receiver) = tokio::sync::watch::channel(None);
    let whistle_context = context.clone();
    tasks.spawn(async move { crate::whistle::run(&whistle_context, receiver).await });
    let raw_game = node
        .subscriber::<Option<types::game_controller_state::GameControllerState>>(
            "game_controller_state",
        )
        .cache(1)
        .build()
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(60), async {
        for (stage, penalized) in [("ready", false), ("penalized", true), ("unpenalized", false)] {
            loop {
                tokio::select! {
                    state = game.recv() => {
                        let state = state.unwrap();
                        if state.game_state == FilteredGameState::Ready && state.penalties[PlayerNumber::Three].is_some() == penalized {
                            std::fs::write(exchange.join(stage), b"observed").unwrap();
                            eprintln!("Observed upstream {stage}");
                            break;
                        }
                    }
                    failure = tasks.join_next() => panic!("network node exited: {failure:?}"),
                }
            }
        }
        loop {
            let state = game.recv().await.unwrap();
            if state.game_state == FilteredGameState::Set { eprintln!("Received upstream Set; injecting whistle"); break; }
        }
        pulse.send_replace(Some(std::time::Instant::now()+crate::whistle::PULSE_DURATION));
        loop {
            let state = game.recv().await.unwrap();
            if matches!(state.game_state, FilteredGameState::Playing {..}) { break; }
        }
        assert!(raw_game.get_latest().is_some_and(|game| game.as_ref().as_ref().is_some_and(|game| game.game_state == hsl_network_messages::GameState::Set)));
        std::fs::write(exchange.join("whistle_in_set"), b"observed").unwrap();
        // Let the upstream controller check that returns remain live after removal.
        tokio::time::sleep(Duration::from_secs(1)).await;
    }).await.expect("upstream GameController roundtrip");
    tasks.abort_all();
    while tasks.join_next().await.is_some() {}
    context.shutdown().unwrap();
}

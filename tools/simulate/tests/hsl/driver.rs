//! Exercises the upstream runtime used by the HSL GameController GUI.
use anyhow::{ensure, Result};
use clap::Parser;
use game_controller_core::{
    action::VAction,
    actions::{FreeSetPlay, Penalize, StartSetPlay, Unpenalize, WaitForSetPlay},
    types::{PenaltyCall, PlayerNumber, SetPlay, Side},
};
use game_controller_runtime::{
    cli::Args, launch::make_launch_data, shutdown_runtime, start_runtime,
};
use std::{path::PathBuf, time::Duration};

#[tokio::main]
async fn main() -> Result<()> {
    let root = PathBuf::from(std::env::var("HSL_CHECKOUT")?);
    let exchange = PathBuf::from(std::env::var("HSL_EXCHANGE")?);
    let launch = make_launch_data(
        &root.join("config"),
        Args::parse_from([
            "roundtrip",
            "--competition",
            "middle_advanced",
            "--home-team",
            "24",
            "--away-team",
            "5",
            "--interface",
            "gc",
            "--no-delay",
        ]),
    )?;
    let (sender, mut receiver) = tokio::sync::watch::channel(serde_json::Value::Null);
    let runtime = start_runtime(
        &root.join("config"),
        &exchange.join("logs"),
        &launch.default_settings,
        &launch.teams,
        &launch.network_interfaces,
        Box::new(move |state| {
            sender.send_replace(serde_json::to_value(state)?);
            Ok(())
        }),
    )
    .await?;
    runtime.ui_notify.notify_one();
    let players: Vec<usize> = if std::env::var("HSL_PLAYERS").as_deref() == Ok("5") { (0..5).collect() } else { vec![2] };
    let all_good = |state: &serde_json::Value| players.iter().all(|&player| state["connectionStatus"]["home"][player] == 2);
    let result: Result<()> = async {
        runtime
            .action_sender
            .send(VAction::StartSetPlay(StartSetPlay {
                side: Some(Side::Home),
                set_play: SetPlay::KickOff,
            }))?;
        tokio::time::timeout(Duration::from_secs(20), async {
            loop {
                receiver.changed().await?;
                if all_good(&receiver.borrow()) {
                    break;
                }
            }
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        println!("GameController accepted all configured HULKs return messages (connection=Good)");
        for stage in ["ready", "penalized", "unpenalized", "whistle_in_set", "paused_clock"] {
            tokio::time::timeout(Duration::from_secs(20), async {
                while !exchange.join(stage).exists() {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            })
            .await?;
            match stage {
                "ready" => runtime.action_sender.send(VAction::Penalize(Penalize {
                    side: Side::Home,
                    player: PlayerNumber::new(3),
                    call: PenaltyCall::RequestForPickUp,
                }))?,
                "penalized" => runtime.action_sender.send(VAction::Unpenalize(Unpenalize {
                    side: Side::Home,
                    player: PlayerNumber::new(3),
                    force: true,
                }))?,
                "unpenalized" => runtime.action_sender.send(VAction::WaitForSetPlay(WaitForSetPlay))?,
                "whistle_in_set" => runtime.action_sender.send(VAction::FreeSetPlay(FreeSetPlay))?,
                _ => {}
            }
        }
        // Require loss and recovery of Good status, rather than accepting a
        // still-fresh return packet from before the pause.
        tokio::time::timeout(Duration::from_secs(10), async {
            while all_good(&receiver.borrow()) {
                receiver.changed().await?;
            }
            Ok::<_, anyhow::Error>(())
        }).await??;
        std::fs::write(exchange.join("returns_paused"), b"observed")?;
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                receiver.changed().await?;
                if all_good(&receiver.borrow()) {
                    break;
                }
            }
            Ok::<_, anyhow::Error>(())
        }).await??;
        std::fs::write(exchange.join("returns_resumed"), b"accepted")?;
        ensure!(
            all_good(&receiver.borrow()),
            "robot stopped returning status"
        );
        println!("Upstream HSL runtime roundtrip passed: Ready, penalty, removal, whistle in Set, independent match clock while robotics paused, resumed returns");
        Ok(())
    }
    .await;
    shutdown_runtime(&runtime).await;
    result
}

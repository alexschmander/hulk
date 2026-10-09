//! Simulator referee. The upstream core owns match state; this module judges physics.
mod engine;
mod physics;
#[cfg(test)]
mod tests;
pub(crate) mod transport;

use crate::{RobotId, TeamId};
pub(crate) use engine::{Ball, Engine, Robot, Snapshot};
use game_controller_core::{
    action::VAction,
    actions::*,
    types::{PenaltyCall, SetPlay, Side},
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefereeMode {
    #[default]
    External,
    Automatic,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Competition {
    Small,
    #[default]
    Middle,
    Large,
}
impl Competition {
    pub fn label(self) -> &'static str {
        match self {
            Self::Small => "Small · Advanced",
            Self::Middle => "Middle · Advanced",
            Self::Large => "Large · Advanced",
        }
    }
    fn yaml(self) -> &'static str {
        match self {
            Self::Small => include_str!("small_advanced.yaml"),
            Self::Middle => include_str!("middle_advanced.yaml"),
            Self::Large => include_str!("large_advanced.yaml"),
        }
    }
}
pub(crate) fn side(team: TeamId) -> Side {
    match team {
        TeamId::Hulks => Side::Home,
        TeamId::Opponents => Side::Away,
    }
}

pub(crate) fn team(side: Side) -> TeamId {
    match side {
        Side::Home => TeamId::Hulks,
        Side::Away => TeamId::Opponents,
    }
}

pub(crate) struct AutoRef {
    pub engine: Arc<Mutex<Engine>>,
    transport: transport::Transport,
    observer: Mutex<physics::Observer>,
}
impl AutoRef {
    pub async fn new(configuration: &crate::Configuration) -> color_eyre::Result<Self> {
        let engine = Arc::new(Mutex::new(Engine::new(
            configuration.competition,
            configuration.field_configuration.as_ref().unwrap().clone(),
        )?));
        let transport = transport::Transport::new(engine.clone()).await?;
        Ok(Self {
            engine,
            transport,
            observer: Mutex::new(physics::Observer::default()),
        })
    }
    pub fn network(&self) -> transport::Endpoints {
        self.transport.endpoints()
    }
    pub fn connect(&mut self, destination: std::net::SocketAddr) {
        self.transport.connect(destination, self.engine.clone());
    }
    pub fn poll(&self) -> Option<String> {
        self.transport.poll()
    }
    pub fn update(
        &self,
        world: &mut crate::bevy_mujoco::MujocoWorld,
        members: &[crate::team::Member],
        dt: std::time::Duration,
    ) -> Result<(), String> {
        self.observer
            .lock()
            .unwrap()
            .update(&mut self.engine.lock().unwrap(), world, members, dt)
    }
    pub fn ui(&self, ui: &mut eframe::egui::Ui, selected: Option<RobotId>) {
        let mut engine = self.engine.lock().unwrap();
        let game = engine.core.get_game(false);
        let label = format!(
            "Referee · {}–{} · {:?}",
            game.teams[Side::Home].score,
            game.teams[Side::Away].score,
            game.state
        );
        ui.menu_button(label, |ui| {
            ui.checkbox(&mut engine.enabled, "Automatic decisions");
            ui.label("Manual calls run on the next simulation step.");
            if ui.button("Stop / resume play").clicked() {
                let resume = engine.core.get_game(false).stopped;
                engine.commands.push(VAction::StopPlay(StopPlay { resume }));
            }
            for (name, side) in [("HULKs goal", Side::Home), ("Opponents goal", Side::Away)] {
                if ui.button(name).clicked() {
                    engine.commands.push(VAction::Goal(Goal { side }));
                }
            }
            if ui.button("Dropped ball").clicked() {
                engine
                    .commands
                    .push(VAction::GlobalGameStuck(GlobalGameStuck));
            }
            if ui.button("Whistle / start play").clicked() {
                engine.manual_whistle = true;
            }
            if let Some(id) = selected {
                ui.separator();
                ui.label(id.to_string());
                ui.menu_button("Award restart", |ui| {
                    for (label, kind) in [
                        ("Kick-in", SetPlay::ThrowIn),
                        ("Corner", SetPlay::CornerKick),
                        ("Goal kick", SetPlay::GoalKick),
                        ("Direct free kick", SetPlay::DirectFreeKick),
                        ("Indirect free kick", SetPlay::IndirectFreeKick),
                    ] {
                        if ui.button(label).clicked() {
                            engine.restarts.push((kind, side(id.team)));
                            ui.close();
                        }
                    }
                });
                for (label, call) in [
                    ("Pick up", PenaltyCall::RequestForPickUp),
                    ("Illegal position", PenaltyCall::IllegalPosition),
                    ("Pushing", PenaltyCall::Pushing),
                    ("Send off", PenaltyCall::SendOff),
                ] {
                    if ui.button(label).clicked() {
                        engine.commands.push(VAction::Penalize(Penalize {
                            side: side(id.team),
                            player: game_controller_core::types::PlayerNumber::new(id.number),
                            call,
                        }));
                    }
                }
                if ui.button("Release penalty now").clicked() {
                    engine.commands.push(VAction::Unpenalize(Unpenalize {
                        side: side(id.team),
                        player: game_controller_core::types::PlayerNumber::new(id.number),
                        force: true,
                    }));
                }
            }
            ui.separator();
            ui.label("Recent referee decisions");
            eframe::egui::ScrollArea::vertical()
                .max_height(180.0)
                .show(ui, |ui| {
                    for event in engine.events.iter().rev().take(30) {
                        ui.small(event);
                    }
                });
        });
    }
}

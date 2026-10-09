//! Simulator referee. The upstream core owns match state; this module judges physics.
mod calls;
mod desk;
mod engine;
mod physics;
#[cfg(test)]
mod tests;
pub(crate) mod transport;

use crate::{RobotId, TeamId};
pub(crate) use calls::short;
pub(crate) use desk::team_color;
pub(crate) use engine::{Ball, Engine, Robot, Snapshot};
use game_controller_core::types::Side;
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
    pub fn match_strip(&self, ui: &mut eframe::egui::Ui, desk: &mut bool, compact: bool) {
        desk::match_strip(ui, &mut self.engine.lock().unwrap(), desk, compact);
    }
    /// Returns a player selected from the desk.
    pub fn desk(
        &self,
        ui: &mut eframe::egui::Ui,
        robots: &[RobotId],
        selected: Option<RobotId>,
    ) -> Option<RobotId> {
        desk::desk(ui, &mut self.engine.lock().unwrap(), robots, selected)
    }
    /// Penalty state and calls for the selected player in the panel header.
    pub fn player_calls(&self, ui: &mut eframe::egui::Ui, id: RobotId) {
        let mut engine = self.engine.lock().unwrap();
        desk::penalty_status(ui, engine.core.get_game(false), id);
        desk::penalty_menu(ui, &mut engine, id, false);
        desk::release_button(ui, &mut engine, id);
    }
    pub fn shortcut(&self, stop: bool, next: bool) {
        desk::shortcut(&mut self.engine.lock().unwrap(), stop, next);
    }
    pub fn toast(&self, ui: &eframe::egui::Ui, rect: eframe::egui::Rect) {
        desk::toast(ui, rect, &self.engine.lock().unwrap());
    }
}

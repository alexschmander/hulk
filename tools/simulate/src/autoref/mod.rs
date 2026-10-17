//! Adapt upstream match state and referee decisions to simulator physics and transport.
mod calls;
mod desk;
mod engine;
mod physics;
mod stuck;
#[cfg(test)]
mod tests;
pub(crate) mod transport;

use crate::{RobotId, TeamId};
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
/// HSL division; field size is chosen separately because divisions share fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Division {
    Small,
    Middle,
    Large,
}
impl Division {
    pub const ALL: [Self; 3] = [Self::Small, Self::Middle, Self::Large];
    pub fn label(self) -> &'static str {
        match self {
            Self::Small => "Small",
            Self::Middle => "Middle",
            Self::Large => "Large",
        }
    }
}
/// Upstream competition presets. Saved Advanced names stay unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Competition {
    Small,
    #[default]
    Middle,
    Large,
    SmallFoundation,
    MiddleFoundation,
    LargeFoundation,
}
impl Competition {
    pub fn new(division: Division, advanced: bool) -> Self {
        match (division, advanced) {
            (Division::Small, true) => Self::Small,
            (Division::Middle, true) => Self::Middle,
            (Division::Large, true) => Self::Large,
            (Division::Small, false) => Self::SmallFoundation,
            (Division::Middle, false) => Self::MiddleFoundation,
            (Division::Large, false) => Self::LargeFoundation,
        }
    }
    pub fn division(self) -> Division {
        match self {
            Self::Small | Self::SmallFoundation => Division::Small,
            Self::Middle | Self::MiddleFoundation => Division::Middle,
            Self::Large | Self::LargeFoundation => Division::Large,
        }
    }
    pub fn advanced(self) -> bool {
        matches!(self, Self::Small | Self::Middle | Self::Large)
    }
    pub fn label(self) -> String {
        format!(
            "{} {}",
            self.division().label(),
            if self.advanced() {
                "Advanced"
            } else {
                "Foundation"
            }
        )
    }
    /// Players per team; the controller starts higher numbers as substitutes.
    pub fn players(self) -> u8 {
        match (self.division(), self.advanced()) {
            (Division::Small, true) => 7,
            (Division::Small, false) => 4,
            (_, true) => 5,
            (_, false) => 3,
        }
    }
    fn yaml(self) -> &'static str {
        match self {
            Self::Small => include_str!("small_advanced.yaml"),
            Self::Middle => include_str!("middle_advanced.yaml"),
            Self::Large => include_str!("large_advanced.yaml"),
            Self::SmallFoundation => include_str!("small_foundation.yaml"),
            Self::MiddleFoundation => include_str!("middle_foundation.yaml"),
            Self::LargeFoundation => include_str!("large_foundation.yaml"),
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
            configuration.opponent.clone(),
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
    pub fn connect(&mut self, destination: std::net::SocketAddr, clock: ros_z::time::Clock) {
        self.transport
            .connect(destination, self.engine.clone(), clock);
    }
    pub(crate) fn command(&self, name: &str, arguments: &serde_json::Value) -> Result<(), String> {
        use calls::Call;
        let side = || match arguments["team"].as_str() {
            Some("hulks") => Ok(TeamId::Hulks),
            Some("opponents") => Ok(TeamId::Opponents),
            _ => Err("Expected team hulks or opponents".to_owned()),
        };
        let robot = || -> Result<RobotId, String> {
            let number = arguments["number"]
                .as_u64()
                .filter(|n| (1..=5).contains(n))
                .ok_or("Expected robot number 1..5")?;
            Ok(RobotId {
                team: side()?,
                number: number as u8,
            })
        };
        let call = match name {
            "stop_play" => Call::StopPlay,
            "resume_play" => Call::ResumePlay,
            "kickoff" => Call::Kickoff,
            "set" => Call::Set,
            "whistle" => Call::Whistle,
            "ball_free" => Call::BallFree,
            "dropped_ball" => Call::DroppedBall,
            "finish_half" => Call::FinishHalf,
            "second_half" => Call::SecondHalf,
            "referee_timeout" => Call::RefereeTimeout,
            "add_minute" => Call::AddMinute,
            "goal" => Call::Goal(side()?),
            "timeout" => Call::Timeout(side()?),
            "release" => Call::Release(robot()?),
            "restart" => Call::Restart(
                side()?,
                calls::RESTARTS
                    .into_iter()
                    .find(|v| {
                        Some(format!("{v:?}")) == arguments["set_play"].as_str().map(str::to_owned)
                    })
                    .ok_or("Unknown set play")?,
            ),
            "penalize" => Call::Penalize(
                robot()?,
                calls::PENALTIES
                    .into_iter()
                    .find(|v| {
                        Some(format!("{v:?}")) == arguments["penalty"].as_str().map(str::to_owned)
                    })
                    .ok_or("Unknown penalty")?,
            ),
            _ => return Err(format!("Unknown referee call {name}")),
        };
        self.engine.lock().unwrap().call(call)
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

/// Runtime catalog for the same calls accepted by the referee command adapter.
pub(crate) fn command_schemas() -> serde_json::Value {
    use serde_json::json;
    let mut schemas = serde_json::Map::new();
    for name in [
        "stop_play",
        "resume_play",
        "kickoff",
        "set",
        "whistle",
        "ball_free",
        "dropped_ball",
        "finish_half",
        "second_half",
        "referee_timeout",
        "add_minute",
        "goal",
        "timeout",
        "release",
        "restart",
        "penalize",
    ] {
        let mut fields = serde_json::Map::new();
        if matches!(
            name,
            "goal" | "timeout" | "release" | "restart" | "penalize"
        ) {
            fields.insert(
                "team".into(),
                json!({"type":"string","enum":["hulks","opponents"]}),
            );
        }
        if matches!(name, "release" | "penalize") {
            fields.insert(
                "number".into(),
                json!({"type":"integer","minimum":1,"maximum":5}),
            );
        }
        if name == "restart" {
            fields.insert("set_play".into(),json!({"type":"string","enum":calls::RESTARTS.iter().map(|v|format!("{v:?}")).collect::<Vec<_>>()}));
        }
        if name == "penalize" {
            fields.insert("penalty".into(),json!({"type":"string","enum":calls::PENALTIES.iter().map(|v|format!("{v:?}")).collect::<Vec<_>>()}));
        }
        schemas.insert(name.into(),json!({"type":"object","required":fields.keys().collect::<Vec<_>>(),"properties":fields,"additionalProperties":false}));
    }
    schemas.into()
}

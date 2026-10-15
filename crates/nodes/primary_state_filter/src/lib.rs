use std::{boxed::Box, future::Future, pin::Pin};
use std::{collections::HashSet, sync::Arc, time::Duration};

mod manual_stand_up;

use color_eyre::Result;
use serde::{Deserialize, Serialize};

use hsl_network_messages::{GamePhase, PlayerNumber};
use ros_z::{prelude::*, qos::QosDurability, time::Time};
use tracing::info;
use types::{
    buttons::{ButtonPressType, Buttons},
    controller_input::ControllerInput,
    fall_detection::FallDetection,
    filtered_game_controller_state::FilteredGameControllerState,
    filtered_game_state::FilteredGameState,
    primary_state::PrimaryState,
};

#[derive(Debug, Clone, Serialize, Deserialize, Message)]
#[serde(deny_unknown_fields)]
pub struct Parameters {
    pub injected_primary_state: Option<PrimaryState>,
    pub recorded_primary_states: HashSet<PrimaryState>,
}

pub fn run_boxed(ctx: Arc<Context>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
    Box::pin(run(ctx))
}

async fn run(ctx: Arc<Context>) -> Result<()> {
    let node = ctx.create_node("primary_state_filter").build().await?;

    let parameters = node.bind_parameter_as::<Parameters>("primary_state_filter")?;
    let player_number_cache = node
        .subscriber::<PlayerNumber>("player_number")
        .qos(QosProfile {
            durability: QosDurability::TransientLocal,
            ..Default::default()
        })
        .cache(1)
        .build()
        .await?;

    let filtered_game_controller_state_sub = node
        .subscriber::<FilteredGameControllerState>("filtered_game_controller_state")
        .build()
        .await?;
    let buttons_sub = node
        .subscriber::<Buttons<Option<ButtonPressType>>>("buttons")
        .build()
        .await?;
    let is_safe_pose_cache = node
        .subscriber::<bool>("is_safe_pose")
        .cache(1)
        .build()
        .await?;
    let controller_input_sub = node
        .subscriber::<ControllerInput>("inputs/controller_input")
        .build()
        .await?;
    let fall_detection_cache = node
        .subscriber::<FallDetection>("fall_detection/status")
        .cache(1)
        .build()
        .await?;

    let primary_state_pub = node
        .publisher::<PrimaryState>("primary_state")
        .qos(QosProfile {
            durability: QosDurability::TransientLocal,
            ..Default::default()
        })
        .build()
        .await?;

    let mut primary_state_filter = PrimaryStateFilter::default();
    primary_state_pub
        .publish(&primary_state_filter.primary_state)
        .await?;

    let mut watchdog = tokio::time::interval(Duration::from_millis(25));
    watchdog.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut previous_injected_primary_state = None;

    loop {
        let previous_state = primary_state_filter.primary_state;
        let mut watchdog_tick = false;
        let parameters_snapshot = parameters.snapshot();
        let parameters = parameters_snapshot.typed();
        tokio::select! {
            _ = watchdog.tick() => {
                watchdog_tick = true;
            }
            received_filtered_game_controller_state = filtered_game_controller_state_sub.recv() => {
                let Some(player_number) = player_number_cache.get_latest() else {continue};

                let filtered_game_controller_state = received_filtered_game_controller_state?;

                let previous_primary_state = primary_state_filter.primary_state;
                primary_state_filter.update_with_filtered_game_controller_state(
                    &filtered_game_controller_state,
                    *player_number,
                );
                if primary_state_filter.primary_state != previous_primary_state {
                    info!(
                        target: "primary_state_filter::input",
                        previous_primary_state = ?previous_primary_state,
                        primary_state = ?primary_state_filter.primary_state,
                        game_state = ?filtered_game_controller_state.game_state,
                        player_number = ?*player_number,
                        penalty = ?filtered_game_controller_state.penalties[*player_number],
                        "primary state changed from game controller"
                    );
                }
            }
            received_buttons = buttons_sub.recv() => {
                let buttons = received_buttons?;
                let is_safe_pose = is_safe_pose_cache.get_latest().is_some_and(|safe| *safe);

                let previous_primary_state = primary_state_filter.primary_state;
                let has_button_event = buttons.f1.is_some() || buttons.stand.is_some() || buttons.walking.is_some();
                if has_button_event {
                    info!(
                        target: "primary_state_filter::input",
                        ?buttons,
                        is_safe_pose,
                        previous_primary_state = ?previous_primary_state,
                        "received button input"
                    );
                }
                primary_state_filter.update_with_buttons(&buttons, is_safe_pose, node.clock().now());
                if primary_state_filter.primary_state != previous_primary_state {
                    info!(
                        target: "primary_state_filter::input",
                        ?buttons,
                        is_safe_pose,
                        previous_primary_state = ?previous_primary_state,
                        primary_state = ?primary_state_filter.primary_state,
                        "primary state changed from buttons"
                    );
                }
            }
            received_controller_input = controller_input_sub.recv_with_metadata() => {
                let controller_input = received_controller_input?;
                primary_state_filter.update_with_controller_input(
                    &controller_input,
                    controller_input.source_time,
                    node.clock().now(),
                    fall_detection_cache.get_latest().as_deref().copied(),
                );
            }
        }

        let injection_changed =
            parameters.injected_primary_state != previous_injected_primary_state;
        previous_injected_primary_state = parameters.injected_primary_state;
        if let Some(injected_primary_state) = parameters.injected_primary_state
            && (injection_changed || injected_primary_state != PrimaryState::Damping)
        {
            let previous_primary_state = primary_state_filter.primary_state;
            primary_state_filter
                .update_with_injected_primary_state(injected_primary_state, node.clock().now());
            if primary_state_filter.primary_state != previous_primary_state {
                info!(
                    target: "primary_state_filter::input",
                    previous_primary_state = ?previous_primary_state,
                    primary_state = ?primary_state_filter.primary_state,
                    "primary state changed from injected parameter"
                );
            }
        }

        primary_state_filter.update_manual_stand_up(
            fall_detection_cache.get_latest().as_deref().copied(),
            node.clock().now(),
        );
        if watchdog_tick
            && primary_state_filter.primary_state == previous_state
            && primary_state_filter.primary_state != PrimaryState::ManualStandUp
        {
            continue;
        }
        primary_state_pub
            .publish(&primary_state_filter.primary_state)
            .await?;
    }
}

#[derive(Default)]
struct PrimaryStateFilter {
    pub primary_state: PrimaryState,
    controller_time: Option<Time>,
    south_pressed: Option<bool>,
    last_stop: Option<Time>,
    manual_started: Option<Time>,
}

impl PrimaryStateFilter {
    fn update_with_filtered_game_controller_state(
        &mut self,
        filtered_game_controller_state: &FilteredGameControllerState,
        player_number: PlayerNumber,
    ) {
        let is_penalized = filtered_game_controller_state.penalties[player_number].is_some();
        let filtered_game_state = filtered_game_controller_state.game_state;
        let is_penalty_shootout = matches!(
            filtered_game_controller_state.game_phase,
            GamePhase::PenaltyShootout { .. }
        );

        self.primary_state = match (self.primary_state, filtered_game_state) {
            (PrimaryState::Damping | PrimaryState::ManualStandUp, _) => self.primary_state,
            (PrimaryState::Initial, FilteredGameState::Set) if is_penalty_shootout => {
                PrimaryState::Set
            }
            (PrimaryState::Initial, game_state) => {
                game_state_to_primary_state(game_state, is_penalized)
            }
            (PrimaryState::Ready, FilteredGameState::Set) if !is_penalized => PrimaryState::Set,
            (PrimaryState::Set, FilteredGameState::Playing { .. }) if !is_penalized => {
                PrimaryState::Playing
            }
            (PrimaryState::Playing, FilteredGameState::Ready) if !is_penalized => {
                PrimaryState::Ready
            }
            (state, FilteredGameState::Finished)
                if !matches!(state, PrimaryState::Damping | PrimaryState::Prepare) =>
            {
                PrimaryState::Finished
            }
            (state, _)
                if is_penalized
                    && !matches!(state, PrimaryState::Damping | PrimaryState::Prepare) =>
            {
                PrimaryState::Penalized
            }
            (state, FilteredGameState::Stop)
                if !matches!(state, PrimaryState::Damping | PrimaryState::Prepare) =>
            {
                PrimaryState::Stop
            }
            (PrimaryState::Stop, game_state) => {
                game_state_to_primary_state(game_state, is_penalized)
            }
            (PrimaryState::Penalized, game_state) if !is_penalized => {
                game_state_to_primary_state(game_state, false)
            }
            _ => self.primary_state,
        }
    }

    fn update_with_buttons(
        &mut self,
        buttons: &Buttons<Option<ButtonPressType>>,
        is_safe_pose: bool,
        now: Time,
    ) {
        if matches!(buttons.f1, Some(ButtonPressType::Short)) {
            self.request_damping(now);
            return;
        }
        self.primary_state = match (self.primary_state, buttons) {
            (PrimaryState::ManualStandUp, _) => PrimaryState::ManualStandUp,
            (
                _,
                Buttons {
                    stand: Some(ButtonPressType::Short),
                    ..
                },
            ) => PrimaryState::Prepare,
            (
                PrimaryState::Prepare,
                Buttons {
                    stand: Some(ButtonPressType::Long),
                    ..
                },
            ) if is_safe_pose => PrimaryState::Initial,
            (
                PrimaryState::Initial,
                Buttons {
                    walking: Some(ButtonPressType::Long),
                    ..
                },
            ) => PrimaryState::Playing,
            _ => self.primary_state,
        }
    }

    fn update_with_injected_primary_state(
        &mut self,
        injected_primary_state: PrimaryState,
        now: Time,
    ) {
        if injected_primary_state == PrimaryState::Damping {
            self.request_damping(now);
            return;
        }
        if injected_primary_state == PrimaryState::ManualStandUp {
            return;
        }
        if matches!(
            self.primary_state,
            PrimaryState::Damping | PrimaryState::Prepare | PrimaryState::ManualStandUp
        ) {
            return;
        }
        self.primary_state = injected_primary_state
    }
}

fn game_state_to_primary_state(game_state: FilteredGameState, is_penalized: bool) -> PrimaryState {
    if is_penalized {
        if game_state == FilteredGameState::Finished {
            return PrimaryState::Finished;
        }
        PrimaryState::Penalized
    } else {
        match game_state {
            FilteredGameState::Initial => PrimaryState::Initial,
            FilteredGameState::Ready => PrimaryState::Ready,
            FilteredGameState::Set => PrimaryState::Set,
            FilteredGameState::Playing { .. } => PrimaryState::Playing,
            FilteredGameState::Finished => PrimaryState::Finished,
            FilteredGameState::Stop => PrimaryState::Stop,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_with_buttons_enters_playing_from_initial_with_safe_long_stand_press() {
        let mut primary_state_filter = PrimaryStateFilter {
            primary_state: PrimaryState::Initial,
            ..Default::default()
        };
        let buttons = Buttons {
            f1: None,
            stand: None,
            walking: Some(ButtonPressType::Long),
        };

        primary_state_filter.update_with_buttons(&buttons, true, Time::zero());

        assert_eq!(primary_state_filter.primary_state, PrimaryState::Playing);
    }
}

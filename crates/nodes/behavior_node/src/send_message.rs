use std::{collections::HashMap, net::SocketAddr, time::Duration};

use hsl_network_messages::{
    GameControllerReturnMessage, GamePhase, Half, HulkMessage, StateMessage,
};
use ros_z::time::Time;
use types::{
    fall_detection::Posture, filtered_game_controller_state::FilteredGameControllerState,
    messages::OutgoingMessage, parameters::HslNetworkParameters, primary_state::PrimaryState,
};

use crate::node::Blackboard;

impl Blackboard {
    pub fn game_controller_return_message(
        &mut self,
        game_controller_address: Option<&SocketAddr>,
    ) -> Option<OutgoingMessage> {
        let now = self.world_state.now;

        if !self.is_return_message_cooldown_elapsed(now, &self.parameters.network) {
            return None;
        }
        let address = game_controller_address?;

        let ground_to_field = self.world_state.robot.ground_to_field.unwrap_or_default();

        let ball_position = self
            .world_state
            .ball
            .map(|ball| hsl_network_messages::BallPosition {
                age: now
                    .to_wallclock()
                    .duration_since(ball.last_seen_ball)
                    .unwrap(),
                position: ball.ball_in_ground,
            });

        self.last_sent_game_controller_return_message_time = Some(now);

        Some(OutgoingMessage::GameController(
            *address,
            GameControllerReturnMessage {
                player_number: self.player_number,
                fallen: self
                    .world_state
                    .fall_detection
                    .is_none_or(|fall_detection| {
                        !fall_detection.is_valid(now) || fall_detection.posture != Posture::Upright
                    }),
                pose: ground_to_field.as_pose(),
                ball: ball_position,
            },
        ))
    }

    fn is_return_message_cooldown_elapsed(
        &self,
        now: Time,
        hsl_network_parameters: &HslNetworkParameters,
    ) -> bool {
        is_cooldown_elapsed(
            now,
            self.last_sent_game_controller_return_message_time,
            hsl_network_parameters.game_controller_return_message_interval,
        )
    }

    pub fn try_sending_state_message(&mut self) -> Option<OutgoingMessage> {
        let primary_state = self.world_state.robot.primary_state;
        if !primary_state.sends_team_messages() {
            return None;
        }

        let now = self.world_state.now;
        let game_controller_state = self.world_state.filtered_game_controller_state.as_ref()?;
        let send_interval =
            state_message_send_interval(game_controller_state, &self.parameters.network)?;
        if !is_state_message_due(
            primary_state,
            now,
            self.last_sent_hsl_message_time,
            self.last_sent_hsl_message_state,
            send_interval,
            &self.parameters.network,
        ) {
            return None;
        }
        if let Some(ground_to_field) = self.world_state.robot.ground_to_field {
            let pose = ground_to_field.as_pose();

            let ball_position =
                self.world_state
                    .ball
                    .map(|ball| hsl_network_messages::BallPosition {
                        age: now
                            .to_wallclock()
                            .duration_since(ball.last_seen_ball)
                            .unwrap(),
                        position: ball.ball_in_field,
                    });

            let message = HulkMessage::State(StateMessage {
                player_number: self.player_number,
                pose,
                head_yaw: self.head_yaw,
                pose_age: self
                    .localization_time
                    .map(|localization_time| now.duration_since(localization_time)),
                ball_position,
            });

            self.last_sent_hsl_message_time = Some(now);
            self.last_sent_hsl_message_state = Some(primary_state);
            Some(OutgoingMessage::Hsl(message))
        } else {
            None
        }
    }
}

/// Whether a state message is due, given the time and primary state of the last one.
///
/// While playing, messages are paced by the remaining budget. Before a kick-off they are sparser,
/// because teammates only need to be told apart from opponents. One is sent right away on entering
/// ready, so that teammates are identified early, and one on entering set, when everybody has
/// reached their kick-off position.
fn is_state_message_due(
    primary_state: PrimaryState,
    now: Time,
    last_time: Option<Time>,
    last_state: Option<PrimaryState>,
    send_interval: Duration,
    parameters: &HslNetworkParameters,
) -> bool {
    match primary_state {
        PrimaryState::Playing => is_cooldown_elapsed(now, last_time, send_interval),
        PrimaryState::Ready | PrimaryState::Set if last_state != Some(primary_state) => true,
        _ => is_cooldown_elapsed(
            now,
            last_time,
            send_interval.max(parameters.ready_and_set_message_interval),
        ),
    }
}

pub(crate) fn recent_game_controller_address(
    contacts: &HashMap<SocketAddr, Time>,
    now: Time,
    timeout: Duration,
) -> Option<SocketAddr> {
    contacts
        .iter()
        .max_by_key(|(_, time)| *time)
        .filter(|(_, time)| now.duration_since(**time) < timeout)
        .map(|(address, _)| *address)
}

pub(crate) fn state_message_send_interval(
    game_controller_state: &FilteredGameControllerState,
    parameters: &HslNetworkParameters,
) -> Option<Duration> {
    let second_half_duration = match (game_controller_state.game_phase, game_controller_state.half)
    {
        (GamePhase::Normal, Half::First) => parameters.half_duration,
        (GamePhase::Extratime, Half::First) => parameters.extra_half_duration,
        _ => Duration::ZERO,
    };
    let remaining_time_in_game =
        game_controller_state.remaining_time_in_half + second_half_duration;
    let available_messages = game_controller_state
        .remaining_number_of_messages
        .saturating_sub(parameters.remaining_amount_of_messages_to_stop_sending);
    let active_player_count = game_controller_state
        .penalties
        .iter()
        .filter(|(_, penalty)| penalty.is_none())
        .count() as u32;

    if available_messages == 0 || active_player_count == 0 || remaining_time_in_game.is_zero() {
        return None;
    }

    Some(remaining_time_in_game * active_player_count / u32::from(available_messages))
}

fn is_cooldown_elapsed(now: Time, last: Option<Time>, cooldown: Duration) -> bool {
    match last {
        None => true,
        Some(last_time) => now.duration_since(last_time) > cooldown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters() -> HslNetworkParameters {
        HslNetworkParameters {
            ready_and_set_message_interval: Duration::from_secs(2),
            ..Default::default()
        }
    }

    fn at(seconds: u64) -> Time {
        Time::zero() + Duration::from_secs(seconds)
    }

    #[test]
    fn playing_is_paced_by_the_budget() {
        let last_sent = (Some(at(0)), Some(PrimaryState::Playing));
        let interval = Duration::from_secs(3);
        assert!(!is_state_message_due(
            PrimaryState::Playing,
            at(2),
            last_sent.0,
            last_sent.1,
            interval,
            &parameters()
        ));
        assert!(is_state_message_due(
            PrimaryState::Playing,
            at(4),
            last_sent.0,
            last_sent.1,
            interval,
            &parameters()
        ));
    }

    #[test]
    fn ready_sends_at_once_and_then_sparsely() {
        let interval = Duration::from_millis(500);
        let after_playing = (Some(at(10)), Some(PrimaryState::Playing));
        assert!(is_state_message_due(
            PrimaryState::Ready,
            at(10),
            after_playing.0,
            after_playing.1,
            interval,
            &parameters()
        ));
        let after_ready = (Some(at(10)), Some(PrimaryState::Ready));
        assert!(!is_state_message_due(
            PrimaryState::Ready,
            at(11),
            after_ready.0,
            after_ready.1,
            interval,
            &parameters()
        ));
        assert!(is_state_message_due(
            PrimaryState::Ready,
            at(13),
            after_ready.0,
            after_ready.1,
            interval,
            &parameters()
        ));
    }

    #[test]
    fn entering_set_sends_once() {
        let interval = Duration::from_secs(3);
        let after_ready = (Some(at(10)), Some(PrimaryState::Ready));
        assert!(is_state_message_due(
            PrimaryState::Set,
            at(11),
            after_ready.0,
            after_ready.1,
            interval,
            &parameters()
        ));
        let after_set = (Some(at(11)), Some(PrimaryState::Set));
        assert!(!is_state_message_due(
            PrimaryState::Set,
            at(12),
            after_set.0,
            after_set.1,
            interval,
            &parameters()
        ));
    }
}

use std::time::Duration;

use ros_z::time::Time;
use tracing::info;
use types::{
    controller_input::{Button, ControllerInput},
    fall_detection::{FallDetection, Posture},
    primary_state::PrimaryState,
};

use super::PrimaryStateFilter;

const CONTROLLER_MAXIMUM_AGE: Duration = Duration::from_millis(250);

impl PrimaryStateFilter {
    fn controller_is_fresh(&self, now: Time) -> bool {
        self.controller_time
            .is_some_and(|time| time <= now && now.duration_since(time) <= CONTROLLER_MAXIMUM_AGE)
    }

    pub(super) fn request_damping(&mut self, now: Time) {
        self.primary_state = PrimaryState::Damping;
        self.manual_started = None;
        self.south_pressed = None;
        self.last_stop = Some(self.last_stop.map_or(now, |last| last.max(now)));
    }

    pub(super) fn update_with_controller_input(
        &mut self,
        input: &ControllerInput,
        source_time: Time,
        now: Time,
        fall_detection: Option<FallDetection>,
    ) {
        // A stop wins even if A/Cross is pressed in the same sample.
        if input.is_pressed(Button::West) {
            self.request_damping(now);
            return;
        }
        let was_released = self.controller_is_fresh(now) && self.south_pressed == Some(false);
        if !input.connected
            || source_time > now
            || now.duration_since(source_time) > CONTROLLER_MAXIMUM_AGE
            || self.controller_time.is_some_and(|last| source_time <= last)
            || self.last_stop.is_some_and(|stop| source_time <= stop)
        {
            self.controller_time = None;
            self.south_pressed = None;
            return;
        }
        self.controller_time = Some(source_time);
        let pressed = input.is_pressed(Button::South);
        self.south_pressed = Some(pressed);
        if was_released && pressed && self.primary_state != PrimaryState::ManualStandUp {
            if fall_detection.is_some_and(|detection| {
                detection.is_valid(now)
                    && detection.posture
                        == Posture::Fallen {
                            ready_for_standup: true,
                        }
            }) {
                self.primary_state = PrimaryState::ManualStandUp;
                self.manual_started = Some(now);
                info!("manual stand-up activated from detector readiness");
            } else {
                info!("manual stand-up rejected: requires fresh Fallen with ready_for_standup");
            }
        }
    }

    pub(super) fn update_manual_stand_up(
        &mut self,
        fall_detection: Option<FallDetection>,
        now: Time,
    ) {
        if !self.controller_is_fresh(now) {
            self.south_pressed = None;
        }
        if self.primary_state != PrimaryState::ManualStandUp {
            return;
        }
        let Some(started) = self.manual_started else {
            self.request_damping(now);
            return;
        };
        let detection = fall_detection.filter(|detection| detection.is_valid(now));
        if !self.controller_is_fresh(now) || now < started {
            self.request_damping(now);
            return;
        }
        match detection.map(|detection| detection.posture) {
            Some(
                Posture::Fallen {
                    ready_for_standup: true,
                }
                | Posture::StandingUp,
            ) => {}
            Some(Posture::Upright)
                if detection.is_some_and(|detection| detection.time > started) =>
            {
                self.primary_state = PrimaryState::Initial;
                self.manual_started = None;
                info!("manual stand-up complete; returning to Initial");
            }
            _ => self.request_damping(now),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::{
        buttons::{ButtonPressType, Buttons},
        controller_input::ControllerButton,
    };

    const READY: Posture = Posture::Fallen {
        ready_for_standup: true,
    };

    fn time(milliseconds: i64) -> Time {
        Time::from_nanos(milliseconds * 1_000_000)
    }

    fn report(posture: Posture, milliseconds: i64) -> FallDetection {
        FallDetection {
            time: time(milliseconds),
            valid_until: time(milliseconds + 100),
            posture,
        }
    }

    fn input(south: bool, west: bool) -> ControllerInput {
        ControllerInput {
            connected: true,
            buttons: [(Button::South, south), (Button::West, west)]
                .into_iter()
                .map(|(name, pressed)| ControllerButton {
                    name,
                    pressed,
                    value: if pressed { 1.0 } else { 0.0 },
                })
                .collect(),
            ..Default::default()
        }
    }

    fn press(filter: &mut PrimaryStateFilter, detection: Option<FallDetection>) {
        filter.update_with_controller_input(&input(false, false), time(990), time(990), detection);
        filter.update_with_controller_input(&input(true, false), time(1000), time(1000), detection);
    }

    #[test]
    fn only_valid_fallen_ready_accepts_a_press_without_state_history() {
        for initial in [
            PrimaryState::Damping,
            PrimaryState::Prepare,
            PrimaryState::Initial,
        ] {
            for (detection, accepted) in [
                (None, false),
                (Some(report(Posture::Upright, 1000)), false),
                (Some(report(Posture::Falling, 1000)), false),
                (
                    Some(report(
                        Posture::Fallen {
                            ready_for_standup: false,
                        },
                        1000,
                    )),
                    false,
                ),
                (Some(report(Posture::StandingUp, 1000)), false),
                (Some(report(READY, 899)), false),
                (Some(report(READY, 1001)), false),
                (Some(report(READY, 1000)), true),
            ] {
                let mut filter = PrimaryStateFilter {
                    primary_state: initial,
                    ..Default::default()
                };
                press(&mut filter, detection);
                assert_eq!(
                    filter.primary_state,
                    if accepted {
                        PrimaryState::ManualStandUp
                    } else {
                        initial
                    }
                );
            }
        }
    }

    #[test]
    fn held_and_rejected_presses_are_not_queued() {
        let mut filter = PrimaryStateFilter::default();
        filter.update_with_controller_input(
            &input(true, false),
            time(980),
            time(980),
            Some(report(READY, 980)),
        );
        assert_eq!(filter.primary_state, PrimaryState::Damping);
        press(
            &mut filter,
            Some(report(
                Posture::Fallen {
                    ready_for_standup: false,
                },
                1000,
            )),
        );
        filter.update_with_controller_input(
            &input(true, false),
            time(1010),
            time(1010),
            Some(report(READY, 1010)),
        );
        assert_eq!(filter.primary_state, PrimaryState::Damping);
        filter.update_with_controller_input(
            &input(false, false),
            time(1020),
            time(1020),
            Some(report(READY, 1020)),
        );
        filter.update_with_controller_input(
            &input(true, false),
            time(1030),
            time(1030),
            Some(report(READY, 1030)),
        );
        assert_eq!(filter.primary_state, PrimaryState::ManualStandUp);
    }

    #[test]
    fn invalid_controller_samples_cannot_start_an_attempt() {
        for (connected, source_time) in [
            (false, 1000),
            (true, 1001),
            (true, 749),
            (true, 900),
            (true, 899),
        ] {
            let mut filter = PrimaryStateFilter::default();
            filter.update_with_controller_input(&input(false, false), time(900), time(900), None);
            let mut controller = input(true, false);
            controller.connected = connected;
            filter.update_with_controller_input(
                &controller,
                time(source_time),
                time(1000),
                Some(report(READY, 1000)),
            );
            assert_eq!(filter.primary_state, PrimaryState::Damping);
        }
    }

    #[test]
    fn stops_win_and_require_a_new_release_and_press_afterwards() {
        for stop_kind in 0..3 {
            let mut filter = PrimaryStateFilter::default();
            press(&mut filter, Some(report(READY, 1000)));
            match stop_kind {
                0 => filter.update_with_controller_input(
                    &input(true, true),
                    time(1010),
                    time(1010),
                    Some(report(READY, 1010)),
                ),
                1 => filter.update_with_buttons(
                    &Buttons {
                        f1: Some(ButtonPressType::Short),
                        stand: None,
                        walking: None,
                    },
                    false,
                    time(1010),
                ),
                _ => filter.update_with_injected_primary_state(PrimaryState::Damping, time(1010)),
            }
            assert_eq!(filter.primary_state, PrimaryState::Damping);
            for (pressed, source) in [(false, 1005), (true, 1006), (true, 1020)] {
                filter.update_with_controller_input(
                    &input(pressed, false),
                    time(source),
                    time(1020),
                    Some(report(READY, 1020)),
                );
                assert_eq!(filter.primary_state, PrimaryState::Damping);
            }
            filter.update_with_controller_input(&input(false, false), time(1030), time(1030), None);
            filter.update_with_controller_input(
                &input(true, false),
                time(1040),
                time(1040),
                Some(report(READY, 1040)),
            );
            assert_eq!(filter.primary_state, PrimaryState::ManualStandUp);
        }
    }

    #[test]
    fn detector_controls_attempt_completion_and_failure() {
        for (detection, now, expected) in [
            (Some(report(READY, 1010)), 1010, PrimaryState::ManualStandUp),
            (
                Some(report(Posture::StandingUp, 1010)),
                1010,
                PrimaryState::ManualStandUp,
            ),
            (
                Some(report(Posture::Upright, 1010)),
                1010,
                PrimaryState::Initial,
            ),
            (
                Some(report(Posture::Upright, 1000)),
                1010,
                PrimaryState::Damping,
            ),
            (
                Some(report(Posture::Falling, 1010)),
                1010,
                PrimaryState::Damping,
            ),
            (
                Some(report(
                    Posture::Fallen {
                        ready_for_standup: false,
                    },
                    1010,
                )),
                1010,
                PrimaryState::Damping,
            ),
            (None, 1010, PrimaryState::Damping),
            (
                Some(report(Posture::StandingUp, 900)),
                1010,
                PrimaryState::Damping,
            ),
            (
                Some(report(Posture::StandingUp, 1020)),
                1010,
                PrimaryState::Damping,
            ),
            (
                Some(report(Posture::StandingUp, 1300)),
                1300,
                PrimaryState::Damping,
            ),
            (Some(report(READY, 990)), 990, PrimaryState::Damping),
        ] {
            let mut filter = PrimaryStateFilter::default();
            press(&mut filter, Some(report(READY, 1000)));
            filter.update_manual_stand_up(detection, time(now));
            assert_eq!(filter.primary_state, expected);
            if expected == PrimaryState::Damping {
                filter.update_with_controller_input(
                    &input(true, false),
                    time(1310),
                    time(1310),
                    Some(report(READY, 1310)),
                );
                filter.update_manual_stand_up(Some(report(READY, 1310)), time(1310));
                assert_eq!(filter.primary_state, PrimaryState::Damping);
            }
        }
    }

    #[test]
    fn injected_states_cannot_start_or_override_an_attempt() {
        let mut filter = PrimaryStateFilter {
            primary_state: PrimaryState::Initial,
            ..Default::default()
        };
        filter.update_with_injected_primary_state(PrimaryState::ManualStandUp, time(900));
        assert_eq!(filter.primary_state, PrimaryState::Initial);
        press(&mut filter, Some(report(READY, 1000)));
        for state in [
            PrimaryState::Prepare,
            PrimaryState::Initial,
            PrimaryState::Playing,
        ] {
            filter.update_with_injected_primary_state(state, time(1010));
            assert_eq!(filter.primary_state, PrimaryState::ManualStandUp);
        }
    }
}

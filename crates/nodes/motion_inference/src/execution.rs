use std::time::Duration;

use ros_z::time::Time;

use crate::inference::{ExecutionState, InferenceRequest};

/// A stopped generation cannot be reopened by an out-of-order service request.
#[derive(Default)]
pub(crate) struct ExecutionGate {
    state: Option<ExecutionState>,
}

impl ExecutionGate {
    /// Returns whether the controller must discard its previous execution.
    pub fn observe(&mut self, next: ExecutionState) -> bool {
        match self.state {
            Some(old) if next.generation < old.generation => false,
            Some(old) if next.generation == old.generation => {
                if old.active && !next.active {
                    self.state = Some(next);
                    true
                } else {
                    false
                }
            }
            _ => {
                self.state = Some(next);
                true
            }
        }
    }

    pub fn accepts<C>(&self, request: &InferenceRequest<C>, now: Time) -> bool {
        request.is_current(now)
            && self
                .state
                .is_some_and(|state| state.active && state.generation == request.generation)
    }
}

/// Scheduling uses actual starts, so a late cycle never creates catch-up samples.
pub(crate) struct PolicyCadence {
    period: Duration,
    last_start: Option<Time>,
    last_sensor: Option<Time>,
}

impl PolicyCadence {
    pub fn new(period: Duration) -> Self {
        Self {
            period,
            last_start: None,
            last_sensor: None,
        }
    }

    pub fn next_start(&self, now: Time) -> Time {
        self.last_start
            .map_or(now, |last| (last + self.period).max(now))
    }

    pub fn has_new_sensor(&self, time: Time) -> bool {
        self.last_sensor.is_none_or(|last| time > last)
    }

    pub fn started(&mut self, now: Time, sensor: Time) {
        self.last_start = Some(now);
        self.last_sensor = Some(sensor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn time(ms: i64) -> Time {
        Time::from_nanos(ms * 1_000_000)
    }

    fn request(generation: u64) -> InferenceRequest<()> {
        InferenceRequest {
            generation,
            requested_at: time(0),
            valid_until: time(40),
            command: (),
        }
    }

    #[test]
    fn delayed_requests_and_completions_cannot_reopen_stopped_execution() {
        let mut gate = ExecutionGate::default();
        assert!(gate.observe(ExecutionState {
            generation: 1,
            active: true
        }));
        assert!(gate.accepts(&request(1), time(20)));
        assert!(gate.observe(ExecutionState {
            generation: 2,
            active: false
        }));
        assert!(!gate.observe(ExecutionState {
            generation: 1,
            active: true
        }));
        assert!(!gate.accepts(&request(1), time(25)));
        assert!(!gate.observe(ExecutionState {
            generation: 2,
            active: true
        }));
        assert!(!gate.accepts(&request(2), time(25)));
        assert!(gate.observe(ExecutionState {
            generation: 3,
            active: true
        }));
        assert!(!gate.observe(ExecutionState {
            generation: 2,
            active: false
        }));
        assert!(gate.accepts(&request(3), time(30)));
        assert!(!gate.accepts(&request(3), time(40)));
        assert!(!gate.accepts(&request(3), time(45)));
    }

    #[test]
    fn late_cycles_do_not_compress_history_or_reuse_sensor_frames() {
        let mut cadence = PolicyCadence::new(Duration::from_millis(20));
        cadence.started(time(20), time(18));
        assert_eq!(cadence.next_start(time(55)), time(55));
        cadence.started(time(55), time(54));
        assert_eq!(cadence.next_start(time(56)), time(75));
        assert!(!cadence.has_new_sensor(time(54)));
        assert!(!cadence.has_new_sensor(time(52)));
        assert!(cadence.has_new_sensor(time(56)));
    }
}

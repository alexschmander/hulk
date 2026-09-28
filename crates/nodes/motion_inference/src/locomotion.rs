use std::{collections::VecDeque, f32::consts::TAU};

use coordinate_systems::{Ground, Robot};
use kinematics::{
    forward,
    joints::{Joints, leg::LegJoints},
};
use linear_algebra::{Point2, Point3, Vector2, point};
use ros_z::time::Time;
use types::joint_limits::JointLimits;
use types::motor_command::MotorCommand;

use crate::{
    config::{HISTORY_FRAME_SIZE, HISTORY_LENGTH, LEGS, Parameters, Policy},
    inference::{InferenceCommand, WalkCommand, position_targets},
    observation::SensorFrame,
};

pub mod kick;
pub mod walk;

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize, ros_z::Message)]
pub struct KickRequest {
    pub ball_position: Point2<Ground>,
    pub ball_velocity: Vector2<Ground>,
    pub direction: f32,
    pub target_speed: f32,
    pub strong: bool,
    pub quick: bool,
}

impl KickRequest {
    /// Change only the orientation of the reference frame, as in B-Human.
    pub fn rotated(mut self, angle: f32) -> Self {
        let rotation = nalgebra::UnitComplex::new(angle);
        self.ball_position = Point2::wrap(rotation * self.ball_position.inner);
        self.ball_velocity = Vector2::wrap(rotation * self.ball_velocity.inner);
        self.direction =
            (self.direction + angle + std::f32::consts::PI).rem_euclid(TAU) - std::f32::consts::PI;
        self
    }

    pub fn is_finite(self) -> bool {
        self.ball_position
            .inner
            .iter()
            .chain(self.ball_velocity.inner.iter())
            .copied()
            .chain([self.direction, self.target_speed])
            .all(f32::is_finite)
            && self.target_speed >= 0.0
    }
}

pub struct Locomotion {
    pub(crate) parameters: std::sync::Arc<Parameters>,
    history: VecDeque<[f32; HISTORY_FRAME_SIZE]>,
    previous_target: Joints<f32>,
    pub phase: f32,
    pub frequency_offset: f32,
    previous_ball: Option<(Point2<Ground>, Point2<Ground>)>,
    last_fast_motion: Option<Time>,
}

impl Locomotion {
    pub fn new(
        sensor: &SensorFrame,
        parameters: std::sync::Arc<Parameters>,
        joints: &JointLimits,
    ) -> Self {
        Self {
            history: VecDeque::from(vec![
                walk::history_frame(
                    sensor,
                    &sensor.last_commanded_position,
                    true,
                    &parameters,
                    joints
                );
                HISTORY_LENGTH
            ]),
            previous_target: sensor.last_commanded_position,
            phase: 0.0,
            frequency_offset: parameters.locomotion.initial_frequency_offset,
            parameters,
            previous_ball: None,
            last_fast_motion: None,
        }
    }

    pub(crate) fn update_parameters(&mut self, parameters: std::sync::Arc<Parameters>) {
        let old_offset = Policy::Walk.offset(&self.parameters);
        let new_offset = Policy::Walk.offset(&parameters);
        for (index, joint) in LEGS.into_iter().enumerate() {
            if old_offset[joint] == new_offset[joint] {
                continue;
            }
            for frame in &mut self.history {
                frame[6 + index] = (frame[6 + index] + old_offset[joint]) - new_offset[joint];
                frame[18 + index] = (frame[18 + index] + old_offset[joint]) - new_offset[joint];
            }
        }
        self.parameters = parameters;
    }

    pub fn advance(&mut self, seconds: f32, standing: bool) {
        if standing {
            self.phase = 0.0;
        } else {
            let limit = self.parameters.locomotion.frequency_offset_limit;
            self.phase = (self.phase
                + seconds
                    * (self.parameters.locomotion.base_frequency
                        + self.frequency_offset.clamp(-limit, limit)))
            .rem_euclid(1.0);
        }
    }

    pub fn standing(&mut self, now: Time, request: InferenceCommand) -> bool {
        let InferenceCommand::Walk(WalkCommand {
            velocity,
            angular_velocity,
        }) = request
        else {
            self.last_fast_motion = Some(now);
            return false;
        };
        let command = [velocity.x(), velocity.y(), angular_velocity];
        let stopped = command == [0.0; 3];
        let slow = command
            .into_iter()
            .zip(self.parameters.locomotion.slow_velocity_thresholds)
            .all(|(value, threshold)| value.abs() < threshold);
        if !slow {
            self.last_fast_motion = Some(now);
        }
        stopped
            && self.last_fast_motion.is_none_or(|last| {
                now.duration_since(last) >= self.parameters.locomotion.stand_delay
            })
    }

    pub fn record_walk_sample(&mut self, sensor: &SensorFrame, joints: &JointLimits) {
        self.record_history(sensor, joints);
        self.previous_ball = None;
    }

    pub fn record_kick_sample(
        &mut self,
        sensor: &SensorFrame,
        request: KickRequest,
        soft: bool,
        joints: &JointLimits,
    ) -> (Point2<Ground>, Point2<Ground>) {
        self.record_history(sensor, joints);
        let ball = Point2::wrap(
            request
                .ball_position
                .inner
                .coords
                .cap_magnitude(self.parameters.kick.ball_position_limit)
                .into(),
        );
        let shifted = kick::shifted_ball(sensor, request, &self.parameters.kick);
        let previous = self
            .previous_ball
            .filter(|(previous, _)| {
                (ball - *previous).norm() <= self.parameters.kick.ball_jump_distance
            })
            .unwrap_or((ball, shifted));
        self.previous_ball = Some((ball, shifted));
        if soft {
            (ball, previous.0)
        } else {
            (shifted, previous.1)
        }
    }

    fn record_history(&mut self, sensor: &SensorFrame, joints: &JointLimits) {
        self.history.pop_front();
        self.history.push_back(walk::history_frame(
            sensor,
            &self.previous_target,
            false,
            &self.parameters,
            joints,
        ));
    }

    fn phase_encoding(&self, standing: bool) -> [f32; 2] {
        if standing {
            [0.0, 0.0]
        } else {
            [(TAU * self.phase).cos(), (TAU * self.phase).sin()]
        }
    }

    pub fn decode(
        &mut self,
        policy: Policy,
        actions: &[f32],
        sensor: &SensorFrame,
    ) -> Joints<MotorCommand> {
        let offset = policy.offset(&self.parameters);
        let (kp, kd) = policy.gains(&self.parameters);
        let action_limit = policy.action_limit(&self.parameters);
        let mut position = sensor.last_commanded_position;
        for (index, joint) in LEGS.into_iter().enumerate() {
            position[joint] = actions[index].clamp(-action_limit, action_limit) + offset[joint];
            // RLWalkPhase::calcJoints retains these targets before downstream composition/clipping.
            self.previous_target[joint] = position[joint];
        }
        let frequency_limit = if policy == Policy::Walk {
            self.parameters.locomotion.frequency_offset_limit
        } else {
            action_limit
        };
        self.frequency_offset = actions[LEGS.len()].clamp(-frequency_limit, frequency_limit);
        position_targets(position, kp, kd)
    }
}

pub fn leg(angles: &LegJoints<f32>, left: bool) -> (Point3<Robot>, Point3<Robot>) {
    if left {
        let tibia_to_robot = forward::left_pelvis_to_robot(angles)
            * forward::left_hip_to_left_pelvis(angles)
            * forward::left_thigh_to_left_hip(angles)
            * forward::left_tibia_to_left_thigh(angles);
        let foot_to_robot = tibia_to_robot
            * forward::left_ankle_to_left_tibia(angles)
            * forward::left_foot_to_left_ankle(angles);
        (
            foot_to_robot * point![0.026, 0.0, -0.038],
            tibia_to_robot.translation(),
        )
    } else {
        let tibia_to_robot = forward::right_pelvis_to_robot(angles)
            * forward::right_hip_to_right_pelvis(angles)
            * forward::right_thigh_to_right_hip(angles)
            * forward::right_tibia_to_right_thigh(angles);
        let foot_to_robot = tibia_to_robot
            * forward::right_ankle_to_right_tibia(angles)
            * forward::right_foot_to_right_ankle(angles);
        (
            foot_to_robot * point![0.026, 0.0, -0.038],
            tibia_to_robot.translation(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        inference::KickCommand,
        test_support::{limits, parameters, sensor, time},
    };
    use linear_algebra::vector;

    fn kick_request() -> KickRequest {
        KickRequest {
            ball_position: point![0.2, 0.2],
            ball_velocity: vector![1.0, 0.0],
            direction: 0.0,
            target_speed: 1.0,
            strong: true,
            quick: true,
        }
    }

    #[test]
    fn walk_clips_frequency_feedback_independently_of_joint_actions() {
        let parameters = parameters();
        let sensor = sensor();
        let mut state = Locomotion::new(&sensor, parameters.clone(), &limits());
        for (raw, expected) in [(1.8, 0.5), (-1.8, -0.5), (0.2, 0.2)] {
            let mut actions = [0.0; 13];
            actions[0] = 10.0;
            actions[12] = raw;
            let target = state.decode(Policy::Walk, &actions, &sensor);
            assert_eq!(
                target[LEGS[0]].position,
                Policy::Walk.offset(&parameters)[LEGS[0]] + Policy::Walk.action_limit(&parameters)
            );
            let tensor =
                walk::Observation::new(&state, &Joints::fill(0.0), false, Vector2::zeros(), 0.0)
                    .to_tensor();
            assert_eq!(tensor[337], expected);
            state.decode(Policy::Kick, &actions, &sensor);
            assert_eq!(state.frequency_offset, raw);
            state.phase = 0.0;
            state.advance(0.02, false);
            assert!((state.phase - 0.02 * (1.5 + expected)).abs() < 1e-6);
        }
    }

    #[test]
    fn standing_requires_slow_command_dwell_after_walking_or_kicking() {
        let mut state = Locomotion::new(&sensor(), parameters(), &limits());
        let stand = InferenceCommand::Walk(WalkCommand::stand());
        let moving = InferenceCommand::Walk(WalkCommand {
            velocity: vector![0.2, 0.0],
            angular_velocity: 0.0,
        });
        assert!(state.standing(time(0), stand));
        assert!(!state.standing(time(20), moving));
        assert!(!state.standing(time(1000), stand));
        assert!(state.standing(time(1020), stand));
        let tiny = InferenceCommand::Walk(WalkCommand {
            velocity: vector![0.001, 0.0],
            angular_velocity: 0.0,
        });
        assert!(!state.standing(time(2000), tiny));
        assert!(state.standing(time(2020), stand));
        let kick = InferenceCommand::Kick(KickCommand {
            reference_time: time(2040),
            soft: false,
            request: kick_request(),
        });
        assert!(!state.standing(time(2040), kick));
        assert!(!state.standing(time(3020), stand));
        assert!(state.standing(time(3040), stand));
    }

    #[test]
    fn kick_variants_keep_separate_shifted_and_unshifted_history() {
        let mut parameters = (*parameters()).clone();
        // Force the shift to be observable regardless of the fixture's stance.
        parameters.kick.shift_foot_distance = [0.0, 0.001];
        let sensor = sensor();
        let limits = limits();
        let mut state = Locomotion::new(&sensor, std::sync::Arc::new(parameters), &limits);
        let first = kick_request();
        let (shifted, previous) = state.record_kick_sample(&sensor, first, false, &limits);
        assert_eq!(shifted, previous);
        assert!((shifted - first.ball_position).norm() > 0.001);
        let mut second = first;
        second.ball_position = point![0.21, 0.2];
        let (raw, old_raw) = state.record_kick_sample(&sensor, second, true, &limits);
        assert_eq!(raw, second.ball_position);
        assert_eq!(old_raw, first.ball_position);
        let expected_previous = kick::shifted_ball(&sensor, second, &state.parameters.kick);
        let (_, old_shifted) = state.record_kick_sample(&sensor, first, false, &limits);
        assert_eq!(old_shifted, expected_previous);
        second.ball_position = point![-0.5, -0.5];
        let (current, previous) = state.record_kick_sample(&sensor, second, true, &limits);
        assert_eq!(current, previous);
        state.record_walk_sample(&sensor, &limits);
        let (current, previous) = state.record_kick_sample(&sensor, first, false, &limits);
        assert_eq!(current, previous);
    }

    #[test]
    fn kick_rotation_transforms_position_velocity_and_direction_together() {
        let transformed = kick_request().rotated(-std::f32::consts::FRAC_PI_2);
        assert!((transformed.ball_position - point![0.2, -0.2]).norm() < 1e-6);
        assert!((transformed.ball_velocity - vector![0.0, -1.0]).norm() < 1e-6);
        assert!((transformed.direction + std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        let wrapped = KickRequest {
            direction: 3.0,
            ..kick_request()
        }
        .rotated(0.4);
        assert!((wrapped.direction - (3.4 - TAU)).abs() < 1e-6);
    }
}

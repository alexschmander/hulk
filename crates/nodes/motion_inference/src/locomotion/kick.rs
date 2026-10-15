use std::f32::consts::{PI, TAU};
use types::joint_limits::JointLimits;

use ::kinematics::{forward, joints::Joints};
use coordinate_systems::{Ground, Robot};
use linear_algebra::{Orientation2, Point2, Vector3};
use nalgebra::{UnitComplex, UnitQuaternion};

use super::{KickRequest, Locomotion, leg};
use crate::{
    config::{KickParameters, LEGS, Policy, clip_measurement},
    observation::SensorFrame,
};

/// Strong and soft kick share this tensor contract, but not their flag/speed rules.
pub struct Observation {
    pub gravity: Vector3<Robot>,
    pub angular_velocity: Vector3<Robot>,
    pub ball_position: Point2<Ground>,
    pub direction: Orientation2<Ground>,
    pub phase: [f32; 2],
    pub position_offsets: Joints<f32>,
    pub joint_velocity: Joints<f32>,
    pub previous_target_offsets: Joints<f32>,
    pub frequency_offset: f32,
    pub strong: bool,
    pub quick: bool,
    pub ball_velocity: linear_algebra::Vector2<Ground>,
    pub target_speed: f32,
    pub previous_ball_position: Point2<Ground>,
    pub joint_velocity_scale: f32,
    pub ball_position_scale: f32,
    pub ball_velocity_scale: f32,
    pub target_speed_scale: f32,
}

impl Observation {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        state: &Locomotion,
        sensor: &SensorFrame,
        joint_velocity: &Joints<f32>,
        standing: bool,
        request: KickRequest,
        soft: bool,
        (ball_position, previous_ball_position): (Point2<Ground>, Point2<Ground>),
        joints: &JointLimits,
    ) -> Self {
        let strong = request.strong && !soft;
        let parameters = &state.parameters;
        let [minimum_speed, maximum_speed] = if soft {
            parameters.kick.soft_speed_limits
        } else {
            parameters.kick.speed_limits
        };
        let offset = if soft { Policy::SoftKick } else { Policy::Kick }.offset(parameters);
        Self {
            joint_velocity_scale: parameters.observation.joint_velocity_scale,
            ball_position_scale: parameters.kick.ball_position_scale,
            ball_velocity_scale: parameters.kick.ball_velocity_scale,
            target_speed_scale: parameters.kick.target_speed_scale,
            gravity: Vector3::wrap(sensor.gravity().into()),
            angular_velocity: sensor.gyro,
            ball_position,
            direction: request.direction,
            phase: state.phase_encoding(standing),
            position_offsets: clip_measurement(sensor.position, joints.position) - offset,
            joint_velocity: *joint_velocity,
            previous_target_offsets: state.previous_target - offset,
            frequency_offset: state.frequency_offset,
            strong,
            quick: request.quick && !soft,
            ball_velocity: request
                .ball_velocity
                .cap_magnitude(parameters.kick.ball_velocity_limit),
            target_speed: if strong {
                maximum_speed
            } else {
                request.target_speed.clamp(minimum_speed, maximum_speed)
            },
            previous_ball_position,
        }
    }

    pub fn to_tensor(&self) -> [f32; 59] {
        let direction = self.direction.angle();
        let mut input = [0.0; 59];
        input[..3].copy_from_slice(self.gravity.inner.as_slice());
        input[3..6].copy_from_slice(self.angular_velocity.inner.as_slice());
        input[6..9].copy_from_slice(&[
            self.ball_position.x() * self.ball_position_scale,
            self.ball_position.y() * self.ball_position_scale,
            normalize_angle(direction) / PI,
        ]);
        input[9..11].copy_from_slice(&self.phase);
        input[11..23].copy_from_slice(&LEGS.map(|joint| self.position_offsets[joint]));
        input[23..35].copy_from_slice(
            &LEGS.map(|joint| self.joint_velocity[joint] * self.joint_velocity_scale),
        );
        input[35..47].copy_from_slice(&LEGS.map(|joint| self.previous_target_offsets[joint]));
        input[47] = self.frequency_offset;
        input[48..51].copy_from_slice(&[f32::from(self.strong), 0.0, f32::from(self.quick)]);
        input[51..54].copy_from_slice(&[
            self.ball_velocity.x() * self.ball_velocity_scale,
            self.ball_velocity.y() * self.ball_velocity_scale,
            0.0,
        ]);
        input[54..56].copy_from_slice(&[direction.sin(), direction.cos()]);
        input[56] = self.target_speed * self.target_speed_scale;
        input[57..59].copy_from_slice(&[
            self.previous_ball_position.x() * self.ball_position_scale,
            self.previous_ball_position.y() * self.ball_position_scale,
        ]);
        input
    }
}

fn normalize_angle(angle: f32) -> f32 {
    // The trained feature uses [-PI, PI); atan2-based orientations can return +PI.
    (angle + PI).rem_euclid(TAU) - PI
}

fn ramp(value: f32, minimum: f32, maximum: f32) -> f32 {
    ((value - minimum) / (maximum - minimum)).clamp(0.0, 1.0)
}

pub(super) fn shifted_ball(
    sensor: &SensorFrame,
    kick: KickRequest,
    parameters: &KickParameters,
    soft: bool,
) -> Point2<Ground> {
    let direction = kick.direction.angle();
    let to_kick_direction = UnitComplex::new(-direction);
    let (roll, pitch, _) = sensor.rotation().euler_angles();
    let level = UnitQuaternion::from_euler_angles(roll, pitch, 0.0);
    let sole_midpoint = (forward::left_sole_to_robot(&sensor.position.left_leg)
        .translation()
        .inner
        .coords
        + forward::right_sole_to_robot(&sensor.position.right_leg)
            .translation()
            .inner
            .coords)
        * 0.5;
    let origin = (level * sole_midpoint).xy();
    let left = to_kick_direction
        * ((level * leg(&sensor.position.left_leg, true).0.inner.coords).xy() - origin);
    let right = to_kick_direction
        * ((level * leg(&sensor.position.right_leg, false).0.inner.coords).xy() - origin);
    let mut ball = to_kick_direction * kick.ball_position.inner.coords;
    let angle_weight = 1.0
        - ramp(
            normalize_angle(direction).abs(),
            parameters.shift_direction_degrees[0].to_radians(),
            parameters.shift_direction_degrees[1].to_radians(),
        );
    let foot_weight = ramp(
        (left.y - right.y).abs(),
        parameters.shift_foot_distance[0],
        parameters.shift_foot_distance[1],
    );
    let [minimum, maximum] = parameters.shift_ball_distance;
    let shift = parameters.shift_distance
        * (ramp(ball.y - left.y, minimum, maximum)
            - (1.0 - ramp(ball.y - right.y, -maximum, -minimum)));
    ball.y += angle_weight * foot_weight * shift;
    // A learned strike can brake before reaching the physical ball. Extending
    // its target along the requested shot gives it room to follow through.
    // The minimum-speed soft touch retains its original short stroke.
    let extension = if soft {
        parameters.contact.soft_forward_offset
            * ramp(
                kick.target_speed,
                parameters.contact.soft_speed_range[0],
                parameters.contact.soft_speed_range[1],
            )
    } else {
        parameters.contact.forward_offset
    };
    // An incoming ball already closes the gap. Extending that strike makes the
    // policy wait too long and can turn a clean interception into a glancing hit.
    let incoming_speed = -(to_kick_direction * kick.ball_velocity.inner).x;
    ball.x +=
        extension * (1.0 - ramp(incoming_speed, 0.0, parameters.contact.incoming_speed_limit));
    Point2::wrap(
        (UnitComplex::new(direction) * ball)
            .cap_magnitude(parameters.ball_position_limit)
            .into(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Parameters;

    #[test]
    fn contact_extension_rotates_with_aim_and_preserves_position_history() {
        let mut parameters: Parameters = serde_json::from_value(
            json5::from_str::<serde_json::Value>(include_str!(
                "../../../../../etc/parameters/base/motion_inference.json5"
            ))
            .unwrap(),
        )
        .unwrap();
        parameters.kick.shift_distance = 0.0;
        parameters.kick.contact.forward_offset = 0.12;
        parameters.kick.contact.soft_forward_offset = 0.12;
        parameters.kick.contact.soft_speed_range = [0.1, 0.3];
        let position = Policy::Walk.offset(&parameters);
        let sensor = SensorFrame {
            timestamp: ros_z::time::Time::zero(),
            position,
            velocity: Joints::fill(0.0),
            orientation: nalgebra::Quaternion::identity(),
            gyro: Vector3::zeros(),
            last_commanded_position: position,
        };
        let joints = JointLimits {
            position: Joints::fill([-3.0, 3.0]),
            maximum_torque: Joints::fill(50.0),
        };
        let parameters = std::sync::Arc::new(parameters);
        for angle in [0.0, PI / 4.0, -PI / 2.0] {
            for (soft, speed, offset) in [
                (false, 3.4, 0.12),
                (true, 0.1, 0.0),
                (true, 0.2, 0.06),
                (true, 0.3, 0.12),
            ] {
                let mut state = Locomotion::new(&sensor, parameters.clone(), &joints);
                let mut request = KickRequest {
                    ball_position: linear_algebra::point![0.4, 0.05],
                    ball_velocity: linear_algebra::vector![0.2, 0.0],
                    direction: Orientation2::new(angle),
                    target_speed: speed,
                    strong: false,
                    quick: false,
                };
                let expected = request.ball_position
                    + linear_algebra::vector![angle.cos() * offset, angle.sin() * offset];
                let (current, previous) = state.record_kick_sample(&sensor, request, &joints, soft);
                assert!((current - expected).norm() < 1e-6);
                assert_eq!(previous, current);
                request.ball_position += linear_algebra::vector![0.01, 0.0];
                let positions = state.record_kick_sample(&sensor, request, &joints, soft);
                assert!((positions.1 - expected).norm() < 1e-6);
                let observation = Observation::new(
                    &state,
                    &sensor,
                    &sensor.velocity,
                    false,
                    request,
                    soft,
                    positions,
                    &joints,
                );
                assert_eq!(observation.ball_velocity, request.ball_velocity);
                assert_eq!(observation.direction, request.direction);
                request.ball_velocity = linear_algebra::vector![-angle.cos(), -angle.sin()] * 0.3;
                assert!(
                    (shifted_ball(&sensor, request, &parameters.kick, soft)
                        - request.ball_position)
                        .norm()
                        < 1e-6
                );
            }
        }
    }

    #[test]
    fn foot_shift_uses_the_sole_midpoint_instead_of_the_torso_origin() {
        let parameters: Parameters = serde_json::from_value(
            json5::from_str::<serde_json::Value>(include_str!(
                "../../../../../etc/parameters/base/motion_inference.json5"
            ))
            .unwrap(),
        )
        .unwrap();
        let position = Joints::fill(0.0);
        let mut sensor = SensorFrame {
            timestamp: ros_z::time::Time::zero(),
            position,
            velocity: Joints::fill(0.0),
            orientation: nalgebra::Quaternion::identity(),
            gyro: Vector3::zeros(),
            last_commanded_position: position,
        };
        let request = KickRequest {
            ball_position: linear_algebra::point![0.5, 0.24],
            ball_velocity: linear_algebra::Vector2::zeros(),
            direction: Orientation2::new(0.0),
            target_speed: 1.0,
            strong: false,
            quick: false,
        };
        let centered = shifted_ball(&sensor, request, &parameters.kick, false);
        // Move both soles sideways relative to the torso while preserving
        // their orientation and separation in the Ground frame.
        sensor.position.left_leg.hip_roll = 0.25;
        sensor.position.right_leg.hip_roll = 0.25;
        sensor.position.left_leg.ankle_down = -0.25;
        sensor.position.right_leg.ankle_down = -0.25;
        let translated = shifted_ball(&sensor, request, &parameters.kick, false);
        assert!(
            (centered - translated).norm() < 1e-5,
            "moving the sole midpoint relative to the torso changed the foot bias: {centered:?} vs {translated:?}"
        );
    }
}

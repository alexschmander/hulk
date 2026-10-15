use coordinate_systems::Odometry;
use linear_algebra::Pose2;
use motion_inference::locomotion::KickRequest;
use std::time::Duration;

pub(super) fn compensate(
    mut request: KickRequest,
    age: Duration,
    previous: Pose2<Odometry>,
    current: Pose2<Odometry>,
) -> KickRequest {
    let previous_to_current = types::odometry::previous_to_current(previous, current);
    request.ball_position =
        previous_to_current * (request.ball_position + request.ball_velocity * age.as_secs_f32());
    request.ball_velocity = previous_to_current * request.ball_velocity;
    request.direction = linear_algebra::Orientation2::new(
        previous_to_current.orientation().angle() + request.direction.angle(),
    );
    request
}

#[cfg(test)]
mod tests {
    use super::*;
    use linear_algebra::{Orientation2, point, vector};

    #[test]
    fn moving_ball_and_robot_are_advanced_to_the_policy_frame() {
        let request = KickRequest {
            ball_position: point![0.5, 0.1],
            ball_velocity: vector![-1.0, 0.5],
            direction: Orientation2::new(0.0),
            target_speed: 2.0,
            strong: false,
            quick: true,
        };
        let request = compensate(
            request,
            Duration::from_millis(40),
            Pose2::new(point![0.0, 0.0], 0.0),
            Pose2::new(point![0.04, 0.0], std::f32::consts::FRAC_PI_2),
        );
        assert!((request.ball_position.x() - 0.12).abs() < 1e-6);
        assert!((request.ball_position.y() + 0.42).abs() < 1e-6);
        assert!((request.ball_velocity.x() - 0.5).abs() < 1e-6);
        assert!((request.ball_velocity.y() - 1.0).abs() < 1e-6);
        assert!((request.direction.angle() + std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    }
}

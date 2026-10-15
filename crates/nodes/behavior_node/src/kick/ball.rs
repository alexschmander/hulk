use coordinate_systems::Ground;
use linear_algebra::{Isometry2, Vector2};
use ros_z::time::Time;
use types::{ball_position::BallPosition, parameters::KickingParameters};

pub(crate) fn in_current_frame(
    mut ball: BallPosition<Ground>,
    tracked: Option<BallPosition<Ground>>,
    observation_to_current: Isometry2<Ground, Ground>,
    now: Time,
    parameters: &KickingParameters,
) -> Option<BallPosition<Ground>> {
    let age = ball.age_at(now)?;
    if age > parameters.ball_observation_timeout {
        return None;
    }
    ball.position = observation_to_current * ball.position;
    // Keep percept position authoritative. The caller supplies our local filter,
    // and a different tracked ball must not lend this percept its velocity.
    ball.velocity = tracked
        .filter(|tracked| {
            (tracked.position - (ball.position + tracked.velocity * age.as_secs_f32())).norm()
                <= parameters.ball_velocity_matching_distance
        })
        .map_or(Vector2::zeros(), |tracked| tracked.velocity);
    ball.position += ball.velocity * age.as_secs_f32();
    Some(ball)
}

#[cfg(test)]
mod tests {
    use super::*;
    use linear_algebra::{point, vector};
    use std::time::Duration;

    #[test]
    fn observation_age_and_ego_motion_are_compensated_together() {
        let ball = BallPosition {
            position: point![0.8, -0.1],
            velocity: Vector2::zeros(),
            last_seen: Time::zero(),
        };
        let tracked = BallPosition {
            position: point![0.66, -0.1],
            velocity: vector![-1.0, 0.0],
            last_seen: Time::zero(),
        };
        let updated = in_current_frame(
            ball,
            Some(tracked),
            Isometry2::from(vector![-0.08, 0.0]),
            Time::zero() + Duration::from_millis(60),
            &KickingParameters::default(),
        )
        .unwrap();
        assert!((updated.position.x() - 0.66).abs() < 1e-6);
        assert_eq!(updated.velocity.x(), -1.0);
    }

    #[test]
    fn a_different_ball_cannot_supply_the_selected_balls_velocity() {
        let ball = BallPosition {
            position: point![0.5, 0.0],
            velocity: Vector2::zeros(),
            last_seen: Time::zero(),
        };
        let tracked = BallPosition {
            position: point![2.0, 1.0],
            velocity: vector![-1.0, 0.0],
            last_seen: Time::zero(),
        };
        let updated = in_current_frame(
            ball,
            Some(tracked),
            Isometry2::identity(),
            Time::zero() + Duration::from_millis(60),
            &KickingParameters::default(),
        )
        .unwrap();
        assert_eq!(updated.position, ball.position);
        assert_eq!(updated.velocity, Vector2::zeros());
    }

    #[test]
    fn an_unrefreshed_visual_topic_cannot_keep_a_stale_kick_target_alive() {
        let ball = BallPosition {
            position: point![0.5, 0.0],
            velocity: Vector2::zeros(),
            last_seen: Time::zero(),
        };
        assert!(
            in_current_frame(
                ball,
                None,
                Isometry2::identity(),
                Time::zero() + Duration::from_millis(101),
                &KickingParameters::default()
            )
            .is_none()
        );
    }
}

//! Motion models in the continuous odometry frame.
//!
//! Robot ego-motion does not appear here: tracks live in the odometry frame, so only the
//! object's own motion and the drift of the odometry frame itself enter the prediction.

use nalgebra::{Matrix2, Matrix4, SMatrix, Vector2};
use serde::{Deserialize, Serialize};

pub trait MotionModel<const N: usize> {
    /// Returns the transition matrix and process noise for a step of `delta_time` seconds.
    fn transition(&self, delta_time: f32) -> (SMatrix<f32, N, N>, SMatrix<f32, N, N>);
}

/// Physically stationary object with position-only state `[x, y]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stationary;

impl MotionModel<2> for Stationary {
    fn transition(&self, _delta_time: f32) -> (Matrix2<f32>, Matrix2<f32>) {
        (Matrix2::identity(), Matrix2::zeros())
    }
}

/// Nearly constant velocity with continuous white acceleration noise and state `[x, y, vx, vy]`.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, ros_z::Message)]
pub struct ConstantVelocity {
    /// Power spectral density of the acceleration noise per axis in m²/s³.
    pub acceleration_noise_density: f32,
}

impl MotionModel<4> for ConstantVelocity {
    fn transition(&self, delta_time: f32) -> (Matrix4<f32>, Matrix4<f32>) {
        let dt = delta_time.max(0.0);
        let transition = Matrix4::new(
            1.0, 0.0, dt, 0.0, //
            0.0, 1.0, 0.0, dt, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        );
        let q = self.acceleration_noise_density;
        let position = q * dt.powi(3) / 3.0;
        let cross = q * dt.powi(2) / 2.0;
        let velocity = q * dt;
        let process_noise = Matrix4::new(
            position, 0.0, cross, 0.0, //
            0.0, position, 0.0, cross, //
            cross, 0.0, velocity, 0.0, //
            0.0, cross, 0.0, velocity,
        );
        (transition, process_noise)
    }
}

/// Velocity that reverts to zero, with state `[x, y, vx, vy]`.
///
/// Each velocity axis is an Ornstein-Uhlenbeck process with stationary standard deviation
/// `velocity_standard_deviation` and correlation time `velocity_time_constant`. Within the time
/// constant it behaves like [`ConstantVelocity`] with acceleration noise density `2σ²/τ`, so
/// maneuvers are followed quickly. Over longer gaps the predicted velocity decays and the position
/// uncertainty grows linearly instead of cubically, which suits robots and humans that start, stop
/// and turn while out of view.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, ros_z::Message)]
pub struct DampedVelocity {
    /// Stationary velocity standard deviation per axis in m/s.
    pub velocity_standard_deviation: f32,
    /// Correlation time of the velocity in seconds.
    pub velocity_time_constant: f32,
}

impl MotionModel<4> for DampedVelocity {
    fn transition(&self, delta_time: f32) -> (Matrix4<f32>, Matrix4<f32>) {
        let dt = f64::from(delta_time.max(0.0));
        let tau = f64::from(self.velocity_time_constant.max(1e-3));
        let variance = f64::from(self.velocity_standard_deviation).powi(2);
        let decay = (-dt / tau).exp();
        let gain = (tau * (1.0 - decay)) as f32;
        #[rustfmt::skip]
        let transition = Matrix4::new(
            1.0, 0.0, gain, 0.0,
            0.0, 1.0, 0.0, gain,
            0.0, 0.0, decay as f32, 0.0,
            0.0, 0.0, 0.0, decay as f32,
        );
        // Exact discretization of the integrated Ornstein-Uhlenbeck process.
        let x = dt / tau;
        let position =
            (variance * tau * tau * (2.0 * x - 3.0 + 4.0 * decay - decay * decay)).max(0.0) as f32;
        let cross = (variance * tau * (1.0 - decay).powi(2)) as f32;
        let velocity = (variance * (1.0 - decay * decay)) as f32;
        let process_noise = Matrix4::new(
            position, 0.0, cross, 0.0, //
            0.0, position, 0.0, cross, //
            cross, 0.0, velocity, 0.0, //
            0.0, cross, 0.0, velocity,
        );
        (transition, process_noise)
    }
}

/// Error growth of the odometry frame relative to the physical world.
///
/// Objects that are fixed in the world appear to move in the odometry frame as odometry drifts.
/// Translation drift grows with the distance walked; heading drift grows with the angle turned and
/// with time, and moves objects perpendicular to their lever arm from the robot.
///
/// Every track receives this noise independently. That ignores the correlation of the common
/// frame error between tracks, which is a documented approximation.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, ros_z::Message)]
pub struct OdometryDrift {
    /// Position variance per meter walked, in m²/m.
    pub translation_variance_per_meter: f32,
    /// Heading variance per radian turned, in rad²/rad.
    pub heading_variance_per_radian: f32,
    /// Heading variance per second, in rad²/s.
    pub heading_variance_per_second: f32,
}

impl OdometryDrift {
    /// Position noise added to an object at `relative_position` from the robot after the robot
    /// moved by `translation` meters and turned by `rotation` radians during `delta_time` seconds.
    pub fn position_noise(
        &self,
        relative_position: Vector2<f32>,
        translation: f32,
        rotation: f32,
        delta_time: f32,
    ) -> Matrix2<f32> {
        let translation_variance = self.translation_variance_per_meter * translation.abs();
        let heading_variance = self.heading_variance_per_radian * rotation.abs()
            + self.heading_variance_per_second * delta_time.max(0.0);
        let perpendicular = Vector2::new(-relative_position.y, relative_position.x);
        Matrix2::identity() * translation_variance
            + perpendicular * perpendicular.transpose() * heading_variance
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{Vector2, Vector4};

    use super::*;
    use crate::gaussian::{Gaussian, predict};

    #[test]
    fn constant_velocity_moves_mean_and_grows_uncertainty() {
        let model = ConstantVelocity {
            acceleration_noise_density: 0.5,
        };
        let mut gaussian = Gaussian::<4> {
            mean: Vector4::new(0.0, 0.0, 1.0, -0.5),
            covariance: Matrix4::identity() * 0.01,
        };
        let (transition, noise) = model.transition(2.0);
        predict(&mut gaussian, &transition, &noise);

        assert_relative_eq!(gaussian.mean, Vector4::new(2.0, -1.0, 1.0, -0.5));
        assert_relative_eq!(
            gaussian.covariance[(0, 0)],
            0.01 + 4.0 * 0.01 + 0.5 * 8.0 / 3.0,
            epsilon = 1e-5
        );
    }

    #[test]
    fn zero_step_is_identity() {
        let model = ConstantVelocity {
            acceleration_noise_density: 1.0,
        };
        let (transition, noise) = model.transition(0.0);
        assert_eq!(transition, Matrix4::identity());
        assert_eq!(noise, Matrix4::zeros());
    }

    #[test]
    fn damped_velocity_matches_constant_velocity_for_short_steps() {
        let damped = DampedVelocity {
            velocity_standard_deviation: 0.4,
            velocity_time_constant: 1.0,
        };
        let constant = ConstantVelocity {
            acceleration_noise_density: 2.0 * 0.4 * 0.4 / 1.0,
        };
        let (damped_transition, damped_noise) = damped.transition(0.01);
        let (constant_transition, constant_noise) = constant.transition(0.01);

        assert_relative_eq!(damped_transition, constant_transition, epsilon = 0.011);
        assert_relative_eq!(damped_noise, constant_noise, max_relative = 0.02);
    }

    #[test]
    fn damped_velocity_stops_and_bounds_velocity_uncertainty() {
        let model = DampedVelocity {
            velocity_standard_deviation: 0.4,
            velocity_time_constant: 1.0,
        };
        let mut gaussian = Gaussian::<4> {
            mean: Vector4::new(0.0, 0.0, 1.0, 0.0),
            covariance: Matrix4::identity() * 1e-4,
        };
        for _ in 0..300 {
            let (transition, noise) = model.transition(1.0 / 30.0);
            predict(&mut gaussian, &transition, &noise);
        }

        // The object coasts for about one time constant, then stops.
        assert_relative_eq!(gaussian.mean.x, 1.0, epsilon = 1e-3);
        assert!(gaussian.mean.z.abs() < 1e-3);
        assert_relative_eq!(gaussian.covariance[(2, 2)], 0.16, epsilon = 1e-3);
        // Position variance approaches 2σ²τ(t - τ) instead of growing with t³.
        assert_relative_eq!(
            gaussian.covariance[(0, 0)],
            2.0 * 0.16 * (10.0 - 1.5),
            max_relative = 0.05
        );
    }

    #[test]
    fn heading_drift_acts_perpendicular_to_lever_arm() {
        let drift = OdometryDrift {
            translation_variance_per_meter: 0.0,
            heading_variance_per_radian: 0.01,
            heading_variance_per_second: 0.0,
        };
        let noise = drift.position_noise(Vector2::new(4.0, 0.0), 0.0, 1.0, 0.0);

        assert_relative_eq!(noise[(0, 0)], 0.0);
        assert_relative_eq!(noise[(1, 1)], 0.16, epsilon = 1e-6);
    }
}

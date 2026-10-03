//! Measurement models with analytical right-increment Jacobians.

mod common;
mod imu;
mod imu_bias;
mod motion;
mod prior;

pub use imu::{ImuKinematics, RelativeYaw};
pub use imu_bias::{ImuBiasPrior, ImuBiasWalk};
pub use motion::MotionPrior;
pub use prior::{CameraIntrinsicsPrior, TrajectoryPrior};

#[cfg(test)]
mod tests;

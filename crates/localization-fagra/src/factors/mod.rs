//! Measurement models with analytical right-increment Jacobians.

mod common;
mod field_containment;
mod ground;
mod imu;
mod imu_bias;
mod kinematic_odometry;
mod motion;
mod preintegrated_imu;
mod prior;

pub use field_containment::FieldContainment;
pub use ground::{FootGround, FootObservation};
pub use imu::{ImuKinematics, RelativeYaw};
pub use imu_bias::{ImuBiasPrior, ImuBiasWalk};
pub use kinematic_odometry::{AdjacentKinematicOdometry, KinematicOdometry};
pub use motion::MotionPrior;
pub use preintegrated_imu::PreintegratedImu;
pub use prior::{CameraIntrinsicsPrior, TrajectoryPrior};

#[cfg(test)]
mod tests;

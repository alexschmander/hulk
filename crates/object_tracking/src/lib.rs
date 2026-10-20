//! Multi-object tracking building blocks shared by the humanoid and goal post filters.
//!
//! Tracks live in the continuous [`coordinate_systems::Odometry`] frame. Own camera detections are
//! transformed into it with the odometry pose of their image, so localization never enters the
//! track state.

pub mod association;
pub mod camera;
pub mod classification;
pub mod gaussian;
pub mod measurement;
pub mod motion;
pub mod multi_bernoulli;
pub mod odometry_history;

use std::time::SystemTime;

use factrs::{
    core::Vector3,
    variables::{SE2, SE23},
};

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// The timestamp of the most recent variable in the optimized graph.
    pub time: SystemTime,
    /// Generation assigned by the latest reset.
    pub generation: u64,
    /// The latest pose estimate from the optimized graph.
    pub latest_robot_to_local: SE23<f64>,
    /// Planar alignment into the field frame, once initialized by vision.
    pub local_to_field: Option<SE2<f64>>,
    /// IRLS Laplace tangent covariance of the composed robot-to-field pose.
    pub robot_to_field_covariance: Option<nalgebra::SMatrix<f64, 6, 6>>,
    /// The current optimized camera intrinsics estimate.
    pub camera_intrinsics: crate::camera_intrinsics::CameraIntrinsics<f64>,
    /// Latest visual frame represented by factors in this result.
    pub latest_visual_measurement_time: Option<SystemTime>,
    /// Optimized pose at `latest_visual_measurement_time`, when its spline interval is retained.
    pub latest_visual_robot_to_local: Option<SE23<f64>>,
    /// Outcome of the optimizer invocation that produced this result.
    pub optimizer_status: super::BackendOptimizerStatus,
}

impl OptimizationResult {
    pub fn position(&self) -> Vector3<f64> {
        self.latest_robot_to_local.xyz().into_owned()
    }
}

mod backend_task;
mod camera;
mod diagnostics;
mod event_handlers;
mod ingest;
mod live_odometry;
mod node;
mod parameters;
mod pose;
mod publish;
mod synchronous;
mod visual_localization;

pub use camera::{camera_intrinsics_from_matrix, intrinsic_from_camera_intrinsics};
pub use diagnostics::{SolveDiagnostics, SolveOptimizerStatus, SolveResidualDiagnostics};
pub use ingest::{ingest_foot_heights, ingest_visual_odometry};
pub use node::{run, run_boxed};
pub use parameters::Localization3dParameters;
pub use pose::initial_state_from_camera_matrix_and_imu;
pub use synchronous::{SynchronousLocalization, SynchronousLocalizationOutput};
pub use visual_localization::GlobalVisualLock;

#[cfg(test)]
fn test_result(time: ros_z::time::Time) -> localization_factrs::OptimizationResult {
    use linear_algebra::IntoTransform;
    localization_factrs::OptimizationResult {
        time: time.to_wallclock(),
        generation: 0,
        robot_to_local: nalgebra::Isometry3::identity().framed_transform(),
        local_to_field: Some(nalgebra::Isometry2::identity().framed_transform()),
        robot_to_field: Some(nalgebra::Isometry3::identity().framed_transform()),
        robot_to_field_covariance: Some(nalgebra::SMatrix::identity()),
        velocity: nalgebra::Vector3::zeros(),
        camera_intrinsics: localization_factrs::CameraIntrinsics::new(
            nalgebra::vector![200.0, 200.0],
            nalgebra::vector![320.0, 240.0],
        ),
        latest_visual_measurement_time: None,
        latest_visual_robot_to_local: None,
        latest_visual_robot_to_field: None,
        optimizer_status: localization_factrs::backend::BackendOptimizerStatus::Converged,
    }
}

pub use backend::{
    BackendConfiguration, BackendConfigurationError, FieldContainmentConfiguration, VinsBackend,
    VinsBackendError,
};
pub use camera_intrinsics::CameraIntrinsics;
pub use factors::visual_reprojection::MIN_REPROJECTION_DEPTH;
pub use frontend::{OptimizationResult, VinsFrontend, VinsFrontendError};
pub use initial_state::InitialState;
pub use splines::{SE23Kinematics, SE23Spline};
pub use symbols::{LocalToField, State};
pub use utils::{interval_dt, tau};

pub mod backend;
mod camera_intrinsics;
pub mod conversions;
mod factors;
mod frontend;
mod initial_state;
mod interval_measurement;
mod measurements;
mod schur_marginalization;
mod splines;
mod symbols;
mod utils;

/// Creates the nonblocking frontend and worker-owned backend.
///
/// # Panics
/// Panics for invalid configuration. Call `BackendConfiguration::validate` first for fallible setup.
pub fn initialize(
    config: BackendConfiguration,
    initial_state: InitialState,
) -> (VinsFrontend, VinsBackend) {
    let (measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, result_receiver) = tokio::sync::watch::channel(None);

    let frontend = VinsFrontend::new(measurement_sender, result_receiver, &config);
    let backend = VinsBackend::new(config, initial_state, measurement_receiver, result_sender);
    (frontend, backend)
}

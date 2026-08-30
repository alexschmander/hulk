use color_eyre::Result;
use localization_factrs::{InitialState, VinsFrontend, VinsFrontendError};
use projection::camera_matrix::CameraMatrix;
use ros_z::{cache::Cache, time::Time};
use types::{
    field_dimensions::FieldDimensions, primary_state::PrimaryState, time_wrapper::TimeWrapper,
};

use crate::{
    live_odometry::LiveVisualOdometryLocalization, pose::initial_state_from_camera_matrix,
    publish::LocalizationPublishers, visual_localization::GlobalVisualLockTracker,
};

pub(crate) fn localization_is_damping(primary_state_cache: &Cache<PrimaryState>) -> bool {
    primary_state_cache
        .get_latest()
        .is_none_or(|state| *state == PrimaryState::Damping)
}

pub(crate) fn initial_state_for_reset(
    camera_matrix_cache: &Cache<TimeWrapper<CameraMatrix>>,
    field_dimensions: &FieldDimensions,
    fallback: &InitialState,
) -> InitialState {
    camera_matrix_cache
        .get_latest()
        .map(|camera_matrix| {
            initial_state_from_camera_matrix(&camera_matrix.inner, field_dimensions)
        })
        .unwrap_or_else(|| fallback.clone())
}

pub(crate) fn reset_localization_for_damping(
    frontend: &mut VinsFrontend,
    live_localization: &mut LiveVisualOdometryLocalization,
    global_visual_lock: &mut GlobalVisualLockTracker,
    initial_state: InitialState,
    time: Time,
) -> Result<(), VinsFrontendError> {
    frontend.reset(time.to_wallclock(), initial_state)?;
    live_localization.clear();
    global_visual_lock.reset_for_damping();
    Ok(())
}

pub(crate) async fn reset_and_publish_startup_prior(
    frontend: &mut VinsFrontend,
    live_localization: &mut LiveVisualOdometryLocalization,
    global_visual_lock: &mut GlobalVisualLockTracker,
    initial_state: InitialState,
    time: Time,
    field_dimensions: &FieldDimensions,
    publishers: LocalizationPublishers<'_>,
) -> Result<()> {
    reset_localization_for_damping(
        frontend,
        live_localization,
        global_visual_lock,
        initial_state,
        time,
    )?;
    publishers
        .publish_startup_prior(time, field_dimensions)
        .await
}

pub(crate) async fn publish_damping_optimization_result(
    frontend: &mut VinsFrontend,
    live_localization: &mut LiveVisualOdometryLocalization,
    global_visual_lock: &mut GlobalVisualLockTracker,
    time: Time,
    field_dimensions: &FieldDimensions,
    publishers: LocalizationPublishers<'_>,
) -> Result<()> {
    let _ = frontend.last_optimization_result();
    live_localization.clear();
    global_visual_lock.reset_for_damping();
    publishers
        .publish_startup_prior(time, field_dimensions)
        .await
}

#[cfg(test)]
mod tests {
    use localization_factrs::{
        CameraIntrinsics, OptimizationResult, backend::BackendOptimizerStatus,
    };

    use super::*;

    #[test]
    fn damping_reset_discards_pending_bootstrap() {
        let time = Time::from_nanos(1_000_000_000);
        let mut tracker = GlobalVisualLockTracker::default();
        tracker.track_associations(time, linear_algebra::Isometry3::identity(), &[]);
        tracker.reset_for_damping();
        let matching_result = OptimizationResult {
            time: time.to_wallclock(),
            transform: nalgebra::Isometry3::identity(),
            velocity: nalgebra::Vector3::zeros(),
            camera_intrinsics: CameraIntrinsics::new(
                nalgebra::vector![200.0, 200.0],
                nalgebra::vector![320.0, 240.0],
            ),
            latest_visual_measurement_time: Some(time.to_wallclock()),
            latest_visual_transform: Some(nalgebra::Isometry3::identity()),
            optimizer_status: BackendOptimizerStatus::Converged,
        };

        assert_eq!(tracker.status(), crate::GlobalVisualLock::Unlocked);
        assert!(!tracker.handle_backend_result(&matching_result));
    }
}

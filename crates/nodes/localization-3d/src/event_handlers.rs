use std::future::ready;

use color_eyre::{Result, eyre::Context as _};
use localization_factrs::VinsFrontend;
use projection::{camera_matrix::CameraMatrix, intrinsic::Intrinsic};
use ros_z::{cache::Cache, pubsub::Publisher, time::Time};
use types::{
    time_wrapper::TimeWrapper,
    visual_localization::AssociationPoseHintSource,
    visual_odometry::{VisualOdometer, VisualOdometryDelta as VisualOdometryDeltaMessage},
};

use crate::{
    camera::{fresh_camera_matrix, intrinsic_from_camera_intrinsics},
    ingest::ingest_visual_odometry,
    live_odometry::LiveVisualOdometryLocalization,
    pose::backend_localization_for_result,
    publish::LocalizationPublishers,
    visual_localization::GlobalVisualLockTracker,
};

pub(crate) fn handle_visual_odometry(
    frontend: &mut VinsFrontend,
    visual_odometry: VisualOdometryDeltaMessage,
    camera_matrix_cache: &Cache<TimeWrapper<CameraMatrix>>,
) -> Result<()> {
    let Some(previous_camera_matrix) =
        fresh_camera_matrix(camera_matrix_cache, visual_odometry.previous_time)
    else {
        return Ok(());
    };
    let Some(current_camera_matrix) =
        fresh_camera_matrix(camera_matrix_cache, visual_odometry.current_time)
    else {
        return Ok(());
    };

    ingest_visual_odometry(
        frontend,
        visual_odometry,
        &previous_camera_matrix.inner,
        &current_camera_matrix.inner,
    )
    .wrap_err("failed to ingest visual odometry measurement into frontend")
}

pub(crate) async fn handle_visual_odometer(
    live_localization: &mut LiveVisualOdometryLocalization,
    global_visual_lock: &GlobalVisualLockTracker,
    visual_odometer: VisualOdometer,
    visual_odometer_cache: &Cache<VisualOdometer>,
    camera_matrix_cache: &Cache<TimeWrapper<CameraMatrix>>,
    publishers: LocalizationPublishers<'_>,
) -> Result<()> {
    if !global_visual_lock.has_backend_result() {
        return Ok(());
    }

    live_localization.try_reset_pending(visual_odometer_cache, camera_matrix_cache);
    if let Some(transform) =
        live_localization.update_from_odometer(&visual_odometer, camera_matrix_cache)
    {
        publishers
            .publish_outputs(
                visual_odometer.time,
                Some(transform),
                Some(transform),
                AssociationPoseHintSource::LiveVisualOdometry,
            )
            .await?;
    }
    Ok(())
}

pub(crate) async fn handle_optimization_result(
    frontend: &mut VinsFrontend,
    live_localization: &mut LiveVisualOdometryLocalization,
    global_visual_lock: &mut GlobalVisualLockTracker,
    visual_odometer_cache: &Cache<VisualOdometer>,
    camera_matrix_cache: &Cache<TimeWrapper<CameraMatrix>>,
    publishers: LocalizationPublishers<'_>,
    calibrated_intrinsics_publisher: &Publisher<Intrinsic>,
) -> Result<()> {
    let Some(result) = frontend.last_optimization_result() else {
        return Ok(());
    };
    let result_time = Time::from_wallclock(result.time);
    let camera_matrix = fresh_camera_matrix(camera_matrix_cache, result_time);
    let backend_transform = backend_localization_for_result(&result, camera_matrix.as_deref());
    let transform = if global_visual_lock.handle_backend_result(&result) {
        live_localization.reset(&result, visual_odometer_cache, camera_matrix_cache);
        Some(
            live_localization
                .field_to_robot_latest(visual_odometer_cache, camera_matrix_cache)
                .unwrap_or(backend_transform),
        )
    } else {
        None
    };

    publishers
        .publish_outputs(
            result_time,
            transform,
            Some(backend_transform),
            AssociationPoseHintSource::BackendBranch,
        )
        .await?;
    calibrated_intrinsics_publisher
        .publish_if_subscribed(|| {
            ready(intrinsic_from_camera_intrinsics(&result.camera_intrinsics))
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use localization_factrs::{
        CameraIntrinsics, OptimizationResult, backend::BackendOptimizerStatus,
    };

    use super::*;

    #[test]
    fn unrelated_backend_result_does_not_acquire_global_lock() {
        let pending_time = Time::from_nanos(2_000_000_000);
        let mut tracker = GlobalVisualLockTracker::default();
        tracker.track_associations(pending_time, linear_algebra::Isometry3::identity(), &[]);
        let stale_result = OptimizationResult {
            time: Time::from_nanos(1_000_000_000).to_wallclock(),
            transform: nalgebra::Isometry3::identity(),
            velocity: nalgebra::Vector3::zeros(),
            camera_intrinsics: CameraIntrinsics::new(
                nalgebra::vector![200.0, 200.0],
                nalgebra::vector![320.0, 240.0],
            ),
            latest_visual_measurement_time: None,
            latest_visual_transform: None,
            optimizer_status: BackendOptimizerStatus::Converged,
        };

        assert!(!tracker.handle_backend_result(&stale_result));
        assert_eq!(tracker.status(), crate::GlobalVisualLock::WaitingForBackend);
    }
}

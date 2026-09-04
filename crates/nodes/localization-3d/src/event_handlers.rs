use std::{future::ready, time::Duration};

use color_eyre::{Result, eyre::Context as _};
use linear_algebra::IntoTransform;
use localization_factrs::{VinsFrontend, backend::BackendOptimizerStatus};
use projection::{camera_matrix::CameraMatrix, intrinsic::Intrinsic};
use ros_z::{cache::Cache, pubsub::Publisher, time::Time};
use types::{
    localization::{LocalizationEstimate3D, LocalizationState3D},
    time_wrapper::TimeWrapper,
    visual_localization::AssociationPoseHintSource,
    visual_odometry::{VisualOdometer, VisualOdometryDelta as VisualOdometryDeltaMessage},
};

use crate::{
    camera::{fresh_camera_matrix, intrinsic_from_camera_intrinsics},
    ingest::ingest_visual_odometry,
    live_odometry::LiveVisualOdometryLocalization,
    pose::{
        backend_localization_for_result, compose_robot_to_field, constrain_localization_to_ground,
    },
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
    live: &mut LiveVisualOdometryLocalization,
    state: LocalizationState3D,
    epoch: u64,
    visual_odometer: VisualOdometer,
    visual_odometer_cache: &Cache<VisualOdometer>,
    camera_matrix_cache: &Cache<TimeWrapper<CameraMatrix>>,
    publishers: LocalizationPublishers<'_>,
) -> Result<()> {
    live.try_reset_pending(visual_odometer_cache, camera_matrix_cache);
    let Some((robot_to_local, local_to_field)) =
        live.update_from_odometer(&visual_odometer, camera_matrix_cache)
    else {
        return Ok(());
    };
    let localization = if matches!(state, LocalizationState3D::Tracking { .. }) {
        local_to_field.and_then(|alignment| {
            let camera = fresh_camera_matrix(camera_matrix_cache, visual_odometer.time)?;
            let field_to_robot = compose_robot_to_field(robot_to_local, alignment).inverse();
            Some(constrain_localization_to_ground(
                field_to_robot,
                &camera.inner.ground_to_robot,
            ))
        })
    } else {
        None
    };
    publishers
        .publish_outputs(
            visual_odometer.time,
            epoch,
            robot_to_local,
            local_to_field,
            localization,
            state,
            AssociationPoseHintSource::LiveVisualOdometry,
        )
        .await
}

pub(crate) async fn handle_optimization_result(
    frontend: &mut VinsFrontend,
    live: &mut LiveVisualOdometryLocalization,
    global_visual_lock: &mut GlobalVisualLockTracker,
    epoch: u64,
    epoch_start: Time,
    now: Time,
    tracking_timeout: Duration,
    state: &mut LocalizationState3D,
    visual_odometer_cache: &Cache<VisualOdometer>,
    camera_matrix_cache: &Cache<TimeWrapper<CameraMatrix>>,
    publishers: LocalizationPublishers<'_>,
    calibrated_intrinsics_publisher: &Publisher<Intrinsic>,
) -> Result<Option<Time>> {
    let Some(result) = frontend.last_optimization_result() else {
        return Ok(None);
    };
    if result.generation != epoch || Time::from_wallclock(result.time) < epoch_start {
        return Ok(None);
    }
    calibrated_intrinsics_publisher
        .publish_if_subscribed(|| {
            ready(intrinsic_from_camera_intrinsics(&result.camera_intrinsics))
        })
        .await?;
    if result.optimizer_status != BackendOptimizerStatus::Converged {
        return Ok(None);
    }

    let time = Time::from_wallclock(result.time);
    let expired = time.saturating_add(tracking_timeout) <= now;
    let has_visual_lock = if expired {
        false
    } else {
        global_visual_lock.handle_backend_result(&result)
    };
    let was_tracking = matches!(state, LocalizationState3D::Tracking { .. });
    let successful_solve = refresh_tracking_state(
        state,
        &result,
        has_visual_lock,
        epoch_start,
        now,
        tracking_timeout,
    );
    if successful_solve.is_some() {
        live.reset(&result, visual_odometer_cache, camera_matrix_cache);
    } else if was_tracking {
        return Ok(None);
    }
    let localization = successful_solve
        .is_some()
        .then(|| {
            backend_localization_for_result(
                &result,
                fresh_camera_matrix(camera_matrix_cache, time).as_deref(),
            )
        })
        .flatten();
    publishers
        .publish_outputs(
            time,
            epoch,
            result.robot_to_local.inner.cast().framed_transform(),
            result
                .local_to_field
                .map(|pose| pose.inner.cast().framed_transform()),
            localization,
            *state,
            AssociationPoseHintSource::BackendBranch,
        )
        .await?;
    Ok(successful_solve)
}

fn refresh_tracking_state(
    state: &mut LocalizationState3D,
    result: &localization_factrs::OptimizationResult,
    has_visual_lock: bool,
    epoch_start: Time,
    now: Time,
    tracking_timeout: Duration,
) -> Option<Time> {
    let time = Time::from_wallclock(result.time);
    if result.optimizer_status != BackendOptimizerStatus::Converged
        || !has_visual_lock
        || time < epoch_start
        || time.saturating_add(tracking_timeout) <= now
    {
        return None;
    }
    let robot_to_field = result
        .robot_to_field
        .as_ref()?
        .inner
        .cast::<f32>()
        .framed_transform();
    let covariance = result.robot_to_field_covariance?.cast::<f32>();
    if !robot_to_field
        .inner
        .to_homogeneous()
        .iter()
        .chain(covariance.iter())
        .all(|value| value.is_finite())
    {
        return None;
    }
    *state = LocalizationState3D::Tracking {
        estimate: LocalizationEstimate3D {
            robot_to_field,
            covariance,
        },
        last_successful_solve: time,
    };
    Some(time)
}

pub(crate) fn lose_track(state: LocalizationState3D) -> LocalizationState3D {
    match state {
        LocalizationState3D::Tracking {
            estimate,
            last_successful_solve,
        } => LocalizationState3D::LostTrack {
            last_known_estimate: estimate,
            last_successful_solve,
        },
        state => state,
    }
}

#[cfg(test)]
mod tests {
    use localization_factrs::{CameraIntrinsics, OptimizationResult};

    use super::*;

    fn result(time: Time) -> OptimizationResult {
        OptimizationResult {
            time: time.to_wallclock(),
            generation: 0,
            robot_to_local: nalgebra::Isometry3::identity().framed_transform(),
            local_to_field: Some(nalgebra::Isometry2::identity().framed_transform()),
            robot_to_field: Some(nalgebra::Isometry3::identity().framed_transform()),
            robot_to_field_covariance: Some(nalgebra::SMatrix::identity()),
            velocity: nalgebra::Vector3::zeros(),
            camera_intrinsics: CameraIntrinsics::new(
                nalgebra::vector![200.0, 200.0],
                nalgebra::vector![320.0, 240.0],
            ),
            latest_visual_measurement_time: None,
            latest_visual_robot_to_local: None,
            latest_visual_robot_to_field: None,
            optimizer_status: BackendOptimizerStatus::Converged,
        }
    }

    fn tracking(time: Time) -> LocalizationState3D {
        LocalizationState3D::Tracking {
            estimate: LocalizationEstimate3D {
                robot_to_field: linear_algebra::Isometry3::identity(),
                covariance: nalgebra::SMatrix::identity(),
            },
            last_successful_solve: time,
        }
    }

    #[test]
    fn max_iterations_does_not_refresh_tracking() {
        let old_time = Time::from_nanos(1);
        let mut state = tracking(old_time);
        let mut result = result(Time::from_nanos(2));
        result.optimizer_status = BackendOptimizerStatus::MaxIterations;

        assert_eq!(
            refresh_tracking_state(
                &mut state,
                &result,
                true,
                old_time,
                Time::from_nanos(2),
                Duration::from_secs(1),
            ),
            None
        );
        assert_eq!(state, tracking(old_time));
    }

    #[test]
    fn timeout_retains_last_estimate_and_solve_time() {
        let state = tracking(Time::from_nanos(3));
        let LocalizationState3D::Tracking {
            estimate,
            last_successful_solve,
        } = state
        else {
            unreachable!()
        };

        assert_eq!(
            lose_track(state),
            LocalizationState3D::LostTrack {
                last_known_estimate: estimate,
                last_successful_solve,
            }
        );
    }

    #[test]
    fn missing_or_invalid_covariance_does_not_establish_tracking() {
        let epoch_start = Time::from_nanos(1);
        let mut missing = result(Time::from_nanos(2));
        missing.robot_to_field_covariance = None;
        let mut invalid = result(Time::from_nanos(2));
        invalid.robot_to_field_covariance.as_mut().unwrap()[(0, 0)] = f64::NAN;

        for result in [missing, invalid] {
            let mut state = LocalizationState3D::Startup;
            assert_eq!(
                refresh_tracking_state(
                    &mut state,
                    &result,
                    true,
                    epoch_start,
                    Time::from_nanos(2),
                    Duration::from_secs(1),
                ),
                None
            );
            assert_eq!(state, LocalizationState3D::Startup);
        }
    }

    #[test]
    fn expired_result_does_not_establish_or_refresh_tracking() {
        let solve_time = Time::from_nanos(1_000_000_000);
        let timeout = Duration::from_secs(2);
        for mut state in [LocalizationState3D::Startup, tracking(Time::from_nanos(1))] {
            let original = state;
            assert_eq!(
                refresh_tracking_state(
                    &mut state,
                    &result(solve_time),
                    true,
                    Time::from_nanos(0),
                    solve_time.saturating_add(timeout),
                    timeout,
                ),
                None
            );
            assert_eq!(state, original);
        }
    }
}

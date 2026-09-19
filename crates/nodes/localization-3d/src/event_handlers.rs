use std::time::Duration;

use color_eyre::{Result, eyre::Context as _};
use linear_algebra::IntoTransform;
use localization_factrs::{OptimizationResult, VinsFrontend, backend::BackendOptimizerStatus};
use projection::camera_matrix::CameraMatrix;
use ros_z::{cache::Cache, time::Time};
use types::{
    localization::{LocalizationEstimate3D, LocalizationState3D},
    time_wrapper::TimeWrapper,
    visual_odometry::{VisualOdometer, VisualOdometryDelta as VisualOdometryDeltaMessage},
};

use crate::{
    camera::fresh_camera_matrix,
    ingest::ingest_visual_odometry,
    live_odometry::{LiveVisualOdometryLocalization, PoseCorrection},
    publish::OutputMailbox,
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

pub(crate) fn handle_visual_odometer(
    live: &mut LiveVisualOdometryLocalization,
    state: LocalizationState3D,
    epoch: u64,
    visual_odometer: VisualOdometer,
    visual_odometer_cache: &Cache<VisualOdometer>,
    camera_matrix_cache: &Cache<TimeWrapper<CameraMatrix>>,
    publishers: &OutputMailbox,
) {
    live.try_reset_pending(visual_odometer_cache, camera_matrix_cache);
    live.update_from_odometer(&visual_odometer, camera_matrix_cache);
    if let Some(geometry) = live.association_geometry(state, epoch) {
        publishers.publish_outputs(geometry);
    }
}

pub(crate) fn handle_odometer_discontinuity(
    live: &LiveVisualOdometryLocalization,
    state: &mut LocalizationState3D,
    visual_lock: &mut GlobalVisualLockTracker,
    current: &VisualOdometer,
    now: Time,
) {
    if live.odometer_epoch_changed(current) {
        *state = lose_track(*state, visual_lock, now);
    }
}

#[derive(Clone, Copy)]
pub(crate) struct AcceptanceWindow {
    pub epoch: u64,
    pub epoch_start: Time,
    pub now: Time,
    pub tracking_timeout: Duration,
    pub visual_tracking_timeout: Duration,
}

pub(crate) struct BackendAcceptance {
    pub state: LocalizationState3D,
    pub correction: Option<PoseCorrection>,
}

pub(crate) fn backend_result_is_eligible(
    result: &OptimizationResult,
    state: LocalizationState3D,
    window: AcceptanceWindow,
) -> bool {
    let time = Time::from_wallclock(result.time);
    let last_solve = match state {
        LocalizationState3D::Startup => None,
        LocalizationState3D::Tracking {
            last_successful_solve,
            ..
        }
        | LocalizationState3D::LostTrack {
            last_successful_solve,
            ..
        } => Some(last_successful_solve),
    };
    result.generation == window.epoch
        && time >= window.epoch_start
        && time <= window.now
        && last_solve.is_none_or(|last| time > last)
}

/// Owns backend acceptance policy; only the visual-lock owner is mutated here.
/// Ineligible results return `None`; eligible non-converged results leave state untouched.
pub(crate) fn accept_backend_result(
    result: &OptimizationResult,
    state: LocalizationState3D,
    visual_lock: &mut GlobalVisualLockTracker,
    window: AcceptanceWindow,
) -> Option<BackendAcceptance> {
    if !backend_result_is_eligible(result, state, window) {
        return None;
    }
    let mut acceptance = BackendAcceptance {
        state,
        correction: None,
    };
    if result.optimizer_status != BackendOptimizerStatus::Converged {
        return Some(acceptance);
    }

    let time = Time::from_wallclock(result.time);
    let solve_expired = time.saturating_add(window.tracking_timeout) <= window.now;
    let visual_expired = result
        .latest_visual_measurement_time
        .is_none_or(|visual_time| {
            Time::from_wallclock(visual_time).saturating_add(window.visual_tracking_timeout)
                <= window.now
        });
    let has_visual_lock =
        !solve_expired && !visual_expired && visual_lock.handle_backend_result(result);
    let was_tracking = matches!(state, LocalizationState3D::Tracking { .. });
    if let Some(estimate) = has_visual_lock.then(|| tracking_estimate(result)).flatten() {
        acceptance.state = LocalizationState3D::Tracking {
            estimate,
            last_successful_solve: time,
        };
        acceptance.correction = Some(PoseCorrection {
            time,
            robot_to_local: result.robot_to_local,
            local_to_field: result.local_to_field,
        });
        if !was_tracking {
            visual_lock.reject_new_frames_through(window.now);
        }
    } else {
        if was_tracking {
            acceptance.state = lose_track(state, visual_lock, window.now);
        } else if has_visual_lock {
            visual_lock.invalidate(window.now);
        }
        // Startup may use a local solve, never an unaccepted global branch.
        if matches!(state, LocalizationState3D::Startup)
            && !solve_expired
            && result
                .robot_to_local
                .inner
                .cast::<f32>()
                .to_homogeneous()
                .iter()
                .all(|v| v.is_finite())
        {
            acceptance.correction = Some(PoseCorrection {
                time,
                robot_to_local: result.robot_to_local,
                local_to_field: None,
            });
        }
    }
    Some(acceptance)
}

fn tracking_estimate(result: &OptimizationResult) -> Option<LocalizationEstimate3D> {
    let robot_to_field = result
        .robot_to_field
        .as_ref()?
        .inner
        .cast::<f32>()
        .framed_transform();
    let covariance = result.robot_to_field_covariance?.cast::<f32>();
    let alignment = result.local_to_field?;
    if !robot_to_field
        .inner
        .to_homogeneous()
        .iter()
        .chain(covariance.iter())
        .chain(
            result
                .robot_to_local
                .inner
                .cast::<f32>()
                .to_homogeneous()
                .iter(),
        )
        .chain(alignment.inner.cast::<f32>().to_homogeneous().iter())
        .all(|value| value.is_finite())
        || covariance.diagonal().iter().any(|variance| *variance < 0.0)
        || (covariance - covariance.transpose()).amax() > 1.0e-5
        || covariance.symmetric_eigen().eigenvalues.min() < -1.0e-6
    {
        return None;
    }
    Some(LocalizationEstimate3D {
        robot_to_field,
        covariance,
    })
}

pub(crate) fn lose_track(
    state: LocalizationState3D,
    visual_lock: &mut GlobalVisualLockTracker,
    now: Time,
) -> LocalizationState3D {
    visual_lock.invalidate(now);
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

pub(crate) fn tracking_deadline(
    state: LocalizationState3D,
    visual_lock: &GlobalVisualLockTracker,
    tracking_timeout: Duration,
    visual_tracking_timeout: Duration,
) -> Option<Time> {
    let LocalizationState3D::Tracking {
        last_successful_solve,
        ..
    } = state
    else {
        return None;
    };
    Some(
        last_successful_solve.saturating_add(tracking_timeout).min(
            visual_lock
                .last_accepted_time()?
                .saturating_add(visual_tracking_timeout),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_result as result;

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
        let state = tracking(old_time);
        let mut result = result(Time::from_nanos(2));
        result.optimizer_status = BackendOptimizerStatus::MaxIterations;

        let acceptance = accept_backend_result(
            &result,
            state,
            &mut GlobalVisualLockTracker::default(),
            AcceptanceWindow {
                epoch: 0,
                epoch_start: old_time,
                now: Time::from_nanos(2),
                tracking_timeout: Duration::from_secs(1),
                visual_tracking_timeout: Duration::from_secs(1),
            },
        )
        .unwrap();
        assert!(acceptance.correction.is_none());
        assert_eq!(acceptance.state, state);
    }

    #[test]
    fn startup_rejects_wrong_epochs_and_uses_only_finite_local_corrections() {
        let mut result = result(Time::from_nanos(20));
        let mut lock = GlobalVisualLockTracker::default();
        let timeout = Duration::from_secs(1);
        for (epoch, start, now) in [(1, 0, 30), (0, 21, 30), (0, 0, 19)] {
            assert!(
                accept_backend_result(
                    &result,
                    LocalizationState3D::Startup,
                    &mut lock,
                    AcceptanceWindow {
                        epoch,
                        epoch_start: Time::from_nanos(start),
                        now: Time::from_nanos(now),
                        tracking_timeout: timeout,
                        visual_tracking_timeout: timeout,
                    },
                )
                .is_none()
            );
        }
        let window = AcceptanceWindow {
            epoch: 0,
            epoch_start: Time::from_nanos(0),
            now: Time::from_nanos(30),
            tracking_timeout: timeout,
            visual_tracking_timeout: timeout,
        };
        let accepted =
            accept_backend_result(&result, LocalizationState3D::Startup, &mut lock, window)
                .unwrap();
        assert_eq!(accepted.state, LocalizationState3D::Startup);
        let correction = accepted.correction.unwrap();
        assert_eq!(correction.time, Time::from_nanos(20));
        assert_eq!(correction.robot_to_local, result.robot_to_local);
        assert!(correction.local_to_field.is_none());
        assert!(result.local_to_field.is_some());

        result.robot_to_local.inner.translation.vector.x = f64::INFINITY;
        let rejected =
            accept_backend_result(&result, LocalizationState3D::Startup, &mut lock, window)
                .unwrap();
        assert_eq!(rejected.state, LocalizationState3D::Startup);
        assert!(rejected.correction.is_none());
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
            lose_track(
                state,
                &mut GlobalVisualLockTracker::default(),
                Time::from_nanos(4)
            ),
            LocalizationState3D::LostTrack {
                last_known_estimate: estimate,
                last_successful_solve,
            }
        );
    }

    #[test]
    fn shared_odometer_discontinuity_invalidates_tracking_but_ignores_old_samples() {
        let time = Time::from_nanos(10);
        let mut live = LiveVisualOdometryLocalization::default();
        live.set_initial(time, linear_algebra::Isometry3::identity());
        let mut odometer = VisualOdometer {
            time,
            epoch: 0,
            current_left_camera_to_visual_odometer: nalgebra::Isometry3::identity(),
        };
        live.update_with_exact_sample(&odometer, &CameraMatrix::default())
            .unwrap();
        let mut state = tracking(time);
        let mut visual_lock = GlobalVisualLockTracker::default();
        odometer.epoch = 1;
        odometer.time = Time::from_nanos(9);
        handle_odometer_discontinuity(
            &live,
            &mut state,
            &mut visual_lock,
            &odometer,
            Time::from_nanos(20),
        );
        assert_eq!(state, tracking(time));
        odometer.time = Time::from_nanos(20);
        handle_odometer_discontinuity(
            &live,
            &mut state,
            &mut visual_lock,
            &odometer,
            Time::from_nanos(25),
        );
        assert!(
            matches!(state, LocalizationState3D::LostTrack { last_successful_solve, .. } if last_successful_solve == time)
        );
        assert!(!visual_lock.accepts_frame(Time::from_nanos(25), state));
        assert!(visual_lock.accepts_frame(Time::from_nanos(26), state));
    }

    #[test]
    fn missing_or_invalid_covariance_does_not_establish_tracking() {
        let mut missing = result(Time::from_nanos(2));
        missing.robot_to_field_covariance = None;
        let mut invalid = result(Time::from_nanos(2));
        invalid.robot_to_field_covariance.as_mut().unwrap()[(0, 0)] = f64::NAN;
        let mut indefinite = result(Time::from_nanos(2));
        indefinite.robot_to_field_covariance.as_mut().unwrap()[(0, 1)] = 2.0;
        indefinite.robot_to_field_covariance.as_mut().unwrap()[(1, 0)] = 2.0;
        let mut asymmetric = result(Time::from_nanos(2));
        asymmetric.robot_to_field_covariance.as_mut().unwrap()[(0, 1)] = 0.1;

        for result in [missing, invalid, indefinite, asymmetric] {
            assert!(tracking_estimate(&result).is_none());
        }
    }

    #[test]
    fn expired_result_does_not_establish_or_refresh_tracking() {
        let solve_time = Time::from_nanos(1_000_000_000);
        let timeout = Duration::from_secs(2);
        for state in [LocalizationState3D::Startup, tracking(Time::from_nanos(1))] {
            let now = solve_time.saturating_add(timeout);
            let acceptance = accept_backend_result(
                &result(solve_time),
                state,
                &mut GlobalVisualLockTracker::default(),
                AcceptanceWindow {
                    epoch: 0,
                    epoch_start: Time::from_nanos(0),
                    now,
                    tracking_timeout: timeout,
                    visual_tracking_timeout: timeout,
                },
            )
            .unwrap();
            assert!(acceptance.correction.is_none());
            assert_eq!(
                acceptance.state,
                lose_track(state, &mut GlobalVisualLockTracker::default(), now)
            );
        }
    }

    #[test]
    fn late_and_unaccepted_results_never_replace_the_trusted_prior() {
        let time = Time::from_nanos(100);
        let lost = lose_track(
            tracking(time),
            &mut GlobalVisualLockTracker::default(),
            Time::from_nanos(110),
        );
        for original in [tracking(time), lost] {
            for solve_time in [99, 100, 120, 140] {
                let mut result = result(Time::from_nanos(solve_time));
                result
                    .robot_to_field
                    .as_mut()
                    .unwrap()
                    .inner
                    .translation
                    .vector
                    .x = 50.0;
                let now = Time::from_nanos(130);
                let acceptance = accept_backend_result(
                    &result,
                    original,
                    &mut GlobalVisualLockTracker::default(),
                    AcceptanceWindow {
                        epoch: 0,
                        epoch_start: Time::from_nanos(0),
                        now,
                        tracking_timeout: Duration::from_secs(1),
                        visual_tracking_timeout: Duration::from_secs(1),
                    },
                );
                if solve_time == 120 {
                    let acceptance = acceptance.unwrap();
                    assert!(acceptance.correction.is_none());
                    assert_eq!(
                        acceptance.state,
                        lose_track(original, &mut GlobalVisualLockTracker::default(), now)
                    );
                } else {
                    assert!(acceptance.is_none());
                }
            }
        }
    }

    #[test]
    fn tracking_requires_a_visual_deadline_even_with_fresh_solves() {
        use linear_algebra::Isometry3;
        let visual_time = Time::from_nanos(1_000_000_000);
        let solve_time = Time::from_nanos(2_000_000_000);
        let mut tracker = GlobalVisualLockTracker::default();
        let matches = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]].map(|[x, y]| {
            types::visual_localization::FieldMarkAssociation {
                detection: linear_algebra::point![320.0 + 200.0 * x, 240.0 + 200.0 * y],
                field_point: linear_algebra::point![x, y, 1.0],
            }
        });
        tracker.track_associations(visual_time, Isometry3::identity(), Vec::from(matches));
        let mut result = result(solve_time);
        result.latest_visual_measurement_time = Some(visual_time.to_wallclock());
        result.latest_visual_robot_to_field =
            Some(nalgebra::Isometry3::identity().framed_transform());
        assert!(tracker.handle_backend_result(&result));
        assert_eq!(
            tracking_deadline(
                tracking(solve_time),
                &tracker,
                Duration::from_secs(2),
                Duration::from_secs(2)
            ),
            Some(Time::from_nanos(3_000_000_000))
        );
        let lost = lose_track(
            tracking(solve_time),
            &mut tracker,
            Time::from_nanos(3_000_000_000),
        );
        assert!(!tracker.has_backend_result());
        assert_eq!(
            tracking_deadline(
                lost,
                &tracker,
                Duration::from_secs(2),
                Duration::from_secs(2)
            ),
            None
        );
    }
}

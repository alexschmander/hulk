use std::{
    future::ready,
    sync::Arc,
    time::{Duration, Instant},
};

use color_eyre::{Result, eyre::Context as _};
use coordinate_systems::{Camera, Robot};
use linear_algebra::Isometry3;
use projection::intrinsic::Intrinsic;
use ros_z::{
    cache::{Cache, CacheInner},
    parameter::NodeParameters,
    pubsub::Publisher,
    time::{Clock, Time},
};
use types::{
    camera_geometry::{CameraGeometry, camera_geometry_at},
    field_dimensions::FieldDimensions,
    localization::{LocalizationEstimate, LocalizationState3D, LocalizationStatus},
    object_detection::{Object, RobocupObjectLabel},
    time_wrapper::TimeWrapper,
    visual_localization::{AssociationGeometry, GlobalLocalizationDebug, VisualLocalizationFrame},
};

use crate::{
    api::{
        AssociationInput, GlobalAssociationInput, associate_global_visual_features,
        associate_visual_features,
    },
    parameters::FieldMarkAssociationParameters,
};

type DetectedObjects = TimeWrapper<Vec<Object<RobocupObjectLabel>>>;

pub(crate) struct DetectionProcessingContext<'a> {
    pub(crate) parameters: &'a NodeParameters<FieldMarkAssociationParameters>,
    pub(crate) camera_geometry_cache: &'a Cache<TimeWrapper<CameraGeometry>>,
    pub(crate) field_dimensions_cache: &'a Cache<FieldDimensions>,
    pub(crate) estimates: &'a Cache<LocalizationEstimate>,
    pub(crate) status: &'a Cache<LocalizationStatus>,
    pub(crate) attitudes: std::sync::Mutex<CacheInner<TimeWrapper<nalgebra::UnitQuaternion<f32>>>>,
    pub(crate) tracking_reference: std::sync::Mutex<Option<LocalizationEstimate>>,
    pub(crate) associations_publisher: Arc<Publisher<TimeWrapper<VisualLocalizationFrame>>>,
    pub(crate) global_localization_publisher: Arc<Publisher<Option<GlobalLocalizationDebug>>>,
    pub(crate) clock: &'a Clock,
}

impl DetectionProcessingContext<'_> {
    pub(crate) fn record_attitude(&self, time: Time, imu: &booster::ImuState) {
        let rpy = imu.roll_pitch_yaw.inner;
        if rpy.iter().all(|v| v.is_finite()) {
            self.attitudes.lock().unwrap().insert(
                time,
                TimeWrapper {
                    time,
                    inner: nalgebra::UnitQuaternion::from_euler_angles(rpy.x, rpy.y, 0.0),
                },
            );
        }
    }

    fn startup_geometry(
        &self,
        time: Time,
        status: &LocalizationStatus,
    ) -> Option<AssociationGeometry> {
        let attitudes = self.attitudes.lock().unwrap();
        let before = attitudes.get_before(time)?;
        let after = attitudes.get_after(time)?;
        startup_geometry(&before, &after, time, status.epoch)
    }

    fn tracking_reference(
        &self,
        estimate: &LocalizationEstimate,
        status: &LocalizationStatus,
    ) -> Option<LocalizationEstimate> {
        use types::localization::LocalizationState;
        let mut reference = self.tracking_reference.lock().unwrap();
        if reference
            .as_ref()
            .is_some_and(|prior| prior.epoch != status.epoch)
        {
            *reference = None;
        }
        if estimate.epoch == status.epoch && status.state == LocalizationState::Tracking {
            *reference = Some(*estimate);
        } else if reference.is_none() && status.state == LocalizationState::LostTrack {
            // A late-starting consumer may recover a pre-loss anchor from history.
            *reference = self
                .estimates
                .get_interval(status.time - Duration::from_secs(2), status.time)
                .iter()
                .filter(|e| e.epoch == status.epoch && e.robot_to_field.is_some())
                .max_by_key(|e| e.time)
                .map(|e| **e);
        }
        *reference
    }
}

struct PreparedDetectionFrame {
    image_time: Time,
    objects: Vec<Object<RobocupObjectLabel>>,
    robot_to_camera: Isometry3<Robot, Camera>,
    camera_intrinsic: Intrinsic,
    geometry: AssociationGeometry,
    field_dimensions: Arc<FieldDimensions>,
    parameters: Arc<FieldMarkAssociationParameters>,
}

pub(crate) fn keep_latest_detection(
    pending: &mut Option<DetectedObjects>,
    objects: DetectedObjects,
) {
    if pending
        .as_ref()
        .is_none_or(|previous| objects.time > previous.time)
    {
        *pending = Some(objects);
    }
}

// The node polls this future alongside recv; neither solving nor publishing suspends ingestion.
pub(crate) async fn process_detected_objects(
    objects: DetectedObjects,
    ctx: &DetectionProcessingContext<'_>,
) -> Result<()> {
    let Some(frame) = prepare_detection_frame(objects, ctx) else {
        return Ok(());
    };
    let started = Instant::now();
    let (frame, localization) = tokio::task::spawn_blocking(move || {
        let visual_features = crate::find_detected_visual_features(&frame.objects);
        let input = AssociationInput {
            visual_features: &visual_features,
            robot_to_camera: frame.robot_to_camera,
            geometry: &frame.geometry,
            camera_intrinsic: frame.camera_intrinsic,
            field_dimensions: &frame.field_dimensions,
            time: frame.image_time,
        };
        let mut localization = associate_visual_features(input, &frame.parameters);
        if localization.associations.is_empty()
            && matches!(frame.geometry.state, LocalizationState3D::LostTrack { .. })
        {
            localization = associate_global_visual_features(GlobalAssociationInput {
                visual_features: &visual_features,
                robot_to_local: frame.geometry.robot_to_local,
                robot_to_camera: frame.robot_to_camera,
                camera_intrinsic: frame.camera_intrinsic,
                field_dimensions: &frame.field_dimensions,
                parameters: &frame.parameters.global_localizer,
            });
        }
        (frame, localization)
    })
    .await
    .wrap_err("field-mark association worker failed")?;

    // A blocking job cannot be cancelled: retain its slot until it returns, then discard late work.
    // Reuse the calibrated age limit for the wall-time budget, including blocking-pool queue time.
    let max_age = frame.parameters.max_pose_hint_age;
    if started.elapsed() > max_age || !frame_is_current(&frame, ctx, max_age) {
        return Ok(());
    }
    if !localization.associations.is_empty() {
        let publisher = Arc::clone(&ctx.associations_publisher);
        let message = TimeWrapper {
            time: frame.image_time,
            inner: VisualLocalizationFrame {
                epoch: frame.geometry.epoch,
                source: localization.source,
                robot_to_camera: frame.robot_to_camera,
                robot_to_local: frame.geometry.robot_to_local,
                camera_intrinsic: frame.camera_intrinsic,
                associations: localization.associations,
            },
        };
        let runtime = tokio::runtime::Handle::current();
        // Await each blocking send before starting another: publication keeps the single frame
        // slot, but never occupies an ingestion worker. Cancellation cannot abort an active send.
        tokio::task::spawn_blocking(move || runtime.block_on(publisher.publish(&message)))
            .await
            .wrap_err("field-mark association publisher failed")??;
    }
    if started.elapsed() <= max_age && frame_is_current(&frame, ctx, max_age) {
        let publisher = Arc::clone(&ctx.global_localization_publisher);
        let runtime = tokio::runtime::Handle::current();
        tokio::task::spawn_blocking(move || {
            runtime.block_on(publisher.publish_if_subscribed(|| ready(localization.debug)))
        })
        .await
        .wrap_err("field-mark association debug publisher failed")??;
    }
    Ok(())
}

fn prepare_detection_frame(
    objects: DetectedObjects,
    ctx: &DetectionProcessingContext<'_>,
) -> Option<PreparedDetectionFrame> {
    let image_time = objects.time;
    let camera = camera_geometry_at(ctx.camera_geometry_cache, image_time)?;
    let field_dimensions = ctx.field_dimensions_cache.get_latest()?;
    let parameters = ctx.parameters.snapshot().typed.clone();
    let status = ctx.status.get_latest()?;
    let geometry = if status.state == types::localization::LocalizationState::Startup {
        ctx.startup_geometry(image_time, &status)?
    } else {
        let latest_estimate = ctx.estimates.get_latest()?;
        let reference = ctx.tracking_reference(&latest_estimate, &status);
        let latest = TimeWrapper {
            time: latest_estimate.time,
            inner: AssociationGeometry::from_estimate(
                &latest_estimate,
                &status,
                reference.as_ref(),
            )?,
        };
        let history: Vec<_> = ctx
            .estimates
            .get_interval(
                image_time - parameters.max_pose_hint_age,
                image_time + parameters.max_pose_hint_age,
            )
            .iter()
            .filter_map(|estimate| {
                Some(Arc::new(TimeWrapper {
                    time: estimate.time,
                    inner: AssociationGeometry::from_estimate(
                        estimate,
                        &status,
                        reference.as_ref(),
                    )?,
                }))
            })
            .collect();
        geometry_for_image(image_time, &history, &latest, parameters.max_pose_hint_age)?
    };
    let frame = PreparedDetectionFrame {
        image_time,
        objects: objects.inner,
        robot_to_camera: camera.robot_to_camera,
        camera_intrinsic: camera.intrinsics,
        geometry,
        field_dimensions,
        parameters,
    };
    frame_is_current(&frame, ctx, frame.parameters.max_pose_hint_age).then_some(frame)
}

fn startup_geometry(
    before: &TimeWrapper<nalgebra::UnitQuaternion<f32>>,
    after: &TimeWrapper<nalgebra::UnitQuaternion<f32>>,
    time: Time,
    epoch: u64,
) -> Option<AssociationGeometry> {
    if time < before.time || time > after.time {
        return None;
    }
    let gap = after.time.duration_since(before.time);
    if gap > types::camera_geometry::MAX_CAMERA_GEOMETRY_GAP {
        return None;
    }
    let mut orientation = if gap.is_zero() {
        before.inner
    } else {
        before.inner.slerp(
            &after.inner,
            time.duration_since(before.time).as_secs_f32() / gap.as_secs_f32(),
        )
    };
    orientation.renormalize();
    Some(AssociationGeometry {
        epoch,
        state: LocalizationState3D::Startup,
        robot_to_local: Isometry3::wrap(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::identity(),
            orientation,
        )),
        local_to_field: None,
    })
}

fn geometry_for_image(
    image_time: Time,
    history: &[Arc<TimeWrapper<AssociationGeometry>>],
    latest: &TimeWrapper<AssociationGeometry>,
    max_age: Duration,
) -> Option<AssociationGeometry> {
    // Select complete snapshots; never attach a new state/covariance to an old branch's pose.
    // Include latest because the independently populated history may not contain it yet.
    std::iter::once(latest)
        .chain(history.iter().rev().map(Arc::as_ref))
        .filter(|geometry| {
            geometry.time.abs_diff(image_time) <= max_age
                && same_lifecycle(&geometry.inner, &latest.inner)
        })
        .min_by_key(|geometry| (geometry.time.abs_diff(image_time), geometry.time))
        .map(|geometry| geometry.inner.clone())
}

fn frame_is_current(
    frame: &PreparedDetectionFrame,
    ctx: &DetectionProcessingContext<'_>,
    max_age: Duration,
) -> bool {
    // Both sources replace their immutable Arc on update, including runtime parameter reloads.
    if !Arc::ptr_eq(&frame.parameters, &ctx.parameters.snapshot().typed)
        || ctx
            .field_dimensions_cache
            .get_latest()
            .is_none_or(|dimensions| !Arc::ptr_eq(&frame.field_dimensions, &dimensions))
    {
        return false;
    }
    if matches!(frame.geometry.state, LocalizationState3D::Startup) {
        return frame.image_time.abs_diff(ctx.clock.now()) <= max_age
            && ctx.status.get_latest().is_some_and(|status| {
                status.epoch == frame.geometry.epoch
                    && status.state == types::localization::LocalizationState::Startup
            });
    }
    let latest = ctx
        .estimates
        .get_latest()
        .zip(ctx.status.get_latest())
        .and_then(|(estimate, status)| {
            Some(TimeWrapper {
                time: estimate.time,
                inner: AssociationGeometry::from_estimate(
                    &estimate,
                    &status,
                    ctx.tracking_reference(&estimate, &status).as_ref(),
                )?,
            })
        });
    result_is_current(
        frame.image_time,
        &frame.geometry,
        latest.as_ref(),
        ctx.clock.now(),
        max_age,
    )
}

fn result_is_current(
    image_time: Time,
    geometry: &AssociationGeometry,
    latest: Option<&TimeWrapper<AssociationGeometry>>,
    now: Time,
    max_age: Duration,
) -> bool {
    image_time.abs_diff(now) <= max_age
        && latest.is_some_and(|latest| {
            latest.time.abs_diff(now) <= max_age && same_lifecycle(geometry, &latest.inner)
        })
}

fn same_lifecycle(geometry: &AssociationGeometry, latest: &AssociationGeometry) -> bool {
    geometry.epoch == latest.epoch
        && matches!(
            (geometry.state, latest.state),
            (LocalizationState3D::Startup, LocalizationState3D::Startup)
                | (
                    LocalizationState3D::Tracking { .. },
                    LocalizationState3D::Tracking { .. }
                )
                | (
                    LocalizationState3D::LostTrack { .. },
                    LocalizationState3D::LostTrack { .. }
                )
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::localization::LocalizationEstimate3D;

    #[test]
    fn startup_uses_bracketed_imu_without_a_localization_estimate() {
        let before = TimeWrapper {
            time: Time::from_nanos(0),
            inner: nalgebra::UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0),
        };
        let after = TimeWrapper {
            time: Time::from_nanos(10_000_000),
            inner: nalgebra::UnitQuaternion::from_euler_angles(0.2, 0.0, 0.0),
        };
        let time = Time::from_nanos(5_000_000);
        let geometry = startup_geometry(&before, &after, time, 7).unwrap();
        assert_eq!(geometry.epoch, 7);
        assert!(matches!(geometry.state, LocalizationState3D::Startup));
        assert_eq!(
            geometry.robot_to_local.inner.translation.vector,
            nalgebra::Vector3::zeros()
        );
        assert!((geometry.robot_to_local.inner.rotation.euler_angles().0 - 0.1).abs() < 1e-6);
        assert!(startup_geometry(&before, &after, Time::from_nanos(11_000_000), 7).is_none());
        let late = TimeWrapper {
            time: Time::from_nanos(30_000_000),
            ..after
        };
        assert!(startup_geometry(&before, &late, time, 7).is_none());
    }

    #[test]
    fn pending_detection_keeps_only_the_newest_timestamp() {
        let mut pending = None;
        for nanos in [1, 3, 2, 4, 4] {
            keep_latest_detection(
                &mut pending,
                TimeWrapper {
                    time: Time::from_nanos(nanos),
                    inner: Vec::new(),
                },
            );
        }
        assert_eq!(pending.take().unwrap().time, Time::from_nanos(4));
        assert!(pending.is_none());
    }

    #[test]
    fn newer_publication_with_older_pose_invalidates_tracking_and_supplies_coherent_geometry() {
        use ros_z::cache::CacheInner;

        let time = Time::from_nanos(1_000_000_000);
        let age = Duration::from_millis(250);
        let estimate = LocalizationEstimate3D {
            robot_to_field: Isometry3::identity(),
            covariance: nalgebra::SMatrix::identity(),
        };
        let tracking = TimeWrapper {
            time,
            inner: AssociationGeometry {
                epoch: 7,
                state: LocalizationState3D::Tracking {
                    estimate,
                    last_successful_solve: time,
                },
                robot_to_local: Isometry3::identity(),
                local_to_field: None,
            },
        };
        let mut lost = tracking.clone();
        lost.time = time - Duration::from_millis(100);
        lost.inner.state = LocalizationState3D::LostTrack {
            last_known_estimate: LocalizationEstimate3D {
                covariance: estimate.covariance * 2.0,
                ..estimate
            },
            last_successful_solve: lost.time,
        };
        lost.inner.robot_to_local = Isometry3::from_translation(1.0, 2.0, 0.0);

        let mut history = CacheInner::new(128);
        history.insert(tracking.time, tracking.clone());
        history.insert(lost.time, lost.clone());
        let mut publications = CacheInner::new(1);
        publications.insert(time, tracking.clone());
        publications.insert(time + Duration::from_millis(1), lost.clone());
        let latest = publications.get_latest().unwrap();
        assert!(result_is_current(
            time,
            &tracking.inner,
            history.get_latest().as_deref(),
            time,
            age
        ));
        assert!(!result_is_current(
            time,
            &tracking.inner,
            Some(&latest),
            time,
            age
        ));

        let selected = geometry_for_image(
            time,
            &history.get_interval(time - age, time + age),
            &latest,
            age,
        )
        .unwrap();
        assert_eq!(selected.state, lost.inner.state);
        assert_eq!(selected.robot_to_local, lost.inner.robot_to_local);
        // Latest remains usable before the history subscriber receives the transition.
        let old_branch = [Arc::new(tracking)];
        assert_eq!(
            geometry_for_image(time, &old_branch, &latest, age)
                .unwrap()
                .state,
            lost.inner.state
        );
        lost.time = time - age - Duration::from_nanos(1);
        assert!(geometry_for_image(time, &old_branch, &lost, age).is_none());
    }

    #[test]
    fn late_results_require_fresh_matching_epoch_and_explicit_state() {
        let time = Time::from_nanos(1_000_000_000);
        let age = Duration::from_millis(250);
        let estimate = LocalizationEstimate3D {
            robot_to_field: Isometry3::identity(),
            covariance: nalgebra::SMatrix::identity(),
        };
        let states = [
            LocalizationState3D::Startup,
            LocalizationState3D::Tracking {
                estimate,
                last_successful_solve: time,
            },
            LocalizationState3D::LostTrack {
                last_known_estimate: estimate,
                last_successful_solve: time,
            },
        ];
        for (index, state) in states.into_iter().enumerate() {
            let geometry = AssociationGeometry {
                epoch: 7,
                state,
                robot_to_local: Isometry3::identity(),
                local_to_field: None,
            };
            let mut latest = TimeWrapper {
                time,
                inner: geometry.clone(),
            };
            let valid = |latest: Option<&TimeWrapper<AssociationGeometry>>, now| {
                result_is_current(time, &geometry, latest, now, age)
            };
            for (latest_index, latest_state) in states.into_iter().enumerate() {
                latest.inner.state = latest_state;
                assert_eq!(valid(Some(&latest), time), index == latest_index);
            }
            latest.inner.state = state;
            let expired = age + Duration::from_nanos(1);
            assert!(valid(Some(&latest), time + age));
            assert!(!valid(None, time));
            assert!(!valid(Some(&latest), time + expired));
            assert!(!valid(Some(&latest), time - expired));
            latest.time = time - expired;
            assert!(!valid(Some(&latest), time));
            latest.time = time;
            latest.inner.epoch += 1;
            assert!(!valid(Some(&latest), time));
        }
    }
}

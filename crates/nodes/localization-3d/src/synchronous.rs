use std::time::Duration;

use booster::ImuState;
use color_eyre::{Result, eyre::eyre};
use coordinate_systems::{Field, Local, Robot};
use linear_algebra::Isometry3;
use localization_factrs::{InitialState, VinsBackend, VinsFrontend, VinsFrontendError, initialize};
use projection::camera_matrix::CameraMatrix;
use ros_z::time::Time;
use types::{
    field_dimensions::FieldDimensions,
    localization::LocalizationState3D,
    time_wrapper::TimeWrapper,
    visual_localization::{AssociationGeometry, VisualLocalizationFrame},
    visual_odometry::{VisualOdometer, VisualOdometryDelta},
};

use crate::{
    camera::camera_intrinsics_from_matrix,
    diagnostics::SolveDiagnostics,
    event_handlers::{
        AcceptanceWindow, accept_backend_result, backend_result_is_eligible,
        handle_odometer_discontinuity, lose_track, tracking_deadline,
    },
    ingest::ingest_visual_odometry,
    live_odometry::{LiveVisualOdometryLocalization, PoseCorrection},
    parameters::{
        Localization3dParameters, backend_configuration_from_parameters_and_field_dimensions,
    },
    pose::backend_localization_for_result,
    publish::field_to_robot,
    visual_localization::{
        GlobalVisualLock, GlobalVisualLockTracker, handle_visual_localization_frame,
    },
};

#[derive(Clone, Debug)]
pub struct SynchronousLocalizationOutput {
    pub raw_backend_robot_to_field: Option<Isometry3<Robot, Field, f64>>,
    pub backend_field_to_robot: Option<Isometry3<Field, Robot>>,
    pub diagnostics: SolveDiagnostics,
}

pub struct SynchronousLocalization {
    frontend: VinsFrontend,
    backend: VinsBackend,
    live: LiveVisualOdometryLocalization,
    visual_lock: GlobalVisualLockTracker,
    epoch: u64,
    state: LocalizationState3D,
    now: Time,
    tracking_timeout: Duration,
    visual_tracking_timeout: Duration,
}

impl SynchronousLocalization {
    pub fn new(
        parameters: &Localization3dParameters,
        field_dimensions: &FieldDimensions,
        initial_camera_matrix: &CameraMatrix,
        initial_robot_to_local: Isometry3<Robot, Local>,
    ) -> Result<Self> {
        parameters.validate().map_err(|message| eyre!(message))?;
        let initial_state = InitialState::from_robot_to_local_and_intrinsics(
            initial_robot_to_local,
            camera_intrinsics_from_matrix(initial_camera_matrix),
        );
        let (frontend, backend) = initialize(
            backend_configuration_from_parameters_and_field_dimensions(
                parameters,
                field_dimensions,
            ),
            initial_state,
        );
        let mut live = LiveVisualOdometryLocalization::default();
        live.set_initial(Time::from_nanos(0), initial_robot_to_local);
        Ok(Self {
            frontend,
            backend,
            live,
            visual_lock: GlobalVisualLockTracker::default(),
            epoch: 0,
            state: LocalizationState3D::Startup,
            now: Time::from_nanos(0),
            tracking_timeout: parameters.tracking_timeout,
            visual_tracking_timeout: parameters.visual_tracking_timeout,
        })
    }

    pub fn global_visual_lock(&self) -> GlobalVisualLock {
        self.visual_lock.status()
    }

    pub fn state(&self) -> LocalizationState3D {
        self.state
    }

    pub fn association_geometry(&self) -> Option<TimeWrapper<AssociationGeometry>> {
        self.live.association_geometry(self.state, self.epoch)
    }

    /// Updates next-factor weights and lifecycle timeouts without resetting poses or alignment.
    pub fn set_parameters(
        &mut self,
        parameters: &Localization3dParameters,
        field_dimensions: &FieldDimensions,
    ) -> Result<()> {
        parameters.validate().map_err(|message| eyre!(message))?;
        self.frontend.update_configuration(
            backend_configuration_from_parameters_and_field_dimensions(
                parameters,
                field_dimensions,
            ),
        )?;
        self.tracking_timeout = parameters.tracking_timeout;
        self.visual_tracking_timeout = parameters.visual_tracking_timeout;
        self.advance_time(self.now);
        Ok(())
    }

    /// Advances lifecycle time even when no measurements or backend results arrive.
    pub fn advance_time(&mut self, now: Time) {
        self.now = self.now.max(now);
        if tracking_deadline(
            self.state,
            &self.visual_lock,
            self.tracking_timeout,
            self.visual_tracking_timeout,
        )
        .is_some_and(|deadline| deadline <= self.now)
        {
            self.state = lose_track(self.state, &mut self.visual_lock, self.now);
        }
    }

    pub fn ingest_imu(&mut self, time: Time, imu: ImuState) -> Result<(), VinsFrontendError> {
        self.advance_time(time);
        self.frontend.ingest_imu(time.to_wallclock(), imu)
    }

    pub fn ingest_visual_odometry(
        &mut self,
        delta: VisualOdometryDelta,
        previous_camera_matrix: &CameraMatrix,
        current_camera_matrix: &CameraMatrix,
    ) -> Result<(), VinsFrontendError> {
        self.advance_time(delta.current_time);
        ingest_visual_odometry(
            &mut self.frontend,
            delta,
            previous_camera_matrix,
            current_camera_matrix,
        )
    }

    pub fn ingest_visual_localization_frame(
        &mut self,
        frame: TimeWrapper<VisualLocalizationFrame>,
    ) -> Result<(), VinsFrontendError> {
        self.advance_time(frame.time);
        handle_visual_localization_frame(
            &mut self.frontend,
            &mut self.visual_lock,
            self.epoch,
            self.state,
            self.now,
            self.visual_tracking_timeout,
            frame,
        )
    }

    pub fn solve_once(
        &mut self,
        camera_matrix: TimeWrapper<&CameraMatrix>,
        exact_visual_odometer: Option<&VisualOdometer>,
    ) -> Result<Option<SynchronousLocalizationOutput>> {
        self.advance_time(camera_matrix.time);
        let Some(backend_result) = self.backend.solve_once()? else {
            return Ok(None);
        };
        let result = localization_factrs::OptimizationResult::from(&backend_result);
        if !backend_result_is_eligible(&result, self.state, self.acceptance_window()) {
            return Ok(None);
        }
        let time = Time::from_wallclock(result.time);
        if camera_matrix.time != time {
            return Err(eyre!("camera matrix timestamp does not match checkpoint"));
        }
        if exact_visual_odometer.is_some_and(|odometer| odometer.time != time) {
            return Err(eyre!("visual odometer timestamp does not match checkpoint"));
        }
        if let Some(odometer) = exact_visual_odometer {
            self.update_live_odometry(
                TimeWrapper {
                    time,
                    inner: camera_matrix.inner,
                },
                odometer,
            )?;
        }

        let raw_backend_robot_to_field = result.robot_to_field;
        let correction = self.accept_result(&result);
        if let Some(correction) = correction
            && let Some(odometer) = exact_visual_odometer
        {
            self.live
                .reset_with_exact_samples(correction, odometer, camera_matrix.inner);
        }
        let backend_field_to_robot = (correction.is_some()
            && matches!(self.state, LocalizationState3D::Tracking { .. }))
        .then(|| backend_localization_for_result(&result))
        .flatten();
        let diagnostics = self
            .backend
            .compute_last_solve_diagnostics()
            .ok_or_else(|| eyre!("diagnostics are unavailable after a backend result"))?
            .into();
        Ok(Some(SynchronousLocalizationOutput {
            raw_backend_robot_to_field,
            backend_field_to_robot,
            diagnostics,
        }))
    }

    fn acceptance_window(&self) -> AcceptanceWindow {
        AcceptanceWindow {
            epoch: self.epoch,
            epoch_start: Time::from_nanos(0),
            now: self.now,
            tracking_timeout: self.tracking_timeout,
            visual_tracking_timeout: self.visual_tracking_timeout,
        }
    }

    fn accept_result(
        &mut self,
        result: &localization_factrs::OptimizationResult,
    ) -> Option<PoseCorrection> {
        let window = self.acceptance_window();
        let acceptance = accept_backend_result(result, self.state, &mut self.visual_lock, window)?;
        self.state = acceptance.state;
        if let Some(correction) = acceptance.correction {
            self.live.defer_reset(correction);
        }
        acceptance.correction
    }

    pub fn update_live_odometry(
        &mut self,
        current_camera_matrix: TimeWrapper<&CameraMatrix>,
        current_visual_odometer: &VisualOdometer,
    ) -> Result<Option<Isometry3<Field, Robot>>> {
        if current_camera_matrix.time != current_visual_odometer.time {
            return Err(eyre!(
                "camera matrix timestamp does not match visual odometer"
            ));
        }
        self.advance_time(current_visual_odometer.time);
        handle_odometer_discontinuity(
            &self.live,
            &mut self.state,
            &mut self.visual_lock,
            current_visual_odometer,
            self.now,
        );
        if self
            .live
            .update_with_exact_sample(current_visual_odometer, current_camera_matrix.inner)
            .is_none()
        {
            return Ok(None);
        }
        let Some(geometry) = self.association_geometry() else {
            return Ok(None);
        };
        Ok((geometry.time == current_visual_odometer.time)
            .then(|| field_to_robot(&geometry.inner))
            .flatten())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linear_algebra::{IntoTransform, point};
    use projection::intrinsic::Intrinsic;
    use std::time::Duration;

    fn parameters() -> Localization3dParameters {
        Localization3dParameters {
            accelerometer_process_noise_variance: 10.0,
            visual_feature_noise_variance: 1.0,
            field_containment_sigma: 0.1,
            tracking_timeout: Duration::from_secs(2),
            visual_tracking_timeout: Duration::from_secs(2),
        }
    }

    fn camera() -> CameraMatrix {
        CameraMatrix {
            intrinsics: Intrinsic::new(nalgebra::vector![220.0, 220.0], point![320.0, 240.0]),
            ..Default::default()
        }
    }

    fn localization() -> SynchronousLocalization {
        SynchronousLocalization::new(
            &parameters(),
            &FieldDimensions::SPL_2025,
            &camera(),
            nalgebra::Isometry3::translation(0.0, 0.0, 0.5).framed_transform(),
        )
        .unwrap()
    }

    fn frame(epoch: u64, time: Time) -> TimeWrapper<VisualLocalizationFrame> {
        let robot_to_camera =
            nalgebra::Isometry3::rotation(nalgebra::vector![std::f32::consts::PI, 0.0, 0.0,])
                .framed_transform();
        let associations = [
            ([320.0, 240.0], [-1.0, 0.0, 0.0]),
            ([540.0, 240.0], [-0.5, 0.0, 0.0]),
            ([320.0, 460.0], [-1.0, -0.5, 0.0]),
        ]
        .into_iter()
        .map(
            |(pixel, field)| types::visual_localization::FieldMarkAssociation {
                detection: linear_algebra::point![<coordinate_systems::Pixel>, pixel[0], pixel[1]],
                field_point: linear_algebra::point![<Field>, field[0], field[1], field[2]],
            },
        )
        .collect();
        TimeWrapper {
            time,
            inner: VisualLocalizationFrame {
                epoch,
                robot_to_camera,
                associations,
                robot_to_local: nalgebra::Isometry3::translation(0.0, 0.0, 0.5).framed_transform(),
                camera_intrinsic: camera().intrinsics,
            },
        }
    }

    fn update_odometry(
        localization: &mut SynchronousLocalization,
        camera: &CameraMatrix,
        odometer: &VisualOdometer,
    ) -> Option<Isometry3<Field, Robot>> {
        localization
            .update_live_odometry(
                TimeWrapper {
                    time: odometer.time,
                    inner: camera,
                },
                odometer,
            )
            .unwrap()
    }

    #[test]
    fn mismatched_visual_epoch_is_ignored() {
        let mut localization = localization();
        localization
            .ingest_visual_localization_frame(frame(1, Time::from_nanos(1)))
            .unwrap();
        assert_eq!(
            localization.global_visual_lock(),
            GlobalVisualLock::Unlocked
        );
    }

    #[test]
    fn expired_startup_frame_cannot_seed_alignment() {
        let mut localization = localization();
        localization.advance_time(Time::from_nanos(10_000_000_000));
        for time in [1_000_000_000, 8_000_000_000] {
            localization
                .ingest_visual_localization_frame(frame(0, Time::from_nanos(time)))
                .unwrap();
            assert_eq!(
                localization.global_visual_lock(),
                GlobalVisualLock::Unlocked
            );
            assert!(localization.backend.solve_once().unwrap().is_none());
        }
    }

    #[test]
    fn matching_visual_epoch_waits_for_converged_backend() {
        let mut localization = localization();
        localization
            .ingest_visual_localization_frame(frame(0, Time::from_nanos(1)))
            .unwrap();
        assert_eq!(
            localization.global_visual_lock(),
            GlobalVisualLock::WaitingForBackend
        );
    }

    #[test]
    fn unaligned_imu_solve_has_no_global_pose() {
        let mut localization = localization();
        let time = Time::from_nanos(1_000_000_000);
        localization.ingest_imu(time, ImuState::default()).unwrap();
        let output = localization
            .solve_once(
                TimeWrapper {
                    time,
                    inner: &camera(),
                },
                None,
            )
            .unwrap()
            .unwrap();
        assert!(output.raw_backend_robot_to_field.is_none());
        assert!(output.backend_field_to_robot.is_none());
    }

    fn acknowledged_result(time: Time) -> localization_factrs::OptimizationResult {
        let pose = nalgebra::Isometry3::translation(0.0, 0.0, 0.5).framed_transform();
        let field_pose = nalgebra::Isometry3::translation(-1.0, 0.0, 0.5).framed_transform();
        localization_factrs::OptimizationResult {
            robot_to_local: pose,
            local_to_field: Some(nalgebra::Isometry2::translation(-1.0, 0.0).framed_transform()),
            robot_to_field: Some(field_pose),
            camera_intrinsics: camera_intrinsics_from_matrix(&camera()),
            latest_visual_measurement_time: Some(time.to_wallclock()),
            latest_visual_robot_to_local: Some(pose),
            latest_visual_robot_to_field: Some(field_pose),
            ..crate::test_result(time)
        }
    }

    #[test]
    fn parameter_updates_preserve_pose_and_apply_timeouts_without_reset() {
        let mut localization = localization();
        let time = Time::from_nanos(1_000_000_000);
        localization
            .ingest_visual_localization_frame(frame(0, time))
            .unwrap();
        assert!(
            localization
                .accept_result(&acknowledged_result(time))
                .is_some()
        );
        localization.backend.solve_once().unwrap();
        let before = localization.association_geometry().unwrap();
        let values = localization.backend.values().len();
        let mut updated = parameters();
        updated.accelerometer_process_noise_variance *= 2.0;
        updated.visual_feature_noise_variance *= 3.0;
        updated.field_containment_sigma *= 2.0;
        updated.tracking_timeout = Duration::from_secs(4);
        updated.visual_tracking_timeout = Duration::from_secs(3);
        localization
            .set_parameters(&updated, &FieldDimensions::SPL_2025)
            .unwrap();
        assert!(localization.backend.solve_once().unwrap().is_none());
        assert_eq!(localization.backend.values().len(), values);
        let after = localization.association_geometry().unwrap();
        assert_eq!(after.time, before.time);
        assert_eq!(after.inner.epoch, before.inner.epoch);
        assert_eq!(after.inner.state, before.inner.state);
        assert_eq!(after.inner.robot_to_local, before.inner.robot_to_local);
        assert_eq!(after.inner.local_to_field, before.inner.local_to_field);

        localization.advance_time(time.saturating_add(Duration::from_secs(2)));
        assert_eq!(localization.state(), before.inner.state);
        let mut invalid = updated.clone();
        invalid.visual_feature_noise_variance = f64::NAN;
        invalid.tracking_timeout = Duration::from_secs(1);
        assert!(
            localization
                .set_parameters(&invalid, &FieldDimensions::SPL_2025)
                .is_err()
        );
        assert_eq!(localization.tracking_timeout, updated.tracking_timeout);
        assert_eq!(localization.state(), before.inner.state);

        updated.visual_tracking_timeout = Duration::from_secs(1);
        localization
            .set_parameters(&updated, &FieldDimensions::SPL_2025)
            .unwrap();
        assert!(matches!(
            localization.state(),
            LocalizationState3D::LostTrack { .. }
        ));
        let lost = localization.association_geometry().unwrap();
        assert_eq!(lost.time, time);
        assert_eq!(lost.inner.robot_to_local, before.inner.robot_to_local);
        assert_eq!(lost.inner.local_to_field, before.inner.local_to_field);
    }

    #[test]
    fn deferred_anchor_keeps_backend_geometry_and_replays_measured_motion() {
        let mut localization = localization();
        let camera = camera();
        let mut odometer = VisualOdometer {
            time: Time::from_nanos(10),
            epoch: 0,
            current_left_camera_to_visual_odometer: nalgebra::Isometry3::identity(),
        };
        update_odometry(&mut localization, &camera, &odometer);
        odometer.time = Time::from_nanos(30);
        odometer
            .current_left_camera_to_visual_odometer
            .translation
            .vector
            .x = 0.4;
        update_odometry(&mut localization, &camera, &odometer);

        let time = Time::from_nanos(20);
        localization
            .ingest_visual_localization_frame(frame(0, time))
            .unwrap();
        assert!(
            localization
                .accept_result(&acknowledged_result(time))
                .is_some()
        );
        let backend_geometry = localization.association_geometry().unwrap().inner;
        assert!(matches!(
            localization.state(),
            LocalizationState3D::Tracking { .. }
        ));
        assert!(backend_geometry.local_to_field.is_some());
        assert_eq!(localization.live.latest_time(), Some(Time::from_nanos(30)));
        assert_eq!(localization.association_geometry().unwrap().time, time);

        odometer.time = Time::from_nanos(15);
        assert!(update_odometry(&mut localization, &camera, &odometer).is_none());
        assert_eq!(localization.live.latest_time(), Some(Time::from_nanos(30)));
        odometer.time = Time::from_nanos(40);
        odometer
            .current_left_camera_to_visual_odometer
            .translation
            .vector
            .x = 0.8;
        assert!(update_odometry(&mut localization, &camera, &odometer).is_none());
        let pending = localization.association_geometry().unwrap();
        assert_eq!(
            pending.inner.robot_to_local,
            backend_geometry.robot_to_local
        );
        assert_eq!(
            pending.inner.local_to_field,
            backend_geometry.local_to_field
        );
        assert_eq!(pending.time, time);

        // The late exact sample installs the correction and replays the retained sample at 40.
        odometer.time = time;
        odometer
            .current_left_camera_to_visual_odometer
            .translation
            .vector
            .x = 0.2;
        update_odometry(&mut localization, &camera, &odometer);
        let replayed = localization.association_geometry().unwrap();
        assert_eq!(replayed.time, Time::from_nanos(40));
        assert!((replayed.inner.robot_to_local.translation().x() - 0.6).abs() < 1.0e-6);
        odometer.time = Time::from_nanos(50);
        odometer
            .current_left_camera_to_visual_odometer
            .translation
            .vector
            .x = 1.0;
        let global = update_odometry(&mut localization, &camera, &odometer)
            .unwrap()
            .inverse();
        assert!((global.translation().x() + 0.2).abs() < 1.0e-6);
    }

    #[test]
    fn deferred_checkpoint_across_vo_reset_keeps_lost_geometry_live_until_fresh_ack() {
        let mut localization = localization();
        let camera = camera();
        let start = Time::from_nanos(1_000_000_000);
        let mut odometer = VisualOdometer {
            time: start,
            epoch: 0,
            current_left_camera_to_visual_odometer: nalgebra::Isometry3::identity(),
        };
        update_odometry(&mut localization, &camera, &odometer);
        odometer.time = start.saturating_add(Duration::from_millis(10));
        odometer
            .current_left_camera_to_visual_odometer
            .translation
            .vector
            .x = 0.4;
        update_odometry(&mut localization, &camera, &odometer);
        let live = localization.live.latest().unwrap();
        assert!((live.robot_to_local.translation().x() - 0.4).abs() < 1.0e-6);
        assert!(live.local_to_field.is_none());

        // No sample can anchor this checkpoint: it lies between the two VO epochs.
        let time = start.saturating_add(Duration::from_millis(20));
        localization
            .ingest_visual_localization_frame(frame(0, time))
            .unwrap();
        assert!(
            localization
                .accept_result(&acknowledged_result(time))
                .is_some()
        );
        let LocalizationState3D::Tracking {
            estimate,
            last_successful_solve,
        } = localization.state()
        else {
            panic!("expected tracking")
        };
        let prior = localization.association_geometry().unwrap().inner;
        assert_eq!(localization.association_geometry().unwrap().time, time);
        assert_eq!(localization.live.latest_time(), Some(odometer.time));
        odometer.time = start.saturating_add(Duration::from_millis(30));
        odometer.epoch += 1;
        // A discontinuous odometer coordinate must not become motion across the reset.
        odometer
            .current_left_camera_to_visual_odometer
            .translation
            .vector
            .x = 100.0;
        assert!(update_odometry(&mut localization, &camera, &odometer).is_none());
        let lost = LocalizationState3D::LostTrack {
            last_known_estimate: estimate,
            last_successful_solve,
        };
        assert_eq!(localization.state(), lost);
        assert_eq!(
            localization.global_visual_lock(),
            GlobalVisualLock::Unlocked
        );
        let reset = localization.association_geometry().unwrap();
        assert_eq!(reset.inner.robot_to_local, prior.robot_to_local);
        assert_eq!(reset.inner.local_to_field, prior.local_to_field);
        assert_eq!(reset.time, odometer.time);
        let mut old_ack = acknowledged_result(time);
        old_ack.time = odometer.time.to_wallclock();
        assert!(localization.accept_result(&old_ack).is_none());
        assert_eq!(localization.state(), lost);

        odometer.time = odometer.time.saturating_add(Duration::from_millis(300));
        odometer
            .current_left_camera_to_visual_odometer
            .translation
            .vector
            .x += 0.25;
        assert!(update_odometry(&mut localization, &camera, &odometer).is_none());
        assert!(odometer.time.abs_diff(time) > Duration::from_millis(250));
        let moved = localization.association_geometry().unwrap();
        assert_eq!(moved.time, odometer.time);
        assert!((moved.inner.robot_to_local.translation().x() - 0.25).abs() < 1.0e-6);
        assert_eq!(moved.inner.local_to_field, prior.local_to_field);
        assert_eq!(moved.inner.state, lost);
        assert_eq!(localization.state(), lost);
        assert_eq!(
            localization.global_visual_lock(),
            GlobalVisualLock::Unlocked
        );

        let fresh = odometer.time.saturating_add(Duration::from_millis(1));
        localization
            .ingest_visual_localization_frame(frame(0, fresh))
            .unwrap();
        assert_eq!(localization.state(), lost);
        assert_eq!(
            localization.global_visual_lock(),
            GlobalVisualLock::WaitingForBackend
        );
        assert!(
            localization
                .accept_result(&acknowledged_result(fresh))
                .is_some()
        );
        assert!(matches!(
            localization.state(),
            LocalizationState3D::Tracking { .. }
        ));
    }

    #[test]
    fn visual_expiration_keeps_trusted_geometry_and_requires_fresh_acknowledgement() {
        let mut localization = localization();
        localization.tracking_timeout = Duration::from_secs(4);
        let time = Time::from_nanos(1_000_000_000);
        localization
            .ingest_visual_localization_frame(frame(0, time))
            .unwrap();
        let mut result = acknowledged_result(time);
        assert!(localization.accept_result(&result).is_some());
        let LocalizationState3D::Tracking {
            estimate,
            last_successful_solve,
        } = localization.state()
        else {
            panic!("valid visual acknowledgement must establish tracking");
        };
        assert_eq!(last_successful_solve, time);
        let alignment = localization
            .association_geometry()
            .unwrap()
            .inner
            .local_to_field;

        let later = Time::from_nanos(2_000_000_000);
        localization.advance_time(later);
        result.time = later.to_wallclock();
        assert!(localization.accept_result(&result).is_some());
        let loss = Time::from_nanos(3_000_000_000);
        localization.advance_time(loss);
        let expected = LocalizationState3D::LostTrack {
            last_known_estimate: estimate,
            last_successful_solve: later,
        };
        assert_eq!(localization.state(), expected);
        let expired = localization.association_geometry().unwrap();
        assert_eq!(expired.inner.state, expected);
        assert_eq!(expired.inner.local_to_field, alignment);
        assert_eq!(
            localization.global_visual_lock(),
            GlobalVisualLock::Unlocked
        );

        localization
            .ingest_visual_localization_frame(frame(0, time))
            .unwrap();
        assert_eq!(
            localization.global_visual_lock(),
            GlobalVisualLock::Unlocked
        );
        result.time = loss.to_wallclock();
        result
            .robot_to_field
            .as_mut()
            .unwrap()
            .inner
            .translation
            .vector
            .x = 50.0;
        assert!(localization.accept_result(&result).is_none());
        assert_eq!(localization.state(), expected);

        let fresh = Time::from_nanos(3_100_000_000);
        localization
            .ingest_visual_localization_frame(frame(0, fresh))
            .unwrap();
        assert_eq!(
            localization.global_visual_lock(),
            GlobalVisualLock::WaitingForBackend
        );
        result.time = fresh.to_wallclock();
        assert!(localization.accept_result(&result).is_none());
        assert_eq!(localization.state(), expected);
        assert!(
            localization
                .accept_result(&acknowledged_result(fresh))
                .is_some()
        );
        assert!(matches!(
            localization.state(),
            LocalizationState3D::Tracking { .. }
        ));
    }

    #[test]
    fn startup_and_lost_live_odometry_propagate_local_pose_without_ground_clamps() {
        let mut localization = localization();
        let mut camera = camera();
        camera.ground_to_robot =
            nalgebra::Isometry3::translation(0.0, 0.0, -99.0).framed_transform();
        let mut odometer = VisualOdometer {
            time: Time::from_nanos(1),
            epoch: 0,
            current_left_camera_to_visual_odometer: nalgebra::Isometry3::identity(),
        };
        assert!(update_odometry(&mut localization, &camera, &odometer).is_none());
        odometer.time = Time::from_nanos(2);
        odometer.current_left_camera_to_visual_odometer = nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.3, 0.2, 0.1),
            nalgebra::UnitQuaternion::from_euler_angles(0.2, -0.1, 0.3),
        );
        assert!(update_odometry(&mut localization, &camera, &odometer).is_none());
        let pose = localization
            .association_geometry()
            .unwrap()
            .inner
            .robot_to_local;
        assert!((pose.translation().z() - 0.6).abs() < 1.0e-6);
        let (roll, pitch, _) = pose.inner.rotation.euler_angles();
        assert!((roll - 0.2).abs() < 1.0e-6);
        assert!((pitch + 0.1).abs() < 1.0e-6);

        let estimate = types::localization::LocalizationEstimate3D {
            robot_to_field: pose.inner.framed_transform(),
            covariance: nalgebra::SMatrix::identity(),
        };
        localization.state = LocalizationState3D::Tracking {
            estimate,
            last_successful_solve: odometer.time,
        };
        let correction = PoseCorrection {
            time: odometer.time,
            robot_to_local: pose.inner.cast().framed_transform(),
            local_to_field: Some(nalgebra::Isometry2::identity().framed_transform()),
        };
        localization
            .live
            .reset_with_exact_samples(correction, &odometer, &camera);
        let global = update_odometry(&mut localization, &camera, &odometer)
            .unwrap()
            .inverse();
        assert!((global.inner.to_homogeneous() - pose.inner.to_homogeneous()).norm() < 1.0e-6);
        localization.state = lose_track(
            localization.state,
            &mut localization.visual_lock,
            odometer.time,
        );
        let lost = localization.state;
        odometer.time = Time::from_nanos(3);
        odometer
            .current_left_camera_to_visual_odometer
            .translation
            .vector
            .x += 0.4;
        assert!(update_odometry(&mut localization, &camera, &odometer).is_none());
        assert_eq!(localization.state(), lost);
        let moved = localization.association_geometry().unwrap();
        assert!((moved.inner.robot_to_local.translation().x() - 0.7).abs() < 1.0e-6);
        assert!(moved.inner.local_to_field.is_some());
    }
}

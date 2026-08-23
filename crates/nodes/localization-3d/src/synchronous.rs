use booster::ImuState;
use color_eyre::{Result, eyre::eyre};
use coordinate_systems::{Field, Robot};
use linear_algebra::{IntoTransform, Isometry3};
use localization_factrs::{InitialState, VinsBackend, VinsFrontend, VinsFrontendError, initialize};
use projection::camera_matrix::CameraMatrix;
use ros_z::time::Time;
use types::{
    field_dimensions::FieldDimensions,
    time_wrapper::TimeWrapper,
    visual_localization::VisualLocalizationFrame,
    visual_odometry::{VisualOdometer, VisualOdometryDelta},
};

use crate::{
    camera::camera_intrinsics_from_matrix,
    diagnostics::SolveDiagnostics,
    ingest::ingest_visual_odometry,
    live_odometry::LiveVisualOdometryLocalization,
    parameters::{
        Localization3dParameters, backend_configuration_from_parameters_and_field_dimensions,
    },
    pose::localization_transform_constrained_to_ground,
    visual_localization::{GlobalVisualLock, handle_visual_localization_frame},
};

/// One deterministic backend checkpoint and its publication-ready poses.
#[derive(Clone, Debug)]
pub struct SynchronousLocalizationOutput {
    /// Backend result timestamp.
    pub time: Time,
    /// Unconstrained backend robot-to-field pose.
    pub raw_backend_robot_to_field: Isometry3<Robot, Field, f64>,
    /// Backend pose constrained using the exact ground-to-robot camera sample.
    pub backend_field_to_robot: Isometry3<Field, Robot>,
    /// Global visual lock state after consuming this result.
    pub global_visual_lock: GlobalVisualLock,
    /// Optimizer and residual diagnostics for this solve.
    pub diagnostics: SolveDiagnostics,
}

/// Synchronous localization frontend/backend pair for deterministic simulation.
pub struct SynchronousLocalization {
    frontend: VinsFrontend,
    backend: VinsBackend,
    live_localization: LiveVisualOdometryLocalization,
    global_visual_lock: GlobalVisualLock,
}

impl SynchronousLocalization {
    /// Creates an isolated estimator using the production backend configuration.
    pub fn new(
        parameters: &Localization3dParameters,
        field_dimensions: &FieldDimensions,
        initial_camera_matrix: &CameraMatrix,
        initial_robot_to_field: Isometry3<Robot, Field>,
    ) -> Result<Self> {
        parameters.validate().map_err(|message| eyre!(message))?;
        let initial_state = InitialState::from_robot_to_field_and_intrinsics(
            initial_robot_to_field,
            camera_intrinsics_from_matrix(initial_camera_matrix),
        );
        let (frontend, backend) = initialize(
            backend_configuration_from_parameters_and_field_dimensions(
                parameters,
                field_dimensions,
            ),
            initial_state,
        );

        Ok(Self {
            frontend,
            backend,
            live_localization: LiveVisualOdometryLocalization::default(),
            global_visual_lock: GlobalVisualLock::Unlocked,
        })
    }

    /// Returns whether global visual localization is unlocked, pending, or locked.
    pub fn global_visual_lock(&self) -> GlobalVisualLock {
        self.global_visual_lock
    }

    /// Ingests one timestamped IMU sample.
    pub fn ingest_imu(&mut self, time: Time, imu: ImuState) -> Result<(), VinsFrontendError> {
        self.frontend.ingest_imu(time.to_wallclock(), imu)
    }

    /// Ingests one camera-frame visual-odometry delta with both endpoint extrinsics.
    pub fn ingest_visual_odometry(
        &mut self,
        delta: VisualOdometryDelta,
        previous_camera_matrix: &CameraMatrix,
        current_camera_matrix: &CameraMatrix,
    ) -> Result<(), VinsFrontendError> {
        ingest_visual_odometry(
            &mut self.frontend,
            delta,
            previous_camera_matrix,
            current_camera_matrix,
        )
    }

    /// Ingests one production visual-localization frame, including global reset semantics.
    pub fn ingest_visual_localization_frame(
        &mut self,
        frame: TimeWrapper<VisualLocalizationFrame>,
    ) -> Result<(), VinsFrontendError> {
        handle_visual_localization_frame(
            &mut self.frontend,
            &mut self.live_localization,
            &mut self.global_visual_lock,
            frame,
        )
    }

    /// Solves one deterministic backend checkpoint.
    ///
    /// `camera_matrix.time` and an optional visual odometer must exactly match the resulting
    /// checkpoint time. `Ok(None)` means no backend result is currently available. The raw pose is
    /// always the backend branch; consumers must inspect `global_visual_lock` before treating it as
    /// an accepted global localization. A locked solve without an exact visual odometer
    /// intentionally invalidates the previous live anchor rather than propagating from a stale
    /// backend pose.
    pub fn solve_once(
        &mut self,
        camera_matrix: TimeWrapper<&CameraMatrix>,
        exact_visual_odometer: Option<&VisualOdometer>,
    ) -> Result<Option<SynchronousLocalizationOutput>> {
        let Some(backend_result) = self.backend.solve_once()? else {
            return Ok(None);
        };
        let result = self
            .frontend
            .last_optimization_result()
            .ok_or_else(|| eyre!("backend result was not delivered to the frontend"))?;
        let time = Time::from_wallclock(result.time);

        if camera_matrix.time != time {
            return Err(eyre!(
                "camera matrix timestamp {:?} does not match checkpoint {:?}",
                camera_matrix.time,
                time
            ));
        }

        if let Some(visual_odometer) = exact_visual_odometer
            && visual_odometer.time != time
        {
            return Err(eyre!(
                "visual odometer timestamp {:?} does not match checkpoint {:?}",
                visual_odometer.time,
                time
            ));
        }

        let backend_field_to_robot = localization_transform_constrained_to_ground(
            &result.transform,
            &camera_matrix.inner.ground_to_robot,
        );
        if self.global_visual_lock.mark_backend_result() {
            self.live_localization.clear();
            if let Some(visual_odometer) = exact_visual_odometer {
                let reset = self.live_localization.reset_with_exact_samples(
                    &result,
                    visual_odometer,
                    camera_matrix.inner,
                );
                debug_assert!(reset, "timestamp was checked above");
            }
        }
        let diagnostics = self
            .backend
            .compute_last_solve_diagnostics()
            .ok_or_else(|| eyre!("diagnostics are unavailable after a backend result"))?
            .into();

        debug_assert_eq!(backend_result.time, result.time);
        Ok(Some(SynchronousLocalizationOutput {
            time,
            raw_backend_robot_to_field: result.transform.framed_transform(),
            backend_field_to_robot,
            global_visual_lock: self.global_visual_lock,
            diagnostics,
        }))
    }

    /// Propagates the latest locked backend pose with an exact accumulated visual-odometer sample.
    ///
    /// The camera matrix and odometer must have identical timestamps.
    pub fn update_live_odometry(
        &mut self,
        current_camera_matrix: TimeWrapper<&CameraMatrix>,
        current_visual_odometer: &VisualOdometer,
    ) -> Result<Option<Isometry3<Field, Robot>>> {
        if current_camera_matrix.time != current_visual_odometer.time {
            return Err(eyre!(
                "camera matrix timestamp {:?} does not match visual odometer {:?}",
                current_camera_matrix.time,
                current_visual_odometer.time
            ));
        }
        if !self.global_visual_lock.has_backend_result() {
            return Ok(None);
        }
        Ok(self
            .live_localization
            .update_with_exact_sample(current_visual_odometer, current_camera_matrix.inner))
    }
}

#[cfg(test)]
mod tests {
    use linear_algebra::{IntoTransform, point};
    use projection::intrinsic::Intrinsic;

    use super::*;

    fn parameters() -> Localization3dParameters {
        Localization3dParameters {
            accelerometer_process_noise_variance: 10.0,
            visual_feature_noise_variance: 1.0,
            pose_hint_visual_feature_noise_variance: 4.0,
            pose_hint_visual_huber_threshold: 2.0,
            field_containment_sigma: 0.1,
        }
    }

    fn camera_matrix() -> CameraMatrix {
        CameraMatrix {
            intrinsics: Intrinsic::new(nalgebra::vector![220.0, 220.0], point![320.0, 240.0]),
            ..Default::default()
        }
    }

    fn initial_robot_to_field() -> Isometry3<Robot, Field> {
        nalgebra::Isometry3::translation(-2.0, 0.5, 0.5).framed_transform()
    }

    fn localization() -> SynchronousLocalization {
        SynchronousLocalization::new(
            &parameters(),
            &FieldDimensions::SPL_2025,
            &camera_matrix(),
            initial_robot_to_field(),
        )
        .expect("valid localization configuration")
    }

    fn global_reset(time: Time) -> TimeWrapper<VisualLocalizationFrame> {
        TimeWrapper {
            time,
            inner: VisualLocalizationFrame {
                robot_to_camera: nalgebra::Isometry3::identity().framed_transform(),
                associations: Vec::new(),
                backend_reset: Some(initial_robot_to_field()),
            },
        }
    }

    #[test]
    fn unlocked_solve_returns_backend_branch_without_global_lock() {
        let mut localization = localization();
        localization
            .ingest_imu(Time::from_nanos(1_000_000_000), ImuState::default())
            .expect("IMU ingestion succeeds");

        let output = localization
            .solve_once(
                TimeWrapper {
                    time: Time::from_nanos(1_000_000_000),
                    inner: &camera_matrix(),
                },
                None,
            )
            .expect("solve succeeds")
            .expect("IMU initializes a backend checkpoint");

        assert_eq!(output.global_visual_lock, GlobalVisualLock::Unlocked);
        assert!(
            output
                .raw_backend_robot_to_field
                .inner
                .translation
                .vector
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn global_reset_locks_on_backend_result() {
        let mut localization = localization();
        let time = Time::from_nanos(2_000_000_000);
        localization
            .ingest_visual_localization_frame(global_reset(time))
            .expect("global reset ingestion succeeds");

        let output = localization
            .solve_once(
                TimeWrapper {
                    time,
                    inner: &camera_matrix(),
                },
                None,
            )
            .expect("solve succeeds")
            .expect("global reset creates a backend checkpoint");

        assert_eq!(output.time, time);
        assert_eq!(output.global_visual_lock, GlobalVisualLock::Locked);
    }

    #[test]
    fn identity_live_odometry_preserves_locked_backend_pose() {
        let mut localization = localization();
        let time = Time::from_nanos(3_000_000_000);
        let visual_odometer = VisualOdometer {
            time,
            epoch: 7,
            current_left_camera_to_visual_odometer: nalgebra::Isometry3::identity(),
        };
        localization
            .ingest_visual_localization_frame(global_reset(time))
            .expect("global reset ingestion succeeds");
        let output = localization
            .solve_once(
                TimeWrapper {
                    time,
                    inner: &camera_matrix(),
                },
                Some(&visual_odometer),
            )
            .expect("solve succeeds")
            .expect("global reset creates a backend checkpoint");

        let live_pose = localization
            .update_live_odometry(
                TimeWrapper {
                    time,
                    inner: &camera_matrix(),
                },
                &visual_odometer,
            )
            .expect("matching exact samples are accepted")
            .expect("locked localization has a live odometry anchor");
        let accepted_pose = output.backend_field_to_robot;

        assert!(
            (live_pose.inner.translation.vector - accepted_pose.inner.translation.vector).norm()
                < 1.0e-6
        );
        assert!(
            live_pose
                .inner
                .rotation
                .angle_to(&accepted_pose.inner.rotation)
                < 1.0e-6
        );
    }

    #[test]
    fn solve_rejects_a_mismatched_camera_timestamp() {
        let mut localization = localization();
        let measurement_time = Time::from_nanos(4_000_000_000);
        localization
            .ingest_imu(measurement_time, ImuState::default())
            .expect("IMU ingestion succeeds");

        let error = localization
            .solve_once(
                TimeWrapper {
                    time: Time::from_nanos(4_100_000_000),
                    inner: &camera_matrix(),
                },
                None,
            )
            .expect_err("mismatched camera time is rejected");

        assert!(error.to_string().contains("camera matrix timestamp"));
    }

    #[test]
    fn locked_solve_without_odometer_invalidates_the_live_anchor() {
        let mut localization = localization();
        let anchor_time = Time::from_nanos(5_000_000_000);
        let anchor_odometer = VisualOdometer {
            time: anchor_time,
            epoch: 1,
            current_left_camera_to_visual_odometer: nalgebra::Isometry3::identity(),
        };
        localization
            .ingest_visual_localization_frame(global_reset(anchor_time))
            .expect("global reset ingestion succeeds");
        localization
            .solve_once(
                TimeWrapper {
                    time: anchor_time,
                    inner: &camera_matrix(),
                },
                Some(&anchor_odometer),
            )
            .expect("anchor solve succeeds");

        let next_time = Time::from_nanos(5_200_000_000);
        localization
            .ingest_imu(next_time, ImuState::default())
            .expect("IMU ingestion succeeds");
        localization
            .solve_once(
                TimeWrapper {
                    time: next_time,
                    inner: &camera_matrix(),
                },
                None,
            )
            .expect("backend-only solve succeeds");
        let current_odometer = VisualOdometer {
            time: next_time,
            ..anchor_odometer
        };

        assert!(
            localization
                .update_live_odometry(
                    TimeWrapper {
                        time: next_time,
                        inner: &camera_matrix(),
                    },
                    &current_odometer,
                )
                .expect("matching exact samples are accepted")
                .is_none()
        );
    }
}

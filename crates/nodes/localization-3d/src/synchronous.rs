use booster::ImuState;
use color_eyre::{Result, eyre::eyre};
use coordinate_systems::{Field, Local, Robot};
use linear_algebra::Isometry3;
use localization_factrs::{
    InitialState, VinsBackend, VinsFrontend, VinsFrontendError, backend::BackendOptimizerStatus,
    initialize,
};
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
    pose::{compose_robot_to_field, localization_transform_constrained_to_ground},
    visual_localization::{
        GlobalVisualLock, GlobalVisualLockTracker, handle_visual_localization_frame,
    },
};

#[derive(Clone, Debug)]
pub struct SynchronousLocalizationOutput {
    pub time: Time,
    pub raw_backend_robot_to_local: Isometry3<Robot, Local, f64>,
    pub raw_backend_robot_to_field: Option<Isometry3<Robot, Field, f64>>,
    pub backend_field_to_robot: Option<Isometry3<Field, Robot>>,
    pub global_visual_lock: GlobalVisualLock,
    pub diagnostics: SolveDiagnostics,
}

pub struct SynchronousLocalization {
    frontend: VinsFrontend,
    backend: VinsBackend,
    live: LiveVisualOdometryLocalization,
    visual_lock: GlobalVisualLockTracker,
    epoch: u64,
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
        live.set_initial(initial_robot_to_local);
        Ok(Self {
            frontend,
            backend,
            live,
            visual_lock: GlobalVisualLockTracker::default(),
            epoch: 0,
        })
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }
    pub fn global_visual_lock(&self) -> GlobalVisualLock {
        self.visual_lock.status()
    }

    pub fn ingest_imu(&mut self, time: Time, imu: ImuState) -> Result<(), VinsFrontendError> {
        self.frontend.ingest_imu(time.to_wallclock(), imu)
    }

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

    pub fn ingest_visual_localization_frame(
        &mut self,
        frame: TimeWrapper<VisualLocalizationFrame>,
    ) -> Result<(), VinsFrontendError> {
        handle_visual_localization_frame(
            &mut self.frontend,
            &mut self.visual_lock,
            self.epoch,
            frame,
        )
    }

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
        if result.generation != self.epoch {
            return Ok(None);
        }
        let time = Time::from_wallclock(result.time);
        if camera_matrix.time != time {
            return Err(eyre!("camera matrix timestamp does not match checkpoint"));
        }
        if exact_visual_odometer.is_some_and(|odometer| odometer.time != time) {
            return Err(eyre!("visual odometer timestamp does not match checkpoint"));
        }

        if result.optimizer_status == BackendOptimizerStatus::Converged {
            self.visual_lock.handle_backend_result(&result);
            self.live.clear();
            if let Some(odometer) = exact_visual_odometer {
                self.live
                    .reset_with_exact_samples(&result, odometer, camera_matrix.inner);
            }
        }
        let backend_field_to_robot = result.robot_to_field.as_ref().map(|pose| {
            localization_transform_constrained_to_ground(
                &pose.inner,
                &camera_matrix.inner.ground_to_robot,
            )
        });
        let diagnostics = self
            .backend
            .compute_last_solve_diagnostics()
            .ok_or_else(|| eyre!("diagnostics are unavailable after a backend result"))?
            .into();
        debug_assert_eq!(backend_result.time, result.time);
        Ok(Some(SynchronousLocalizationOutput {
            time,
            raw_backend_robot_to_local: result.robot_to_local,
            raw_backend_robot_to_field: result.robot_to_field,
            backend_field_to_robot,
            global_visual_lock: self.visual_lock.status(),
            diagnostics,
        }))
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
        if !self.visual_lock.has_backend_result() {
            return Ok(None);
        }
        Ok(self
            .live
            .update_with_exact_sample(current_visual_odometer, current_camera_matrix.inner)
            .and_then(|(robot_to_local, alignment)| {
                alignment.map(|alignment| {
                    let robot_to_field = compose_robot_to_field(robot_to_local, alignment);
                    localization_transform_constrained_to_ground(
                        &robot_to_field.inner.cast(),
                        &current_camera_matrix.inner.ground_to_robot,
                    )
                })
            }))
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
        TimeWrapper { time, inner: VisualLocalizationFrame {
            epoch,
            robot_to_camera: nalgebra::Isometry3::identity().framed_transform(),
            local_to_field: nalgebra::Isometry2::identity().framed_transform(),
            associations: [
                ([320.0, 240.0], [0.0, 0.0, 1.0]),
                ([540.0, 240.0], [1.0, 0.0, 1.0]),
                ([320.0, 460.0], [0.0, 1.0, 1.0]),
            ].into_iter().map(|(pixel, field)| types::visual_localization::FieldMarkAssociation {
                detection: linear_algebra::point![<coordinate_systems::Pixel>, pixel[0], pixel[1]],
                field_point: linear_algebra::point![<Field>, field[0], field[1], field[2]],
            }).collect(),
        }}
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
}

use std::time::{Duration, SystemTime};

use booster::ImuState;
use coordinate_systems::{Camera, Field, Local, Robot};
use factrs::{
    core::{SE3, SO3},
    traits::Variable,
    variables::{MatrixLieGroup, SE23},
};
use linear_algebra::{IntoTransform, Isometry2, Isometry3, Point3};
use thiserror::Error;
use tokio::sync::{mpsc::UnboundedSender, watch};
use types::visual_localization::FieldMarkAssociation;

use crate::backend::OptimizationResult as BackendOptimizationResult;
use crate::camera_intrinsics::CameraIntrinsics;
use crate::factors::{
    foot_above_ground::FootHeightMeasurement, visual_odometry::VisualOdometryMeasurement,
};
use crate::measurements::{
    ImuMeasurement, ResetMeasurement, SensorMeasurement, VisualFrameMeasurement,
    VisualReprojectionMeasurement,
};
use crate::{BackendConfiguration, BackendConfigurationError, InitialState};

pub struct VinsFrontend {
    measurement_sender: UnboundedSender<SensorMeasurement>,
    result_receiver: watch::Receiver<Option<BackendOptimizationResult>>,
    structural_configuration: (Duration, Duration, usize),
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub time: SystemTime,
    pub generation: u64,
    pub robot_to_local: Isometry3<Robot, Local, f64>,
    pub local_to_field: Option<Isometry2<Local, Field, f64>>,
    pub robot_to_field: Option<Isometry3<Robot, Field, f64>>,
    pub robot_to_field_covariance: Option<nalgebra::SMatrix<f64, 6, 6>>,
    pub velocity: nalgebra::Vector3<f64>,
    pub camera_intrinsics: CameraIntrinsics<f64>,
    pub latest_visual_measurement_time: Option<SystemTime>,
    pub latest_visual_robot_to_local: Option<Isometry3<Robot, Local, f64>>,
    pub latest_visual_robot_to_field: Option<Isometry3<Robot, Field, f64>>,
    pub optimizer_status: crate::backend::BackendOptimizerStatus,
}

impl VinsFrontend {
    pub fn new(
        measurement_sender: UnboundedSender<SensorMeasurement>,
        result_receiver: watch::Receiver<Option<BackendOptimizationResult>>,
        config: &BackendConfiguration,
    ) -> Self {
        Self {
            measurement_sender,
            result_receiver,
            structural_configuration: (
                config.knot_spacing,
                config.max_optimization_window,
                config.optimizer_max_iterations,
            ),
        }
    }

    pub async fn wait_for_optimization_result(&mut self) -> Result<(), VinsFrontendError> {
        self.result_receiver
            .changed()
            .await
            .map_err(|_| VinsFrontendError::BackendDisconnected)
    }

    /// Returns the latest backend result and marks it observed for `wait_for_optimization_result`.
    pub fn last_optimization_result(&mut self) -> Option<OptimizationResult> {
        self.result_receiver
            .borrow_and_update()
            .as_ref()
            .map(OptimizationResult::from)
    }

    /// Returns the latest backend result without marking it observed.
    pub fn peek_last_optimization_result(&self) -> Option<OptimizationResult> {
        self.result_receiver
            .borrow()
            .as_ref()
            .map(OptimizationResult::from)
    }

    /// Adds an IMU measurement to the optimization pipeline.
    pub fn ingest_imu(
        &mut self,
        time: SystemTime,
        state: ImuState,
    ) -> Result<(), VinsFrontendError> {
        let measurement = ImuMeasurement { time, state };
        self.measurement_sender
            .send(SensorMeasurement::Imu(measurement))
            .map_err(|_| VinsFrontendError::BackendDisconnected)
    }

    /// Reinitializes the backend to a known startup state.
    pub fn reset(
        &mut self,
        time: SystemTime,
        generation: u64,
        initial_state: InitialState,
    ) -> Result<(), VinsFrontendError> {
        let measurement = ResetMeasurement {
            time,
            generation,
            initial_state,
        };

        self.measurement_sender
            .send(SensorMeasurement::Reset(measurement))
            .map_err(|_| VinsFrontendError::BackendDisconnected)
    }

    /// Enqueues tuning without waiting for the worker or resetting the graph.
    ///
    /// The last update in each drained batch applies before constructing factors in that batch,
    /// regardless of sensor timestamps or resets. Existing factors (including interval factors
    /// extended with more samples) and marginal priors keep their original weights. Old factors
    /// age out over the configured window; their marginalized information is not reweighted.
    /// Configuration-only batches do not solve or publish. Reset preserves the latest tuning.
    /// Knot spacing, window size, and iteration limit must match initialization.
    pub fn update_configuration(
        &mut self,
        config: BackendConfiguration,
    ) -> Result<(), VinsFrontendError> {
        config.validate()?;
        let (spacing, window, iterations) = self.structural_configuration;
        for (changed, field) in [
            (config.knot_spacing != spacing, "knot_spacing"),
            (
                config.max_optimization_window != window,
                "max_optimization_window",
            ),
            (
                config.optimizer_max_iterations != iterations,
                "optimizer_max_iterations",
            ),
        ] {
            if changed {
                return Err(BackendConfigurationError::UnsupportedStructuralChange(field).into());
            }
        }
        self.measurement_sender
            .send(SensorMeasurement::Configuration(Box::new(config)))
            .map_err(|_| VinsFrontendError::BackendDisconnected)
    }

    /// Adds fixed visual feature associations to the optimization pipeline.
    /// A candidate is only needed to bootstrap alignment; unseeded frames require an existing one.
    pub fn ingest_visual_reprojection_associations(
        &mut self,
        time: SystemTime,
        associations: impl IntoIterator<Item = FieldMarkAssociation>,
        robot_to_camera: Isometry3<Robot, Camera>,
        local_to_field_candidate: Option<Isometry2<Local, Field>>,
    ) -> Result<(), VinsFrontendError> {
        let robot_to_camera = isometry3_to_se3(robot_to_camera.inner);
        let measurements: Vec<_> = associations
            .into_iter()
            .map(|association| VisualReprojectionMeasurement {
                detection: association.detection.inner.cast(),
                field_point: association.field_point.inner.cast(),
            })
            .collect();

        if measurements.is_empty() {
            return Ok(());
        }
        self.measurement_sender
            .send(SensorMeasurement::Visual(VisualFrameMeasurement {
                time,
                robot_to_camera,
                local_to_field_candidate: local_to_field_candidate
                    .map(crate::conversions::local_to_field_to_se2),
                measurements,
            }))
            .map_err(|_| VinsFrontendError::BackendDisconnected)
    }

    /// Adds a frame-to-frame visual odometry delta to the optimization pipeline.
    pub fn ingest_visual_odometry_delta(
        &mut self,
        previous_time: SystemTime,
        current_time: SystemTime,
        previous_robot_to_left_camera: Isometry3<Robot, Camera>,
        current_robot_to_left_camera: Isometry3<Robot, Camera>,
        current_left_camera_to_previous_left_camera: nalgebra::Isometry3<f32>,
    ) -> Result<(), VinsFrontendError> {
        let current_robot_to_previous_robot = previous_robot_to_left_camera.inner.inverse()
            * current_left_camera_to_previous_left_camera
            * current_robot_to_left_camera.inner;
        let measurement = VisualOdometryMeasurement {
            previous_time,
            current_time,
            robot_delta: isometry3_to_se3(current_robot_to_previous_robot),
        };

        self.measurement_sender
            .send(SensorMeasurement::VisualOdometry(measurement))
            .map_err(|_| VinsFrontendError::BackendDisconnected)
    }

    /// Adds sole positions to keep the estimated feet above the ground plane.
    pub fn ingest_foot_heights(
        &mut self,
        time: SystemTime,
        left_sole_in_robot: Point3<Robot>,
        right_sole_in_robot: Point3<Robot>,
    ) -> Result<(), VinsFrontendError> {
        let measurement = FootHeightMeasurement {
            time,
            left_sole_in_robot,
            right_sole_in_robot,
        };

        self.measurement_sender
            .send(SensorMeasurement::FootHeights(measurement))
            .map_err(|_| VinsFrontendError::BackendDisconnected)
    }
}

#[derive(Debug, Error)]
pub enum VinsFrontendError {
    #[error(transparent)]
    InvalidConfiguration(#[from] BackendConfigurationError),
    #[error("the localization backend is disconnected")]
    BackendDisconnected,
}

fn isometry3_to_se3(isometry: nalgebra::Isometry3<f32>) -> SE3 {
    isometry3_f64_to_se3(isometry.cast())
}

fn isometry3_f64_to_se3(isometry: nalgebra::Isometry3<f64>) -> SE3 {
    let rotation = isometry.rotation.quaternion();
    let translation = isometry.translation.vector;

    SE3::from_rot_trans(
        SO3::from_xyzw(rotation.i, rotation.j, rotation.k, rotation.w),
        translation,
    )
}

impl From<&BackendOptimizationResult> for OptimizationResult {
    fn from(backend_result: &BackendOptimizationResult) -> Self {
        let (robot_to_local, velocity) =
            se23_to_isometry3_and_velocity(&backend_result.latest_robot_to_local);
        let robot_to_local = robot_to_local.framed_transform();
        let local_to_field = backend_result
            .local_to_field
            .as_ref()
            .map(crate::conversions::se2_to_local_to_field);
        let robot_to_field = local_to_field
            .as_ref()
            .map(|alignment| alignment.to_3d() * robot_to_local);
        let latest_visual_robot_to_local = backend_result
            .latest_visual_robot_to_local
            .as_ref()
            .map(|pose| se23_to_isometry3_and_velocity(pose).0.framed_transform());
        let latest_visual_robot_to_field = latest_visual_robot_to_local
            .as_ref()
            .zip(local_to_field.as_ref())
            .map(|(pose, alignment)| alignment.to_3d() * *pose);
        OptimizationResult {
            time: backend_result.time,
            generation: backend_result.generation,
            robot_to_local,
            local_to_field,
            robot_to_field,
            robot_to_field_covariance: backend_result.robot_to_field_covariance,
            velocity,
            camera_intrinsics: backend_result.camera_intrinsics.clone(),
            latest_visual_measurement_time: backend_result.latest_visual_measurement_time,
            latest_visual_robot_to_local,
            latest_visual_robot_to_field,
            optimizer_status: backend_result.optimizer_status,
        }
    }
}

pub(crate) fn se23_to_isometry3_and_velocity(
    se23: &SE23,
) -> (nalgebra::Isometry3<f64>, nalgebra::Vector3<f64>) {
    let rotation = se23.rot();
    let local_velocity = rotation.inverse().apply(se23.uvw());

    let isometry = nalgebra::Isometry3::from_parts(
        nalgebra::Translation3::from(se23.xyz().into_owned()),
        nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            rotation.w(),
            rotation.x(),
            rotation.y(),
            rotation.z(),
        )),
    );

    (isometry, local_velocity)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime};

    use crate::measurements::SensorMeasurement;

    use super::*;

    fn translation(x: f32, y: f32, z: f32) -> Isometry3<Robot, Camera> {
        Isometry3::from_translation(x, y, z)
    }

    #[test]
    fn configuration_rejects_invalid_tuning_and_structural_changes_before_enqueueing() {
        use BackendConfigurationError::*;
        let config = crate::backend::tests::backend_configuration();
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let (_result_sender, result_receiver) = tokio::sync::watch::channel(None);
        let mut frontend = VinsFrontend::new(sender, result_receiver, &config);
        type ConfigurationChange = fn(&mut BackendConfiguration);
        let cases: &[(ConfigurationChange, BackendConfigurationError)] = &[
            (
                |c| c.foot_ground_sigma = f64::NAN,
                InvalidPositiveValue("foot_ground_sigma"),
            ),
            (
                |c| c.field_containment.sigma = f64::INFINITY,
                InvalidPositiveValue("field_containment.sigma"),
            ),
            (
                |c| c.field_containment.x_limit = -1.0,
                InvalidPositiveValue("field_containment.x_limit"),
            ),
            (
                |c| c.field_containment.y_limit = 0.0,
                InvalidPositiveValue("field_containment.y_limit"),
            ),
            (
                |c| c.gyroscope_noise[(0, 0)] = f64::NAN,
                InvalidCovariance("gyroscope_noise"),
            ),
            (
                |c| c.accelerometer_noise[(0, 0)] = -1.0,
                InvalidCovariance("accelerometer_noise"),
            ),
            (
                |c| c.gyroscope_process_noise[(0, 0)] = 0.0,
                InvalidCovariance("gyroscope_process_noise"),
            ),
            (
                |c| c.accelerometer_process_noise[(0, 1)] = 0.1,
                InvalidCovariance("accelerometer_process_noise"),
            ),
            (
                |c| c.roll_pitch_yaw_noise[(0, 0)] = f64::INFINITY,
                InvalidCovariance("roll_pitch_yaw_noise"),
            ),
            (
                |c| c.visual_feature_noise.fill(1.0),
                InvalidCovariance("visual_feature_noise"),
            ),
            (
                |c| c.visual_odometry_noise.fill(0.0),
                InvalidCovariance("visual_odometry_noise"),
            ),
            (|c| c.gravity.fill(0.0), InvalidGravity),
            (|c| c.gravity.x = f64::NAN, InvalidGravity),
            (
                |c| c.knot_spacing *= 2,
                UnsupportedStructuralChange("knot_spacing"),
            ),
            (
                |c| c.max_optimization_window *= 2,
                UnsupportedStructuralChange("max_optimization_window"),
            ),
            (
                |c| c.optimizer_max_iterations += 1,
                UnsupportedStructuralChange("optimizer_max_iterations"),
            ),
        ];
        for (mutate, expected) in cases {
            let mut invalid = config.clone();
            mutate(&mut invalid);
            let Err(VinsFrontendError::InvalidConfiguration(actual)) =
                frontend.update_configuration(invalid)
            else {
                panic!("expected {expected}");
            };
            assert_eq!(&actual, expected);
            assert!(matches!(
                receiver.try_recv(),
                Err(tokio::sync::mpsc::error::TryRecvError::Empty)
            ));
        }
        drop(receiver);
        assert!(matches!(
            frontend.update_configuration(config),
            Err(VinsFrontendError::BackendDisconnected)
        ));
    }

    #[test]
    fn visual_odometry_delta_uses_endpoint_camera_extrinsics() {
        let (measurement_sender, mut measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
        let (_result_sender, result_receiver) = tokio::sync::watch::channel(None);
        let mut frontend = VinsFrontend::new(
            measurement_sender,
            result_receiver,
            &crate::backend::tests::backend_configuration(),
        );
        let previous_time = SystemTime::UNIX_EPOCH;
        let current_time = previous_time + Duration::from_millis(33);

        frontend
            .ingest_visual_odometry_delta(
                previous_time,
                current_time,
                translation(1.0, 0.0, 0.0),
                translation(2.0, 0.0, 0.0),
                nalgebra::Isometry3::translation(0.5, 0.0, 0.0),
            )
            .expect("visual odometry should ingest");

        let SensorMeasurement::VisualOdometry(measurement) = measurement_receiver
            .try_recv()
            .expect("measurement should be queued")
        else {
            panic!("expected visual odometry measurement");
        };

        assert_eq!(measurement.previous_time, previous_time);
        assert_eq!(measurement.current_time, current_time);
        assert!((measurement.robot_delta.xyz().x - 1.5).abs() < 1.0e-9);
    }

    #[test]
    fn visual_frames_preserve_optional_bootstrap_candidates() {
        let (measurement_sender, mut measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
        let (_result_sender, result_receiver) = tokio::sync::watch::channel(None);
        let mut frontend = VinsFrontend::new(
            measurement_sender,
            result_receiver,
            &crate::backend::tests::backend_configuration(),
        );
        for candidate in [
            None,
            Some(nalgebra::Isometry2::translation(1.0, 2.0).framed_transform()),
        ] {
            frontend
                .ingest_visual_reprojection_associations(
                    SystemTime::UNIX_EPOCH,
                    [FieldMarkAssociation {
                        detection: linear_algebra::point![320.0, 240.0],
                        field_point: linear_algebra::point![1.0, 2.0, 0.0],
                    }],
                    Isometry3::identity(),
                    candidate,
                )
                .unwrap();
            let SensorMeasurement::Visual(frame) = measurement_receiver.try_recv().unwrap() else {
                panic!("expected visual frame");
            };
            assert_eq!(
                frame
                    .local_to_field_candidate
                    .as_ref()
                    .map(|seed| (seed.x(), seed.y())),
                candidate.map(|seed| (
                    seed.inner.translation.x as f64,
                    seed.inner.translation.y as f64
                ))
            );
            assert_eq!(frame.measurements.len(), 1);
            assert_eq!(frame.time, SystemTime::UNIX_EPOCH);
            assert!(frame.robot_to_camera.ominus(&SE3::identity()).norm() < 1.0e-12);
            assert_eq!(
                frame.measurements[0].field_point,
                nalgebra::point![1.0, 2.0, 0.0]
            );
        }
    }

    #[test]
    fn reset_sends_generation() {
        let (measurement_sender, mut measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
        let (_result_sender, result_receiver) = tokio::sync::watch::channel(None);
        let mut frontend = VinsFrontend::new(
            measurement_sender,
            result_receiver,
            &crate::backend::tests::backend_configuration(),
        );

        frontend
            .reset(SystemTime::UNIX_EPOCH, 7, InitialState::default())
            .expect("reset should send");

        let SensorMeasurement::Reset(reset) = measurement_receiver
            .try_recv()
            .expect("reset should be queued")
        else {
            panic!("expected reset measurement");
        };
        assert_eq!(reset.generation, 7);
    }
}

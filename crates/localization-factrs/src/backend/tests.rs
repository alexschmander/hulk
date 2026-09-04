use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, SystemTime};

use super::*;
use crate::{
    factors::{
        field_containment::FieldContainmentFactor,
        foot_above_ground::{FootHeightMeasurement, IntervalFootAboveGroundFactor},
        visual_odometry::{
            AdjacentVisualOdometryFactor, VisualOdometryFactor, VisualOdometryMeasurement,
        },
        visual_reprojection::VisualReprojectionFactor,
    },
    measurements::{VisualFrameMeasurement, VisualReprojectionMeasurement},
    symbols::{CameraIntrinsics, LocalToField, State},
};
use booster::ImuState;
use factrs::{
    containers::FactorBuilder,
    core::{PriorResidual, SE3, SO3, Values, Vector3},
    linalg::Matrix3,
    noise::GaussianNoise,
    optimizers::OptObserver,
    traits::{Optimizer, Variable},
    variables::{SE2, SE23},
};
use linear_algebra::IntoFramed;
use nalgebra::{Matrix2, SMatrix};

struct StatePresenceObserver {
    state: State,
    seen: Arc<AtomicBool>,
}

impl OptObserver for StatePresenceObserver {
    fn on_step(&self, values: &Values, _time: i64) {
        if values.get_raw(self.state).is_some() {
            self.seen.store(true, Ordering::SeqCst);
        }
    }
}

fn backend_configuration() -> BackendConfiguration {
    BackendConfiguration {
        knot_spacing: Duration::from_millis(200),
        max_optimization_window: Duration::from_secs(3),
        optimizer_max_iterations: 1,
        gyroscope_noise: Matrix3::identity() * 0.05_f64.powi(2),
        accelerometer_noise: Matrix3::identity() * 0.5_f64.powi(2),
        use_accelerometer_measurements: false,
        gyroscope_process_noise: Matrix3::identity() * 0.01,
        accelerometer_process_noise: Matrix3::identity() * 0.01,
        roll_pitch_yaw_noise: Matrix3::identity() * 0.01,
        visual_feature_noise: Matrix2::identity() * 5.0,
        visual_odometry_noise: SMatrix::<f64, 6, 6>::identity() * 0.05,
        foot_ground_sigma: 0.01,
        field_containment: FieldContainmentConfiguration::default(),
        gravity: Vector3::new(0.0, 0.0, 9.81),
    }
}

#[test]
fn new_interval_state_is_initialized_with_constant_velocity() {
    let (_measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, _result_receiver) = tokio::sync::watch::channel(None);
    let mut backend = VinsBackend::new(
        backend_configuration(),
        InitialState::default(),
        measurement_receiver,
        result_sender,
    );
    backend.init_intervals_through(0);
    let state = backend.values.get_mut(State(1)).unwrap();
    *state = SE23::from_rot_vel_trans(
        SO3::identity(),
        Vector3::new(2.0, -1.0, 0.5),
        Vector3::new(1.0, 2.0, 3.0),
    );

    backend.init_intervals_through(1);

    let initialized = backend.values.get(State(2)).unwrap();
    assert!((initialized.uvw() - Vector3::new(2.0, -1.0, 0.5)).norm() < 1.0e-12);
    assert!((initialized.xyz() - Vector3::new(1.4, 1.8, 3.1)).norm() < 1.0e-12);
}

fn stationary_imu(time: SystemTime) -> SensorMeasurement {
    SensorMeasurement::Imu(ImuMeasurement {
        time,
        state: ImuState {
            roll_pitch_yaw: Vector3::zeros().framed(),
            angular_velocity: Vector3::zeros().framed(),
            linear_acceleration: Vector3::new(0.0, 0.0, 9.81).framed(),
        },
    })
}

fn visual_odometry(
    previous_time: SystemTime,
    current_time: SystemTime,
    x: f64,
) -> SensorMeasurement {
    SensorMeasurement::VisualOdometry(VisualOdometryMeasurement {
        previous_time,
        current_time,
        robot_delta: SE3::from_rot_trans(SO3::identity(), Vector3::new(x, 0.0, 0.0)),
    })
}

fn reset(time: SystemTime, generation: u64, initial_state: InitialState) -> SensorMeasurement {
    SensorMeasurement::Reset(crate::measurements::ResetMeasurement {
        time,
        generation,
        initial_state,
    })
}

fn visual_reprojection_measurement(
    time: SystemTime,
    detection_x: f64,
) -> VisualReprojectionMeasurement {
    VisualReprojectionMeasurement {
        time,
        detection: nalgebra::point![detection_x, 0.0],
        field_point: nalgebra::point![0.0, 0.0, 2.0],
        robot_to_camera: SE3::identity(),
    }
}

fn visual_reprojection(time: SystemTime) -> SensorMeasurement {
    visual_frame(
        vec![visual_reprojection_measurement(time, 0.0)],
        SE2::identity(),
    )
}

fn visual_frame(
    measurements: Vec<VisualReprojectionMeasurement>,
    local_to_field_candidate: SE2,
) -> SensorMeasurement {
    SensorMeasurement::Visual(VisualFrameMeasurement {
        local_to_field_candidate,
        measurements,
    })
}

fn visual_reprojection_factor_count(backend: &mut VinsBackend, state: State) -> usize {
    backend
        .optimizer
        .graph_mut()
        .factors_for_residual::<VisualReprojectionFactor, _>((
            state,
            State(state.0 + 1),
            LocalToField(0),
            CameraIntrinsics(0),
        ))
        .count()
}

fn visual_reprojection_factor_dimensions(backend: &VinsBackend, state: State) -> Vec<usize> {
    backend
        .optimizer
        .graph()
        .factors_for_residual::<VisualReprojectionFactor, _>((
            state,
            State(state.0 + 1),
            LocalToField(0),
            CameraIntrinsics(0),
        ))
        .map(|factor| {
            factor
                .try_dim_out(backend.values())
                .expect("visual reprojection factor should have a dimension")
        })
        .collect()
}

fn visual_odometry_factor_count(backend: &mut VinsBackend, state: State) -> usize {
    backend
        .optimizer
        .graph_mut()
        .factors_for_residual::<VisualOdometryFactor, _>((state, State(state.0 + 1)))
        .count()
}

fn visual_odometry_factor_dimensions(backend: &VinsBackend, state: State) -> Vec<usize> {
    backend
        .optimizer
        .graph()
        .factors_for_residual::<VisualOdometryFactor, _>((state, State(state.0 + 1)))
        .map(|factor| {
            factor
                .try_dim_out(backend.values())
                .expect("visual odometry factor should have a dimension")
        })
        .collect()
}

fn adjacent_visual_odometry_factor_count(backend: &mut VinsBackend, state: State) -> usize {
    backend
        .optimizer
        .graph_mut()
        .factors_for_residual::<AdjacentVisualOdometryFactor, _>((
            state,
            State(state.0 + 1),
            State(state.0 + 2),
        ))
        .count()
}

fn adjacent_visual_odometry_factor_dimensions(backend: &VinsBackend, state: State) -> Vec<usize> {
    backend
        .optimizer
        .graph()
        .factors_for_residual::<AdjacentVisualOdometryFactor, _>((
            state,
            State(state.0 + 1),
            State(state.0 + 2),
        ))
        .map(|factor| {
            factor
                .try_dim_out(backend.values())
                .expect("adjacent visual odometry factor should have a dimension")
        })
        .collect()
}

fn field_containment_factor_count(backend: &VinsBackend, state: State) -> usize {
    backend
        .optimizer
        .graph()
        .factors_for_residual::<FieldContainmentFactor, _>((state, LocalToField(0)))
        .count()
}

fn foot_heights(time: SystemTime) -> SensorMeasurement {
    SensorMeasurement::FootHeights(FootHeightMeasurement {
        time,
        left_sole_in_robot: linear_algebra::point![<coordinate_systems::Robot>, 0.0, 0.05, -0.2],
        right_sole_in_robot: linear_algebra::point![<coordinate_systems::Robot>, 0.0, -0.05, -0.2],
    })
}

fn moving_initial_state(velocity: Vector3<f64>) -> InitialState {
    InitialState {
        robot_to_local: SE23::from_rot_vel_trans(SO3::identity(), velocity, Vector3::zeros()),
        ..InitialState::default()
    }
}

fn initial_state_with_intrinsics(
    focal_lengths: nalgebra::Vector2<f64>,
    optical_center: nalgebra::Vector2<f64>,
) -> InitialState {
    InitialState {
        camera_intrinsics: crate::camera_intrinsics::CameraIntrinsics::new(
            focal_lengths,
            optical_center,
        ),
        ..InitialState::default()
    }
}

fn solve_aligned_graph(alignment_sigma: f64) -> OptimizationResult {
    let (_measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, _result_receiver) = tokio::sync::watch::channel(None);
    let mut config = backend_configuration();
    config.optimizer_max_iterations = 10;
    let mut backend = VinsBackend::new(
        config,
        InitialState::default(),
        measurement_receiver,
        result_sender,
    );
    let time = SystemTime::UNIX_EPOCH + Duration::from_millis(50);
    backend
        .ingest_sensor_measurements([visual_reprojection(time)])
        .expect("visual measurement should ingest");
    backend.optimizer.graph_mut().add_factor(
        FactorBuilder::new(PriorResidual::new(SE2::identity()), LocalToField(0))
            .noise(GaussianNoise::<3>::from_scalar_sigma(alignment_sigma))
            .build(),
    );

    backend
        .optimize()
        .expect("aligned graph should produce a result")
}

#[test]
fn converged_aligned_result_has_joint_robot_to_field_covariance() {
    let tight = solve_aligned_graph(0.01);
    let loose = solve_aligned_graph(0.2);

    assert_eq!(tight.optimizer_status, BackendOptimizerStatus::Converged);
    assert_eq!(loose.optimizer_status, BackendOptimizerStatus::Converged);
    let tight = tight
        .robot_to_field_covariance
        .expect("constrained aligned graph should have covariance");
    let loose = loose
        .robot_to_field_covariance
        .expect("constrained aligned graph should have covariance");
    assert!(
        tight
            .iter()
            .chain(loose.iter())
            .all(|value| value.is_finite())
    );
    assert!((tight - tight.transpose()).norm() < 1.0e-12);
    assert!((loose - loose.transpose()).norm() < 1.0e-12);
    assert!(loose.trace() > tight.trace());
}

#[test]
fn solve_once_before_measurements_preserves_initial_values() {
    let (_measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, _result_receiver) = tokio::sync::watch::channel(None);
    let mut backend = VinsBackend::new(
        backend_configuration(),
        InitialState::default(),
        measurement_receiver,
        result_sender,
    );

    let _ = backend.solve_once().expect("empty solve should succeed");
    let _ = backend
        .solve_once()
        .expect("repeated empty solve should succeed");

    assert!(backend.values().get_raw(CameraIntrinsics(0)).is_some());
    assert!(backend.values().get_raw(State(0)).is_some());
}

#[test]
fn field_containment_factors_are_added_to_existing_states_after_alignment() {
    let (_measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, _result_receiver) = tokio::sync::watch::channel(None);
    let mut backend = VinsBackend::new(
        backend_configuration(),
        InitialState::default(),
        measurement_receiver,
        result_sender,
    );
    let start = SystemTime::UNIX_EPOCH;

    assert!(backend.values().get(LocalToField(0)).is_none());
    assert_eq!(field_containment_factor_count(&backend, State(0)), 0);

    backend
        .ingest_sensor_measurements([stationary_imu(start)])
        .expect("IMU should ingest");
    backend
        .ingest_sensor_measurements([stationary_imu(start + Duration::from_millis(50))])
        .expect("IMU should ingest");

    assert_eq!(field_containment_factor_count(&backend, State(0)), 0);
    assert_eq!(field_containment_factor_count(&backend, State(1)), 0);

    backend
        .ingest_sensor_measurements([visual_reprojection(start + Duration::from_millis(50))])
        .expect("visual frame should ingest");

    assert!(backend.values().get(LocalToField(0)).is_some());
    assert_eq!(field_containment_factor_count(&backend, State(0)), 1);
    assert_eq!(field_containment_factor_count(&backend, State(1)), 1);
}

#[test]
fn solve_once_result_includes_camera_intrinsics() {
    let (measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, _result_receiver) = tokio::sync::watch::channel(None);
    let initial_state = initial_state_with_intrinsics(
        nalgebra::vector![200.0, 210.0],
        nalgebra::vector![250.0, 240.0],
    );
    let mut backend = VinsBackend::new(
        backend_configuration(),
        initial_state,
        measurement_receiver,
        result_sender,
    );

    measurement_sender
        .send(stationary_imu(SystemTime::UNIX_EPOCH))
        .expect("IMU should send");

    let result = backend
        .solve_once()
        .expect("solve should succeed")
        .expect("result should be available");

    assert_eq!(result.generation, 0);
    assert_eq!(
        result.camera_intrinsics.focals(),
        nalgebra::vector![200.0, 210.0]
    );
    assert_eq!(
        result.camera_intrinsics.optical_center(),
        nalgebra::vector![250.0, 240.0]
    );
}

#[test]
fn visual_reprojection_measurements_create_interval_factor() {
    let (measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, _result_receiver) = tokio::sync::watch::channel(None);
    let mut backend = VinsBackend::new(
        backend_configuration(),
        InitialState::default(),
        measurement_receiver,
        result_sender,
    );
    let start = SystemTime::UNIX_EPOCH;

    measurement_sender
        .send(visual_reprojection(start))
        .expect("visual reprojection should send");

    let _ = backend.solve_once().expect("solve should succeed");

    assert!(backend.values().get_raw(State(0)).is_some());
    assert!(backend.values().get_raw(State(1)).is_some());
    assert_eq!(visual_reprojection_factor_count(&mut backend, State(0)), 1);
}

#[test]
fn first_visual_batch_measurement_is_ingested() {
    let (_measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, _result_receiver) = tokio::sync::watch::channel(None);
    let mut backend = VinsBackend::new(
        backend_configuration(),
        InitialState::default(),
        measurement_receiver,
        result_sender,
    );
    let start = SystemTime::UNIX_EPOCH;

    backend
        .ingest_sensor_measurements([visual_reprojection(start)])
        .expect("first measurement should ingest");

    assert!(backend.values().get_raw(State(0)).is_some());
    assert!(backend.values().get_raw(State(1)).is_some());
    assert_eq!(visual_reprojection_factor_count(&mut backend, State(0)), 1);
}

#[test]
fn visual_reprojection_huber_is_per_measurement_residual_block() {
    let (_measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, _result_receiver) = tokio::sync::watch::channel(None);
    let mut backend = VinsBackend::new(
        backend_configuration(),
        InitialState::default(),
        measurement_receiver,
        result_sender,
    );
    let start = SystemTime::UNIX_EPOCH;

    backend
        .ingest_sensor_measurements([visual_frame(
            vec![
                visual_reprojection_measurement(start, 0.0),
                visual_reprojection_measurement(start, 1.0),
            ],
            SE2::identity(),
        )])
        .expect("visual measurements should ingest");

    assert_eq!(visual_reprojection_factor_count(&mut backend, State(0)), 2);
    assert_eq!(
        visual_reprojection_factor_dimensions(&backend, State(0)),
        vec![2, 2]
    );
}

#[test]
fn visual_odometry_measurements_create_interval_factor() {
    let (measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, _result_receiver) = tokio::sync::watch::channel(None);
    let mut backend = VinsBackend::new(
        backend_configuration(),
        InitialState::default(),
        measurement_receiver,
        result_sender,
    );
    let start = SystemTime::UNIX_EPOCH;

    measurement_sender
        .send(visual_odometry(
            start,
            start + Duration::from_millis(100),
            0.1,
        ))
        .expect("visual odometry should send");

    let _ = backend.solve_once().expect("solve should succeed");

    assert!(backend.values().get_raw(State(0)).is_some());
    assert!(backend.values().get_raw(State(1)).is_some());
    assert_eq!(visual_odometry_factor_count(&mut backend, State(0)), 1);
}

#[test]
fn visual_odometry_huber_is_per_delta_residual_block() {
    let (_measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, _result_receiver) = tokio::sync::watch::channel(None);
    let mut backend = VinsBackend::new(
        backend_configuration(),
        InitialState::default(),
        measurement_receiver,
        result_sender,
    );
    let start = SystemTime::UNIX_EPOCH;

    backend
        .ingest_sensor_measurements([
            visual_odometry(start, start + Duration::from_millis(50), 0.05),
            visual_odometry(
                start + Duration::from_millis(60),
                start + Duration::from_millis(100),
                0.04,
            ),
        ])
        .expect("visual odometry measurements should ingest");

    assert_eq!(visual_odometry_factor_count(&mut backend, State(0)), 2);
    assert_eq!(
        visual_odometry_factor_dimensions(&backend, State(0)),
        vec![6, 6]
    );
}

#[test]
fn visual_odometry_measurements_create_adjacent_interval_factor() {
    let (measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, _result_receiver) = tokio::sync::watch::channel(None);
    let mut backend = VinsBackend::new(
        backend_configuration(),
        InitialState::default(),
        measurement_receiver,
        result_sender,
    );
    let start = SystemTime::UNIX_EPOCH;

    measurement_sender
        .send(stationary_imu(start))
        .expect("IMU should send");
    measurement_sender
        .send(visual_odometry(
            start + Duration::from_millis(100),
            start + Duration::from_millis(200),
            0.1,
        ))
        .expect("visual odometry should send");

    let _ = backend.solve_once().expect("solve should succeed");

    assert_eq!(visual_odometry_factor_count(&mut backend, State(0)), 0);
    assert_eq!(
        adjacent_visual_odometry_factor_count(&mut backend, State(0)),
        1
    );
}

#[test]
fn adjacent_visual_odometry_huber_is_per_delta_residual_block() {
    let (_measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, _result_receiver) = tokio::sync::watch::channel(None);
    let mut backend = VinsBackend::new(
        backend_configuration(),
        InitialState::default(),
        measurement_receiver,
        result_sender,
    );
    let start = SystemTime::UNIX_EPOCH;

    backend
        .ingest_sensor_measurements([
            stationary_imu(start),
            visual_odometry(
                start + Duration::from_millis(100),
                start + Duration::from_millis(200),
                0.1,
            ),
            visual_odometry(
                start + Duration::from_millis(120),
                start + Duration::from_millis(200),
                0.08,
            ),
        ])
        .expect("adjacent visual odometry measurements should ingest");

    assert_eq!(
        adjacent_visual_odometry_factor_count(&mut backend, State(0)),
        2
    );
    assert_eq!(
        adjacent_visual_odometry_factor_dimensions(&backend, State(0)),
        vec![6, 6]
    );
}

#[test]
fn invalid_visual_odometry_measurement_does_not_create_empty_factor() {
    let (measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, _result_receiver) = tokio::sync::watch::channel(None);
    let mut backend = VinsBackend::new(
        backend_configuration(),
        InitialState::default(),
        measurement_receiver,
        result_sender,
    );
    let start = SystemTime::UNIX_EPOCH;

    measurement_sender
        .send(visual_odometry(start, start, 0.0))
        .expect("visual odometry should send");

    let _ = backend.solve_once().expect("solve should succeed");

    assert_eq!(visual_odometry_factor_count(&mut backend, State(0)), 0);

    measurement_sender
        .send(visual_odometry(
            start,
            start + Duration::from_millis(100),
            0.1,
        ))
        .expect("visual odometry should send");

    let _ = backend.solve_once().expect("solve should succeed");

    assert_eq!(visual_odometry_factor_count(&mut backend, State(0)), 1);
}

#[test]
fn reset_measurement_resets_backend_to_initial_state() {
    let (measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, _result_receiver) = tokio::sync::watch::channel(None);
    let mut backend = VinsBackend::new(
        backend_configuration(),
        InitialState::default(),
        measurement_receiver,
        result_sender,
    );
    let start = SystemTime::UNIX_EPOCH;
    measurement_sender
        .send(visual_odometry(
            start,
            start + Duration::from_millis(100),
            3.0,
        ))
        .expect("visual odometry should send");
    let _ = backend.solve_once().expect("initial solve should succeed");

    let reset_time = start + Duration::from_secs(1);
    let initial_state = InitialState::new(
        SE23::from_rot_vel_trans(
            SO3::identity(),
            Vector3::zeros(),
            Vector3::new(1.0, 2.0, 0.45),
        ),
        backend.initial_state.camera_intrinsics.clone(),
    );
    measurement_sender
        .send(reset(reset_time, 7, initial_state))
        .expect("reset should send");

    let result = backend
        .solve_once()
        .expect("reset solve should succeed")
        .expect("reset should produce a result");

    assert_eq!(result.time, reset_time);
    assert_eq!(result.generation, 7);
    assert!((result.latest_robot_to_local.xyz() - Vector3::new(1.0, 2.0, 0.45)).norm() < 1.0e-9);
    assert_eq!(visual_odometry_factor_count(&mut backend, State(0)), 0);
}

#[test]
fn foot_height_measurements_create_interval_factor() {
    let (measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, _result_receiver) = tokio::sync::watch::channel(None);
    let mut backend = VinsBackend::new(
        backend_configuration(),
        InitialState::default(),
        measurement_receiver,
        result_sender,
    );
    let start = SystemTime::UNIX_EPOCH;

    measurement_sender
        .send(foot_heights(start))
        .expect("foot heights should send");

    let _ = backend.solve_once().expect("solve should succeed");

    assert!(backend.values().get_raw(State(0)).is_some());
    assert!(backend.values().get_raw(State(1)).is_some());
    assert_eq!(
        backend
            .optimizer
            .graph_mut()
            .factors_for_residual::<IntervalFootAboveGroundFactor, _>((State(0), State(1)))
            .count(),
        1
    );
}

#[test]
fn foot_height_measurements_before_solver_start_are_dropped() {
    let (measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, _result_receiver) = tokio::sync::watch::channel(None);
    let mut backend = VinsBackend::new(
        backend_configuration(),
        InitialState::default(),
        measurement_receiver,
        result_sender,
    );
    let solver_start = SystemTime::UNIX_EPOCH + Duration::from_secs(1);

    measurement_sender
        .send(stationary_imu(solver_start))
        .expect("IMU should send");
    measurement_sender
        .send(foot_heights(solver_start - Duration::from_millis(100)))
        .expect("foot heights should send");

    let _ = backend.solve_once().expect("solve should succeed");

    assert_eq!(
        backend
            .optimizer
            .graph_mut()
            .factors_for_residual::<IntervalFootAboveGroundFactor, _>((State(0), State(1)))
            .count(),
        0
    );
}

#[test]
fn imu_gaps_are_bridged_with_empty_intervals() {
    let (measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, _result_receiver) = tokio::sync::watch::channel(None);
    let mut backend = VinsBackend::new(
        backend_configuration(),
        InitialState::default(),
        measurement_receiver,
        result_sender,
    );

    let start = SystemTime::UNIX_EPOCH;
    measurement_sender
        .send(stationary_imu(start))
        .expect("first IMU should send");
    measurement_sender
        .send(stationary_imu(start + Duration::from_secs(1)))
        .expect("second IMU should send");

    let _ = backend.solve_once().expect("solve should succeed");

    for index in 0..=6 {
        assert!(
            backend.values().get_raw(State(index)).is_some(),
            "state {index} should be initialized"
        );
    }
    assert_eq!(backend.highest_initialized_interval, Some(5));
}

#[test]
fn long_gap_bridge_states_do_not_inherit_stale_velocity() {
    let (_measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, _result_receiver) = tokio::sync::watch::channel(None);
    let mut backend = VinsBackend::new(
        backend_configuration(),
        moving_initial_state(Vector3::new(0.0, 3.0, 0.0)),
        measurement_receiver,
        result_sender,
    );

    backend
        .interval_assigner
        .assign_or_initialize_interval(SystemTime::UNIX_EPOCH)
        .expect("first interval should be assigned");

    backend.init_intervals_through(0);
    backend.init_intervals_through(LONG_GAP_MIN_EMPTY_INTERVALS + 1);

    let boundary_state = backend
        .values()
        .get(State(1))
        .expect("gap boundary state should be initialized");
    let bridge_state = backend
        .values()
        .get(State(2))
        .expect("bridge state should be initialized");
    let target_state = backend
        .values()
        .get(State(LONG_GAP_MIN_EMPTY_INTERVALS + 1))
        .expect("target state should be initialized");

    assert!(boundary_state.uvw().norm() < 1.0e-9);
    assert!(bridge_state.uvw().norm() < 1.0e-9);
    assert!(target_state.uvw().norm() < 1.0e-9);
}

#[test]
fn marginalization_happens_after_optimizer_step() {
    let (measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, _result_receiver) = tokio::sync::watch::channel(None);
    let mut config = backend_configuration();
    config.max_optimization_window = Duration::from_millis(400);
    let mut backend = VinsBackend::new(
        config,
        InitialState::default(),
        measurement_receiver,
        result_sender,
    );
    let state_seen_during_optimization = Arc::new(AtomicBool::new(false));
    backend.optimizer.add_observer(StatePresenceObserver {
        state: State(0),
        seen: Arc::clone(&state_seen_during_optimization),
    });

    let start = SystemTime::UNIX_EPOCH;
    measurement_sender
        .send(stationary_imu(start))
        .expect("first IMU should send");
    measurement_sender
        .send(stationary_imu(start + Duration::from_secs(2)))
        .expect("later IMU should send");
    measurement_sender
        .send(visual_odometry(
            start,
            start + Duration::from_millis(100),
            0.1,
        ))
        .expect("visual odometry should send");

    let _ = backend.solve_once().expect("solve should succeed");

    assert!(
        state_seen_during_optimization.load(Ordering::SeqCst),
        "state 0 must still be available while optimizing the current batch"
    );
    assert!(
        backend.values().get_raw(State(0)).is_none(),
        "state 0 should be marginalized after optimization"
    );
}

#[test]
fn local_to_field_is_initialized_once_and_retained_by_marginalization() {
    let (_measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, _result_receiver) = tokio::sync::watch::channel(None);
    let mut config = backend_configuration();
    config.max_optimization_window = Duration::from_millis(400);
    let mut backend = VinsBackend::new(
        config,
        InitialState::default(),
        measurement_receiver,
        result_sender,
    );
    let start = SystemTime::UNIX_EPOCH;

    assert!(backend.values().get(LocalToField(0)).is_none());
    backend
        .ingest_sensor_measurements([
            visual_frame(
                vec![visual_reprojection_measurement(start, 0.0)],
                SE2::new(0.2, 1.0, 2.0),
            ),
            visual_frame(
                vec![visual_reprojection_measurement(
                    start + Duration::from_millis(50),
                    0.0,
                )],
                SE2::new(-0.4, 9.0, 8.0),
            ),
        ])
        .expect("visual frames should ingest");

    let alignment = backend
        .values()
        .get(LocalToField(0))
        .expect("first visual candidate should initialize alignment");
    assert!((alignment.x() - 1.0).abs() < 1.0e-12);
    assert!((alignment.y() - 2.0).abs() < 1.0e-12);

    backend
        .ingest_sensor_measurements([stationary_imu(start + Duration::from_secs(2))])
        .expect("later IMU should ingest");
    let _ = backend.optimize();

    assert!(backend.values().get(State(0)).is_none());
    assert!(backend.values().get(LocalToField(0)).is_some());
}

#[test]
fn late_measurements_for_marginalized_intervals_are_skipped() {
    let (measurement_sender, measurement_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (result_sender, _result_receiver) = tokio::sync::watch::channel(None);
    let mut config = backend_configuration();
    config.max_optimization_window = Duration::from_millis(400);
    let mut backend = VinsBackend::new(
        config,
        InitialState::default(),
        measurement_receiver,
        result_sender,
    );

    let start = SystemTime::UNIX_EPOCH;
    measurement_sender
        .send(stationary_imu(start))
        .expect("first IMU should send");
    measurement_sender
        .send(stationary_imu(start + Duration::from_secs(2)))
        .expect("later IMU should send");
    let _ = backend.solve_once().expect("solve should succeed");
    assert!(
        backend.values().get_raw(State(0)).is_none(),
        "state 0 should have been marginalized"
    );

    measurement_sender
        .send(stationary_imu(start + Duration::from_millis(100)))
        .expect("late IMU should send");
    let _ = backend
        .solve_once()
        .expect("late marginalized measurement should be skipped");

    assert!(
        backend.values().get_raw(State(0)).is_none(),
        "late measurement must not reintroduce marginalized state 0"
    );
}

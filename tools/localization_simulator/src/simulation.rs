use booster::ImuState;
use color_eyre::{Result, eyre::eyre};
use coordinate_systems::{Camera, Field, Ground, Head, Robot};
use field_mark_association::{
    DetectedVisualFeatures, FieldMarkAssociationParameters, FieldMarkAssociationState,
};
use linear_algebra::{Framed, IntoTransform, Isometry3 as FramedIsometry3};
use localization_3d::{
    GlobalVisualLock, SolveDiagnostics, SynchronousLocalization, SynchronousLocalizationOutput,
    initial_robot_to_field_from_field_dimensions,
};
use nalgebra::{Isometry3, Point2, Translation3, UnitQuaternion, Vector3};
use projection::camera_matrix::CameraMatrix;
use ros_z::time::Time;
use types::{
    field_dimensions::FieldDimensions,
    time_wrapper::TimeWrapper,
    visual_localization::{FieldMarkAssociation, VisualLocalizationFrame},
    visual_odometry::VisualOdometryDelta,
};

use crate::{
    config::{
        AssociationMode, FIELD_MARK_INTERVAL, SOLVE_INTERVAL, SimulationConfig, TICK_INTERVAL,
        production_association_parameters, production_localization_parameters,
    },
    sensors::SyntheticSensors,
    trajectory::{Scenario, fixed_robot_to_camera, robot_to_field_from_camera_to_field},
};

#[derive(Clone, Debug)]
/// Counts from one generated field-mark frame.
pub struct LandmarkFrameCounts {
    /// Ground-truth landmarks geometrically visible before synthetic sensor effects.
    pub ideal_visible: usize,
    /// In-bounds detections emitted after dropout and pixel noise.
    pub emitted_detections: usize,
    /// Correspondences accepted for estimator ingestion.
    pub associated: usize,
    /// Pixel detections emitted after noise and dropout, including their semantic class.
    pub detections: Vec<LandmarkDetection>,
    /// Pixel-to-field correspondences actually passed to localization.
    pub associations: Vec<FieldMarkAssociation>,
    /// Global-association reset pose passed to the backend, if one was accepted.
    pub backend_reset_robot_to_field: Option<FramedIsometry3<Robot, Field>>,
}

#[derive(Clone, Copy, Debug)]
pub struct LandmarkDetection {
    pub class: LandmarkClass,
    pub pixel: [f32; 2],
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug)]
pub enum LandmarkClass {
    GoalPost,
    LSpot,
    TSpot,
    XSpot,
    PenaltySpot,
}

#[derive(Clone, Debug)]
/// One fixed-tick snapshot retained for inspection and regression tests.
pub struct SimulationHistorySample {
    /// Logical sample time.
    pub time: Time,
    /// Synthetic IMU input passed to the localization frontend.
    pub imu: ImuState,
    /// Ground-truth robot-to-field pose.
    pub truth_robot_to_field: FramedIsometry3<Robot, Field>,
    /// Latest unconstrained backend robot-to-field pose.
    pub raw_backend_robot_to_field: Option<FramedIsometry3<Robot, Field, f64>>,
    /// Latest locked, live-odometry-propagated robot-to-field pose.
    pub live_robot_to_field: Option<FramedIsometry3<Robot, Field>>,
    /// Global visual lock state at this tick.
    pub global_visual_lock: GlobalVisualLock,
    /// Diagnostics from the latest backend solve.
    pub diagnostics: Option<SolveDiagnostics>,
    /// Field-mark counts when this tick emitted a field-mark frame.
    pub landmark_frame: Option<LandmarkFrameCounts>,
    /// Accumulated noisy camera-to-visual-odometer transform.
    pub noisy_cumulative_camera_to_visual_odometer: Isometry3<f32>,
    /// Noisy VO transition passed to the frontend at this tick.
    pub visual_odometry_delta: Option<VisualOdometryDelta>,
}

/// Deterministic synchronous runner that retains every fixed-tick result.
pub struct LocalizationSimulation {
    scenario: Scenario,
    config: SimulationConfig,
    history: Vec<SimulationHistorySample>,
    localization: SynchronousLocalization,
    sensors: SyntheticSensors,
    association_state: FieldMarkAssociationState,
    field_dimensions: FieldDimensions,
    association_parameters: FieldMarkAssociationParameters,
    step_index: usize,
    previous_camera_matrix: Option<CameraMatrix>,
    latest_backend_output: Option<SynchronousLocalizationOutput>,
    latest_live_robot_to_field: Option<FramedIsometry3<Robot, Field>>,
    latest_pose_hint: FramedIsometry3<Robot, Field>,
}

impl LocalizationSimulation {
    /// Creates an isolated deterministic localization simulation.
    pub fn new(scenario: Scenario, config: SimulationConfig) -> Result<Self> {
        config.validate().map_err(|message| eyre!(message))?;
        let field_dimensions = FieldDimensions::SPL_2025;
        let initial_camera_to_field = scenario.sample_camera_to_field(0.0);
        let truth_robot_to_field = robot_to_field_from_camera_to_field(&initial_camera_to_field);
        let initial_camera_matrix = camera_matrix(&truth_robot_to_field);
        let initial_robot_to_field =
            initial_robot_to_field_from_field_dimensions(&field_dimensions);
        let localization_parameters =
            production_localization_parameters().map_err(|message| eyre!(message))?;
        let association_parameters =
            production_association_parameters().map_err(|message| eyre!(message))?;
        let localization = SynchronousLocalization::new(
            &localization_parameters,
            &field_dimensions,
            &initial_camera_matrix,
            initial_robot_to_field,
        )?;
        Ok(Self {
            scenario,
            sensors: SyntheticSensors::new(&config, &field_dimensions),
            config,
            history: Vec::new(),
            localization,
            association_state: FieldMarkAssociationState::default(),
            field_dimensions,
            association_parameters,
            step_index: 0,
            previous_camera_matrix: None,
            latest_backend_output: None,
            latest_live_robot_to_field: None,
            latest_pose_hint: initial_robot_to_field,
        })
    }

    /// Recreates all estimator and synthetic-sensor state from the original inputs.
    pub fn reset(&mut self) -> Result<()> {
        *self = Self::new(self.scenario.clone(), self.config.clone())?;
        Ok(())
    }

    /// Returns the active validated trajectory.
    pub fn scenario(&self) -> &Scenario {
        &self.scenario
    }

    /// Returns the immutable synthetic sensor configuration.
    pub fn config(&self) -> &SimulationConfig {
        &self.config
    }

    /// Returns all generated history samples.
    pub fn history(&self) -> &[SimulationHistorySample] {
        &self.history
    }

    /// Resets and deterministically runs the complete scenario.
    pub fn replay(&mut self) -> Result<&[SimulationHistorySample]> {
        self.reset()?;
        self.run_to_end()
    }

    /// Runs the remaining bounded scenario synchronously.
    pub fn run_to_end(&mut self) -> Result<&[SimulationHistorySample]> {
        while self.step()? {}
        Ok(&self.history)
    }

    /// Processes one fixed simulation tick, returning `false` after the final tick.
    pub fn step(&mut self) -> Result<bool> {
        if self.step_index > self.scenario.tick_count() {
            return Ok(false);
        }
        let elapsed = self.step_index as f32 * TICK_INTERVAL.as_secs_f32();
        let time = Time::from_nanos(
            i64::try_from(self.step_index).unwrap_or(i64::MAX) * TICK_INTERVAL.as_nanos() as i64,
        );
        let camera_to_field = self.scenario.sample_camera_to_field(elapsed);
        let robot_to_field = robot_to_field_from_camera_to_field(&camera_to_field);
        let current_camera_matrix = camera_matrix(&robot_to_field);
        let imu = imu_state(&self.scenario, elapsed, &robot_to_field, &camera_to_field);
        self.localization.ingest_imu(time, imu)?;

        let visual_odometry = self.sensors.measure_visual_odometry(time, &camera_to_field);
        if let (Some(delta), Some(previous_camera_matrix)) = (
            visual_odometry.delta.clone(),
            self.previous_camera_matrix.as_ref(),
        ) {
            self.localization.ingest_visual_odometry(
                delta,
                previous_camera_matrix,
                &current_camera_matrix,
            )?;
        }

        let mut landmark_frame = None;
        if self.step_index % interval_steps(FIELD_MARK_INTERVAL) == 0 {
            let observations = self
                .sensors
                .observe_landmarks(&camera_to_field, &current_camera_matrix);
            let ideal_visible = observations.ideal_visible_count;
            let emitted_detections = observations.detections.supported_feature_count();
            let association_count;
            let frame = match self.config.association_mode {
                AssociationMode::KnownCorrespondences if self.step_index == 0 => {
                    association_count = 0;
                    VisualLocalizationFrame {
                        robot_to_camera: framed_robot_to_camera(),
                        associations: Vec::new(),
                        backend_reset: Some(robot_to_field.framed_transform()),
                    }
                }
                AssociationMode::KnownCorrespondences => {
                    association_count = observations.true_associations.len();
                    VisualLocalizationFrame {
                        robot_to_camera: framed_robot_to_camera(),
                        associations: observations.true_associations,
                        backend_reset: None,
                    }
                }
                AssociationMode::ProductionAssociation => {
                    let result = self.association_state.associate_visual_features_with_debug(
                        &observations.detections,
                        &current_camera_matrix,
                        &self.field_dimensions,
                        Some(self.latest_pose_hint),
                        &self.association_parameters,
                        false,
                    );
                    association_count = result.associations.len();
                    VisualLocalizationFrame {
                        robot_to_camera: framed_robot_to_camera(),
                        associations: result.associations,
                        backend_reset: result.accepted_global_pose,
                    }
                }
            };
            landmark_frame = Some(LandmarkFrameCounts {
                ideal_visible,
                emitted_detections,
                associated: association_count,
                detections: flatten_detections(&observations.detections),
                associations: frame.associations.clone(),
                backend_reset_robot_to_field: frame.backend_reset,
            });
            self.localization
                .ingest_visual_localization_frame(TimeWrapper { time, inner: frame })?;
        }

        if self.step_index % interval_steps(SOLVE_INTERVAL) == 0
            && let Some(output) = self.localization.solve_once(
                TimeWrapper {
                    time,
                    inner: &current_camera_matrix,
                },
                Some(&visual_odometry.odometer),
            )?
        {
            self.latest_backend_output = Some(output);
        }
        self.latest_live_robot_to_field = self
            .localization
            .update_live_odometry(
                TimeWrapper {
                    time,
                    inner: &current_camera_matrix,
                },
                &visual_odometry.odometer,
            )?
            .map(|field_to_robot| field_to_robot.inverse());
        if let Some(live_pose) = self.latest_live_robot_to_field {
            self.latest_pose_hint = live_pose;
        } else if let Some(backend_pose) = self.latest_backend_output.as_ref() {
            self.latest_pose_hint = backend_pose.backend_field_to_robot.inverse();
        }

        let sample = SimulationHistorySample {
            time,
            imu,
            truth_robot_to_field: robot_to_field.framed_transform(),
            raw_backend_robot_to_field: self
                .latest_backend_output
                .as_ref()
                .map(|output| output.raw_backend_robot_to_field),
            live_robot_to_field: self.latest_live_robot_to_field,
            global_visual_lock: self.localization.global_visual_lock(),
            diagnostics: self
                .latest_backend_output
                .as_ref()
                .map(|output| output.diagnostics.clone()),
            landmark_frame,
            noisy_cumulative_camera_to_visual_odometer: visual_odometry
                .odometer
                .current_left_camera_to_visual_odometer,
            visual_odometry_delta: visual_odometry.delta,
        };
        self.previous_camera_matrix = Some(current_camera_matrix);
        self.step_index += 1;
        self.history.push(sample);
        Ok(true)
    }
}

fn flatten_detections(detections: &DetectedVisualFeatures) -> Vec<LandmarkDetection> {
    let mut flattened = Vec::with_capacity(detections.supported_feature_count());
    let groups = [
        (LandmarkClass::GoalPost, detections.goalposts.as_slice()),
        (LandmarkClass::LSpot, detections.l_spots.as_slice()),
        (LandmarkClass::TSpot, detections.t_spots.as_slice()),
        (LandmarkClass::XSpot, detections.x_spots.as_slice()),
        (
            LandmarkClass::PenaltySpot,
            detections.penalty_spots.as_slice(),
        ),
    ];
    for (class, features) in groups {
        flattened.extend(features.iter().map(|feature| LandmarkDetection {
            class,
            pixel: [feature.pixel.inner.x, feature.pixel.inner.y],
            confidence: feature.confidence,
        }));
    }
    flattened
}

fn interval_steps(interval: std::time::Duration) -> usize {
    (interval.as_nanos() / TICK_INTERVAL.as_nanos()) as usize
}

fn framed_robot_to_camera() -> FramedIsometry3<Robot, Camera> {
    fixed_robot_to_camera().framed_transform()
}

pub(crate) fn camera_matrix(robot_to_field: &Isometry3<f32>) -> CameraMatrix {
    let (_, _, yaw) = robot_to_field.rotation.euler_angles();
    let ground_to_field = Isometry3::from_parts(
        Translation3::new(
            robot_to_field.translation.x,
            robot_to_field.translation.y,
            0.0,
        ),
        UnitQuaternion::from_euler_angles(0.0, 0.0, yaw),
    );
    let ground_to_robot: FramedIsometry3<Ground, Robot> =
        (robot_to_field.inverse() * ground_to_field).framed_transform();
    let robot_to_head: FramedIsometry3<Robot, Head> = Isometry3::identity().framed_transform();
    let head_to_camera: FramedIsometry3<Head, Camera> = fixed_robot_to_camera().framed_transform();
    CameraMatrix::from_normalized_focal_and_center(
        nalgebra::vector![0.625, 0.625],
        Point2::new(0.5, 0.5),
        Framed::wrap(nalgebra::vector![640.0, 480.0]),
        ground_to_robot,
        robot_to_head,
        head_to_camera,
    )
}

fn imu_state(
    scenario: &Scenario,
    elapsed: f32,
    robot_to_field: &Isometry3<f32>,
    camera_to_field: &Isometry3<f32>,
) -> ImuState {
    let dt = TICK_INTERVAL.as_secs_f32();
    let (roll, pitch, yaw) = robot_to_field.rotation.euler_angles();
    let (rotation_start, rotation_end, rotation_dt) = if elapsed + dt <= scenario.duration_seconds()
    {
        let next_camera = scenario.sample_camera_to_field(elapsed + dt);
        (
            robot_to_field.rotation,
            robot_to_field_from_camera_to_field(&next_camera).rotation,
            dt,
        )
    } else {
        let previous_camera = scenario.sample_camera_to_field((elapsed - dt).max(0.0));
        (
            robot_to_field_from_camera_to_field(&previous_camera).rotation,
            robot_to_field.rotation,
            dt,
        )
    };
    let angular_velocity = (rotation_start.inverse() * rotation_end).scaled_axis() / rotation_dt;

    let global_acceleration = trajectory_acceleration(scenario, elapsed, camera_to_field, dt);
    let gravity_in_robot = robot_to_field
        .rotation
        .inverse_transform_vector(&(global_acceleration + Vector3::new(0.0, 0.0, 9.81)));
    ImuState {
        roll_pitch_yaw: Framed::wrap(Vector3::new(roll, pitch, yaw)),
        angular_velocity: Framed::wrap(angular_velocity),
        linear_acceleration: Framed::wrap(gravity_in_robot),
    }
}

fn trajectory_acceleration(
    scenario: &Scenario,
    elapsed: f32,
    camera_to_field: &Isometry3<f32>,
    dt: f32,
) -> Vector3<f32> {
    let position = camera_to_field.translation.vector;
    if elapsed + 2.0 * dt <= scenario.duration_seconds() {
        let next = scenario
            .sample_camera_to_field(elapsed + dt)
            .translation
            .vector;
        let after_next = scenario
            .sample_camera_to_field(elapsed + 2.0 * dt)
            .translation
            .vector;
        return (after_next - 2.0 * next + position) / dt.powi(2);
    }
    if elapsed >= 2.0 * dt {
        let previous = scenario
            .sample_camera_to_field(elapsed - dt)
            .translation
            .vector;
        let before_previous = scenario
            .sample_camera_to_field(elapsed - 2.0 * dt)
            .translation
            .vector;
        return (position - 2.0 * previous + before_previous) / dt.powi(2);
    }
    Vector3::zeros()
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn stationary_known_correspondence_simulation_locks_with_finite_output() {
        let mut simulation = LocalizationSimulation::new(
            Scenario::stationary(),
            SimulationConfig {
                landmark_pixel_sigma: 0.0,
                vo_translation_sigma_m: 0.0,
                vo_rotation_sigma_rad: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
        simulation.run_to_end().unwrap();
        let last = simulation.history.last().unwrap();
        assert_eq!(last.global_visual_lock, GlobalVisualLock::Locked);
        let raw = last.raw_backend_robot_to_field.as_ref().unwrap();
        assert!(
            raw.inner
                .translation
                .vector
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(
            raw.inner
                .rotation
                .coords
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(last.live_robot_to_field.is_some());
    }

    #[test]
    fn production_association_acquires_global_lock() {
        let mut simulation = LocalizationSimulation::new(
            Scenario::stationary(),
            SimulationConfig {
                association_mode: AssociationMode::ProductionAssociation,
                landmark_pixel_sigma: 0.0,
                vo_translation_sigma_m: 0.0,
                vo_rotation_sigma_rad: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
        simulation.run_to_end().unwrap();
        assert!(
            simulation.history.iter().any(|sample| sample
                .landmark_frame
                .as_ref()
                .is_some_and(|counts| counts.associated > 0)),
            "maximum visible landmarks: {}",
            simulation
                .history
                .iter()
                .filter_map(|sample| sample.landmark_frame.as_ref())
                .map(|counts| counts.ideal_visible)
                .max()
                .unwrap_or(0)
        );
        assert_eq!(
            simulation.history.last().unwrap().global_visual_lock,
            GlobalVisualLock::Locked
        );
        let last = simulation.history.last().unwrap();
        let truth = last.truth_robot_to_field.inner;
        let estimate = last
            .live_robot_to_field
            .expect("locked production association has a live pose")
            .inner;
        assert!((estimate.translation.vector - truth.translation.vector).norm() < 0.1);
        assert!(estimate.rotation.angle_to(&truth.rotation) < 5.0_f32.to_radians());
    }

    #[test]
    fn complete_simulation_is_deterministic_for_the_same_seed() {
        let config = SimulationConfig {
            seed: 0x5eed,
            landmark_pixel_sigma: 1.0,
            vo_translation_sigma_m: 0.002,
            vo_rotation_sigma_rad: 0.001,
            ..Default::default()
        };
        let mut simulation = LocalizationSimulation::new(Scenario::six_dof_loop(), config)
            .expect("simulation initializes");
        simulation.run_to_end().expect("simulation runs");
        let first = simulation.history.clone();
        simulation.replay().expect("simulation replays");
        let second = &simulation.history;

        assert_eq!(first.len(), second.len());
        for (left, right) in first.iter().zip(second) {
            assert_eq!(left.time, right.time);
            assert_eq!(left.global_visual_lock, right.global_visual_lock);
            assert_eq!(
                left.landmark_frame.as_ref().map(|counts| (
                    counts.ideal_visible,
                    counts.emitted_detections,
                    counts.associated
                )),
                right.landmark_frame.as_ref().map(|counts| (
                    counts.ideal_visible,
                    counts.emitted_detections,
                    counts.associated
                ))
            );
            assert_relative_eq!(
                left.noisy_cumulative_camera_to_visual_odometer
                    .translation
                    .vector,
                right
                    .noisy_cumulative_camera_to_visual_odometer
                    .translation
                    .vector,
                epsilon = 1.0e-6
            );
            match (
                left.raw_backend_robot_to_field.as_ref(),
                right.raw_backend_robot_to_field.as_ref(),
            ) {
                (Some(left), Some(right)) => {
                    assert_relative_eq!(
                        left.inner.translation.vector,
                        right.inner.translation.vector,
                        epsilon = 1.0e-9
                    );
                    assert!(left.inner.rotation.angle_to(&right.inner.rotation) < 1.0e-9);
                }
                (None, None) => {}
                _ => panic!("backend output availability differs"),
            }
            match (left.live_robot_to_field, right.live_robot_to_field) {
                (Some(left), Some(right)) => {
                    assert_relative_eq!(
                        left.inner.translation.vector,
                        right.inner.translation.vector,
                        epsilon = 1.0e-6
                    );
                    assert!(left.inner.rotation.angle_to(&right.inner.rotation) < 1.0e-6);
                }
                (None, None) => {}
                _ => panic!("live output availability differs"),
            }
        }
    }

    #[test]
    fn exact_stationary_scenario_stays_near_truth() {
        let mut simulation = LocalizationSimulation::new(
            Scenario::stationary(),
            SimulationConfig {
                landmark_pixel_sigma: 0.0,
                vo_translation_sigma_m: 0.0,
                vo_rotation_sigma_rad: 0.0,
                ..Default::default()
            },
        )
        .expect("simulation initializes");
        simulation.run_to_end().expect("simulation runs");
        let last = simulation.history.last().expect("history is non-empty");
        let estimate = last
            .live_robot_to_field
            .expect("known reset establishes live localization")
            .inner;
        let truth = last.truth_robot_to_field.inner;

        assert!((estimate.translation.vector - truth.translation.vector).norm() < 0.01);
        assert!(estimate.rotation.angle_to(&truth.rotation) < 0.5_f32.to_radians());
    }
}

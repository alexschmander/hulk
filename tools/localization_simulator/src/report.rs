use booster::ImuState;
use color_eyre::{Result, eyre::eyre};
use localization_3d::{GlobalVisualLock, Localization3dParameters, SolveDiagnostics};
use nalgebra::Isometry3;
use serde::Serialize;
use types::field_dimensions::FieldDimensions;

use field_mark_association::FieldMarkAssociationParameters;

use crate::{
    SimulationConfig,
    config::{production_association_parameters, production_localization_parameters},
    simulation::{LandmarkClass, LandmarkFrameCounts, LocalizationSimulation},
    trajectory::Scenario,
};

pub const REPORT_SCHEMA_VERSION: u32 = 2;

/// Complete deterministic output from one headless localization run.
#[derive(Debug, Serialize)]
pub struct AnalysisReport {
    pub schema_version: u32,
    pub scenario: Scenario,
    pub simulation_config: SimulationConfig,
    pub field_dimensions: FieldDimensions,
    pub localization_parameters: Localization3dParameters,
    pub association_parameters: FieldMarkAssociationParameters,
    pub summary: AnalysisSummary,
    pub samples: Vec<AnalysisSample>,
}

#[derive(Debug, Serialize)]
pub struct AnalysisSummary {
    pub sample_count: usize,
    pub duration_ns: i64,
    pub global_lock_acquired_at_ns: Option<i64>,
    pub backend_error: ErrorStatistics,
    pub live_error: ErrorStatistics,
    pub backend_step: StepStatistics,
    pub live_step: StepStatistics,
}

#[derive(Debug, Serialize)]
pub struct AnalysisSample {
    pub time_ns: i64,
    pub imu: ImuSample,
    pub truth_robot_to_field: Pose3,
    pub raw_backend_robot_to_field: Option<Pose3>,
    pub live_robot_to_field: Option<Pose3>,
    pub noisy_camera_to_visual_odometer: Pose3,
    pub visual_odometry_delta: Option<VisualOdometryDeltaSample>,
    pub global_visual_lock: GlobalLock,
    pub backend_error: Option<PoseError>,
    pub live_error: Option<PoseError>,
    pub landmark_frame: Option<LandmarkFrameCountsReport>,
    pub solve_diagnostics: Option<SolveDiagnostics>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Pose3 {
    pub translation_m: [f64; 3],
    pub quaternion_xyzw: [f64; 4],
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct PoseError {
    pub translation_m: f64,
    pub rotation_rad: f64,
}

#[derive(Debug, Default, Serialize)]
pub struct ErrorStatistics {
    pub sample_count: usize,
    pub translation_rms_m: Option<f64>,
    pub translation_max_m: Option<f64>,
    pub translation_final_m: Option<f64>,
    pub rotation_rms_rad: Option<f64>,
    pub rotation_max_rad: Option<f64>,
    pub rotation_final_rad: Option<f64>,
}

#[derive(Debug, Default, Serialize)]
pub struct StepStatistics {
    pub step_count: usize,
    pub translation_max_m: Option<f64>,
    pub rotation_max_rad: Option<f64>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GlobalLock {
    Unlocked,
    WaitingForBackend,
    Locked,
}

#[derive(Clone, Debug, Serialize)]
pub struct LandmarkFrameCountsReport {
    pub ideal_visible: usize,
    pub emitted_detections: usize,
    pub associated: usize,
    pub detections: Vec<LandmarkDetectionReport>,
    pub associations: Vec<LandmarkAssociationReport>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ImuSample {
    pub roll_pitch_yaw_rad: [f64; 3],
    pub angular_velocity_rad_per_s: [f64; 3],
    pub linear_acceleration_m_per_s2: [f64; 3],
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct VisualOdometryDeltaSample {
    pub previous_time_ns: i64,
    pub current_time_ns: i64,
    pub current_camera_to_previous_camera: Pose3,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct LandmarkDetectionReport {
    pub class: LandmarkClassReport,
    pub pixel_xy: [f64; 2],
    pub confidence: f64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct LandmarkAssociationReport {
    pub detection_pixel_xy: [f64; 2],
    pub field_point_m: [f64; 3],
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LandmarkClassReport {
    GoalPost,
    LSpot,
    TSpot,
    XSpot,
    PenaltySpot,
}

/// Runs one complete deterministic scenario and builds its analysis report.
pub fn run_analysis(scenario: Scenario, config: SimulationConfig) -> Result<AnalysisReport> {
    let report_scenario = scenario.clone();
    let report_config = config.clone();
    let mut simulation = LocalizationSimulation::new(scenario, config)?;
    simulation.run_to_end()?;

    let localization_parameters =
        production_localization_parameters().map_err(|message| eyre!(message))?;
    let association_parameters =
        production_association_parameters().map_err(|message| eyre!(message))?;
    let mut backend_errors = StatisticsAccumulator::default();
    let mut live_errors = StatisticsAccumulator::default();
    let mut backend_steps = StepAccumulator::default();
    let mut live_steps = StepAccumulator::default();
    let mut previous_backend = None;
    let mut previous_live = None;
    let mut lock_acquired_at = None;
    let samples = simulation
        .history()
        .iter()
        .map(|sample| {
            let truth = sample.truth_robot_to_field.inner.cast::<f64>();
            let backend = sample
                .raw_backend_robot_to_field
                .as_ref()
                .map(|pose| pose.inner);
            let live = sample
                .live_robot_to_field
                .as_ref()
                .map(|pose| pose.inner.cast::<f64>());
            let backend_error = backend.map(|pose| pose_error(&truth, &pose));
            let live_error = live.map(|pose| pose_error(&truth, &pose));
            if let Some(error) = backend_error {
                backend_errors.add(error);
            }
            if let Some(error) = live_error {
                live_errors.add(error);
            }
            if let Some(pose) = backend {
                backend_steps.add(previous_backend.as_ref(), &pose);
                previous_backend = Some(pose);
            }
            if let Some(pose) = live {
                live_steps.add(previous_live.as_ref(), &pose);
                previous_live = Some(pose);
            }
            if lock_acquired_at.is_none() && sample.global_visual_lock == GlobalVisualLock::Locked {
                lock_acquired_at = Some(sample.time.as_nanos());
            }

            AnalysisSample {
                time_ns: sample.time.as_nanos(),
                imu: sample.imu.into(),
                truth_robot_to_field: Pose3::from_isometry(&truth),
                raw_backend_robot_to_field: backend.as_ref().map(Pose3::from_isometry),
                live_robot_to_field: live.as_ref().map(Pose3::from_isometry),
                noisy_camera_to_visual_odometer: Pose3::from_isometry(
                    &sample
                        .noisy_cumulative_camera_to_visual_odometer
                        .cast::<f64>(),
                ),
                visual_odometry_delta: sample.visual_odometry_delta.as_ref().map(|delta| {
                    VisualOdometryDeltaSample {
                        previous_time_ns: delta.previous_time.as_nanos(),
                        current_time_ns: delta.current_time.as_nanos(),
                        current_camera_to_previous_camera: Pose3::from_isometry(
                            &delta
                                .current_left_camera_to_previous_left_camera
                                .cast::<f64>(),
                        ),
                    }
                }),
                global_visual_lock: sample.global_visual_lock.into(),
                backend_error,
                live_error,
                landmark_frame: sample.landmark_frame.as_ref().map(Into::into),
                solve_diagnostics: sample.diagnostics.clone(),
            }
        })
        .collect::<Vec<_>>();
    let duration_ns = samples.last().map_or(0, |sample| sample.time_ns);

    Ok(AnalysisReport {
        schema_version: REPORT_SCHEMA_VERSION,
        scenario: report_scenario,
        simulation_config: report_config,
        field_dimensions: FieldDimensions::SPL_2025,
        localization_parameters,
        association_parameters,
        summary: AnalysisSummary {
            sample_count: samples.len(),
            duration_ns,
            global_lock_acquired_at_ns: lock_acquired_at,
            backend_error: backend_errors.finish(),
            live_error: live_errors.finish(),
            backend_step: backend_steps.finish(),
            live_step: live_steps.finish(),
        },
        samples,
    })
}

impl Pose3 {
    fn from_isometry(pose: &Isometry3<f64>) -> Self {
        let translation = pose.translation.vector;
        let quaternion = pose.rotation.quaternion();
        Self {
            translation_m: [translation.x, translation.y, translation.z],
            quaternion_xyzw: [quaternion.i, quaternion.j, quaternion.k, quaternion.w],
        }
    }
}

impl From<GlobalVisualLock> for GlobalLock {
    fn from(value: GlobalVisualLock) -> Self {
        match value {
            GlobalVisualLock::Unlocked => Self::Unlocked,
            GlobalVisualLock::WaitingForBackend => Self::WaitingForBackend,
            GlobalVisualLock::Locked => Self::Locked,
        }
    }
}

impl From<&LandmarkFrameCounts> for LandmarkFrameCountsReport {
    fn from(value: &LandmarkFrameCounts) -> Self {
        Self {
            ideal_visible: value.ideal_visible,
            emitted_detections: value.emitted_detections,
            associated: value.associated,
            detections: value
                .detections
                .iter()
                .map(|detection| LandmarkDetectionReport {
                    class: detection.class.into(),
                    pixel_xy: [detection.pixel[0] as f64, detection.pixel[1] as f64],
                    confidence: detection.confidence as f64,
                })
                .collect(),
            associations: value
                .associations
                .iter()
                .map(|association| LandmarkAssociationReport {
                    detection_pixel_xy: [
                        association.detection.inner.x as f64,
                        association.detection.inner.y as f64,
                    ],
                    field_point_m: [
                        association.field_point.inner.x as f64,
                        association.field_point.inner.y as f64,
                        association.field_point.inner.z as f64,
                    ],
                })
                .collect(),
        }
    }
}

impl From<ImuState> for ImuSample {
    fn from(value: ImuState) -> Self {
        Self {
            roll_pitch_yaw_rad: vector3_to_array(value.roll_pitch_yaw.inner.cast::<f64>()),
            angular_velocity_rad_per_s: vector3_to_array(
                value.angular_velocity.inner.cast::<f64>(),
            ),
            linear_acceleration_m_per_s2: vector3_to_array(
                value.linear_acceleration.inner.cast::<f64>(),
            ),
        }
    }
}

impl From<LandmarkClass> for LandmarkClassReport {
    fn from(value: LandmarkClass) -> Self {
        match value {
            LandmarkClass::GoalPost => Self::GoalPost,
            LandmarkClass::LSpot => Self::LSpot,
            LandmarkClass::TSpot => Self::TSpot,
            LandmarkClass::XSpot => Self::XSpot,
            LandmarkClass::PenaltySpot => Self::PenaltySpot,
        }
    }
}

fn vector3_to_array(vector: nalgebra::Vector3<f64>) -> [f64; 3] {
    [vector.x, vector.y, vector.z]
}

fn pose_error(truth: &Isometry3<f64>, estimate: &Isometry3<f64>) -> PoseError {
    PoseError {
        translation_m: (truth.translation.vector - estimate.translation.vector).norm(),
        rotation_rad: truth.rotation.angle_to(&estimate.rotation),
    }
}

#[derive(Default)]
struct StatisticsAccumulator {
    count: usize,
    translation_squared_sum: f64,
    translation_max: f64,
    translation_final: f64,
    rotation_squared_sum: f64,
    rotation_max: f64,
    rotation_final: f64,
}

impl StatisticsAccumulator {
    fn add(&mut self, error: PoseError) {
        self.count += 1;
        self.translation_squared_sum += error.translation_m.powi(2);
        self.translation_max = self.translation_max.max(error.translation_m);
        self.translation_final = error.translation_m;
        self.rotation_squared_sum += error.rotation_rad.powi(2);
        self.rotation_max = self.rotation_max.max(error.rotation_rad);
        self.rotation_final = error.rotation_rad;
    }

    fn finish(self) -> ErrorStatistics {
        let Some(count) = (self.count > 0).then_some(self.count as f64) else {
            return ErrorStatistics::default();
        };
        ErrorStatistics {
            sample_count: self.count,
            translation_rms_m: Some((self.translation_squared_sum / count).sqrt()),
            translation_max_m: Some(self.translation_max),
            translation_final_m: Some(self.translation_final),
            rotation_rms_rad: Some((self.rotation_squared_sum / count).sqrt()),
            rotation_max_rad: Some(self.rotation_max),
            rotation_final_rad: Some(self.rotation_final),
        }
    }
}

#[derive(Default)]
struct StepAccumulator {
    count: usize,
    translation_max: f64,
    rotation_max: f64,
}

impl StepAccumulator {
    fn add(&mut self, previous: Option<&Isometry3<f64>>, current: &Isometry3<f64>) {
        let Some(previous) = previous else {
            return;
        };
        self.count += 1;
        self.translation_max = self
            .translation_max
            .max((current.translation.vector - previous.translation.vector).norm());
        self.rotation_max = self
            .rotation_max
            .max(current.rotation.angle_to(&previous.rotation));
    }

    fn finish(self) -> StepStatistics {
        StepStatistics {
            step_count: self.count,
            translation_max_m: (self.count > 0).then_some(self.translation_max),
            rotation_max_rad: (self.count > 0).then_some(self.rotation_max),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stationary_report_contains_complete_samples_and_summary() {
        let report = run_analysis(
            Scenario::stationary(),
            SimulationConfig {
                landmark_pixel_sigma: 0.0,
                vo_translation_sigma_m: 0.0,
                vo_rotation_sigma_rad: 0.0,
                ..Default::default()
            },
        )
        .expect("report generation succeeds");

        assert_eq!(report.schema_version, REPORT_SCHEMA_VERSION);
        assert_eq!(report.summary.sample_count, report.samples.len());
        assert_eq!(report.summary.duration_ns, 3_000_000_000);
        assert!(report.summary.global_lock_acquired_at_ns.is_some());
        assert!(report.samples.iter().all(|sample| {
            sample
                .truth_robot_to_field
                .translation_m
                .iter()
                .all(|value| value.is_finite())
        }));
        assert!(
            report
                .samples
                .iter()
                .any(|sample| sample.landmark_frame.is_some())
        );
        assert_eq!(
            report
                .samples
                .iter()
                .filter(|sample| sample.visual_odometry_delta.is_some())
                .count(),
            report.samples.len() - 1
        );
        assert!(report.samples.iter().any(|sample| {
            sample
                .landmark_frame
                .as_ref()
                .is_some_and(|frame| !frame.detections.is_empty() && !frame.associations.is_empty())
        }));
        assert!(
            report
                .samples
                .iter()
                .any(|sample| sample.solve_diagnostics.is_some())
        );
    }

    #[test]
    fn report_serializes_as_json() {
        let report = run_analysis(Scenario::stationary(), SimulationConfig::default())
            .expect("report generation succeeds");
        let json = serde_json::to_string(&report).expect("report is JSON serializable");

        assert!(json.contains("truth_robot_to_field"));
        assert!(json.contains("translation_rms_m"));
        assert!(json.contains("solve_diagnostics"));
    }
}

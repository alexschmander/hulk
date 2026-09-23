use color_eyre::{Result, eyre::eyre};

use crate::{
    diagnostics::SolveDiagnostics,
    estimator::{Estimator, OPTIMIZATION_WINDOW},
    parameters::Localization3dParameters,
};
use booster::ImuState;
use coordinate_systems::{Local, Robot};
use kinematics::robot_kinematics::RobotKinematics;
use linear_algebra::Isometry3;
use ros_z::time::Time;
use types::camera_geometry::CameraGeometry;
use types::{
    field_dimensions::FieldDimensions,
    localization::{LocalizationEstimate, LocalizationState, LocalizationStatus},
    time_wrapper::TimeWrapper,
    visual_localization::VisualLocalizationFrame,
    visual_odometry::VisualOdometer,
};

pub struct SolveOutput {
    pub estimate: Option<LocalizationEstimate>,
    pub diagnostics: SolveDiagnostics,
}

/// Owns the graph and accepted lifecycle. The ROSZ node and deterministic runner
/// use the same operations; neither maintains another pending-measurement queue.
pub struct Localization {
    estimator: Estimator,
    parameters: Localization3dParameters,
    status: LocalizationStatus,
    latest: Option<LocalizationEstimate>,
    latest_visual: Option<Time>,
    pending_visual: Option<Time>,
    last_solve: Option<Time>,
}

impl Localization {
    pub(crate) fn has_measurement_gap(&self, time: Time) -> bool {
        time > self
            .estimator
            .latest_time()
            .saturating_add(OPTIMIZATION_WINDOW)
    }
    pub fn new(
        time: Time,
        epoch: u64,
        parameters: &Localization3dParameters,
        field: &FieldDimensions,
        camera: &CameraGeometry,
        initial_pose: Isometry3<Robot, Local>,
    ) -> Result<Self> {
        parameters.validate().map_err(|message| eyre!(message))?;
        if !camera.intrinsics.is_valid()
            || !initial_pose
                .inner
                .to_homogeneous()
                .iter()
                .all(|v| v.is_finite())
        {
            return Err(eyre!("invalid localization initialization geometry"));
        }
        Ok(Self {
            estimator: Estimator::new(
                time,
                epoch,
                initial_pose,
                camera,
                parameters.clone(),
                field,
            )?,
            parameters: parameters.clone(),
            status: LocalizationStatus {
                time,
                epoch,
                state: LocalizationState::Startup,
            },
            latest: None,
            latest_visual: None,
            pending_visual: None,
            last_solve: None,
        })
    }

    pub fn status(&self) -> LocalizationStatus {
        self.status
    }
    pub fn estimate(&self) -> Option<LocalizationEstimate> {
        self.latest
    }

    pub fn set_parameters(&mut self, parameters: &Localization3dParameters) -> Result<()> {
        parameters.validate().map_err(|message| eyre!(message))?;
        self.parameters = parameters.clone();
        self.estimator.update_parameters(parameters.clone());
        Ok(())
    }

    pub fn deadline(&self) -> Option<Time> {
        if self.status.state != LocalizationState::Tracking {
            return None;
        }
        Some(
            self.last_solve?
                .saturating_add(self.parameters.tracking_timeout)
                .min(
                    self.latest_visual?
                        .saturating_add(self.parameters.visual_tracking_timeout),
                ),
        )
    }

    pub fn advance_time(&mut self, now: Time) {
        if self.deadline().is_some_and(|deadline| deadline <= now) {
            self.status = LocalizationStatus {
                time: now,
                state: LocalizationState::LostTrack,
                ..self.status
            };
        }
    }

    pub fn ingest_imu(&mut self, time: Time, imu: ImuState) -> Result<bool> {
        self.estimator.ingest_imu(time, imu)
    }

    pub fn ingest_kinematics(&mut self, sample: TimeWrapper<RobotKinematics>) -> Result<bool> {
        self.estimator.ingest_kinematics(sample)
    }

    pub fn ingest_kinematic_odometry(
        &mut self,
        sample: types::odometry::KinematicOdometryDelta,
    ) -> Result<bool> {
        self.estimator.ingest_kinematic_odometry(sample)
    }

    pub fn ingest_visual_odometry(
        &mut self,
        sample: VisualOdometer,
        previous: Option<&CameraGeometry>,
        current: Option<&CameraGeometry>,
    ) -> Result<bool> {
        self.estimator
            .ingest_visual_odometry(sample, previous, current)
    }

    pub fn ingest_visual_localization_frame(
        &mut self,
        frame: TimeWrapper<VisualLocalizationFrame>,
    ) -> Result<bool> {
        let time = frame.time;
        // Recovery requires evidence acquired after loss, not delayed pre-loss work.
        if self.status.state == LocalizationState::LostTrack && time <= self.status.time {
            return Ok(false);
        }
        let inserted = self.estimator.ingest_visual(frame)?;
        if inserted && self.latest_visual.is_none_or(|old| time > old) {
            self.pending_visual = Some(self.pending_visual.map_or(time, |old| old.max(time)));
        }
        Ok(inserted)
    }

    pub fn solve(&mut self, now: Time) -> SolveOutput {
        self.advance_time(now);
        let solved = self.estimator.solve();
        let estimate = solved.estimate.map(|mut estimate| {
            let visual_time = self.pending_visual.or(self.latest_visual);
            let visual_valid = visual_time.is_some_and(|time| {
                time.saturating_add(self.parameters.visual_tracking_timeout) > now
                    && (self.status.state != LocalizationState::LostTrack
                        || time > self.status.time)
            });
            // An initialization candidate is not field localization until the
            // optimized frame agrees with its actual pixel observations.
            let field_valid = estimate.robot_to_field.is_some()
                && self.estimator.visual_rms().is_some_and(|rms| rms <= 10.0);
            if solved.converged && visual_valid && field_valid {
                self.latest_visual = visual_time;
                self.pending_visual = None;
                if self.status.state != LocalizationState::Tracking {
                    self.status = LocalizationStatus {
                        time: now,
                        state: LocalizationState::Tracking,
                        ..self.status
                    };
                }
            }
            if self.status.state == LocalizationState::Startup {
                estimate.robot_to_field = None;
            }
            if solved.converged {
                self.last_solve = Some(estimate.time);
            }
            self.latest = Some(estimate);
            estimate
        });
        self.advance_time(now);
        SolveOutput {
            estimate,
            diagnostics: solved.diagnostics,
        }
    }
}

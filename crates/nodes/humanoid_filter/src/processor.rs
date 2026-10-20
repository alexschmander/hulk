//! Node logic independent of the transport, so that recordings can be replayed through it.

use std::{collections::VecDeque, time::Duration};

use coordinate_systems::{Field, Ground, Odometry};
use hsl_network_messages::PlayerNumber;
use linear_algebra::{Isometry2, Point2, Pose2};
use object_tracking::{multi_bernoulli::ScanReport, odometry_history::OdometryHistory};
use projection::camera_matrix::CameraMatrix;
use ros_z::time::Time;
use types::{
    game_controller_state::GameControllerState,
    object_detection::{Object, RobocupObjectLabel},
    obstacles::Obstacle,
    players::Players,
    primary_state::PrimaryState,
    time_wrapper::TimeWrapper,
    tracked_objects::{HumanoidTrack, TrackedObjects},
    world_state::PlayerState,
};

use crate::{
    Parameters,
    filter::{HumanoidFilter, team_report},
};

/// Inputs the node reads from caches.
pub trait Inputs {
    fn primary_state(&self) -> Option<PrimaryState>;
    fn game_controller_state(&self) -> Option<GameControllerState>;
    fn player_number(&self) -> Option<PlayerNumber>;
    fn camera_matrix_nearest(&self, time: Time) -> Option<TimeWrapper<CameraMatrix>>;
}

/// How long odometry poses are kept to pair them with localization estimates of their time.
const ODOMETRY_RETENTION: Duration = Duration::from_secs(2);

pub struct Outputs {
    pub reports: Vec<ScanReport>,
    pub tracks: Option<TrackedObjects<HumanoidTrack>>,
    pub obstacles: Vec<Obstacle>,
}

/// Teammate report waiting until the filter has processed everything up to its receipt time.
struct PendingReport {
    receipt_time: Time,
    player_number: PlayerNumber,
    position_in_field: Point2<Field>,
    pose_age: Option<Duration>,
}

pub struct Processor {
    filter: HumanoidFilter,
    odometry_history: OdometryHistory,
    pending_reports: VecDeque<PendingReport>,
    last_report_times: Players<Option<Time>>,
    last_primary_state: Option<PrimaryState>,
    penalized_teammates: Players<bool>,
}

impl Default for Processor {
    fn default() -> Self {
        Self {
            filter: HumanoidFilter::new(),
            odometry_history: OdometryHistory::new(ODOMETRY_RETENTION),
            pending_reports: VecDeque::new(),
            last_report_times: Players::new(None),
            last_primary_state: None,
            penalized_teammates: Players::new(false),
        }
    }
}

impl Processor {
    pub fn filter(&self) -> &HumanoidFilter {
        &self.filter
    }

    /// Records a ground-to-field estimate referring to `time`, the time of its image.
    pub fn receive_ground_to_field(
        &mut self,
        time: Time,
        ground_to_field: Isometry2<Ground, Field>,
    ) {
        self.odometry_history
            .push_ground_to_field(time, ground_to_field);
    }

    /// Queues new teammate reports until the filter reaches their receipt time.
    pub fn receive_player_states(
        &mut self,
        player_states: &Players<Option<TimeWrapper<PlayerState>>>,
        own_player_number: Option<PlayerNumber>,
    ) {
        for (player_number, state) in player_states.iter() {
            let Some(state) = state else {
                continue;
            };
            if Some(player_number) == own_player_number
                || self.last_report_times[player_number].is_some_and(|last| state.time <= last)
            {
                continue;
            }
            self.last_report_times[player_number] = Some(state.time);
            self.pending_reports.push_back(PendingReport {
                receipt_time: state.time,
                player_number,
                position_in_field: state.inner.pose.position(),
                pose_age: state.inner.pose_age,
            });
        }
        self.pending_reports
            .make_contiguous()
            .sort_by_key(|report| report.receipt_time);
    }

    /// Processes time-ordered odometry and detections; returns outputs if any update happened.
    pub fn process(
        &mut self,
        items: impl IntoIterator<
            Item = (
                Time,
                (
                    Option<Pose2<Odometry>>,
                    Option<TimeWrapper<Vec<Object<RobocupObjectLabel>>>>,
                ),
            ),
        >,
        inputs: &impl Inputs,
        parameters: &Parameters,
    ) -> Option<Outputs> {
        let filter = &mut self.filter;
        let primary_state = inputs.primary_state();
        if self.last_primary_state == Some(PrimaryState::Penalized)
            && primary_state != Some(PrimaryState::Penalized)
        {
            // The robot was carried; the tracks no longer match its odometry.
            filter.clear();
        }
        self.last_primary_state = primary_state;

        let game_controller_state = inputs.game_controller_state();
        let own_player_number = inputs.player_number();
        if let Some(state) = &game_controller_state {
            for (player_number, penalty) in state.penalties.iter() {
                let penalized = penalty.is_some();
                if penalized && !self.penalized_teammates[player_number] {
                    filter.remove_teammate(player_number);
                }
                self.penalized_teammates[player_number] = penalized;
            }
        }
        let counts = game_controller_state
            .as_ref()
            .map(|state| robot_counts(state, own_player_number));

        let mut reports = Vec::new();
        let mut last_time = None;
        let mut updated = false;
        for (time, (odometry_pose, detected_objects)) in items {
            while self
                .pending_reports
                .front()
                .is_some_and(|report| report.receipt_time <= time)
            {
                let report = self.pending_reports.pop_front().expect("front exists");
                updated |= process_team_report(
                    filter,
                    &self.odometry_history,
                    &report,
                    counts.map_or(4, |(teammates, _)| teammates),
                    parameters,
                );
            }
            if let Some(pose) = odometry_pose {
                self.odometry_history.push_pose(time, pose);
            }
            filter.predict(time, odometry_pose, parameters);
            if let Some(detected_objects) = detected_objects
                && let Some(camera_matrix) =
                    inputs.camera_matrix_nearest(time).filter(|camera_matrix| {
                        time.abs_diff(camera_matrix.time) <= parameters.maximum_camera_matrix_age
                    })
                && let Some(report) = filter.update_with_detections(
                    time,
                    &detected_objects.inner,
                    &camera_matrix.inner,
                    counts.map(|(teammates, opponents)| teammates + opponents),
                    parameters,
                )
            {
                reports.push(report);
                updated = true;
            }
            last_time = Some(time);
        }
        // Odometry arrives much faster than images; publish only what changed.
        let last_time = last_time.filter(|_| updated)?;
        let ground_to_field = current_ground_to_field(filter, &self.odometry_history);
        Some(Outputs {
            reports,
            tracks: filter.tracked_objects(last_time, ground_to_field, parameters),
            obstacles: filter.obstacles(parameters),
        })
    }
}

/// Ground-to-field transform at the filter's current odometry pose.
fn current_ground_to_field(
    filter: &HumanoidFilter,
    odometry_history: &OdometryHistory,
) -> Option<Isometry2<Ground, Field>> {
    odometry_history.ground_to_field_at(filter.odometry_pose()?)
}

fn process_team_report(
    filter: &mut HumanoidFilter,
    odometry_history: &OdometryHistory,
    report: &PendingReport,
    active_teammates: usize,
    parameters: &Parameters,
) -> bool {
    let team = &parameters.team;
    let processing_time = filter
        .time()
        .map_or(report.receipt_time, |time| time.max(report.receipt_time));
    let pose_age = report.pose_age.unwrap_or(team.default_pose_age)
        + processing_time.duration_since(report.receipt_time);
    if pose_age > team.maximum_pose_age {
        return false;
    }
    let (Some(ground_to_field), Some(odometry_pose)) = (
        current_ground_to_field(filter, odometry_history),
        filter.odometry_pose(),
    ) else {
        return false;
    };
    filter.predict(processing_time, None, parameters);
    let team_report = team_report(
        report.player_number,
        report.position_in_field,
        pose_age,
        ground_to_field,
        odometry_pose.as_transform::<Ground>(),
        parameters,
    );
    filter.update_with_team_report(processing_time, &team_report, active_teammates, parameters);
    true
}

/// Number of active teammates other than this robot, and of active opponents.
fn robot_counts(
    state: &GameControllerState,
    own_player_number: Option<PlayerNumber>,
) -> (usize, usize) {
    let teammates = state
        .penalties
        .iter()
        .filter(|(player_number, penalty)| {
            penalty.is_none() && Some(*player_number) != own_player_number
        })
        .count();
    let opponents = state
        .opponent_penalties
        .iter()
        .filter(|(_, penalty)| penalty.is_none())
        .count();
    (teammates, opponents)
}

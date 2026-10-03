use color_eyre::Result;
use coordinate_systems::Robot;
use linear_algebra::{Isometry3, Point3, Vector3};
use nalgebra::Matrix2;
use ros_z::time::Time;

use super::Estimator;

/// Accepted, normalized observations. Replay bypasses stream admission, not factor construction.
#[derive(Clone)]
pub(super) enum MotionRecord {
    Imu {
        time: Time,
        angular_velocity: Vector3<Robot, f64>,
        attitude: linear_algebra::Orientation3<coordinate_systems::ImuReference, f64>,
        /// Raw specific force; calibration is applied when rebuilding intervals.
        force: Vector3<Robot, f64>,
    },
    Feet {
        time: Time,
        left: Point3<Robot, f64>,
        right: Point3<Robot, f64>,
    },
    Kinematic {
        previous_time: Time,
        time: Time,
        translation: linear_algebra::Vector2<coordinate_systems::Ground, f64>,
        information_root: Matrix2<f64>,
    },
    Visual {
        previous_time: Time,
        time: Time,
        transform: Isometry3<Robot, Robot, f64>,
    },
}

impl MotionRecord {
    fn start(&self) -> Time {
        match self {
            Self::Imu { time, .. } | Self::Feet { time, .. } => *time,
            Self::Kinematic { previous_time, .. } | Self::Visual { previous_time, .. } => {
                *previous_time
            }
        }
    }

    fn end(&self) -> Time {
        match self {
            Self::Imu { time, .. }
            | Self::Feet { time, .. }
            | Self::Kinematic { time, .. }
            | Self::Visual { time, .. } => *time,
        }
    }

    fn insert(self, estimator: &mut Estimator) -> Result<()> {
        match self {
            Self::Imu {
                time,
                angular_velocity,
                attitude,
                force,
            } => estimator.insert_imu(time, angular_velocity, attitude, force),
            Self::Feet { time, left, right } => estimator.insert_feet(time, left, right),
            Self::Kinematic {
                previous_time,
                time,
                translation,
                information_root,
            } => estimator.insert_kinematic_odometry(
                previous_time,
                time,
                translation,
                information_root,
            ),
            Self::Visual {
                previous_time,
                time,
                transform,
            } => estimator.insert_visual_odometry(previous_time, time, transform),
        }
    }
}

impl Estimator {
    pub(super) fn accept_motion(&mut self, record: MotionRecord) -> Result<bool> {
        record.clone().insert(self)?;
        self.commit_time(record.end());
        let first = (self.oldest_window_segment() - 1).max(0);
        let start =
            Time::from_nanos(self.origin.as_nanos() + first * self.parameters.timing.knot_ns());
        if start > self.history_start {
            self.motion_history.retain(|record| record.start() >= start);
            // Preserve the source sample needed to interpolate the left boundary.
            if let Some((&before, _)) = self.attitudes.range(..=start).next_back() {
                self.attitudes.retain(|time, _| *time >= before);
            }
            self.history_start = start;
        }
        self.motion_history.push(record);
        Ok(true)
    }
}

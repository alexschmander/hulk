//! Pairs localization estimates with the odometry poses of their times.

use std::{collections::VecDeque, time::Duration};

use coordinate_systems::{Field, Ground, Odometry};
use linear_algebra::{Isometry2, Pose2};
use ros_z::time::Time;

use crate::gaussian::wrap_angle;

/// Recent odometry poses and ground-to-field estimates, each with the time it refers to.
///
/// Ground moves with the robot, so a ground-to-field estimate only holds together with the
/// odometry pose of its own time. Localization refers to the time of its image and arrives late,
/// so combining it with the current pose would turn the field around the robot by however much
/// the robot rotated in the meantime. Paired with the pose of its time, it yields an
/// odometry-to-field transform that stays valid while the robot moves, until localization or
/// odometry drift.
#[derive(Clone, Debug)]
pub struct OdometryHistory {
    poses: VecDeque<(Time, Pose2<Odometry>)>,
    /// Estimates whose odometry pose has not been processed yet, ordered by time.
    pending_ground_to_field: VecDeque<(Time, Isometry2<Ground, Field>)>,
    odometry_to_field: Option<Isometry2<Odometry, Field>>,
    retention: Duration,
}

impl OdometryHistory {
    pub fn new(retention: Duration) -> Self {
        Self {
            poses: VecDeque::new(),
            pending_ground_to_field: VecDeque::new(),
            odometry_to_field: None,
            retention,
        }
    }

    /// Appends the odometry pose at `time`; poses must come in time order, others are ignored.
    pub fn push_pose(&mut self, time: Time, pose: Pose2<Odometry>) {
        if self.poses.back().is_some_and(|(latest, _)| time <= *latest) {
            return;
        }
        self.poses.push_back((time, pose));
        while self
            .poses
            .front()
            .is_some_and(|(stamp, _)| time.duration_since(*stamp) > self.retention)
        {
            self.poses.pop_front();
        }
        self.anchor();
    }

    /// Records a ground-to-field estimate referring to `time`, which may lie ahead of the
    /// odometry processed so far.
    pub fn push_ground_to_field(&mut self, time: Time, ground_to_field: Isometry2<Ground, Field>) {
        let index = self
            .pending_ground_to_field
            .partition_point(|(stamp, _)| *stamp <= time);
        self.pending_ground_to_field
            .insert(index, (time, ground_to_field));
        self.anchor();
    }

    pub fn clear(&mut self) {
        self.poses.clear();
        self.pending_ground_to_field.clear();
        self.odometry_to_field = None;
    }

    /// Anchors the newest pending estimate whose odometry pose is known, dropping older ones.
    fn anchor(&mut self) {
        let Some(latest) = self.poses.back().map(|(time, _)| *time) else {
            return;
        };
        let reached = self
            .pending_ground_to_field
            .partition_point(|(stamp, _)| *stamp <= latest);
        let reached: Vec<_> = self.pending_ground_to_field.drain(..reached).collect();
        let anchored = reached
            .into_iter()
            .rev()
            .find_map(|(time, ground_to_field)| {
                let ground_to_odometry = self.pose_at(time)?.as_transform::<Ground>();
                Some(ground_to_field * ground_to_odometry.inverse())
            });
        if anchored.is_some() {
            self.odometry_to_field = anchored;
        }
    }

    /// Pose at `time`, interpolated between the surrounding poses, if `time` lies within the
    /// history.
    pub fn pose_at(&self, time: Time) -> Option<Pose2<Odometry>> {
        let after = self.poses.partition_point(|(stamp, _)| *stamp < time);
        let (after_time, after_pose) = *self.poses.get(after)?;
        if after_time == time || after == 0 {
            return (after_time == time).then_some(after_pose);
        }
        let (before_time, before_pose) = self.poses[after - 1];
        let fraction = (time.duration_since(before_time).as_secs_f32()
            / after_time.duration_since(before_time).as_secs_f32())
        .clamp(0.0, 1.0);
        let position =
            before_pose.position() + (after_pose.position() - before_pose.position()) * fraction;
        let angle =
            before_pose.angle() + wrap_angle(after_pose.angle() - before_pose.angle()) * fraction;
        Some(Pose2::new(position, angle))
    }

    /// Odometry-to-field transform from the latest ground-to-field estimate whose odometry pose
    /// is known.
    pub fn odometry_to_field(&self) -> Option<Isometry2<Odometry, Field>> {
        self.odometry_to_field
    }

    /// Ground-to-field transform at the odometry `pose`.
    pub fn ground_to_field_at(&self, pose: Pose2<Odometry>) -> Option<Isometry2<Ground, Field>> {
        Some(self.odometry_to_field()? * pose.as_transform::<Ground>())
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::{FRAC_PI_2, PI};

    use approx::assert_relative_eq;
    use linear_algebra::{point, vector};

    use super::*;

    fn time(milliseconds: i64) -> Time {
        Time::from_nanos((1_000 + milliseconds) * 1_000_000)
    }

    #[test]
    fn interpolates_between_poses_across_the_angle_wrap() {
        let mut history = OdometryHistory::new(Duration::from_secs(1));
        history.push_pose(time(0), Pose2::new(point![0.0, 0.0], 3.0));
        history.push_pose(time(10), Pose2::new(point![1.0, 2.0], -3.0));

        let pose = history.pose_at(time(5)).unwrap();

        assert_relative_eq!(pose.position(), point![0.5, 1.0]);
        assert_relative_eq!(wrap_angle(pose.angle()).abs(), PI, epsilon = 0.01);
        assert!(history.pose_at(time(11)).is_none());
        assert!(history.pose_at(time(-1)).is_none());
    }

    #[test]
    fn forgets_poses_older_than_the_retention() {
        let mut history = OdometryHistory::new(Duration::from_millis(100));
        for milliseconds in (0..=300).step_by(10) {
            history.push_pose(time(milliseconds), Pose2::zero());
        }

        assert!(history.pose_at(time(150)).is_none());
        assert!(history.pose_at(time(250)).is_some());
    }

    #[test]
    fn late_localization_is_paired_with_the_pose_of_its_time() {
        let mut history = OdometryHistory::new(Duration::from_secs(1));
        // The robot stands at the odometry origin facing +x, which is the field's +y at (1, 2).
        history.push_pose(time(0), Pose2::zero());
        // It walks 1 m forward and turns left before the localization of the first image arrives.
        history.push_pose(time(100), Pose2::new(point![1.0, 0.0], FRAC_PI_2));
        history.push_ground_to_field(
            time(0),
            Isometry2::<Ground, Field>::from_parts(vector![1.0, 2.0], FRAC_PI_2),
        );

        let ground_to_field = history
            .ground_to_field_at(history.pose_at(time(100)).unwrap())
            .unwrap()
            .as_pose();

        assert_relative_eq!(ground_to_field.position(), point![1.0, 3.0], epsilon = 1e-5);
        assert_relative_eq!(
            wrap_angle(ground_to_field.angle()).abs(),
            PI,
            epsilon = 1e-5
        );
    }

    #[test]
    fn anchored_localization_outlives_the_retention() {
        let mut history = OdometryHistory::new(Duration::from_millis(100));
        history.push_pose(time(0), Pose2::zero());
        history.push_ground_to_field(time(0), Isometry2::from_parts(vector![1.0, 0.0], 0.0));
        for milliseconds in (10..=1_000).step_by(10) {
            history.push_pose(time(milliseconds), Pose2::zero());
        }

        assert!(history.odometry_to_field().is_some());
    }

    #[test]
    fn localization_ahead_of_odometry_waits_for_its_pose() {
        let mut history = OdometryHistory::new(Duration::from_secs(1));
        history.push_pose(time(0), Pose2::zero());
        history.push_ground_to_field(time(0), Isometry2::from_parts(vector![1.0, 0.0], 0.0));
        history.push_ground_to_field(time(50), Isometry2::from_parts(vector![5.0, 0.0], 0.0));

        let first = history.odometry_to_field().unwrap();
        assert_relative_eq!(first.as_pose().position(), point![1.0, 0.0]);

        history.push_pose(time(60), Pose2::zero());
        let second = history.odometry_to_field().unwrap();
        assert_relative_eq!(second.as_pose().position(), point![5.0, 0.0]);
    }
}

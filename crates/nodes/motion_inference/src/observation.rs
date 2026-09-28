use ::kinematics::joints::Joints;
use color_eyre::eyre::{Result, ensure};
use coordinate_systems::Robot;
use linear_algebra::Vector3;
use nalgebra::{Quaternion, UnitQuaternion};
use ros_z::time::Time;

use crate::config::Parameters;

/// Odometry uses IMU yaw minus a fixed startup offset. The offset cancels in
/// relative rotations, so retaining source-stamped IMU yaw avoids another async
/// lookup when aligning kick observations with the selected sensor frame.
#[derive(Default)]
pub(crate) struct YawHistory(std::collections::VecDeque<(Time, f32)>);

impl YawHistory {
    pub fn record(&mut self, time: Time, yaw: f32) {
        if self.0.back().is_some_and(|(last, _)| time <= *last) {
            return;
        }
        self.0.push_back((time, yaw));
        while self.0.len() > 256
            || self.0.front().is_some_and(|(old, _)| {
                time.duration_since(*old) > std::time::Duration::from_millis(500)
            })
        {
            self.0.pop_front();
        }
    }

    pub fn at(&self, time: Time, maximum_gap: std::time::Duration) -> Option<f32> {
        let before = self.0.iter().rev().find(|(stamp, _)| *stamp <= time);
        let after = self.0.iter().find(|(stamp, _)| *stamp >= time);
        match (before, after) {
            (Some(&(a, yaw_a)), Some(&(b, yaw_b))) => {
                if time.duration_since(a) > maximum_gap || b.duration_since(time) > maximum_gap {
                    return None;
                }
                if a == b {
                    return Some(yaw_a);
                }
                let delta = (yaw_b - yaw_a).sin().atan2((yaw_b - yaw_a).cos());
                Some(
                    yaw_a
                        + delta * time.duration_since(a).as_secs_f32()
                            / b.duration_since(a).as_secs_f32(),
                )
            }
            (Some(&(stamp, yaw)), None) if time.duration_since(stamp) <= maximum_gap => Some(yaw),
            (None, Some(&(stamp, yaw))) if stamp.duration_since(time) <= maximum_gap => Some(yaw),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SensorFrame {
    pub timestamp: Time,
    pub position: Joints<f32>,
    pub velocity: Joints<f32>,
    pub orientation: Quaternion<f32>,
    pub gyro: Vector3<Robot>,
    pub last_commanded_position: Joints<f32>,
}

impl SensorFrame {
    pub fn validate_at(&self, now: Time, parameters: &Parameters) -> Result<()> {
        self.validate(parameters)?;
        ensure!(self.timestamp <= now, "sensor timestamp is in the future");
        ensure!(
            now.duration_since(self.timestamp) <= parameters.timing.maximum_sensor_age,
            "sensor frame expired"
        );
        Ok(())
    }

    pub fn validate(&self, parameters: &Parameters) -> Result<()> {
        ensure!(
            self.position
                .into_iter()
                .chain(self.velocity)
                .chain(self.last_commanded_position)
                .chain(self.orientation.coords.iter().copied())
                .chain(self.gyro.inner.iter().copied())
                .all(f32::is_finite),
            "non-finite sensor frame"
        );
        let q = self.orientation;
        let norm = (q.w * q.w + q.i * q.i + q.j * q.j + q.k * q.k).sqrt();
        ensure!(
            (norm - 1.0).abs() < parameters.observation.quaternion_norm_tolerance,
            "invalid orientation quaternion norm: {norm}"
        );
        Ok(())
    }

    pub fn rotation(&self) -> UnitQuaternion<f32> {
        UnitQuaternion::new_normalize(self.orientation)
    }

    pub fn gravity(&self) -> [f32; 3] {
        (self.rotation().inverse() * nalgebra::Vector3::new(0.0, 0.0, -1.0)).into()
    }
}

#[derive(Clone, Default)]
pub struct VelocityEstimator {
    previous: Option<(Time, Joints<f32>)>,
    pub walking: Joints<f32>,
    pub get_up: Joints<f32>,
}

impl VelocityEstimator {
    pub fn update(&mut self, sensor: &SensorFrame, parameters: &Parameters) -> Result<()> {
        if let Some((time, _)) = self.previous {
            ensure!(sensor.timestamp >= time, "sensor time moved backwards");
            if sensor.timestamp == time {
                return Ok(());
            }
        }
        let recent_sample = self.previous.and_then(|(time, position)| {
            let elapsed_frames = (sensor.timestamp.duration_since(time).as_secs_f32()
                / parameters.timing.sensor_period.as_secs_f32())
            .floor()
            .max(1.0);
            (elapsed_frames <= parameters.observation.maximum_velocity_sample_gap_frames)
                .then_some((position, elapsed_frames))
        });
        let maximum_change = parameters
            .observation
            .maximum_walking_velocity_change_degrees
            .to_radians();
        let sample_frequency = (1.0 / parameters.timing.sensor_period.as_secs_f64()) as f32;
        for (joint, position) in sensor.position.enumerate() {
            let velocity = match recent_sample {
                Some((previous_position, elapsed_frames)) => {
                    (position - previous_position[joint]) * sample_frequency / elapsed_frames
                }
                None => sensor.velocity[joint],
            };
            self.get_up[joint] = velocity;
            let previous_velocity = self.walking[joint];
            self.walking[joint] += (velocity - previous_velocity).clamp(
                (-maximum_change).min(-previous_velocity),
                maximum_change.max(-previous_velocity),
            );
        }
        self.previous = Some((sensor.timestamp, sensor.position));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::time;
    use std::{f32::consts::PI, time::Duration};

    #[test]
    fn yaw_interpolation_crosses_wrap_and_rejects_old_epochs() {
        let mut history = YawHistory::default();
        history.record(time(100), 179.0_f32.to_radians());
        history.record(time(120), -179.0_f32.to_radians());
        let gap = Duration::from_millis(30);
        assert!((history.at(time(110), gap).unwrap().abs() - PI).abs() < 1e-6);
        assert!(history.at(time(60), gap).is_none());
        assert!(history.at(time(160), gap).is_none());
        history.record(time(110), 0.0); // Out-of-order samples cannot rewrite history.
        assert!((history.at(time(110), gap).unwrap().abs() - PI).abs() < 1e-6);
        history.record(time(700), 0.1);
        assert!(history.at(time(110), gap).is_none());
    }
}

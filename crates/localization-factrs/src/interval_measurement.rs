use std::time::SystemTime;

use crate::{
    factors::{
        foot_above_ground::FootHeightMeasurement, visual_odometry::VisualOdometryMeasurement,
    },
    measurements::{ImuMeasurement, ResetMeasurement, SensorMeasurement, VisualFrameMeasurement},
};

pub struct IntervalMeasurements {
    pub resets: Vec<ResetMeasurement>,
    pub imu: Vec<ImuMeasurement>,
    pub visual: Vec<VisualFrameMeasurement>,
    pub visual_odometry: Vec<VisualOdometryMeasurement>,
    pub foot_heights: Vec<FootHeightMeasurement>,
}

impl IntervalMeasurements {
    pub fn new() -> Self {
        Self {
            resets: Vec::new(),
            imu: Vec::new(),
            visual: Vec::new(),
            visual_odometry: Vec::new(),
            foot_heights: Vec::new(),
        }
    }

    pub fn push_imu(&mut self, imu: ImuMeasurement) {
        insert_sorted(&mut self.imu, imu, |imu| imu.time);
    }

    pub fn push(&mut self, measurement: SensorMeasurement) {
        match measurement {
            SensorMeasurement::Reset(reset) => self.push_reset(reset),
            SensorMeasurement::Imu(imu) => self.push_imu(imu),
            SensorMeasurement::Visual(visual) => self.push_visual(visual),
            SensorMeasurement::VisualOdometry(visual_odometry) => {
                self.push_visual_odometry(visual_odometry)
            }
            SensorMeasurement::FootHeights(foot_heights) => self.push_foot_heights(foot_heights),
        }
    }

    pub fn push_visual(&mut self, visual: VisualFrameMeasurement) {
        insert_visual_frame(&mut self.visual, visual);
    }

    pub fn push_reset(&mut self, reset: ResetMeasurement) {
        insert_sorted(&mut self.resets, reset, |measurement| measurement.time);
    }

    pub fn push_visual_odometry(&mut self, visual_odometry: VisualOdometryMeasurement) {
        insert_sorted(&mut self.visual_odometry, visual_odometry, |measurement| {
            measurement.current_time
        });
    }

    pub fn push_foot_heights(&mut self, foot_heights: FootHeightMeasurement) {
        insert_sorted(&mut self.foot_heights, foot_heights, |measurement| {
            measurement.time
        });
    }

    pub fn latest_reset(&self) -> Option<&ResetMeasurement> {
        self.resets.last()
    }

    pub fn retain_at_or_after(&mut self, time: SystemTime) {
        self.resets.retain(|measurement| measurement.time >= time);
        self.imu.retain(|measurement| measurement.time >= time);
        self.visual
            .retain(|frame| visual_frame_time(frame).is_some_and(|frame_time| frame_time >= time));
        self.visual_odometry.retain(|measurement| {
            measurement.previous_time >= time && measurement.current_time >= time
        });
        self.foot_heights
            .retain(|measurement| measurement.time >= time);
    }
}

fn insert_visual_frame(frames: &mut Vec<VisualFrameMeasurement>, visual: VisualFrameMeasurement) {
    insert_sorted(frames, visual, |visual| {
        visual_frame_time(visual).expect("visual frames must contain at least one measurement")
    });
}

fn visual_frame_time(visual: &VisualFrameMeasurement) -> Option<SystemTime> {
    Some(visual.measurements.first()?.time)
}

fn insert_sorted<T, K: Ord>(vec: &mut Vec<T>, item: T, key: impl Fn(&T) -> K) {
    let index = vec
        .binary_search_by_key(&key(&item), key)
        .unwrap_or_else(|index| index);
    vec.insert(index, item);
}

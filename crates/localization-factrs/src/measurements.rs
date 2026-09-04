use std::time::SystemTime;

use booster::ImuState;
use coordinate_systems::{Field, Pixel};
use factrs::{core::SE3, variables::SE2};
use linear_algebra::{Point2 as FramedPoint2, Point3 as FramedPoint3};
use nalgebra::{Point2, Point3};

use crate::factors::{
    foot_above_ground::FootHeightMeasurement, visual_odometry::VisualOdometryMeasurement,
};
use crate::initial_state::InitialState;

#[derive(Debug, Clone)]
pub enum SensorMeasurement {
    Reset(ResetMeasurement),
    Imu(ImuMeasurement),
    Visual(VisualFrameMeasurement),
    VisualOdometry(VisualOdometryMeasurement),
    FootHeights(FootHeightMeasurement),
}

impl SensorMeasurement {
    pub fn time(&self) -> SystemTime {
        match self {
            SensorMeasurement::Reset(reset) => reset.time,
            SensorMeasurement::Imu(imu) => imu.time,
            SensorMeasurement::Visual(visual) => {
                visual
                    .measurements
                    .first()
                    .expect("visual frames must contain at least one measurement")
                    .time
            }
            SensorMeasurement::VisualOdometry(visual_odometry) => visual_odometry.current_time,
            SensorMeasurement::FootHeights(foot_heights) => foot_heights.time,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VisualFrameMeasurement {
    pub local_to_field_candidate: SE2<f64>,
    pub measurements: Vec<VisualReprojectionMeasurement>,
}

#[derive(Debug, Clone)]
pub struct ResetMeasurement {
    pub time: SystemTime,
    pub generation: u64,
    pub initial_state: InitialState,
}

#[derive(Debug, Clone)]
pub struct ImuMeasurement {
    pub time: SystemTime,
    pub state: ImuState,
}

/// A fixed visual feature association in domain frames.
#[derive(Debug, Clone, Copy)]
pub struct VisualReprojectionAssociation {
    /// Detected feature location in pixel coordinates.
    pub detection: FramedPoint2<Pixel>,
    /// Associated field feature in field coordinates.
    pub field_point: FramedPoint3<Field>,
}

#[derive(Debug, Clone)]
pub struct VisualReprojectionMeasurement {
    /// Time of the detection
    pub time: SystemTime,
    /// The detected feature in image space.
    pub detection: Point2<f64>,
    /// The associated 3d field point.
    pub field_point: Point3<f64>,
    /// Transformation from the robot frame to the camera frame
    pub robot_to_camera: SE3<f64>,
}

use std::time::SystemTime;

use booster::ImuState;
use factrs::{core::SE3, variables::SE2};
use nalgebra::{Point2, Point3};

use crate::factors::{
    foot_above_ground::FootHeightMeasurement, visual_odometry::VisualOdometryMeasurement,
};
use crate::initial_state::InitialState;

#[derive(Debug, Clone)]
pub enum SensorMeasurement {
    Configuration(Box<crate::BackendConfiguration>),
    Reset(ResetMeasurement),
    Imu(ImuMeasurement),
    Visual(VisualFrameMeasurement),
    VisualOdometry(VisualOdometryMeasurement),
    FootHeights(FootHeightMeasurement),
}

#[derive(Debug, Clone)]
pub struct VisualFrameMeasurement {
    pub time: SystemTime,
    pub robot_to_camera: SE3<f64>,
    /// Bootstrap alignment only. Tracking frames leave this unset.
    pub local_to_field_candidate: Option<SE2<f64>>,
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

/// Optimizer-precision association, converted from the wire type by the frontend.
#[derive(Debug, Clone, Copy)]
pub struct VisualReprojectionMeasurement {
    /// The detected feature in image space.
    pub detection: Point2<f64>,
    /// The associated 3d field point.
    pub field_point: Point3<f64>,
}

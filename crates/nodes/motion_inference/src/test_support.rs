use crate::{
    config::{Parameters, Policy},
    observation::SensorFrame,
};
use kinematics::joints::Joints;
use linear_algebra::Vector3;
use ros_z::time::Time;
use std::sync::Arc;
use types::joint_limits::JointLimits;

pub fn parameters() -> Arc<Parameters> {
    let value: serde_json::Value = json5::from_str(include_str!(
        "../../../../etc/parameters/base/motion_inference.json5"
    ))
    .unwrap();
    let parameters: Parameters = serde_json::from_value(value).unwrap();
    parameters.validate().unwrap();
    Arc::new(parameters)
}

pub fn sensor() -> SensorFrame {
    let position = Policy::Walk.offset(&parameters());
    SensorFrame {
        timestamp: Time::zero(),
        position,
        velocity: Joints::fill(0.0),
        orientation: nalgebra::Quaternion::identity(),
        gyro: Vector3::zeros(),
        last_commanded_position: position,
    }
}

pub fn limits() -> JointLimits {
    JointLimits {
        position: Joints::fill([-3.0, 3.0]),
        maximum_torque: Joints::fill(100.0),
    }
}

pub fn time(ms: i64) -> Time {
    Time::from_nanos(ms * 1_000_000)
}

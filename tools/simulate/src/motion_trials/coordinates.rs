//! Compare live policy inputs with physical sole, trunk and ball frames.
use std::collections::HashMap;

use crate::robot_io::Data;
use motion_inference::{
    inference::InferenceCommand,
    locomotion::KickRequest,
    trace::{Event, Record},
};
use ros_z::{service::RequestId, time::Time};

struct Frame {
    time: Time,
    origin: nalgebra::Vector2<f64>,
    yaw: f64,
    ball: nalgebra::Vector2<f64>,
    velocity: nalgebra::Vector2<f64>,
}

#[derive(Default)]
pub(super) struct Audit {
    frames: Vec<Frame>,
    requests: HashMap<Option<RequestId>, KickRequest>,
    position_errors: Vec<f64>,
    velocity_errors: Vec<f64>,
    aim_errors: Vec<f64>,
    tensor_position_errors: Vec<f64>,
    tensor_velocity_errors: Vec<f64>,
    tensor_aim_errors: Vec<f64>,
    worst_aim: Option<serde_json::Value>,
}

impl Audit {
    pub fn capture(&mut self, data: &Data, prefix: &str, ball_prefix: &str, time: Time) {
        // Deliberately independent of RobotBinding::ground_to_world. The model's
        // training origin is halfway between sole points, with trunk yaw.
        let soles = ["left_foot_link", "right_foot_link"].map(|name| {
            let body = data.body(&format!("{prefix}{name}")).unwrap();
            let view = body.view(data);
            let position = nalgebra::Vector3::from_column_slice(&view.xpos);
            let rotation = nalgebra::Matrix3::from_row_slice(&view.xmat);
            position + rotation * nalgebra::vector![0.0, 0.0, -0.038]
        });
        let trunk = data.body(&format!("{prefix}Trunk")).unwrap();
        let q = trunk.view(data).xquat;
        let yaw = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            q[0], q[1], q[2], q[3],
        ))
        .euler_angles()
        .2;
        let ball = data.body(&format!("{ball_prefix}ball")).unwrap();
        let p = ball.view(data).xpos;
        let joint = data
            .joint(&format!("{ball_prefix}ball_free_joint"))
            .unwrap();
        let velocity = joint.view(data).qvel;
        self.frames.push(Frame {
            time,
            origin: ((soles[0] + soles[1]) * 0.5).xy(),
            yaw,
            ball: nalgebra::vector![p[0], p[1]],
            velocity: nalgebra::vector![velocity[0], velocity[1]],
        });
    }

    pub fn inspect(&mut self, record: &Record, start: Time, end: Time, goal_x: f64) {
        match &record.event {
            Event::Request(InferenceCommand::Kick(command)) => {
                self.requests
                    .insert(record.request_id.clone(), command.request);
            }
            Event::Input {
                inference_time,
                values,
                ..
            } if *inference_time >= start && *inference_time < end && values.len() == 59 => {
                let Some(request) = self.requests.get(&record.request_id) else {
                    return;
                };
                let index = self
                    .frames
                    .partition_point(|frame| frame.time < *inference_time);
                let frame = self.frames[index.saturating_sub(1)..self.frames.len().min(index + 1)]
                    .iter()
                    .min_by_key(|frame| frame.time.abs_diff(*inference_time));
                let Some(frame) = frame
                    .filter(|frame| frame.time.abs_diff(*inference_time).as_secs_f64() <= 0.006)
                else {
                    return;
                };
                let rotation = nalgebra::Rotation2::new(-frame.yaw);
                let position = rotation * (frame.ball - frame.origin);
                // Absolute world velocity, rotated only. Ego motion is not subtracted.
                let velocity = rotation * frame.velocity;
                let aim = rotation * (nalgebra::vector![goal_x, 0.0] - frame.ball);
                let angle = aim.y.atan2(aim.x);
                let difference =
                    |a: f64, b: f64| ((a - b).sin().atan2((a - b).cos())).abs().to_degrees();
                let requested_position = nalgebra::vector![
                    request.ball_position.x() as f64,
                    request.ball_position.y() as f64
                ];
                let requested_velocity = nalgebra::vector![
                    request.ball_velocity.x() as f64,
                    request.ball_velocity.y() as f64
                ];
                self.position_errors
                    .push((requested_position - position).norm());
                self.velocity_errors
                    .push((requested_velocity - velocity).norm());
                let aim_error = difference(request.direction.angle() as f64, angle);
                if self.aim_errors.iter().all(|previous| *previous < aim_error) {
                    self.worst_aim = Some(
                        serde_json::json!({"ball":frame.ball.as_slice(),"origin":frame.origin.as_slice(),"yaw":frame.yaw,"requested_angle":request.direction.angle(),"expected_angle":angle,"goal_x":goal_x,"time":inference_time.as_nanos()}),
                    );
                }
                self.aim_errors.push(aim_error);
                self.tensor_position_errors.push(
                    (nalgebra::vector![values[6] as f64, values[7] as f64] * 10.0 - position)
                        .norm(),
                );
                self.tensor_velocity_errors.push(
                    (nalgebra::vector![values[51] as f64, values[52] as f64] * 10.0 - velocity)
                        .norm(),
                );
                self.tensor_aim_errors.push(difference(
                    (values[54] as f64).atan2(values[55] as f64),
                    request.direction.angle() as f64,
                ));
                assert!(
                    difference(
                        values[8] as f64 * std::f64::consts::PI,
                        request.direction.angle() as f64
                    ) < 1e-4,
                    "direction/pi feature disagrees with kick request"
                );
            }
            _ => {}
        }
    }

    pub fn finish(&mut self, filtering: bool) -> serde_json::Value {
        fn stats(samples: &mut [f64]) -> serde_json::Value {
            samples.sort_by(f64::total_cmp);
            serde_json::json!({"count":samples.len(),"p95":samples.get(samples.len()*95/100),"max":samples.last()})
        }
        let result = serde_json::json!({
            "warmup_seconds":0.25,
            "request_position_m":stats(&mut self.position_errors),
            "request_velocity_m_s":stats(&mut self.velocity_errors),
            "request_aim_degrees":stats(&mut self.aim_errors),
            "tensor_position_m":stats(&mut self.tensor_position_errors),
            "tensor_velocity_m_s":stats(&mut self.tensor_velocity_errors),
            "tensor_aim_encoding_degrees":stats(&mut self.tensor_aim_errors),
            "initial_yaw_degrees":self.frames.first().map(|f| f.yaw.to_degrees()),
            "final_yaw_degrees":self.frames.last().map(|f| f.yaw.to_degrees()),
            "worst_aim":self.worst_aim,
        });
        eprintln!("MOTION_COORDINATES {result}");
        assert!(
            self.position_errors.len() >= 10,
            "too few live kick inputs for a coordinate audit"
        );
        let p95 = |key: &str| result[key]["p95"].as_f64().unwrap();
        assert!(
            p95("request_position_m") < if filtering { 0.15 } else { 0.03 },
            "ball position frame mismatch: {result}"
        );
        assert!(
            p95("request_velocity_m_s") < if filtering { 0.30 } else { 0.03 },
            "ball velocity frame mismatch: {result}"
        );
        assert!(
            p95("request_aim_degrees") < if filtering { 3.0 } else { 1.0 },
            "kick direction frame mismatch: {result}"
        );
        assert!(
            p95("tensor_position_m") < if filtering { 0.19 } else { 0.07 },
            "policy ball position mismatch, including the configured foot shift: {result}"
        );
        assert!(
            p95("tensor_velocity_m_s") < if filtering { 0.30 } else { 0.03 },
            "policy ball velocity mismatch: {result}"
        );
        assert!(
            p95("tensor_aim_encoding_degrees") < 1e-4,
            "policy aim encoding mismatch: {result}"
        );
        result
    }
}

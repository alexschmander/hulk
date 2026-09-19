//! One sample per evaluated request, including failed requests. Times are node-clock times.
use kinematics::joints::head::HeadJoints;
use ros_z::{Message, time::Time};
use serde::{Deserialize, Serialize};
use types::{motion_command::HeadMotion, motor_command::MotorCommand};

use crate::{
    head::{HeadController, HeadOutput},
    joint_control::{ConstraintDiagnostic, HeadObservation, KinematicState, MotionProgress},
    parameters::Parameters,
    patterns::ScanStatus,
};

pub const TOPIC: &str = "head_motion/diagnostics";

#[derive(Clone, Serialize, Deserialize, Message)]
pub struct Diagnostics {
    pub evaluation: u64,
    pub request_sequence: i64,
    pub request: HeadMotion,
    pub received_at: Time,
    pub evaluated_at: Time,
    pub completed_at: Time,
    pub observation_time: Option<Time>,
    pub observation_received_at: Option<Time>,
    pub observation: Option<HeadObservation>,
    pub output: Option<Output>,
    pub error: Option<String>,
}

#[derive(Clone, Serialize, Deserialize, Message)]
pub struct Output {
    pub reference: HeadJoints<KinematicState>,
    pub commands: HeadJoints<MotorCommand>,
    pub progress: Option<MotionProgress>,
    pub scan: Option<ScanStatus>,
    /// Describes the abandoned waypoint, while `scan` describes the newly selected one.
    pub scan_timeout: Option<String>,
    pub glance_timeout: Option<String>,
    pub hold_reason: Option<String>,
    pub constraints: Vec<ConstraintDiagnostic>,
    pub reseeded: bool,
    pub injected: bool,
    /// Exact evaluated parameters, including live tuning, for reproducible analysis.
    pub parameters: Parameters,
}

impl Output {
    pub fn capture(
        controller: &HeadController,
        output: &HeadOutput,
        parameters: &Parameters,
    ) -> Self {
        Self {
            reference: output.joint_control.reference,
            commands: output.joint_control.commands.clone(),
            progress: output.joint_control.progress,
            scan: controller.scan_status(),
            scan_timeout: output.scan_timeout.as_ref().map(|s| format!("{s:?}")),
            glance_timeout: output.glance_timeout.as_ref().map(|s| format!("{s:?}")),
            hold_reason: output.hold_reason.map(|s| format!("{s:?}")),
            constraints: output.joint_control.diagnostics.clone(),
            reseeded: output.joint_control.reseeded,
            injected: output.injected,
            parameters: parameters.clone(),
        }
    }
}

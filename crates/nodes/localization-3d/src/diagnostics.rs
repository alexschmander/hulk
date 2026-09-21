use ros_z::{Message, time::Time};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Diagnostics for one solve attempt. Missing costs were not reported by the solver.
#[derive(Debug, Clone, Serialize, Deserialize, Message)]
pub struct SolveDiagnostics {
    pub time: Time,
    pub epoch: u64,
    pub duration: Duration,
    pub iterations: Option<usize>,
    pub initial_cost: Option<f64>,
    pub final_cost: Option<f64>,
    pub termination: String,
    pub state_count: usize,
    pub measurement_count: usize,
    pub failure: Option<String>,
}

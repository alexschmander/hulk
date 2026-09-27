use ros_z::{Message, time::Time};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Diagnostics for one solve attempt. Missing costs were not reported by the solver.
#[derive(Debug, Clone, Serialize, Deserialize, Message)]
pub struct SolveDiagnostics {
    pub time: Time,
    pub epoch: u64,
    pub duration: Duration,
    /// Complete Localization::solve, including discarded recovery candidates.
    #[serde(default)]
    pub estimation_duration: Duration,
    /// Input ingestion/factor construction, populated by the caller.
    #[serde(default)]
    pub ingestion_duration: Duration,
    pub iterations: Option<usize>,
    pub lm_attempts: usize,
    pub lm_rejected_steps: usize,
    pub gradient_norm: Option<f64>,
    /// Replaced a rejected field-conditioned graph with a validated motion-only window.
    pub motion_rebuilt: bool,
    pub initial_cost: Option<f64>,
    pub final_cost: Option<f64>,
    pub termination: String,
    pub state_count: usize,
    pub measurement_count: usize,
    pub failure: Option<String>,
}

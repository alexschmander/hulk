//! Deterministic, headless inputs and runner for exercising 3D localization.

pub mod config;
pub mod report;
mod sensors;
pub mod simulation;
pub mod trajectory;

pub use config::{AssociationMode, SimulationConfig, VisualOdometryOutlier};
pub use simulation::{
    LandmarkClass, LandmarkDetection, LandmarkFrameCounts, LocalizationSimulation,
    SimulationHistorySample,
};
pub use trajectory::{PoseKeyframe, Scenario};

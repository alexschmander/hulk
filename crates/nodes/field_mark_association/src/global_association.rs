mod config;
mod map;
mod solver;
#[cfg(test)]
mod tests;
mod types;

pub(crate) use config::GLOBAL_LOCALIZER_MAX_DETECTIONS;
pub use config::GlobalAssociationConfig;
pub(crate) use map::candidate_points;
#[cfg(test)]
pub(crate) use solver::solve;
pub(crate) use solver::{GlobalAssociationResult, GlobalLocalizationInput, SolverWorkspace};
pub(crate) use types::FEATURE_CLASSES;
pub use types::VisualFeatureClass;

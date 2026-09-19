mod config;
pub(crate) mod solver;
#[cfg(test)]
mod tests;

pub use config::GlobalAssociationConfig;
pub(crate) use config::{GLOBAL_LOCALIZER_MAX_DETECTIONS, GLOBAL_LOCALIZER_MAX_INPUT_DETECTIONS};

use std::time::Duration;

use ros_z::Message;
use serde::{Deserialize, Serialize};

use crate::global_association::GlobalAssociationConfig as GlobalLocalizerParameters;

#[derive(Clone, Debug, Deserialize, Serialize, Message)]
#[serde(deny_unknown_fields)]
pub struct FieldMarkAssociationParameters {
    /// Stateless global-association gates, scoring, and deterministic work limits.
    pub global_localizer: GlobalLocalizerParameters,
    /// Maximum node-side timestamp difference for association geometry.
    ///
    /// Direct API calls receive an untimestamped hint and therefore do not apply this limit.
    pub max_pose_hint_age: Duration,
}

impl Default for FieldMarkAssociationParameters {
    fn default() -> Self {
        Self {
            global_localizer: GlobalLocalizerParameters::default(),
            max_pose_hint_age: Duration::from_millis(250),
        }
    }
}

impl FieldMarkAssociationParameters {
    pub(crate) fn validate(&self) -> std::result::Result<(), String> {
        self.global_localizer.validate()?;
        if self.max_pose_hint_age.is_zero() {
            return Err("max_pose_hint_age must be > 0".to_string());
        }
        Ok(())
    }
}

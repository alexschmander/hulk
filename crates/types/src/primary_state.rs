use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize, ros_z::Message)]
pub enum PrimaryState {
    Damping,
    Prepare,
    Stop,
    Initial,
    Ready,
    Set,
    Playing,
    Penalized,
    Finished,
}

impl PrimaryState {
    /// Whether robots send team messages in this state.
    ///
    /// Teammates walk to and stand at their kick-off positions in ready and set, where others
    /// need their reports to tell them apart from opponents.
    pub fn sends_team_messages(self) -> bool {
        matches!(
            self,
            PrimaryState::Ready | PrimaryState::Set | PrimaryState::Playing
        )
    }
}

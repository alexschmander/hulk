//! Published outputs of the object filters.
//!
//! Track states are expressed in the continuous odometry frame. Ground and field positions are
//! derived at publication time and are not part of the filter state.

use std::time::Duration;

use coordinate_systems::{Field, Ground, Odometry};
use linear_algebra::{Isometry2, Point2};
use ros_z::{Message, time::Time};
use serde::{Deserialize, Serialize};

use crate::multivariate_normal_distribution::MultivariateNormalDistribution;

/// Stable identifier of a maintained track hypothesis. It does not certify physical identity.
pub type TrackId = u64;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, Message)]
pub enum TrackStatus {
    /// Not yet confident enough to be acted upon.
    Tentative,
    /// Confident and recently observed.
    Confirmed,
    /// Confident but not observed for a while, e.g. outside the field of view.
    Dormant,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, Message)]
pub enum StationaryClass {
    GoalPost,
    LSpot,
    TSpot,
    XSpot,
    PenaltySpot,
}

impl StationaryClass {
    pub const ALL: [StationaryClass; 5] = [
        StationaryClass::GoalPost,
        StationaryClass::LSpot,
        StationaryClass::TSpot,
        StationaryClass::XSpot,
        StationaryClass::PenaltySpot,
    ];

    pub fn is_obstacle(self) -> bool {
        matches!(self, StationaryClass::GoalPost)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, Message)]
pub struct StationaryTrack {
    pub id: TrackId,
    pub existence: f32,
    pub status: TrackStatus,
    pub class: StationaryClass,
    /// Probability of each class, in the order of [`StationaryClass::ALL`].
    pub class_probabilities: Vec<f32>,
    pub is_obstacle: bool,
    pub detection_probability: f32,
    pub time_since_detection: Option<Duration>,
    /// Position `[x, y]` in the odometry frame.
    pub state: MultivariateNormalDistribution<2>,
    pub position_in_ground: Point2<Ground>,
    pub position_in_field: Option<Point2<Field>>,
}

/// Tracks of one filter at one time, with the transforms used to derive their ground and field
/// positions.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, Message)]
pub struct TrackedObjects<Track> {
    pub time: Time,
    pub ground_to_odometry: Isometry2<Ground, Odometry>,
    pub ground_to_field: Option<Isometry2<Ground, Field>>,
    pub tracks: Vec<Track>,
}

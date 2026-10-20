use std::time::Duration;

use object_tracking::{
    camera::ObservationParameters, motion::OdometryDrift, multi_bernoulli::ScanSettings,
};
use ros_z::Message;
use serde::{Deserialize, Serialize};
use types::tracked_objects::StationaryClass;

#[derive(Clone, Debug, Deserialize, Serialize, Message)]
#[serde(deny_unknown_fields)]
pub struct Parameters {
    /// Largest time between an image and the camera matrix it is projected with; camera matrices
    /// come at the rate of the inertial measurements, so larger gaps mean missing estimates.
    pub maximum_camera_matrix_age: Duration,
    pub observation: ObservationParameters,
    pub drift: OdometryDrift,
    pub scan: ScanSettings,
    pub classes: ClassesParameters,
    /// Pixels at the image border in which objects count as not visible.
    pub visibility_margin: f32,
    pub minimum_existence: f32,
    pub maximum_tracks: usize,
    /// Tracks without a detection for this long are dropped to bound the local map.
    pub retention_without_detection: Duration,
    pub confirmation_existence: f32,
    /// Confirmed tracks without a detection for this long are published as dormant.
    pub dormant_after: Duration,
    pub obstacle_existence: f32,
    pub goal_post_obstacle_radius: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize, Message)]
#[serde(deny_unknown_fields)]
pub struct ClassesParameters {
    pub goal_post: ClassParameters,
    pub l_spot: ClassParameters,
    pub t_spot: ClassParameters,
    pub x_spot: ClassParameters,
    pub penalty_spot: ClassParameters,
}

impl ClassesParameters {
    pub fn get(&self, class: StationaryClass) -> &ClassParameters {
        match class {
            StationaryClass::GoalPost => &self.goal_post,
            StationaryClass::LSpot => &self.l_spot,
            StationaryClass::TSpot => &self.t_spot,
            StationaryClass::XSpot => &self.x_spot,
            StationaryClass::PenaltySpot => &self.penalty_spot,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Message)]
#[serde(deny_unknown_fields)]
pub struct ClassParameters {
    pub minimum_confidence: f32,
    /// Detection probability of a fully visible object within range.
    pub detection_probability: f32,
    /// Range of the detector; beyond it, tracks are not expected to be detected. Detections from
    /// farther away would be explained by closer tracks along the same ray, so this must cover the
    /// range at which the detector still reports objects.
    pub maximum_distance: f32,
    /// Height of the object for visibility, zero for markings on the ground.
    pub height: f32,
    /// Probability that a detection of this class carries this class's label.
    pub label_accuracy: f32,
    /// False detections with this label per square meter of visible ground.
    pub clutter_density: f32,
    /// False bearing-only detections with this label per radian of view.
    pub bearing_clutter_density: f32,
    /// Previously undetected objects of this class per square meter.
    pub undetected_density: f32,
}

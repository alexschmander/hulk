use std::time::Duration;

use object_tracking::{
    camera::ObservationParameters, motion::OdometryDrift, multi_bernoulli::ScanSettings,
};
use ros_z::Message;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, Message)]
#[serde(deny_unknown_fields)]
pub struct Parameters {
    /// Largest time between an image and the camera matrix it is projected with; camera matrices
    /// come at the rate of the inertial measurements, so larger gaps mean missing estimates.
    pub maximum_camera_matrix_age: Duration,
    pub observation: ObservationParameters,
    pub drift: OdometryDrift,
    pub scan: ScanSettings,
    pub robot: ClassParameters,
    pub human: ClassParameters,
    pub team: TeamParameters,
    pub identity: IdentityParameters,
    /// Pixels at the image border in which objects count as not visible.
    pub visibility_margin: f32,
    /// Lower bound of the birth scale when the expected number of robot tracks reaches the
    /// number of robots on the field. The robot count is a soft constraint only.
    pub minimum_robot_birth_scale: f32,
    /// Tracks closer than this describe one object and are merged; two bodies cannot overlap.
    pub merge_distance: f32,
    pub minimum_existence: f32,
    pub maximum_tracks: usize,
    /// Tracks neither detected nor reported for this long are dropped to bound resources.
    pub retention_without_evidence: Duration,
    pub confirmation_existence: f32,
    /// Confirmed tracks without a camera detection for this long are published as dormant.
    pub dormant_after: Duration,
    pub obstacle_existence: f32,
    /// Tracks with a larger position standard deviation are not published as obstacles.
    pub obstacle_maximum_position_standard_deviation: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize, Message)]
#[serde(deny_unknown_fields)]
pub struct ClassParameters {
    pub minimum_confidence: f32,
    /// Detection probability of a fully visible, unoccluded object within range.
    pub detection_probability: f32,
    /// Range of the detector; beyond it, tracks are not expected to be detected. Detections from
    /// farther away would be unexplained and spawn duplicate tracks, so this must cover the range
    /// at which the detector still reports objects.
    pub maximum_distance: f32,
    pub height: f32,
    /// Radius for occlusion.
    pub radius: f32,
    /// Probability that a detection of this class carries this class's label.
    pub label_accuracy: f32,
    /// False detections with this label per square meter of visible ground.
    pub clutter_density: f32,
    /// False bearing-only detections with this label per radian of view.
    pub bearing_clutter_density: f32,
    /// Previously undetected objects of this class per square meter.
    pub undetected_density: f32,
    /// Typical walking speed per axis, in m/s. Also the velocity uncertainty of new tracks.
    pub velocity_standard_deviation: f32,
    /// How long a velocity persists, in seconds. Shorter allows faster maneuvers.
    pub velocity_time_constant: f32,
    pub obstacle_radius_at_foot_height: f32,
    pub obstacle_radius_at_hip_height: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize, Message)]
#[serde(deny_unknown_fields)]
pub struct TeamParameters {
    /// Position standard deviation of a teammate's own localization, in meters.
    pub sender_position_standard_deviation: f32,
    /// Position standard deviation of our localization, in meters.
    pub registration_position_standard_deviation: f32,
    /// Heading standard deviation of our localization, in radians. Its effect grows with distance.
    pub registration_heading_standard_deviation: f32,
    /// Maximum walking speed, bounding how far a teammate moved since its pose was estimated.
    pub maximum_speed: f32,
    /// Assumed pose age for messages that do not report it.
    pub default_pose_age: Duration,
    /// Messages older than this are ignored.
    pub maximum_pose_age: Duration,
    /// Density of a teammate being somewhere not represented by any track, or of a report being an
    /// outlier, per square meter.
    pub unexplained_density: f32,
    /// Probability that a report that matches no track describes a real robot there.
    pub report_validity: f32,
    /// Lower bound of the prior probability that a reporting teammate is represented by no track,
    /// even if tracks already claim it.
    pub untracked_probability: f32,
    /// Prior probability that an anonymous robot track is any teammate.
    pub teammate_prior: f32,
    /// Teammate reports never shrink a track's position standard deviation below this unless the
    /// camera confirms it, because repeated reports share the registration error.
    pub position_standard_deviation_floor: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize, Message)]
#[serde(deny_unknown_fields)]
pub struct IdentityParameters {
    /// A track is published as a teammate once its probability of being that teammate reaches
    /// this value.
    pub promotion_probability: f32,
    /// A published teammate keeps its role while the probability stays above this value.
    pub retention_probability: f32,
    /// Association probability for a report to count as discriminating evidence.
    pub discriminating_association_probability: f32,
    pub minimum_discriminating_reports: u32,
    /// Rate at which teammate beliefs relax towards the prior, per second.
    pub prior_rate: f32,
    /// Rate at which the identities of tracks with identical position estimates mix, per second.
    pub swap_rate: f32,
}

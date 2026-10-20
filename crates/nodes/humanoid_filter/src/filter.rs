use std::{collections::HashMap, time::Duration};

use coordinate_systems::{Field, Ground, Odometry};
use hsl_network_messages::PlayerNumber;
use linear_algebra::{IntoFramed, Isometry2, Pose2};
use nalgebra::{Matrix2, Matrix4, SVector, Vector2, Vector4};
use object_tracking::{
    camera::{CameraView, GroundAnchor, Occluder, expected_unoccluded_fractions},
    classification::{ClassBelief, ClassEvidence},
    gaussian::{Gaussian, marginal_sigma_points},
    measurement::{MeasurementInnovation, MeasurementModel},
    motion::{DampedVelocity, MotionModel},
    multi_bernoulli::{
        Birth, Component, Measurement, MultiBernoulli, ScanReport, Track, TrackAttributes, TrackId,
        reduce_mixture,
    },
};
use projection::camera_matrix::CameraMatrix;
use ros_z::time::Time;
use serde::{Deserialize, Serialize};
use types::{
    object_detection::{Object, RobocupObjectLabel},
    obstacles::{Obstacle, ObstacleKind},
    odometry,
    tracked_objects::{
        HumanoidRole, HumanoidTrack, TeammateProbability, TrackStatus, TrackedObjects,
    },
};

use crate::parameters::{ClassParameters, Parameters};

pub const CLASS_COUNT: usize = 2;
pub const ROBOT: usize = 0;
pub const HUMAN: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HumanoidLabel {
    Robot,
    Human,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Label {
    pub label: HumanoidLabel,
    pub evidence: ClassEvidence<CLASS_COUNT>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub struct TeammateBelief {
    pub player_number: PlayerNumber,
    pub probability: f32,
    pub discriminating_reports: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Attributes {
    pub class: ClassBelief<CLASS_COUNT>,
    /// Probability of being each teammate. The remainder is the probability of not being a
    /// teammate, i.e. an opponent, an unknown robot or a human.
    pub teammates: Vec<TeammateBelief>,
    pub published_teammate: Option<PlayerNumber>,
    /// Known only from team messages, never confirmed by the camera.
    pub remote_only: bool,
    pub last_detection: Option<Time>,
    pub last_team_report: Option<Time>,
    pub detection_probability: f32,
}

impl Attributes {
    pub fn teammate_probability(&self, player_number: PlayerNumber) -> f32 {
        self.teammates
            .iter()
            .find(|belief| belief.player_number == player_number)
            .map_or(0.0, |belief| belief.probability)
    }

    fn teammate_mut(&mut self, player_number: PlayerNumber) -> &mut TeammateBelief {
        if let Some(index) = self
            .teammates
            .iter()
            .position(|belief| belief.player_number == player_number)
        {
            return &mut self.teammates[index];
        }
        self.teammates.push(TeammateBelief {
            player_number,
            probability: 0.0,
            discriminating_reports: 0,
        });
        self.teammates.last_mut().expect("just pushed")
    }

    fn last_evidence(&self) -> Option<Time> {
        self.last_detection.max(self.last_team_report)
    }

    /// Combines the attributes of a duplicate track of the same object, existing with probability
    /// `existence`, into these.
    ///
    /// The duplicate supports each teammate hypothesis only as far as it exists, so a weak
    /// remote-only track cannot override the identity of an established camera track.
    fn absorb(&mut self, duplicate: Attributes, existence: f32) {
        for belief in duplicate.teammates {
            let supported = belief.probability * existence.clamp(0.0, 1.0);
            let own = self.teammate_mut(belief.player_number);
            own.probability = own.probability.max(supported);
            own.discriminating_reports = own
                .discriminating_reports
                .max(belief.discriminating_reports);
        }
        let total: f32 = self.teammates.iter().map(|belief| belief.probability).sum();
        if total > 1.0 {
            for belief in &mut self.teammates {
                belief.probability /= total;
            }
        }
        if self.published_teammate.is_none() {
            self.published_teammate = duplicate
                .published_teammate
                .filter(|player_number| self.teammate_probability(*player_number) > 0.0);
        }
        self.remote_only &= duplicate.remote_only;
        self.last_detection = self.last_detection.max(duplicate.last_detection);
        self.last_team_report = self.last_team_report.max(duplicate.last_team_report);
    }
}

impl TrackAttributes<Label> for Attributes {
    fn label_likelihood(&self, label: &Label) -> f32 {
        self.class.label_likelihood(&label.evidence)
    }

    fn update(&mut self, missed_weight: f32, observations: &[(f32, &Label)]) {
        let evidence: Vec<_> = observations
            .iter()
            .map(|(weight, label)| (*weight, &label.evidence))
            .collect();
        self.class.update(missed_weight, &evidence);
    }
}

/// Teammate pose report brought into the odometry frame.
#[derive(Clone, Copy, Debug)]
pub struct TeamReport {
    pub player_number: PlayerNumber,
    pub position: Vector2<f32>,
    pub covariance: Matrix2<f32>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct HumanoidFilter {
    pub tracks: MultiBernoulli<4, Attributes>,
    odometry: Option<(Time, Pose2<Odometry>)>,
    time: Option<Time>,
    /// Prior probability of a track being one particular teammate, from the last team report.
    teammate_prior: Option<f32>,
}

impl HumanoidFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.tracks.tracks.clear();
    }

    pub fn time(&self) -> Option<Time> {
        self.time
    }

    pub fn odometry_pose(&self) -> Option<Pose2<Odometry>> {
        self.odometry.map(|(_, pose)| pose)
    }

    /// Predicts all tracks to `time`, adding the drift of the odometry frame if the robot moved.
    pub fn predict(
        &mut self,
        time: Time,
        odometry_pose: Option<Pose2<Odometry>>,
        parameters: &Parameters,
    ) {
        let delta_time = self
            .time
            .map_or(0.0, |last| time.duration_since(last).as_secs_f32());
        let (translation, rotation, robot) = match (self.odometry, odometry_pose) {
            (Some((_, last)), Some(current)) => {
                let delta = odometry::previous_to_current(last, current);
                (
                    delta.translation().coords().norm(),
                    delta.orientation().angle(),
                    current.position().inner.coords,
                )
            }
            (_, Some(current)) => (0.0, 0.0, current.position().inner.coords),
            (Some((_, last)), None) => (0.0, 0.0, last.position().inner.coords),
            (None, None) => (0.0, 0.0, Vector2::zeros()),
        };
        if delta_time > 0.0 || translation > 0.0 || rotation != 0.0 {
            let drift = parameters.drift;
            self.tracks.predict(
                |track| {
                    let model = DampedVelocity {
                        velocity_standard_deviation: expected_class_value(
                            &track.attributes.class,
                            parameters,
                            |class| class.velocity_standard_deviation,
                        ),
                        velocity_time_constant: expected_class_value(
                            &track.attributes.class,
                            parameters,
                            |class| class.velocity_time_constant,
                        ),
                    };
                    model.transition(delta_time)
                },
                |_| 1.0,
                |gaussian| {
                    let position_noise = drift.position_noise(
                        gaussian.mean.fixed_rows::<2>(0) - robot,
                        translation,
                        rotation,
                        delta_time,
                    );
                    let mut noise = Matrix4::zeros();
                    noise
                        .fixed_view_mut::<2, 2>(0, 0)
                        .copy_from(&position_noise);
                    noise
                },
            );
        }
        if delta_time > 0.0 {
            self.mix_identities(delta_time, parameters);
        }
        if self.time.is_none_or(|last| time > last) {
            self.time = Some(time);
        }
        if let Some(pose) = odometry_pose {
            self.odometry = Some((time, pose));
        }
    }

    /// Lets teammate beliefs forget over time and exchange between tracks that come close.
    ///
    /// Tracks of robots that pass each other can swap the robot they follow, so their identities
    /// mix while their position estimates overlap. Well separated estimates, e.g. of robots in a
    /// duel close to the camera, keep their identities. Every belief also relaxes towards the prior, so that a hypothesis
    /// once ruled out can be recovered by later team reports instead of being excluded forever.
    fn mix_identities(&mut self, delta_time: f32, parameters: &Parameters) {
        let identity = &parameters.identity;
        let prior = self
            .teammate_prior
            .unwrap_or(parameters.team.teammate_prior / 4.0);
        let retained = (-identity.prior_rate * delta_time).exp();
        for track in &mut self.tracks.tracks {
            if track.attributes.remote_only {
                continue;
            }
            for belief in &mut track.attributes.teammates {
                belief.probability = belief.probability * retained + prior * (1.0 - retained);
            }
        }

        let positions: Vec<(Vector2<f32>, Matrix2<f32>)> = self
            .tracks
            .tracks
            .iter()
            .map(|track| {
                let estimate = track.estimate();
                (
                    estimate.mean.fixed_rows::<2>(0).into_owned(),
                    estimate.covariance.fixed_view::<2, 2>(0, 0).into_owned(),
                )
            })
            .collect();
        let swap = 1.0 - (-identity.swap_rate * delta_time).exp();
        for first in 0..positions.len() {
            for second in first + 1..positions.len() {
                // Up to complete confusion, i.e. equal beliefs, for indistinguishable tracks.
                let mixing = 0.5 * swap * position_overlap(positions[first], positions[second]);
                if mixing < 1e-4 {
                    continue;
                }
                let (head, tail) = self.tracks.tracks.split_at_mut(second);
                let (a, b) = (&mut head[first].attributes, &mut tail[0].attributes);
                if a.remote_only || b.remote_only {
                    continue;
                }
                let numbers: Vec<PlayerNumber> = a
                    .teammates
                    .iter()
                    .chain(&b.teammates)
                    .map(|belief| belief.player_number)
                    .collect();
                for number in numbers {
                    let (p_a, p_b) = (
                        a.teammate_probability(number),
                        b.teammate_probability(number),
                    );
                    if p_a == p_b {
                        continue;
                    }
                    a.teammate_mut(number).probability = (1.0 - mixing) * p_a + mixing * p_b;
                    b.teammate_mut(number).probability = (1.0 - mixing) * p_b + mixing * p_a;
                }
            }
        }
    }

    /// Updates the tracks with the humanoid detections of one image.
    ///
    /// `robots_on_field` is the number of robots other than this one that can be on the field,
    /// used as a soft constraint on initiating robot tracks.
    pub fn update_with_detections(
        &mut self,
        time: Time,
        detected_objects: &[Object<RobocupObjectLabel>],
        camera_matrix: &CameraMatrix,
        robots_on_field: Option<usize>,
        parameters: &Parameters,
    ) -> Option<ScanReport> {
        let (_, pose) = self.odometry?;
        let view = CameraView {
            camera_matrix,
            ground_to_odometry: pose.as_transform::<Ground>(),
        };

        let measurements: Vec<Measurement<Label>> = detected_objects
            .iter()
            .filter_map(|object| {
                let label = humanoid_label(object.label)?;
                let class_parameters = label_class_parameters(label, parameters);
                if object.bounding_box.confidence < class_parameters.minimum_confidence {
                    return None;
                }
                let model = view.measure(
                    &object.bounding_box,
                    GroundAnchor::BottomCenter,
                    &parameters.observation,
                )?;
                let clutter_density = model.clutter_density(
                    class_parameters.clutter_density,
                    class_parameters.bearing_clutter_density,
                );
                Some(Measurement {
                    model,
                    label: Label {
                        label,
                        evidence: label_evidence(label, parameters),
                    },
                    clutter_density,
                })
            })
            .collect();

        let unoccluded = unoccluded_fractions(&view, &self.tracks.tracks, parameters);
        let robot_birth_scale = robot_birth_scale(&self.tracks.tracks, robots_on_field, parameters);

        let report = self.tracks.update(
            &measurements,
            &parameters.scan,
            |track, gaussian| {
                expected_detection_probability(
                    &view,
                    track,
                    gaussian,
                    unoccluded.get(&track.id).copied().unwrap_or(1.0),
                    parameters,
                )
            },
            |measurement| birth(&view, measurement, robot_birth_scale, parameters),
        );

        for track in &mut self.tracks.tracks {
            track.attributes.detection_probability = expected_detection_probability(
                &view,
                track,
                &track.estimate(),
                unoccluded.get(&track.id).copied().unwrap_or(1.0),
                parameters,
            );
            if track.last_association_probability >= 0.5 {
                track.attributes.last_detection = Some(time);
                track.attributes.remote_only = false;
            }
        }
        self.retire(time, parameters);
        Some(report)
    }

    /// Updates the tracks with one teammate report.
    ///
    /// The report can only be explained by a track that is that teammate. Tracks that are
    /// unlikely to be that teammate are left almost unchanged; a report that no track explains
    /// starts a remote-only track.
    pub fn update_with_team_report(
        &mut self,
        time: Time,
        report: &TeamReport,
        active_teammates: usize,
        parameters: &Parameters,
    ) {
        let team = &parameters.team;
        let measurement = MeasurementModel::Position {
            position: report.position,
            covariance: report.covariance,
        };
        let floor = team.position_standard_deviation_floor.powi(2);
        let teammate_prior = team.teammate_prior / active_teammates.max(1) as f32;
        self.teammate_prior = Some(teammate_prior);

        // A teammate number belongs to at most one object. The hypotheses are that one track is
        // this teammate, with prior probability existence × P(teammate), or that none is.
        let mut weights = Vec::with_capacity(self.tracks.tracks.len());
        let mut innovations = Vec::with_capacity(self.tracks.tracks.len());
        for track in &mut self.tracks.tracks {
            let known = track
                .attributes
                .teammates
                .iter()
                .any(|belief| belief.player_number == report.player_number);
            if !known && !track.attributes.remote_only {
                // Lazily assign the prior of a teammate this track has not heard about yet, taken
                // from the probability of not being a teammate.
                let remaining = 1.0
                    - track
                        .attributes
                        .teammates
                        .iter()
                        .map(|belief| belief.probability)
                        .sum::<f32>();
                track
                    .attributes
                    .teammate_mut(report.player_number)
                    .probability = teammate_prior.min(remaining.max(0.0));
            }
            let existence = track.existence.min(parameters.scan.maximum_probability);
            let explains = track.attributes.teammate_probability(report.player_number)
                * track.attributes.class.probabilities[ROBOT];
            let explains = explains.min(parameters.scan.maximum_probability);
            let component_innovations: Vec<_> = track
                .components
                .iter()
                .map(|component| {
                    MeasurementInnovation::compute(&component.gaussian, &measurement).filter(
                        |innovation| {
                            innovation.squared_mahalanobis_distance()
                                <= parameters.scan.position_gate
                        },
                    )
                })
                .collect();
            let likelihood: f32 = track
                .components
                .iter()
                .zip(&component_innovations)
                .filter_map(|(component, innovation)| {
                    Some(component.weight * innovation.as_ref()?.log_likelihood().exp())
                })
                .sum();
            weights.push(f64::from(existence * explains) * f64::from(likelihood));
            innovations.push((explains, component_innovations));
        }
        let prior_of_no_track = self
            .tracks
            .tracks
            .iter()
            .zip(&innovations)
            .map(|(track, (explains, _))| {
                f64::from(track.existence.min(parameters.scan.maximum_probability) * explains)
            })
            .sum::<f64>();
        let prior_of_no_track =
            (1.0 - prior_of_no_track).max(f64::from(team.untracked_probability));
        let no_track_weight = prior_of_no_track * f64::from(team.unexplained_density);
        let total = no_track_weight + weights.iter().sum::<f64>();
        let association: Vec<f32> = weights
            .iter()
            .map(|weight| (weight / total) as f32)
            .collect();
        let unexplained = (no_track_weight / total) as f32;

        for ((track, (explains, component_innovations)), associated) in self
            .tracks
            .tracks
            .iter_mut()
            .zip(innovations)
            .zip(association)
        {
            if explains <= 0.0 {
                continue;
            }
            let existence = track.existence.min(parameters.scan.maximum_probability);
            let missed = 1.0 - associated;
            let missed_existence = existence * (1.0 - explains) / (1.0 - existence * explains);
            let posterior_existence = (associated + missed * missed_existence).clamp(0.0, 1.0);
            if posterior_existence <= 0.0 {
                track.existence = 0.0;
                continue;
            }
            let missed_weight = missed * missed_existence / posterior_existence;
            let associated_weight = associated / posterior_existence;

            let mut components: Vec<Component<4>> = track
                .components
                .iter()
                .map(|component| Component {
                    weight: component.weight * missed_weight,
                    gaussian: component.gaussian,
                })
                .collect();
            if associated_weight > 0.0 {
                let likelihoods: Vec<f32> = track
                    .components
                    .iter()
                    .zip(&component_innovations)
                    .map(|(component, innovation)| {
                        innovation.as_ref().map_or(0.0, |innovation| {
                            component.weight * innovation.log_likelihood().exp()
                        })
                    })
                    .collect();
                let total: f32 = likelihoods.iter().sum();
                let camera_supported = track.attributes.last_detection.is_some();
                for ((component, innovation), likelihood) in track
                    .components
                    .iter()
                    .zip(&component_innovations)
                    .zip(likelihoods)
                {
                    let (Some(innovation), true) = (innovation, total > 0.0) else {
                        continue;
                    };
                    let Some(mut gaussian) = innovation.update(&component.gaussian) else {
                        continue;
                    };
                    if !camera_supported {
                        apply_position_floor(&mut gaussian, floor);
                    }
                    components.push(Component {
                        weight: associated_weight * likelihood / total,
                        gaussian,
                    });
                }
            }
            components.retain(|component| component.weight > 0.0);
            if components.is_empty() {
                continue;
            }

            let prior_teammate = track.attributes.teammate_probability(report.player_number);
            let others = (1.0 - prior_teammate).max(1e-6);
            for belief in &mut track.attributes.teammates {
                if belief.player_number == report.player_number {
                    belief.probability = associated_weight;
                } else {
                    belief.probability *= missed_weight / others;
                }
            }
            if !track
                .attributes
                .teammates
                .iter()
                .any(|belief| belief.player_number == report.player_number)
            {
                track
                    .attributes
                    .teammate_mut(report.player_number)
                    .probability = associated_weight;
            }
            if associated >= parameters.identity.discriminating_association_probability {
                track
                    .attributes
                    .teammate_mut(report.player_number)
                    .discriminating_reports += 1;
            }
            if associated >= 0.5 {
                track.attributes.last_team_report = Some(time);
                let robot = &mut track.attributes.class.probabilities;
                robot[ROBOT] = robot[ROBOT] * (1.0 - associated_weight) + associated_weight;
                robot[HUMAN] = 1.0 - robot[ROBOT];
            }
            track.existence = posterior_existence;
            track.components = reduce_mixture(components, &parameters.scan.mixture_reduction);
        }
        self.tracks.tracks.retain(|track| track.existence > 0.0);

        let existence = unexplained * team.report_validity;
        if existence >= parameters.minimum_existence {
            let mut covariance = Matrix4::zeros();
            covariance
                .fixed_view_mut::<2, 2>(0, 0)
                .copy_from(&report.covariance);
            let velocity_variance = parameters.robot.velocity_standard_deviation.powi(2);
            covariance[(2, 2)] = velocity_variance;
            covariance[(3, 3)] = velocity_variance;
            let mut gaussian = Gaussian {
                mean: Vector4::new(report.position.x, report.position.y, 0.0, 0.0),
                covariance,
            };
            apply_position_floor(&mut gaussian, floor);
            self.tracks.insert(
                existence,
                gaussian,
                Attributes {
                    class: ClassBelief {
                        probabilities: SVector::from([1.0, 0.0]),
                    },
                    teammates: vec![TeammateBelief {
                        player_number: report.player_number,
                        probability: 1.0,
                        discriminating_reports: 1,
                    }],
                    published_teammate: None,
                    remote_only: true,
                    last_detection: None,
                    last_team_report: Some(time),
                    detection_probability: 0.0,
                },
            );
        }
        self.update_published_teammates(parameters);
    }

    /// A penalized teammate leaves the field: tracks that are probably it lose existence.
    pub fn remove_teammate(&mut self, player_number: PlayerNumber) {
        for track in &mut self.tracks.tracks {
            let probability = track.attributes.teammate_probability(player_number);
            if probability <= 0.0 {
                continue;
            }
            track.existence *= 1.0 - probability;
            let others = (1.0 - probability).max(1e-6);
            track
                .attributes
                .teammates
                .retain(|belief| belief.player_number != player_number);
            for belief in &mut track.attributes.teammates {
                belief.probability /= others;
            }
            if track.attributes.published_teammate == Some(player_number) {
                track.attributes.published_teammate = None;
            }
        }
        self.tracks.tracks.retain(|track| track.existence > 0.0);
    }

    fn retire(&mut self, time: Time, parameters: &Parameters) {
        let retention = parameters.retention_without_evidence;
        self.tracks.tracks.retain(|track| {
            track
                .attributes
                .last_evidence()
                .is_none_or(|last| time.duration_since(last) <= retention)
        });
        self.tracks
            .merge_coincident(parameters.merge_distance, Attributes::absorb);
        self.tracks
            .prune(parameters.minimum_existence, parameters.maximum_tracks);
        self.update_published_teammates(parameters);
    }

    /// Applies promotion and retention hysteresis, publishing each teammate number at most once.
    fn update_published_teammates(&mut self, parameters: &Parameters) {
        let identity = &parameters.identity;
        for track in &mut self.tracks.tracks {
            let keep = track
                .attributes
                .published_teammate
                .is_some_and(|player_number| {
                    track.attributes.teammate_probability(player_number)
                        >= identity.retention_probability
                });
            if !keep {
                track.attributes.published_teammate = None;
            }
        }
        let mut best: HashMap<PlayerNumber, (TrackId, f32)> = HashMap::new();
        for track in &self.tracks.tracks {
            for belief in &track.attributes.teammates {
                let qualifies = if track.attributes.published_teammate == Some(belief.player_number)
                {
                    true
                } else {
                    belief.probability >= identity.promotion_probability
                        && belief.discriminating_reports >= identity.minimum_discriminating_reports
                };
                if !qualifies {
                    continue;
                }
                let entry = best
                    .entry(belief.player_number)
                    .or_insert((track.id, belief.probability));
                if belief.probability > entry.1 {
                    *entry = (track.id, belief.probability);
                }
            }
        }
        for track in &mut self.tracks.tracks {
            track.attributes.published_teammate = best
                .iter()
                .find(|(_, (id, _))| *id == track.id)
                .map(|(player_number, _)| *player_number);
        }
    }

    pub fn tracked_objects(
        &self,
        now: Time,
        ground_to_field: Option<Isometry2<Ground, Field>>,
        parameters: &Parameters,
    ) -> Option<TrackedObjects<HumanoidTrack>> {
        let pose = self.odometry_pose()?;
        let ground_to_odometry = pose.as_transform::<Ground>();
        let odometry_to_ground = ground_to_odometry.inverse();
        let rotation = odometry_to_ground.inner.rotation;
        let tracks = self
            .tracks
            .tracks
            .iter()
            .map(|track| {
                let state = track.estimate();
                let position = state
                    .mean
                    .fixed_rows::<2>(0)
                    .into_owned()
                    .framed::<Odometry>();
                let position_in_ground = odometry_to_ground * position.as_point();
                let velocity_in_ground =
                    (rotation * state.mean.fixed_rows::<2>(2).into_owned()).framed();
                let time_since_detection = track
                    .attributes
                    .last_detection
                    .map(|last| now.duration_since(last));
                HumanoidTrack {
                    id: track.id,
                    existence: track.existence,
                    status: status(track, time_since_detection, parameters),
                    robot_probability: track.attributes.class.probabilities[ROBOT],
                    human_probability: track.attributes.class.probabilities[HUMAN],
                    role: role(&track.attributes),
                    teammate_probabilities: track
                        .attributes
                        .teammates
                        .iter()
                        .map(|belief| TeammateProbability {
                            player_number: belief.player_number,
                            probability: belief.probability,
                        })
                        .collect(),
                    remote_only: track.attributes.remote_only,
                    detection_probability: track.attributes.detection_probability,
                    time_since_detection,
                    state,
                    position_in_ground,
                    velocity_in_ground,
                    position_in_field: ground_to_field
                        .map(|ground_to_field| ground_to_field * position_in_ground),
                }
            })
            .collect();
        Some(TrackedObjects {
            time: now,
            ground_to_odometry,
            ground_to_field,
            tracks,
        })
    }

    pub fn obstacles(&self, parameters: &Parameters) -> Vec<Obstacle> {
        let Some(pose) = self.odometry_pose() else {
            return Vec::new();
        };
        let odometry_to_ground = pose.as_transform::<Ground>().inverse();
        let maximum_variance = parameters
            .obstacle_maximum_position_standard_deviation
            .powi(2);
        self.tracks
            .tracks
            .iter()
            .filter(|track| track.existence >= parameters.obstacle_existence)
            .filter_map(|track| {
                let state = track.estimate();
                let position_covariance = state.covariance.fixed_view::<2, 2>(0, 0);
                if position_covariance.symmetric_eigenvalues().max() > maximum_variance {
                    return None;
                }
                let position = odometry_to_ground
                    * state
                        .mean
                        .fixed_rows::<2>(0)
                        .into_owned()
                        .framed::<Odometry>()
                        .as_point();
                let is_human = track.attributes.class.most_likely() == HUMAN;
                let class = if is_human {
                    &parameters.human
                } else {
                    &parameters.robot
                };
                Some(Obstacle {
                    kind: if is_human {
                        ObstacleKind::Person
                    } else {
                        ObstacleKind::Robot
                    },
                    position,
                    radius_at_foot_height: class.obstacle_radius_at_foot_height,
                    radius_at_hip_height: class.obstacle_radius_at_hip_height,
                })
            })
            .collect()
    }
}

pub fn humanoid_label(label: RobocupObjectLabel) -> Option<HumanoidLabel> {
    match label {
        RobocupObjectLabel::Robot => Some(HumanoidLabel::Robot),
        _ => None,
    }
}

fn label_class_parameters(label: HumanoidLabel, parameters: &Parameters) -> &ClassParameters {
    match label {
        HumanoidLabel::Robot => &parameters.robot,
        HumanoidLabel::Human => &parameters.human,
    }
}

fn class_parameters(class: usize, parameters: &Parameters) -> &ClassParameters {
    if class == HUMAN {
        &parameters.human
    } else {
        &parameters.robot
    }
}

fn expected_class_value(
    belief: &ClassBelief<CLASS_COUNT>,
    parameters: &Parameters,
    value: impl Fn(&ClassParameters) -> f32,
) -> f32 {
    belief.expectation(&SVector::from([
        value(&parameters.robot),
        value(&parameters.human),
    ]))
}

/// `P(label | class)`: correct with the true class's label accuracy, otherwise the other label.
pub fn label_evidence(label: HumanoidLabel, parameters: &Parameters) -> ClassEvidence<CLASS_COUNT> {
    let robot_accuracy = parameters.robot.label_accuracy.clamp(0.0, 1.0);
    let human_accuracy = parameters.human.label_accuracy.clamp(0.0, 1.0);
    let likelihoods = match label {
        HumanoidLabel::Robot => SVector::from([robot_accuracy, 1.0 - human_accuracy]),
        HumanoidLabel::Human => SVector::from([1.0 - robot_accuracy, human_accuracy]),
    };
    ClassEvidence { likelihoods }
}

fn class_detection_probability(
    view: &CameraView,
    position: Vector2<f32>,
    class: &ClassParameters,
    margin: f32,
) -> f32 {
    let point = position.framed::<Odometry>().as_point();
    if view.ground_distance(point) > class.maximum_distance {
        return 0.0;
    }
    class.detection_probability * view.visible_fraction(point, class.height, margin)
}

fn expected_detection_probability(
    view: &CameraView,
    track: &Track<4, Attributes>,
    gaussian: &Gaussian<4>,
    unoccluded: f32,
    parameters: &Parameters,
) -> f32 {
    let points = marginal_sigma_points::<4, 2>(gaussian);
    let per_class = SVector::<f32, CLASS_COUNT>::from_fn(|class, _| {
        let class = class_parameters(class, parameters);
        points
            .iter()
            .map(|point| {
                class_detection_probability(view, *point, class, parameters.visibility_margin)
            })
            .sum::<f32>()
            / points.len() as f32
    });
    track.attributes.class.expectation(&per_class) * unoccluded
}

fn unoccluded_fractions(
    view: &CameraView,
    tracks: &[Track<4, Attributes>],
    parameters: &Parameters,
) -> HashMap<TrackId, f32> {
    let odometry_to_ground = view.odometry_to_ground();
    let occluders: Vec<Occluder> = tracks
        .iter()
        .map(|track| {
            let estimate = track.estimate();
            let position = estimate
                .mean
                .fixed_rows::<2>(0)
                .into_owned()
                .framed::<Odometry>()
                .as_point();
            let rotation = odometry_to_ground.inner.rotation.to_rotation_matrix();
            let covariance = estimate.covariance.fixed_view::<2, 2>(0, 0).into_owned();
            Occluder {
                position: (odometry_to_ground * position).inner.coords,
                position_covariance: rotation.matrix() * covariance * rotation.matrix().transpose(),
                radius: expected_class_value(&track.attributes.class, parameters, |class| {
                    class.radius
                }),
                existence: track.existence,
            }
        })
        .collect();
    let camera = view.camera_position().inner.coords;
    tracks
        .iter()
        .zip(expected_unoccluded_fractions(camera, &occluders))
        .map(|(track, fraction)| (track.id, fraction))
        .collect()
}

/// Soft robot count constraint: fewer robot initiations once the expected number of robot tracks
/// approaches the number of robots on the field. Never fully suppresses initiation.
fn robot_birth_scale(
    tracks: &[Track<4, Attributes>],
    robots_on_field: Option<usize>,
    parameters: &Parameters,
) -> f32 {
    let Some(robots_on_field) = robots_on_field.filter(|count| *count > 0) else {
        return 1.0;
    };
    let expected_robots: f32 = tracks
        .iter()
        .map(|track| track.existence * track.attributes.class.probabilities[ROBOT])
        .sum();
    ((robots_on_field as f32 - expected_robots) / robots_on_field as f32)
        .clamp(parameters.minimum_robot_birth_scale, 1.0)
}

fn birth(
    view: &CameraView,
    measurement: &Measurement<Label>,
    robot_birth_scale: f32,
    parameters: &Parameters,
) -> Option<Birth<4, Attributes>> {
    let (position, covariance) = measurement.model.ground_position()?;
    let class = label_class_parameters(measurement.label.label, parameters);
    let scale = match measurement.label.label {
        HumanoidLabel::Robot => robot_birth_scale,
        HumanoidLabel::Human => 1.0,
    };
    // In the units of the measurement likelihood, like the clutter density.
    let intensity = scale
        * class.undetected_density
        * measurement.model.square_meters_per_unit()
        * class_detection_probability(view, position, class, parameters.visibility_margin);
    let mut state_covariance = Matrix4::zeros();
    state_covariance
        .fixed_view_mut::<2, 2>(0, 0)
        .copy_from(&covariance);
    let velocity_variance = class.velocity_standard_deviation.powi(2);
    state_covariance[(2, 2)] = velocity_variance;
    state_covariance[(3, 3)] = velocity_variance;
    Some(Birth {
        gaussian: Gaussian {
            mean: Vector4::new(position.x, position.y, 0.0, 0.0),
            covariance: state_covariance,
        },
        attributes: Attributes {
            class: ClassBelief::from_evidence(&ClassBelief::uniform(), &measurement.label.evidence),
            teammates: Vec::new(),
            published_teammate: None,
            remote_only: false,
            last_detection: None,
            last_team_report: None,
            detection_probability: 0.0,
        },
        intensity,
    })
}

/// Bhattacharyya coefficient of two position estimates: one for identical estimates, approaching
/// zero for estimates that are far apart relative to their uncertainties.
fn position_overlap(
    (first_mean, first_covariance): (Vector2<f32>, Matrix2<f32>),
    (second_mean, second_covariance): (Vector2<f32>, Matrix2<f32>),
) -> f32 {
    let covariance = (first_covariance + second_covariance) * 0.5;
    let Some(inverse) = covariance.try_inverse() else {
        return 0.0;
    };
    let difference = first_mean - second_mean;
    let mahalanobis = (difference.transpose() * inverse * difference)[0];
    let determinants = first_covariance.determinant() * second_covariance.determinant();
    if determinants <= 0.0 {
        return 0.0;
    }
    let distance = mahalanobis / 8.0 + 0.5 * (covariance.determinant() / determinants.sqrt()).ln();
    (-distance).exp().clamp(0.0, 1.0)
}

/// Inflates the position covariance so that no direction is more certain than `floor`.
fn apply_position_floor(gaussian: &mut Gaussian<4>, floor: f32) {
    let position = gaussian.covariance.fixed_view::<2, 2>(0, 0).into_owned();
    let smallest = position.symmetric_eigenvalues().min();
    if smallest < floor {
        let inflation = Matrix2::identity() * (floor - smallest);
        let inflated = position + inflation;
        gaussian
            .covariance
            .fixed_view_mut::<2, 2>(0, 0)
            .copy_from(&inflated);
    }
}

fn role(attributes: &Attributes) -> HumanoidRole {
    if attributes.class.most_likely() == HUMAN {
        HumanoidRole::Human
    } else if let Some(player_number) = attributes.published_teammate {
        HumanoidRole::Teammate(player_number)
    } else {
        HumanoidRole::Opponent
    }
}

fn status(
    track: &Track<4, Attributes>,
    time_since_detection: Option<Duration>,
    parameters: &Parameters,
) -> TrackStatus {
    if track.existence < parameters.confirmation_existence {
        TrackStatus::Tentative
    } else if time_since_detection.is_none_or(|age| age > parameters.dormant_after) {
        TrackStatus::Dormant
    } else {
        TrackStatus::Confirmed
    }
}

/// Brings a teammate's field pose into the odometry frame with a covariance that accounts for
/// both localizations, our heading error over the lever arm, and the teammate's possible motion
/// since its pose was estimated.
pub fn team_report(
    player_number: PlayerNumber,
    position_in_field: linear_algebra::Point2<Field>,
    pose_age: Duration,
    ground_to_field: Isometry2<Ground, Field>,
    ground_to_odometry: Isometry2<Ground, Odometry>,
    parameters: &Parameters,
) -> TeamReport {
    let team = &parameters.team;
    let position_in_ground = ground_to_field.inverse() * position_in_field;
    let relative = position_in_ground.inner.coords;
    let perpendicular = Vector2::new(-relative.y, relative.x);
    let motion = team.maximum_speed * pose_age.as_secs_f32();
    let covariance_in_ground = Matrix2::identity()
        * (team.sender_position_standard_deviation.powi(2)
            + team.registration_position_standard_deviation.powi(2)
            + motion.powi(2) / 3.0)
        + perpendicular
            * perpendicular.transpose()
            * team.registration_heading_standard_deviation.powi(2);
    let rotation = ground_to_odometry.inner.rotation.to_rotation_matrix();
    TeamReport {
        player_number,
        position: (ground_to_odometry * position_in_ground).inner.coords,
        covariance: rotation.matrix() * covariance_in_ground * rotation.matrix().transpose(),
    }
}

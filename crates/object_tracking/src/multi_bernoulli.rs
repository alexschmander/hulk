//! Labeled multi-Bernoulli (LMB) tracking with belief-propagation association.
//!
//! Each [`Track`] is a Bernoulli component: an existence probability, a Gaussian mixture over the
//! state given existence, and discrete attributes such as a class belief. A scan of measurements is
//! processed by computing association weights, approximating their marginals with loopy belief
//! propagation, and replacing every track by the marginal LMB approximation of its posterior.
//! Unexplained position measurements initiate new tracks with a Poisson-style initiation weight.
//!
//! The independent per-track marginals cannot preserve joint association histories, e.g. which of
//! two robots left a duel on which side. That loss is a known property of this baseline.

use std::fmt::Debug;

use nalgebra::{SMatrix, SVector};
use serde::{Deserialize, Serialize};

use crate::{
    association::{
        AssociationProblem, BeliefPropagationSettings, ClusterStatistics, Edge, clustered_marginals,
    },
    gaussian::{Gaussian, merge, predict, squared_mahalanobis_distance_between},
    measurement::{MeasurementInnovation, MeasurementModel},
};

pub type TrackId = u64;

/// Discrete attributes carried by a track, updated with association probabilities.
pub trait TrackAttributes<Label> {
    /// Probability of the detector reporting `label` for this object, marginalized over the
    /// object's attributes.
    fn label_likelihood(&self, label: &Label) -> f32;

    /// Replaces the attribute belief by the mixture over association hypotheses.
    ///
    /// `missed_weight` is the posterior weight of no measurement (given existence) and each
    /// observation carries the posterior weight of being this track's measurement. The weights
    /// sum to one.
    fn update(&mut self, missed_weight: f32, observations: &[(f32, &Label)]);
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Component<const N: usize> {
    pub weight: f32,
    pub gaussian: Gaussian<N>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Track<const N: usize, A> {
    pub id: TrackId,
    pub existence: f32,
    pub components: Vec<Component<N>>,
    pub attributes: A,
    /// Probability that the last scan contained a measurement of this track.
    pub last_association_probability: f32,
    /// Sum of association probabilities over the track's lifetime.
    pub accumulated_association_probability: f32,
}

impl<const N: usize, A> Track<N, A> {
    pub fn new(id: TrackId, existence: f32, gaussian: Gaussian<N>, attributes: A) -> Self {
        Self {
            id,
            existence,
            components: vec![Component {
                weight: 1.0,
                gaussian,
            }],
            attributes,
            last_association_probability: 0.0,
            accumulated_association_probability: 0.0,
        }
    }

    /// Moment-matched single Gaussian of the state given existence.
    pub fn estimate(&self) -> Gaussian<N> {
        merge(
            self.components
                .iter()
                .map(|component| (component.weight, &component.gaussian)),
        )
        .unwrap_or_else(|| self.components[0].gaussian)
    }

    /// State of the component with the largest weight.
    pub fn dominant_component(&self) -> &Gaussian<N> {
        &self
            .components
            .iter()
            .max_by(|left, right| left.weight.total_cmp(&right.weight))
            .expect("tracks always have at least one component")
            .gaussian
    }
}

/// A measurement with its marks, in the tracking frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Measurement<Label> {
    pub model: MeasurementModel,
    pub label: Label,
    /// Density of false alarms with this label at this measurement, in the measurement's units
    /// (per m² for positions, per radian for bearings).
    pub clutter_density: f32,
}

/// Initiation proposal for a measurement that no existing track explains.
pub struct Birth<const N: usize, A> {
    pub gaussian: Gaussian<N>,
    pub attributes: A,
    /// Density of previously undetected objects that would produce this measurement, already
    /// multiplied by their detection probability, in the same units as the clutter density.
    pub intensity: f32,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, ros_z::Message)]
pub struct MixtureReduction {
    pub maximum_components: usize,
    pub prune_weight: f32,
    pub merge_squared_distance: f32,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, ros_z::Message)]
pub struct ScanSettings {
    /// Gate on the squared Mahalanobis distance of position measurements.
    pub position_gate: f32,
    /// Gate on the squared Mahalanobis distance of bearing measurements.
    pub bearing_gate: f32,
    /// Upper bound of detection probability and existence inside the association, keeping the
    /// missed weight positive.
    pub maximum_probability: f32,
    pub belief_propagation_iterations: usize,
    pub belief_propagation_tolerance: f64,
    /// Clusters with at most this many joint association hypotheses are solved exactly.
    pub maximum_exact_hypotheses: usize,
    pub mixture_reduction: MixtureReduction,
}

/// Diagnostics of one scan update.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, ros_z::Message)]
pub struct ScanReport {
    pub measurement_count: usize,
    pub gated_edge_count: usize,
    pub belief_propagation_iterations: usize,
    pub belief_propagation_converged: bool,
    pub cluster_count: usize,
    pub exact_cluster_count: usize,
    pub largest_cluster_tracks: usize,
    pub largest_cluster_measurements: usize,
    pub numerical_failures: usize,
    pub births: Vec<TrackId>,
    /// Probability of each measurement being explained by no pre-existing track.
    pub measurement_unassociated: Vec<f32>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct MultiBernoulli<const N: usize, A> {
    pub tracks: Vec<Track<N, A>>,
    next_id: TrackId,
}

impl<const N: usize, A> Default for MultiBernoulli<N, A> {
    fn default() -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
        }
    }
}

impl<const N: usize, A: Clone + Debug> MultiBernoulli<N, A> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn allocate_id(&mut self) -> TrackId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn insert(&mut self, existence: f32, gaussian: Gaussian<N>, attributes: A) -> TrackId {
        let id = self.allocate_id();
        self.tracks
            .push(Track::new(id, existence, gaussian, attributes));
        id
    }

    /// Predicts every track by one step.
    ///
    /// `transition` returns the transition matrix and process noise for a track, typically from a
    /// [`crate::motion::MotionModel`] selected by its attributes. `survival_probability` encodes
    /// physical disappearance or an explicit retention policy; a track that is merely out of view
    /// must not lose existence here. `additional_noise` adds per-component process noise, e.g.
    /// odometry drift.
    pub fn predict(
        &mut self,
        transition: impl Fn(&Track<N, A>) -> (SMatrix<f32, N, N>, SMatrix<f32, N, N>),
        survival_probability: impl Fn(&Track<N, A>) -> f32,
        additional_noise: impl Fn(&Gaussian<N>) -> SMatrix<f32, N, N>,
    ) {
        for track in &mut self.tracks {
            track.existence *= survival_probability(track).clamp(0.0, 1.0);
            let (transition, process_noise) = transition(track);
            for component in &mut track.components {
                let noise = process_noise + additional_noise(&component.gaussian);
                predict(&mut component.gaussian, &transition, &noise);
            }
        }
    }

    /// Processes one scan in which every track could have produced at most one measurement.
    ///
    /// `detection_probability` is evaluated per track component and must integrate visibility over
    /// the component's uncertainty. `birth` proposes a new track for a measurement; it is only
    /// consulted for position measurements.
    pub fn update<Label>(
        &mut self,
        measurements: &[Measurement<Label>],
        settings: &ScanSettings,
        detection_probability: impl Fn(&Track<N, A>, &Gaussian<N>) -> f32,
        birth: impl Fn(&Measurement<Label>) -> Option<Birth<N, A>>,
    ) -> ScanReport
    where
        A: TrackAttributes<Label>,
    {
        let mut report = ScanReport {
            measurement_count: measurements.len(),
            ..Default::default()
        };

        let births: Vec<Option<Birth<N, A>>> = measurements
            .iter()
            .map(|measurement| {
                measurement
                    .model
                    .is_position()
                    .then(|| birth(measurement))
                    .flatten()
                    .filter(|birth| birth.intensity > 0.0)
            })
            .collect();
        let birth_ratios: Vec<f64> = measurements
            .iter()
            .zip(&births)
            .map(|(measurement, birth)| {
                birth.as_ref().map_or(0.0, |birth| {
                    f64::from(birth.intensity) / f64::from(measurement.clutter_density.max(1e-12))
                })
            })
            .collect();

        let mut track_terms = Vec::with_capacity(self.tracks.len());
        let mut problem = AssociationProblem {
            missed_weights: Vec::with_capacity(self.tracks.len()),
            measurement_count: measurements.len(),
            edges: Vec::new(),
        };
        for (track_index, track) in self.tracks.iter().enumerate() {
            let existence = track.existence.clamp(0.0, settings.maximum_probability);
            let detection_probabilities: Vec<f32> = track
                .components
                .iter()
                .map(|component| {
                    detection_probability(track, &component.gaussian)
                        .clamp(0.0, settings.maximum_probability)
                })
                .collect();
            let detected_fraction: f32 = track
                .components
                .iter()
                .zip(&detection_probabilities)
                .map(|(component, probability)| component.weight * probability)
                .sum();
            problem
                .missed_weights
                .push(f64::from(1.0 - existence * detected_fraction));

            let mut innovations = Vec::new();
            for (measurement_index, measurement) in measurements.iter().enumerate() {
                let label_likelihood = track.attributes.label_likelihood(&measurement.label);
                if label_likelihood <= 0.0 {
                    continue;
                }
                let mut component_terms = Vec::with_capacity(track.components.len());
                for (component_index, component) in track.components.iter().enumerate() {
                    let probability = detection_probabilities[component_index];
                    if probability <= 0.0 {
                        continue;
                    }
                    let Some(innovation) =
                        MeasurementInnovation::compute(&component.gaussian, &measurement.model)
                    else {
                        report.numerical_failures += 1;
                        continue;
                    };
                    let gate = if innovation.is_position() {
                        settings.position_gate
                    } else {
                        settings.bearing_gate
                    };
                    if innovation.squared_mahalanobis_distance() > gate {
                        continue;
                    }
                    let log_weight =
                        component.weight.ln() + probability.ln() + innovation.log_likelihood();
                    component_terms.push((component_index, log_weight, innovation));
                }
                if component_terms.is_empty() {
                    continue;
                }
                let log_sum = log_sum_exp(component_terms.iter().map(|(_, log, _)| *log));
                let log_edge_weight = f64::from(existence.ln())
                    + f64::from(log_sum)
                    + f64::from(label_likelihood.ln())
                    - f64::from(measurement.clutter_density.max(1e-12).ln())
                    - (1.0 + birth_ratios[measurement_index]).ln();
                problem.edges.push(Edge {
                    track: track_index,
                    measurement: measurement_index,
                    weight: log_edge_weight.min(600.0).exp(),
                });
                innovations.push((measurement_index, log_sum, component_terms));
            }
            track_terms.push((detection_probabilities, detected_fraction, innovations));
        }
        report.gated_edge_count = problem.edges.len();

        let (marginals, clusters) = clustered_marginals(
            &problem,
            BeliefPropagationSettings {
                maximum_iterations: settings.belief_propagation_iterations,
                tolerance: settings.belief_propagation_tolerance,
            },
            settings.maximum_exact_hypotheses,
        );
        let ClusterStatistics {
            cluster_count,
            exact_cluster_count,
            largest_cluster_tracks,
            largest_cluster_measurements,
        } = clusters;
        report.belief_propagation_iterations = marginals.iterations;
        report.belief_propagation_converged = marginals.converged;
        report.cluster_count = cluster_count;
        report.exact_cluster_count = exact_cluster_count;
        report.largest_cluster_tracks = largest_cluster_tracks;
        report.largest_cluster_measurements = largest_cluster_measurements;

        let mut edge_lookup = vec![Vec::new(); self.tracks.len()];
        for (edge, probability) in problem.edges.iter().zip(&marginals.edges) {
            edge_lookup[edge.track].push((edge.measurement, *probability as f32));
        }

        for (track_index, (track, (detection_probabilities, detected_fraction, innovations))) in
            self.tracks.iter_mut().zip(track_terms).enumerate()
        {
            if detected_fraction <= 0.0 && innovations.is_empty() {
                // Invisible tracks receive no evidence from this scan.
                track.last_association_probability = 0.0;
                continue;
            }
            let existence = track.existence.clamp(0.0, settings.maximum_probability);
            let missed_probability = marginals.track_missed[track_index] as f32;
            let missed_fraction = (1.0 - detected_fraction).max(0.0);
            let missed_existence =
                existence * missed_fraction / (1.0 - existence + existence * missed_fraction);
            let associated: f32 = edge_lookup[track_index]
                .iter()
                .map(|(_, probability)| probability)
                .sum();
            let posterior_existence =
                (associated + missed_probability * missed_existence).clamp(0.0, 1.0);

            let mut components = Vec::new();
            if posterior_existence <= 0.0 {
                track.existence = 0.0;
                continue;
            }
            let missed_weight = missed_probability * missed_existence / posterior_existence;
            if missed_fraction > 0.0 && missed_weight > 0.0 {
                for (component, probability) in
                    track.components.iter().zip(&detection_probabilities)
                {
                    let weight =
                        missed_weight * component.weight * (1.0 - probability) / missed_fraction;
                    if weight > 0.0 {
                        components.push(Component {
                            weight,
                            gaussian: component.gaussian,
                        });
                    }
                }
            }

            let mut label_observations = Vec::new();
            for (measurement_index, log_sum, component_terms) in innovations {
                let probability = edge_lookup[track_index]
                    .iter()
                    .find(|(index, _)| *index == measurement_index)
                    .map_or(0.0, |(_, probability)| *probability);
                if probability <= 0.0 {
                    continue;
                }
                let weight = probability / posterior_existence;
                label_observations.push((weight, &measurements[measurement_index].label));
                for (component_index, log_weight, innovation) in component_terms {
                    let Some(gaussian) =
                        innovation.update(&track.components[component_index].gaussian)
                    else {
                        report.numerical_failures += 1;
                        continue;
                    };
                    components.push(Component {
                        weight: weight * (log_weight - log_sum).exp(),
                        gaussian,
                    });
                }
            }

            if components.is_empty() {
                track.existence = 0.0;
                continue;
            }
            track.attributes.update(missed_weight, &label_observations);
            track.existence = posterior_existence;
            track.last_association_probability = associated;
            track.accumulated_association_probability += associated;
            track.components = reduce_mixture(components, &settings.mixture_reduction);
        }
        self.tracks.retain(|track| track.existence > 0.0);

        for ((birth, unassociated), ratio) in births
            .into_iter()
            .zip(&marginals.measurement_unassociated)
            .zip(&birth_ratios)
        {
            let Some(birth) = birth else {
                continue;
            };
            let existence = (unassociated * ratio / (1.0 + ratio)) as f32;
            if existence <= 0.0 {
                continue;
            }
            let id = self.insert(existence, birth.gaussian, birth.attributes);
            if let Some(track) = self.tracks.last_mut() {
                track.last_association_probability = existence;
                track.accumulated_association_probability = existence;
            }
            report.births.push(id);
        }
        report.measurement_unassociated = marginals
            .measurement_unassociated
            .iter()
            .map(|probability| *probability as f32)
            .collect();
        report
    }

    /// Removes tracks below `minimum_existence` and keeps at most `maximum_tracks`, preferring
    /// tracks with higher existence.
    pub fn prune(&mut self, minimum_existence: f32, maximum_tracks: usize) {
        self.tracks
            .retain(|track| track.existence >= minimum_existence);
        if self.tracks.len() > maximum_tracks {
            self.tracks
                .sort_by(|left, right| right.existence.total_cmp(&left.existence));
            self.tracks.truncate(maximum_tracks);
        }
    }

    /// Merges tracks whose position means, the first two state dimensions, are closer than
    /// `distance`.
    ///
    /// Two physical objects cannot occupy the same place, so coincident tracks describe one object.
    /// Objects in contact can still come closer than `distance`, but then both of their tracks are
    /// detected in the same scan. Since a measurement is explained by at most one track, tracks
    /// that were probably both associated in the last scan explained different measurements and
    /// are kept apart.
    /// The track with more accumulated association probability survives with the larger existence.
    /// It takes the state of the track that is supported more by the last scan, and `merge`
    /// combines the attributes of the absorbed track, given with its existence, into the
    /// survivor's.
    pub fn merge_coincident(&mut self, distance: f32, mut merge: impl FnMut(&mut A, A, f32)) {
        let squared_distance = distance * distance;
        let mut index = 0;
        while index < self.tracks.len() {
            let position = self.tracks[index]
                .estimate()
                .mean
                .fixed_rows::<2>(0)
                .into_owned();
            let duplicate = (index + 1..self.tracks.len()).find(|other| {
                let other_position = self.tracks[*other]
                    .estimate()
                    .mean
                    .fixed_rows::<2>(0)
                    .into_owned();
                let both_associated = self.tracks[index].last_association_probability
                    * self.tracks[*other].last_association_probability;
                (other_position - position).norm_squared() < squared_distance
                    && both_associated <= 0.5
            });
            let Some(other) = duplicate else {
                index += 1;
                continue;
            };
            let other = self.tracks.remove(other);
            let survivor = &mut self.tracks[index];
            let mut absorbed = if other.accumulated_association_probability
                > survivor.accumulated_association_probability
            {
                std::mem::replace(survivor, other)
            } else {
                other
            };
            if absorbed.last_association_probability > survivor.last_association_probability {
                std::mem::swap(&mut survivor.components, &mut absorbed.components);
                survivor.last_association_probability = absorbed.last_association_probability;
            }
            survivor.existence = survivor.existence.max(absorbed.existence);
            survivor.accumulated_association_probability +=
                absorbed.accumulated_association_probability;
            merge(
                &mut survivor.attributes,
                absorbed.attributes,
                absorbed.existence,
            );
            // The merged track is compared with the remaining tracks again.
        }
    }
}

/// Prunes, merges and caps a Gaussian mixture, renormalizing the weights to one.
pub fn reduce_mixture<const N: usize>(
    mut components: Vec<Component<N>>,
    settings: &MixtureReduction,
) -> Vec<Component<N>> {
    let total: f32 = components.iter().map(|component| component.weight).sum();
    if total <= 0.0 {
        return components;
    }
    components
        .iter_mut()
        .for_each(|component| component.weight /= total);
    components.sort_by(|left, right| right.weight.total_cmp(&left.weight));
    let strongest = components[0].clone();
    components.retain(|component| component.weight >= settings.prune_weight);
    if components.is_empty() {
        components.push(strongest);
    }

    let mut reduced = Vec::new();
    while !components.is_empty() {
        let leader = components[0].gaussian;
        let (group, rest): (Vec<_>, Vec<_>) = components.into_iter().partition(|component| {
            squared_mahalanobis_distance_between(&leader, &component.gaussian)
                .is_some_and(|distance| distance <= settings.merge_squared_distance)
        });
        components = rest;
        let weight: f32 = group.iter().map(|component| component.weight).sum();
        let gaussian = merge(
            group
                .iter()
                .map(|component| (component.weight, &component.gaussian)),
        )
        .unwrap_or(leader);
        reduced.push(Component { weight, gaussian });
    }

    reduced.sort_by(|left, right| right.weight.total_cmp(&left.weight));
    reduced.truncate(settings.maximum_components.max(1));
    let total: f32 = reduced.iter().map(|component| component.weight).sum();
    reduced
        .iter_mut()
        .for_each(|component| component.weight /= total);
    reduced
}

fn log_sum_exp(values: impl Iterator<Item = f32> + Clone) -> f32 {
    let maximum = values.clone().fold(f32::NEG_INFINITY, f32::max);
    if !maximum.is_finite() {
        return maximum;
    }
    maximum
        + values
            .map(|value| (value - maximum).exp())
            .sum::<f32>()
            .ln()
}

/// Leading position of a state vector.
pub fn position<const N: usize>(state: &SVector<f32, N>) -> nalgebra::Vector2<f32> {
    state.fixed_rows::<2>(0).into_owned()
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{Matrix2, Vector2};

    use super::*;
    use crate::motion::{MotionModel, Stationary};

    #[derive(Clone, Debug, PartialEq)]
    struct NoAttributes;

    impl TrackAttributes<()> for NoAttributes {
        fn label_likelihood(&self, _label: &()) -> f32 {
            1.0
        }

        fn update(&mut self, _missed_weight: f32, _observations: &[(f32, &())]) {}
    }

    fn settings() -> ScanSettings {
        ScanSettings {
            position_gate: 13.82,
            bearing_gate: 10.83,
            maximum_probability: 0.999,
            belief_propagation_iterations: 100,
            belief_propagation_tolerance: 1e-6,
            maximum_exact_hypotheses: 1000,
            mixture_reduction: MixtureReduction {
                maximum_components: 4,
                prune_weight: 1e-3,
                merge_squared_distance: 1.0,
            },
        }
    }

    fn position_measurement(x: f32, y: f32) -> Measurement<()> {
        Measurement {
            model: MeasurementModel::Position {
                position: Vector2::new(x, y),
                covariance: Matrix2::identity() * 0.01,
            },
            label: (),
            clutter_density: 0.01,
        }
    }

    fn gaussian(x: f32, y: f32, variance: f32) -> Gaussian<2> {
        Gaussian {
            mean: Vector2::new(x, y),
            covariance: Matrix2::identity() * variance,
        }
    }

    fn birth_everywhere(measurement: &Measurement<()>) -> Option<Birth<2, NoAttributes>> {
        let MeasurementModel::Position {
            position,
            covariance,
        } = measurement.model
        else {
            return None;
        };
        Some(Birth {
            gaussian: Gaussian {
                mean: position,
                covariance,
            },
            attributes: NoAttributes,
            intensity: 0.01,
        })
    }

    #[test]
    fn invisible_track_keeps_existence_and_state() {
        let mut filter = MultiBernoulli::<2, NoAttributes>::new();
        filter.insert(0.8, gaussian(1.0, 1.0, 0.1), NoAttributes);
        let before = filter.tracks[0].clone();

        let report = filter.update(&[], &settings(), |_, _| 0.0, |_| None);

        assert!(report.births.is_empty());
        assert_relative_eq!(filter.tracks[0].existence, before.existence, epsilon = 1e-6);
        assert_eq!(filter.tracks[0].components, before.components);
    }

    #[test]
    fn visible_miss_lowers_existence_by_bayes_rule() {
        let mut filter = MultiBernoulli::<2, NoAttributes>::new();
        filter.insert(0.9, gaussian(1.0, 1.0, 0.1), NoAttributes);

        filter.update(&[], &settings(), |_, _| 0.9, |_| None);

        assert_relative_eq!(
            filter.tracks[0].existence,
            0.9 * 0.1 / (1.0 - 0.9 * 0.9),
            epsilon = 1e-5
        );
    }

    #[test]
    fn detection_confirms_track_and_moves_state() {
        let mut filter = MultiBernoulli::<2, NoAttributes>::new();
        filter.insert(0.5, gaussian(1.0, 1.0, 0.1), NoAttributes);

        let report = filter.update(
            &[position_measurement(1.2, 1.0)],
            &settings(),
            |_, _| 0.9,
            birth_everywhere,
        );

        let track = &filter.tracks[0];
        assert!(track.existence > 0.95, "existence {}", track.existence);
        assert!(track.estimate().mean.x > 1.1);
        assert!(report.measurement_unassociated[0] < 0.05);
        assert!(
            filter
                .tracks
                .iter()
                .skip(1)
                .all(|track| track.existence < 0.05)
        );
    }

    #[test]
    fn unexplained_measurement_starts_track() {
        let mut filter = MultiBernoulli::<2, NoAttributes>::new();
        let report = filter.update(
            &[position_measurement(3.0, 0.0)],
            &settings(),
            |_, _| 0.9,
            birth_everywhere,
        );

        assert_eq!(report.births.len(), 1);
        assert_relative_eq!(filter.tracks[0].existence, 0.5, epsilon = 1e-5);
    }

    #[test]
    fn stationary_prediction_keeps_mean() {
        let mut filter = MultiBernoulli::<2, NoAttributes>::new();
        filter.insert(0.5, gaussian(1.0, 1.0, 0.1), NoAttributes);
        filter.predict(
            |_| Stationary.transition(1.0),
            |_| 1.0,
            |_| Matrix2::identity() * 0.01,
        );

        assert_eq!(
            filter.tracks[0].components[0].gaussian.mean,
            Vector2::new(1.0, 1.0)
        );
        assert_relative_eq!(
            filter.tracks[0].components[0].gaussian.covariance[(0, 0)],
            0.11
        );
    }

    #[test]
    fn mixture_reduction_merges_close_and_caps_count() {
        let components = vec![
            Component {
                weight: 0.5,
                gaussian: gaussian(0.0, 0.0, 1.0),
            },
            Component {
                weight: 0.3,
                gaussian: gaussian(0.1, 0.0, 1.0),
            },
            Component {
                weight: 0.2,
                gaussian: gaussian(10.0, 0.0, 1.0),
            },
            Component {
                weight: 1e-6,
                gaussian: gaussian(-10.0, 0.0, 1.0),
            },
        ];
        let reduced = reduce_mixture(
            components,
            &MixtureReduction {
                maximum_components: 1,
                prune_weight: 1e-3,
                merge_squared_distance: 1.0,
            },
        );

        assert_eq!(reduced.len(), 1);
        assert_relative_eq!(reduced[0].weight, 1.0);
        assert!(reduced[0].gaussian.mean.x > 0.0 && reduced[0].gaussian.mean.x < 0.1);
    }

    #[test]
    fn coincident_tracks_merge_into_the_established_one() {
        let mut filter = MultiBernoulli::<2, Vec<u32>>::default();
        let established = filter.insert(0.9, gaussian(0.0, 0.0, 0.5), vec![1]);
        let distinct = filter.insert(0.9, gaussian(1.0, 0.0, 0.5), vec![2]);
        filter.insert(0.95, gaussian(0.1, 0.0, 0.01), vec![3]);
        filter.tracks[0].accumulated_association_probability = 10.0;
        filter.tracks[2].accumulated_association_probability = 1.0;
        filter.tracks[2].last_association_probability = 1.0;

        filter.merge_coincident(0.3, |kept, absorbed, _| kept.extend(absorbed));

        assert_eq!(filter.tracks.len(), 2);
        let merged = &filter.tracks[0];
        assert_eq!(merged.id, established);
        assert_eq!(merged.attributes, vec![1, 3]);
        assert_relative_eq!(merged.existence, 0.95);
        assert_relative_eq!(merged.estimate().mean.x, 0.1);
        assert_relative_eq!(merged.accumulated_association_probability, 11.0);
        assert_eq!(filter.tracks[1].id, distinct);
    }

    #[test]
    fn close_tracks_detected_in_the_same_scan_stay_apart() {
        let mut filter = MultiBernoulli::<2, Vec<u32>>::default();
        filter.insert(0.99, gaussian(0.0, 0.0, 0.1), vec![1]);
        filter.insert(0.99, gaussian(0.2, 0.0, 0.1), vec![2]);
        for track in &mut filter.tracks {
            track.last_association_probability = 0.9;
        }

        filter.merge_coincident(0.3, |kept, absorbed, _| kept.extend(absorbed));

        assert_eq!(filter.tracks.len(), 2);
    }
}

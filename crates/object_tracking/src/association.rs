//! Data association marginals for one scan of measurements.
//!
//! The problem is the standard point-target formulation: every track explains at most one
//! measurement and every measurement originates from at most one track. A track's weight for
//! being missed (or not existing) is `missed_weight`; an edge weight is the unnormalized weight of
//! the track explaining that measurement, relative to the measurement being clutter or a new
//! object, whose weight is normalized to one.
//!
//! [`loopy_belief_propagation`] approximates the marginals following Williams and Lau,
//! "Approximate evaluation of marginal association probabilities with belief propagation",
//! IEEE TAES 2014. [`exact_marginals`] enumerates all joint hypotheses. [`clustered_marginals`]
//! splits a problem into independent clusters and uses exact enumeration where it is affordable.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edge {
    pub track: usize,
    pub measurement: usize,
    /// Weight relative to the measurement's own unassociated weight, `>= 0`.
    pub weight: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AssociationProblem {
    /// Weight of each track being missed or not existing, `> 0`.
    pub missed_weights: Vec<f64>,
    pub measurement_count: usize,
    pub edges: Vec<Edge>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AssociationMarginals {
    /// Probability that each track explains no measurement.
    pub track_missed: Vec<f64>,
    /// Probability of each edge, aligned with [`AssociationProblem::edges`].
    pub edges: Vec<f64>,
    /// Probability that each measurement is explained by no existing track.
    pub measurement_unassociated: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BeliefPropagationSettings {
    pub maximum_iterations: usize,
    pub tolerance: f64,
}

impl Default for BeliefPropagationSettings {
    fn default() -> Self {
        Self {
            maximum_iterations: 100,
            tolerance: 1e-4,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClusterStatistics {
    pub cluster_count: usize,
    pub exact_cluster_count: usize,
    pub largest_cluster_tracks: usize,
    pub largest_cluster_measurements: usize,
}

/// Marginals of independent clusters, exact where the number of joint hypotheses of a cluster is
/// bounded by `maximum_exact_hypotheses` and approximated by belief propagation otherwise.
pub fn clustered_marginals(
    problem: &AssociationProblem,
    settings: BeliefPropagationSettings,
    maximum_exact_hypotheses: usize,
) -> (AssociationMarginals, ClusterStatistics) {
    let track_count = problem.missed_weights.len();
    let mut parents: Vec<usize> = (0..track_count + problem.measurement_count).collect();
    for edge in &problem.edges {
        let track_root = find(&mut parents, edge.track);
        let measurement_root = find(&mut parents, track_count + edge.measurement);
        parents[track_root] = measurement_root;
    }

    let mut marginals = AssociationMarginals {
        track_missed: vec![1.0; track_count],
        edges: vec![0.0; problem.edges.len()],
        measurement_unassociated: vec![1.0; problem.measurement_count],
        iterations: 0,
        converged: true,
    };
    let mut statistics = ClusterStatistics::default();

    let mut clusters: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for (index, edge) in problem.edges.iter().enumerate() {
        let root = find(&mut parents, edge.track);
        clusters.entry(root).or_default().push(index);
    }
    for edge_indices in clusters.values() {
        let mut tracks: Vec<usize> = edge_indices
            .iter()
            .map(|&edge| problem.edges[edge].track)
            .collect();
        tracks.sort_unstable();
        tracks.dedup();
        let mut measurements: Vec<usize> = edge_indices
            .iter()
            .map(|&edge| problem.edges[edge].measurement)
            .collect();
        measurements.sort_unstable();
        measurements.dedup();
        let local_track = |track: usize| tracks.binary_search(&track).expect("track in cluster");
        let local_measurement = |measurement: usize| {
            measurements
                .binary_search(&measurement)
                .expect("measurement in cluster")
        };
        let subproblem = AssociationProblem {
            missed_weights: tracks
                .iter()
                .map(|&track| problem.missed_weights[track])
                .collect(),
            measurement_count: measurements.len(),
            edges: edge_indices
                .iter()
                .map(|&edge| {
                    let edge = problem.edges[edge];
                    Edge {
                        track: local_track(edge.track),
                        measurement: local_measurement(edge.measurement),
                        weight: edge.weight,
                    }
                })
                .collect(),
        };

        let mut degrees = vec![1usize; tracks.len()];
        for edge in &subproblem.edges {
            degrees[edge.track] += 1;
        }
        let hypothesis_bound = degrees
            .iter()
            .try_fold(1usize, |product, degree| product.checked_mul(*degree))
            .unwrap_or(usize::MAX);
        let exact = hypothesis_bound <= maximum_exact_hypotheses;
        let local = if exact {
            exact_marginals(&subproblem)
        } else {
            loopy_belief_propagation(&subproblem, settings)
        };

        statistics.cluster_count += 1;
        statistics.exact_cluster_count += usize::from(exact);
        statistics.largest_cluster_tracks = statistics.largest_cluster_tracks.max(tracks.len());
        statistics.largest_cluster_measurements = statistics
            .largest_cluster_measurements
            .max(measurements.len());
        marginals.iterations = marginals.iterations.max(local.iterations);
        marginals.converged &= local.converged;
        for (local_index, &track) in tracks.iter().enumerate() {
            marginals.track_missed[track] = local.track_missed[local_index];
        }
        for (local_index, &measurement) in measurements.iter().enumerate() {
            marginals.measurement_unassociated[measurement] =
                local.measurement_unassociated[local_index];
        }
        for (local_index, &edge) in edge_indices.iter().enumerate() {
            marginals.edges[edge] = local.edges[local_index];
        }
    }
    (marginals, statistics)
}

fn find(parents: &mut [usize], node: usize) -> usize {
    let mut root = node;
    while parents[root] != root {
        root = parents[root];
    }
    let mut current = node;
    while parents[current] != root {
        let next = parents[current];
        parents[current] = root;
        current = next;
    }
    root
}

pub fn loopy_belief_propagation(
    problem: &AssociationProblem,
    settings: BeliefPropagationSettings,
) -> AssociationMarginals {
    let track_count = problem.missed_weights.len();
    let normalized_weights: Vec<f64> = problem
        .edges
        .iter()
        .map(|edge| edge.weight / problem.missed_weights[edge.track])
        .collect();
    let edges_of_track = edges_by(track_count, problem.edges.iter().map(|edge| edge.track));
    let edges_of_measurement = edges_by(
        problem.measurement_count,
        problem.edges.iter().map(|edge| edge.measurement),
    );

    let mut measurement_to_track = vec![1.0_f64; problem.edges.len()];
    let mut track_to_measurement = vec![0.0_f64; problem.edges.len()];
    let mut iterations = 0;
    let mut converged = problem.edges.is_empty();

    while !converged && iterations < settings.maximum_iterations {
        iterations += 1;
        for edges in &edges_of_track {
            let sum = 1.0
                + edges
                    .iter()
                    .map(|&edge| normalized_weights[edge] * measurement_to_track[edge])
                    .sum::<f64>();
            for &edge in edges {
                let others = sum - normalized_weights[edge] * measurement_to_track[edge];
                track_to_measurement[edge] = normalized_weights[edge] / others;
            }
        }
        let mut largest_change = 0.0_f64;
        for edges in &edges_of_measurement {
            let sum = 1.0
                + edges
                    .iter()
                    .map(|&edge| track_to_measurement[edge])
                    .sum::<f64>();
            for &edge in edges {
                let updated = 1.0 / (sum - track_to_measurement[edge]);
                largest_change = largest_change.max((updated - measurement_to_track[edge]).abs());
                measurement_to_track[edge] = updated;
            }
        }
        converged = largest_change < settings.tolerance;
    }

    let mut track_missed = vec![1.0; track_count];
    let mut edge_marginals = vec![0.0; problem.edges.len()];
    for (track, edges) in edges_of_track.iter().enumerate() {
        let sum = 1.0
            + edges
                .iter()
                .map(|&edge| normalized_weights[edge] * measurement_to_track[edge])
                .sum::<f64>();
        track_missed[track] = 1.0 / sum;
        for &edge in edges {
            edge_marginals[edge] = normalized_weights[edge] * measurement_to_track[edge] / sum;
        }
    }
    let measurement_unassociated = edges_of_measurement
        .iter()
        .map(|edges| {
            1.0 / (1.0
                + edges
                    .iter()
                    .map(|&edge| track_to_measurement[edge])
                    .sum::<f64>())
        })
        .collect();

    AssociationMarginals {
        track_missed,
        edges: edge_marginals,
        measurement_unassociated,
        iterations,
        converged,
    }
}

/// Exact marginals by enumerating every joint association. Exponential; for tests and tiny
/// clusters only.
pub fn exact_marginals(problem: &AssociationProblem) -> AssociationMarginals {
    let track_count = problem.missed_weights.len();
    let edges_of_track = edges_by(track_count, problem.edges.iter().map(|edge| edge.track));
    let mut track_missed = vec![0.0; track_count];
    let mut edge_marginals = vec![0.0; problem.edges.len()];
    let mut measurement_associated = vec![0.0; problem.measurement_count];
    let mut total = 0.0;

    let mut choice = vec![None; track_count];
    let mut used = vec![false; problem.measurement_count];
    enumerate(
        0,
        1.0,
        problem,
        &edges_of_track,
        &mut choice,
        &mut used,
        &mut |weight, choice| {
            total += weight;
            for (track, chosen) in choice.iter().enumerate() {
                match chosen {
                    None => track_missed[track] += weight,
                    Some(edge) => {
                        edge_marginals[*edge] += weight;
                        measurement_associated[problem.edges[*edge].measurement] += weight;
                    }
                }
            }
        },
    );

    AssociationMarginals {
        track_missed: track_missed.iter().map(|value| value / total).collect(),
        edges: edge_marginals.iter().map(|value| value / total).collect(),
        measurement_unassociated: measurement_associated
            .iter()
            .map(|value| 1.0 - value / total)
            .collect(),
        iterations: 0,
        converged: true,
    }
}

fn enumerate(
    track: usize,
    weight: f64,
    problem: &AssociationProblem,
    edges_of_track: &[Vec<usize>],
    choice: &mut Vec<Option<usize>>,
    used: &mut Vec<bool>,
    visit: &mut impl FnMut(f64, &[Option<usize>]),
) {
    if track == choice.len() {
        visit(weight, choice);
        return;
    }
    choice[track] = None;
    enumerate(
        track + 1,
        weight * problem.missed_weights[track],
        problem,
        edges_of_track,
        choice,
        used,
        visit,
    );
    for &edge in &edges_of_track[track] {
        let measurement = problem.edges[edge].measurement;
        if used[measurement] {
            continue;
        }
        used[measurement] = true;
        choice[track] = Some(edge);
        enumerate(
            track + 1,
            weight * problem.edges[edge].weight,
            problem,
            edges_of_track,
            choice,
            used,
            visit,
        );
        used[measurement] = false;
    }
    choice[track] = None;
}

fn edges_by(count: usize, keys: impl Iterator<Item = usize>) -> Vec<Vec<usize>> {
    let mut groups = vec![Vec::new(); count];
    for (edge, key) in keys.enumerate() {
        groups[key].push(edge);
    }
    groups
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    fn assert_marginals_close(
        actual: &AssociationMarginals,
        expected: &AssociationMarginals,
        epsilon: f64,
    ) {
        for (a, e) in actual.track_missed.iter().zip(&expected.track_missed) {
            assert_relative_eq!(a, e, epsilon = epsilon);
        }
        for (a, e) in actual.edges.iter().zip(&expected.edges) {
            assert_relative_eq!(a, e, epsilon = epsilon);
        }
        for (a, e) in actual
            .measurement_unassociated
            .iter()
            .zip(&expected.measurement_unassociated)
        {
            assert_relative_eq!(a, e, epsilon = epsilon);
        }
    }

    #[test]
    fn single_track_is_exact() {
        let problem = AssociationProblem {
            missed_weights: vec![0.5],
            measurement_count: 2,
            edges: vec![
                Edge {
                    track: 0,
                    measurement: 0,
                    weight: 2.0,
                },
                Edge {
                    track: 0,
                    measurement: 1,
                    weight: 0.5,
                },
            ],
        };
        let approximate = loopy_belief_propagation(&problem, BeliefPropagationSettings::default());
        let exact = exact_marginals(&problem);

        assert!(approximate.converged);
        assert_marginals_close(&approximate, &exact, 1e-9);
        assert_relative_eq!(exact.edges[0], 2.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn tree_shaped_problem_is_exact() {
        let edge = |track, measurement, weight| Edge {
            track,
            measurement,
            weight,
        };
        let problem = AssociationProblem {
            missed_weights: vec![0.4, 0.2, 0.7],
            measurement_count: 2,
            edges: vec![
                edge(0, 0, 3.0),
                edge(1, 0, 2.0),
                edge(1, 1, 5.0),
                edge(2, 1, 0.5),
            ],
        };
        let approximate = loopy_belief_propagation(&problem, BeliefPropagationSettings::default());
        let exact = exact_marginals(&problem);

        assert!(approximate.converged);
        assert_marginals_close(&approximate, &exact, 1e-4);
    }

    #[test]
    fn loopy_problem_stays_close_to_exact() {
        let edge = |track, measurement, weight| Edge {
            track,
            measurement,
            weight,
        };
        let problem = AssociationProblem {
            missed_weights: vec![0.2, 0.3, 0.1],
            measurement_count: 3,
            edges: vec![
                edge(0, 0, 5.0),
                edge(0, 1, 1.0),
                edge(1, 0, 4.0),
                edge(1, 1, 3.0),
                edge(1, 2, 0.2),
                edge(2, 1, 2.0),
                edge(2, 2, 6.0),
            ],
        };
        let approximate = loopy_belief_propagation(&problem, BeliefPropagationSettings::default());
        let exact = exact_marginals(&problem);

        assert!(approximate.converged);
        // Belief propagation is approximate on loopy graphs; this dense case is deliberately hard.
        assert_marginals_close(&approximate, &exact, 0.15);
        for (track, missed) in exact.track_missed.iter().enumerate() {
            let associated: f64 = problem
                .edges
                .iter()
                .zip(&exact.edges)
                .filter(|(edge, _)| edge.track == track)
                .map(|(_, probability)| probability)
                .sum();
            assert_relative_eq!(missed + associated, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn clusters_are_solved_independently_and_exactly() {
        let edge = |track, measurement, weight| Edge {
            track,
            measurement,
            weight,
        };
        let problem = AssociationProblem {
            missed_weights: vec![0.2, 0.3, 0.1, 0.5, 0.9],
            measurement_count: 5,
            edges: vec![
                edge(0, 0, 5.0),
                edge(0, 1, 1.0),
                edge(1, 0, 4.0),
                edge(1, 1, 3.0),
                edge(1, 2, 0.2),
                edge(2, 1, 2.0),
                edge(2, 2, 6.0),
                edge(3, 4, 1.5),
            ],
        };
        let (clustered, statistics) =
            clustered_marginals(&problem, BeliefPropagationSettings::default(), 1000);
        let exact = exact_marginals(&problem);

        assert_eq!(statistics.cluster_count, 2);
        assert_eq!(statistics.exact_cluster_count, 2);
        assert_eq!(statistics.largest_cluster_tracks, 3);
        assert_marginals_close(&clustered, &exact, 1e-9);
        assert_eq!(clustered.track_missed[4], 1.0);
        assert_eq!(clustered.measurement_unassociated[3], 1.0);
    }

    #[test]
    fn empty_problem_converges_immediately() {
        let problem = AssociationProblem {
            missed_weights: vec![1.0],
            measurement_count: 1,
            edges: vec![],
        };
        let marginals = loopy_belief_propagation(&problem, BeliefPropagationSettings::default());

        assert!(marginals.converged);
        assert_eq!(marginals.track_missed, vec![1.0]);
        assert_eq!(marginals.measurement_unassociated, vec![1.0]);
    }
}

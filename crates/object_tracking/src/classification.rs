//! Soft physical class of a track, observed through a detector confusion model.

use nalgebra::SVector;
use serde::{Deserialize, Serialize};

use crate::multi_bernoulli::TrackAttributes;

/// Likelihood of one detector label under each of `K` physical classes, `P(label | class)`.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub struct ClassEvidence<const K: usize> {
    pub likelihoods: SVector<f32, K>,
}

/// Probability of each of `K` physical classes.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub struct ClassBelief<const K: usize> {
    pub probabilities: SVector<f32, K>,
}

impl<const K: usize> ClassBelief<K> {
    pub fn uniform() -> Self {
        Self {
            probabilities: SVector::repeat(1.0 / K as f32),
        }
    }

    /// Posterior after observing `evidence` starting from `prior`.
    pub fn from_evidence(prior: &Self, evidence: &ClassEvidence<K>) -> Self {
        let mut belief = *prior;
        belief.update(0.0, &[(1.0, evidence)]);
        belief
    }

    pub fn most_likely(&self) -> usize {
        self.probabilities.imax()
    }

    /// Expected value of a per-class quantity.
    pub fn expectation(&self, values: &SVector<f32, K>) -> f32 {
        self.probabilities.dot(values)
    }
}

impl<const K: usize> TrackAttributes<ClassEvidence<K>> for ClassBelief<K> {
    fn label_likelihood(&self, evidence: &ClassEvidence<K>) -> f32 {
        self.probabilities.dot(&evidence.likelihoods)
    }

    fn update(&mut self, missed_weight: f32, observations: &[(f32, &ClassEvidence<K>)]) {
        let mut posterior = self.probabilities * missed_weight.max(0.0);
        for (weight, evidence) in observations {
            let joint = self.probabilities.component_mul(&evidence.likelihoods);
            let normalization = joint.sum();
            if normalization > 0.0 {
                posterior += joint * (*weight / normalization);
            }
        }
        let total = posterior.sum();
        if total > 0.0 && total.is_finite() {
            self.probabilities = posterior / total;
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::Vector3;

    use super::*;

    #[test]
    fn repeated_consistent_labels_concentrate_belief() {
        let evidence = ClassEvidence {
            likelihoods: Vector3::new(0.8, 0.1, 0.1),
        };
        let mut belief = ClassBelief::<3>::uniform();
        for _ in 0..3 {
            belief.update(0.0, &[(1.0, &evidence)]);
        }

        assert_eq!(belief.most_likely(), 0);
        assert!(belief.probabilities[0] > 0.99);
    }

    #[test]
    fn uncertain_association_only_partially_updates() {
        let evidence = ClassEvidence {
            likelihoods: Vector3::new(1.0, 0.0, 0.0),
        };
        let mut belief = ClassBelief::<3>::uniform();
        belief.update(0.5, &[(0.5, &evidence)]);

        assert_relative_eq!(belief.probabilities[0], 0.5 / 3.0 + 0.5, epsilon = 1e-6);
        assert_relative_eq!(belief.probabilities.sum(), 1.0, epsilon = 1e-6);
    }
}

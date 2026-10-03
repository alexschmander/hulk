use color_eyre::Result;
use fagra::{BlockId, StateKey};
use linear_algebra::IntoTransform;
use localization_fagra::variables::PoseControl;
use nalgebra::SMatrix;
use types::localization::{LocalizationEstimate, PoseEstimate};

use super::Estimator;

// Four pose controls, optional planar alignment, and two six-dimensional bias knots.
// The last three rows/columns are unused when no alignment exists.
pub(super) type EstimateCovariance = SMatrix<f64, 39, 39>;

impl Estimator {
    pub fn estimate_covariance_blocks(
        &mut self,
        controls: [StateKey<PoseControl>; 4],
    ) -> Result<Vec<BlockId>> {
        let mut blocks: Vec<BlockId> = controls.iter().map(|key| key.block_id()).collect();
        if let Some(alignment) = self.alignment {
            blocks.push(alignment.block_id());
        }
        let (biases, _) = self.ensure_biases(self.latest_time)?;
        blocks.extend(biases.map(|key| key.block_id()));
        Ok(blocks)
    }

    pub fn estimate_from_covariance(
        &self,
        controls: [StateKey<PoseControl>; 4],
        tau: f64,
        covariance: &EstimateCovariance,
        diagnostics: &mut crate::diagnostics::SolveDiagnostics,
    ) -> Result<LocalizationEstimate> {
        let (_, fraction) = super::bias::bias_segment_and_tau(
            self.origin,
            self.latest_time,
            self.parameters.timing.bias_ns(),
        );
        let bias = self.bias_at(self.latest_time)?;
        let bias_offset = 24 + usize::from(self.alignment.is_some()) * 3;
        // Preserve cross-correlations between all controls and field alignment.
        // The unused alignment rows remain zero before field initialization.
        let joint = SMatrix::<f64, 27, 27>::from_fn(|row, col| {
            if row < bias_offset && col < bias_offset {
                covariance[(row, col)]
            } else {
                0.0
            }
        });
        let weights = [1.0 - fraction, fraction];
        let bias_covariance = SMatrix::<f64, 6, 6>::from_fn(|row, col| {
            let mut value = 0.0;
            for i in 0..2 {
                for j in 0..2 {
                    value += weights[i]
                        * weights[j]
                        * covariance[(bias_offset + i * 6 + row, bias_offset + j * 6 + col)];
                }
            }
            value
        });
        diagnostics.imu_bias = Some(crate::diagnostics::ImuBiasEstimate {
            gyroscope: bias.gyroscope,
            accelerometer: bias.accelerometer,
            covariance: std::array::from_fn(|r| std::array::from_fn(|c| bias_covariance[(r, c)])),
        });
        let spline = self.spline(controls)?;
        let sample = spline.linearize()?.pose(tau)?;
        let local = PoseEstimate {
            pose: sample.pose.inner.framed_transform(),
            covariance: sample.covariance(&joint.fixed_view::<24, 24>(0, 0).into_owned()),
        };
        let field = if let Some(key) = self.alignment {
            let alignment = self.graph.get(key)?;
            Some(PoseEstimate {
                pose: alignment.local_to_field.to_3d() * local.pose,
                covariance: sample.field_covariance(&joint),
            })
        } else {
            None
        };
        Ok(LocalizationEstimate {
            time: self.latest_time,
            epoch: self.epoch,
            generation: self.generation,
            robot_to_local: local,
            robot_to_field: field,
        })
    }
}

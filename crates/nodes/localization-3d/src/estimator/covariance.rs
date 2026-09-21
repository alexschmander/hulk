use color_eyre::Result;
use fagra::{BlockId, StateKey};
use linear_algebra::IntoTransform;
use localization_fagra::variables::PoseControl;
use nalgebra::SMatrix;
use types::localization::{LocalizationEstimate, PoseEstimate};

use super::Estimator;

impl Estimator {
    pub(super) fn estimate_with_covariance(
        &mut self,
        controls: [StateKey<PoseControl>; 4],
        tau: f64,
    ) -> Result<LocalizationEstimate> {
        let mut blocks: Vec<BlockId> = controls.iter().map(|key| key.block_id()).collect();
        if let Some(alignment) = self.alignment {
            blocks.push(alignment.block_id());
        }
        // Preserve cross-correlations between all controls and field alignment.
        // The unused alignment rows remain zero before field initialization.
        let covariance = self.graph.joint_covariance(&blocks)?;
        let joint = SMatrix::<f64, 27, 27>::from_fn(|row, col| {
            if row < covariance.nrows() && col < covariance.ncols() {
                covariance[(row, col)]
            } else {
                0.0
            }
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
            robot_to_local: local,
            robot_to_field: field,
        })
    }
}

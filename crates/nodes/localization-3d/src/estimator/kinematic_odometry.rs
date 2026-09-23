use color_eyre::Result;
use localization_fagra::factors::{AdjacentKinematicOdometry, KinematicOdometry};
use nalgebra::Matrix2;
use types::odometry::KinematicOdometryDelta;

use super::{Estimator, HUBER_THRESHOLD, WINDOW_NS, seconds_per_knot};

impl Estimator {
    pub(crate) fn ingest_kinematic_odometry(
        &mut self,
        sample: KinematicOdometryDelta,
    ) -> Result<bool> {
        if sample.time <= sample.previous_time
            || self
                .latest_kinematic_time
                .is_some_and(|last| sample.previous_time < last)
            || !sample
                .current_to_previous
                .inner
                .to_homogeneous()
                .iter()
                .all(|v| v.is_finite())
            || sample.previous_time.as_nanos()
                < self
                    .latest_time
                    .max(sample.time)
                    .as_nanos()
                    .saturating_sub(WINDOW_NS)
        {
            return Ok(false);
        }
        let Some((a, previous_tau)) =
            self.check_time(sample.previous_time, "kinematic odometry")?
        else {
            return Ok(false);
        };
        let Some((b, current_tau)) = self.check_time(sample.time, "kinematic odometry")? else {
            return Ok(false);
        };
        if b - a > 1 {
            return Ok(false);
        }
        let first = self.ensure_segment(a)?;
        let second = self.ensure_segment(b)?;
        let dt = sample
            .time
            .duration_since(sample.previous_time)
            .as_secs_f64();
        // ponytail: diagonal noise approximates shared encoder/IMU errors; calibrate
        // on recordings, use correlated preintegration if consistency requires it.
        let noise = &self.parameters.kinematic_odometry_noise;
        let variance =
            noise.position_sigma.map(|s| s * s) + noise.translation_variance_per_second * dt;
        let information_root = Matrix2::from_diagonal(&variance.map(|v| v.sqrt().recip()));
        let translation = sample.current_to_previous.inner.translation.vector.cast();
        if a == b {
            self.graph.add_factor(KinematicOdometry {
                controls: first,
                duration: seconds_per_knot(),
                previous_tau,
                current_tau,
                translation,
                information_root,
                huber_threshold: HUBER_THRESHOLD,
            })?;
        } else {
            self.graph.add_factor(AdjacentKinematicOdometry {
                controls: [first[0], first[1], first[2], first[3], second[3]],
                duration: seconds_per_knot(),
                previous_tau,
                current_tau,
                translation,
                information_root,
                huber_threshold: HUBER_THRESHOLD,
            })?;
        }
        self.latest_kinematic_time = Some(sample.time);
        self.commit_time(sample.time);
        *self.measurements.entry(a).or_default() += 1;
        Ok(true)
    }
}

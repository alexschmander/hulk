use super::{Estimator, KNOT_SPACING_NS, WINDOW_NS, control_keys, seconds_per_knot};
use color_eyre::{Result, eyre::eyre};
use fagra::StateKey;
use localization_fagra::{
    factors::{FieldContainment, MotionPrior},
    spline::PoseSpline,
    variables::PoseControl,
};
use ros_z::time::Time;

impl Estimator {
    pub(super) fn check_time(
        &self,
        time: Time,
        sensor: &'static str,
    ) -> Result<Option<(i64, f64)>> {
        if time < self.origin
            || time.as_nanos() < self.latest_time.as_nanos().saturating_sub(WINDOW_NS)
        {
            tracing::warn!(
                sensor,
                ?time,
                "discarding measurement outside optimization window or epoch"
            );
            return Ok(None);
        }
        self.segment_and_tau(time).map(Some)
    }

    pub(super) fn commit_time(&mut self, time: Time) {
        self.latest_time = self.latest_time.max(time);
    }

    pub(super) fn segment_and_tau(&self, time: Time) -> Result<(i64, f64)> {
        let elapsed = time
            .as_nanos()
            .checked_sub(self.origin.as_nanos())
            .filter(|elapsed| *elapsed >= 0)
            .ok_or_else(|| eyre!("timestamp outside trajectory domain"))?;
        Ok((
            elapsed / KNOT_SPACING_NS,
            (elapsed % KNOT_SPACING_NS) as f64 / KNOT_SPACING_NS as f64,
        ))
    }

    pub(super) fn ensure_segment(&mut self, segment: i64) -> Result<[StateKey<PoseControl>; 4]> {
        let largest = *self
            .controls
            .last_key_value()
            .ok_or_else(|| eyre!("empty trajectory"))?
            .0;
        let next_segment = self.segments().end;
        let gap = segment - next_segment >= 5;
        for index in largest + 1..=segment + 2 {
            let previous = self.graph.get(self.controls[&(index - 1)])?;
            let before = self.graph.get(self.controls[&(index - 2)])?;
            let mut prediction = previous.clone();
            if !gap {
                prediction.pose.inner.translation.vector +=
                    previous.pose.inner.translation.vector - before.pose.inner.translation.vector;
                prediction.pose.inner.rotation *=
                    before.pose.inner.rotation.inverse() * previous.pose.inner.rotation;
            }
            prediction.pose.inner.rotation.renormalize();
            self.controls.insert(index, self.graph.add(prediction));
        }
        for current in next_segment..=segment {
            self.add_motion_prior(current, gap && current < segment)?;
            self.add_containment(current)?;
        }
        control_keys(&self.controls, segment)
    }

    pub(super) fn add_motion_prior(&mut self, segment: i64, gap: bool) -> Result<()> {
        let root = MotionPrior::information_root(
            seconds_per_knot(),
            0.01,
            self.parameters.accelerometer_process_noise_variance,
        )? / if gap { 10.0_f64.sqrt() } else { 1.0 };
        self.graph.add_factor(MotionPrior {
            controls: control_keys(&self.controls, segment)?,
            duration: seconds_per_knot(),
            information_root: root,
            use_start_velocity: !gap,
        })?;
        Ok(())
    }

    pub(super) fn add_containment(&mut self, segment: i64) -> Result<()> {
        if let Some(alignment) = self.alignment {
            self.graph.add_factor(FieldContainment {
                controls: control_keys(&self.controls, segment)?,
                duration: seconds_per_knot(),
                tau: 0.5,
                alignment,
                half_extents: self.field_half_extents,
                sigma: self.parameters.field_containment_sigma,
            })?;
        }
        Ok(())
    }

    pub(super) fn spline(&self, controls: [StateKey<PoseControl>; 4]) -> Result<PoseSpline<f64>> {
        Ok(PoseSpline::new(
            [
                self.graph.get(controls[0])?,
                self.graph.get(controls[1])?,
                self.graph.get(controls[2])?,
                self.graph.get(controls[3])?,
            ],
            seconds_per_knot(),
        )?)
    }

    pub(super) fn retire_old_segments(&mut self) -> Result<()> {
        let (latest_segment, _) = self.segment_and_tau(self.latest_time)?;
        let oldest = (latest_segment - WINDOW_NS / KNOT_SPACING_NS).max(0);
        let mut old_states: Vec<_> = self
            .controls
            .range(..oldest - 1)
            .map(|(_, key)| key.block_id())
            .collect();
        if old_states.is_empty() {
            return Ok(());
        }
        let oldest_bias = oldest * KNOT_SPACING_NS / super::bias::BIAS_KNOT_SPACING_NS;
        old_states.extend(
            self.biases
                .range(..oldest_bias)
                .map(|(_, key)| key.block_id()),
        );
        self.graph.marginalize(&old_states)?;
        self.biases.retain(|index, _| *index >= oldest_bias);
        self.controls.retain(|index, _| *index >= oldest - 1);
        self.measurements.retain(|index, _| *index >= oldest);
        self.yaw_factors.retain(|index, _| *index >= oldest);
        self.retire_preintegration(oldest);
        while self
            .foot_batches
            .first_key_value()
            .is_some_and(|(index, _)| *index < oldest)
        {
            let (_, batch) = self.foot_batches.pop_first().unwrap();
            self.graph.remove_batch(batch)?;
        }
        while self
            .odometry_batches
            .first_key_value()
            .is_some_and(|(index, _)| *index < oldest)
        {
            let (_, batch) = self.odometry_batches.pop_first().unwrap();
            self.graph.remove_batch(batch)?;
        }
        for &(index, batch) in &self.reprojection_batches {
            if index < oldest {
                self.graph.remove_batch(batch)?;
            }
        }
        self.reprojection_batches
            .retain(|(index, _)| *index >= oldest);
        Ok(())
    }
}

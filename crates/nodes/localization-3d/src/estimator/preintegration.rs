use std::collections::{BTreeMap, BTreeSet};

use color_eyre::{Result, eyre::WrapErr};
use coordinate_systems::{ImuReference, Robot};
use fagra::FactorKey;
use linear_algebra::Vector3;
use localization_fagra::{
    factors::{ImuKinematics, PreintegratedImu},
    preintegration::{ImuNoise, ImuPreintegrator},
    variables::ImuBias,
};
use nalgebra::Matrix3;
use ros_z::time::Time;
use types::localization::MAX_IMU_ATTITUDE_GAP;

use super::{
    Estimator, KNOT_SPACING_NS,
    bias::{BIAS_KNOT_SPACING_NS, bias_segment_and_tau},
    seconds_per_knot,
};

const INTERVAL_NS: i64 = 100_000_000;
const _: () = assert!(KNOT_SPACING_NS % INTERVAL_NS == 0);

#[derive(Clone, Copy)]
struct Sample {
    gyro: Vector3<Robot, f64>,
    force: Vector3<Robot, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use booster::ImuState;
    use std::time::Duration;

    fn ingest(estimator: &mut Estimator, millis: u64, force: f32) {
        estimator
            .ingest_imu(
                estimator.origin + Duration::from_millis(millis),
                ImuState {
                    linear_acceleration: Vector3::wrap(nalgebra::vector![0.0, 0.0, force]),
                    ..Default::default()
                },
            )
            .unwrap();
    }

    #[test]
    fn impulses_missing_force_gaps_and_late_bridges_keep_distinct_likelihoods() {
        let mut estimator = super::super::tests::estimator();
        estimator.parameters.accelerometer = Some(Default::default());
        for millis in [0, 10, 20, 30, 70, 80, 90, 100] {
            ingest(
                &mut estimator,
                millis,
                match millis {
                    10 => 100.0,
                    20 => f32::NAN,
                    _ => 0.0,
                },
            );
            estimator.prepare_preintegration().unwrap();
        }
        let pieces = &estimator.preintegration.intervals[&0].pieces;
        assert!(
            (pieces.iter().map(|(_, _, p)| p.delta.duration).sum::<f64>() - 0.06).abs() < 1e-12
        );
        assert!(
            (pieces
                .iter()
                .filter(|(_, _, p)| p.acceleration)
                .map(|(_, _, p)| p.delta.duration)
                .sum::<f64>()
                - 0.05)
                .abs()
                < 1e-12
        );
        assert!(
            (pieces
                .iter()
                .map(|(_, _, p)| p.delta.velocity.z())
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-12
        );
        for millis in [50, 60] {
            ingest(&mut estimator, millis, 0.0);
        }
        estimator.prepare_preintegration().unwrap();
        let pieces = &estimator.preintegration.intervals[&0].pieces;
        assert!((pieces.iter().map(|(_, _, p)| p.delta.duration).sum::<f64>() - 0.1).abs() < 1e-12);
        assert_eq!(pieces.len(), 3);
        assert!(!pieces[1].2.acceleration);
        assert_eq!(pieces[2].2.delta.velocity, Vector3::zeros()); // Flight, not missing data.
        let mut parameters = estimator.parameters.clone();
        parameters.accelerometer = None;
        estimator.update_parameters(parameters);
        estimator.prepare_preintegration().unwrap();
        let pieces = &estimator.preintegration.intervals[&0].pieces;
        assert_eq!(pieces.len(), 1);
        assert!(!pieces[0].2.acceleration);
    }

    #[test]
    fn bias_boundary_and_accepted_bias_refresh_reintegrate_the_right_knots() {
        let mut estimator = super::super::tests::estimator();
        estimator.parameters.accelerometer = Some(Default::default());
        for millis in (4950..=5050).step_by(2) {
            ingest(&mut estimator, millis, 9.81);
        }
        estimator.prepare_preintegration().unwrap();
        assert!(
            (estimator.preintegration.intervals[&49].pieces[0]
                .2
                .delta
                .duration
                - 0.05)
                .abs()
                < 1e-12
        );
        assert!(
            (estimator.preintegration.intervals[&50].pieces[0]
                .2
                .delta
                .duration
                - 0.05)
                .abs()
                < 1e-12
        );
        for &key in estimator.biases.values() {
            estimator
                .graph
                .set(
                    key,
                    ImuBias {
                        gyroscope: Vector3::wrap(nalgebra::vector![0.0, 0.0, 0.02]),
                        accelerometer: Vector3::wrap(nalgebra::vector![0.0, 0.0, 0.2]),
                    },
                )
                .unwrap();
        }
        estimator.prepare_preintegration().unwrap();
        for index in [49, 50] {
            let p = &estimator.preintegration.intervals[&index].pieces[0].2;
            assert!((p.delta.rotation.inner.scaled_axis().z + 0.001).abs() < 1e-12);
            assert!((p.delta.velocity.z() - (9.81_f32 as f64 - 0.2) * 0.05).abs() < 1e-12);
            assert_eq!(p.delta.reference_biases[0].gyroscope.z(), 0.02);
        }
        let mut parameters = estimator.parameters.clone();
        parameters.accelerometer.as_mut().unwrap().scale.z = 2.0;
        estimator.update_parameters(parameters);
        estimator.prepare_preintegration().unwrap();
        assert!(
            (estimator.preintegration.intervals[&50].pieces[0]
                .2
                .delta
                .velocity
                .z()
                - (2.0 * 9.81_f32 as f64 - 0.2) * 0.05)
                .abs()
                < 1e-12
        );
    }
}

struct Interval {
    factors: Vec<FactorKey<PreintegratedImu>>,
    boundaries: Vec<FactorKey<ImuKinematics>>,
    reference_biases: [ImuBias; 2],
    end: Option<Time>,
    pieces: Vec<(Time, Time, ImuPreintegrator)>,
    reusable: bool,
}

#[derive(Default)]
pub(super) struct ImuIntervals {
    samples: BTreeMap<Time, Sample>,
    intervals: BTreeMap<i64, Interval>,
    dirty: BTreeSet<i64>,
    terminal: Option<(i64, FactorKey<ImuKinematics>)>,
}

impl ImuIntervals {
    pub(super) fn restore_boundary(&mut self, origin: Time, start: Time, source: &Self) {
        if let Some((&time, &sample)) = source.samples.range(..start).next_back() {
            self.insert(origin, time, sample.gyro, sample.force);
        }
    }
    #[cfg(test)]
    pub(super) fn interval_count(&self) -> usize {
        self.intervals.len()
    }
    pub(super) fn invalidate(&mut self) {
        self.dirty.extend(self.intervals.keys().copied());
        for interval in self.intervals.values_mut() {
            interval.reusable = false;
        }
    }

    pub(super) fn insert(
        &mut self,
        origin: Time,
        time: Time,
        gyro: Vector3<Robot, f64>,
        force: Vector3<Robot, f64>,
    ) {
        // A late/replaced reading changes the two adjacent held-measurement spans.
        // Long gaps carry no inertial likelihood, and need no empty interval cache.
        let before = self.samples.range(..time).next_back().map(|(&t, _)| t);
        let after = self
            .samples
            .range((std::ops::Bound::Excluded(time), std::ops::Bound::Unbounded))
            .next()
            .map(|(&t, _)| t);
        if self
            .samples
            .last_key_value()
            .is_some_and(|(&latest, _)| time <= latest)
        {
            let first = (before.unwrap_or(time).as_nanos() - origin.as_nanos()) / INTERVAL_NS;
            let last = (after.unwrap_or(time).as_nanos() - origin.as_nanos()) / INTERVAL_NS;
            for (_, interval) in self.intervals.range_mut(first..=last) {
                interval.reusable = false;
            }
        }
        self.samples.insert(time, Sample { gyro, force });
        self.dirty
            .insert((time.as_nanos() - origin.as_nanos()) / INTERVAL_NS);
        for (a, b) in before
            .map(|a| (a, time))
            .into_iter()
            .chain(after.map(|b| (time, b)))
        {
            self.dirty
                .insert((a.as_nanos() - origin.as_nanos()) / INTERVAL_NS);
            if b.duration_since(a) <= MAX_IMU_ATTITUDE_GAP {
                let first = (a.as_nanos() - origin.as_nanos()) / INTERVAL_NS;
                let last = (b.as_nanos() - origin.as_nanos() - 1) / INTERVAL_NS;
                self.dirty.extend(first..=last);
            }
        }
    }
}

impl Estimator {
    pub(super) fn prepare_preintegration(&mut self) -> Result<()> {
        let p = &self.parameters.imu_preintegration;
        for (&index, interval) in &mut self.preintegration.intervals {
            let knot = index * INTERVAL_NS / BIAS_KNOT_SPACING_NS;
            for (offset, reference) in interval.reference_biases.iter().enumerate() {
                let bias = self.graph.get(self.biases[&(knot + offset as i64)])?;
                if (bias.gyroscope - reference.gyroscope).norm()
                    > p.gyroscope_reintegration_threshold
                    || (bias.accelerometer - reference.accelerometer).norm()
                        > p.accelerometer_reintegration_threshold
                {
                    self.preintegration.dirty.insert(index);
                    interval.reusable = false;
                }
            }
        }
        let oldest = self.segments().start * KNOT_SPACING_NS / INTERVAL_NS;
        while let Some(&index) = self.preintegration.dirty.first() {
            if index >= oldest {
                self.rebuild_imu_interval(index)?;
            }
            self.preintegration.dirty.remove(&index);
        }
        if let Some((_, boundary)) = self.preintegration.terminal.take() {
            self.graph.remove_factor(boundary)?;
        }
        if let Some((&time, &sample)) = self.preintegration.samples.last_key_value()
            && self.segment_and_tau(time)?.0 >= self.segments().start
        {
            // This sample has no following span yet. Remove its instantaneous
            // gyro constraint before the next solve, when it enters an integral.
            let has_tilt = self
                .preintegration
                .intervals
                .values()
                .any(|i| i.end == Some(time) && !i.boundaries.is_empty());
            let tilt = if has_tilt {
                0.0
            } else {
                self.parameters.imu_preintegration.tilt_sigma.recip()
            };
            let batch = self.add_imu_boundary(
                time,
                sample.gyro,
                self.parameters
                    .imu_preintegration
                    .terminal_gyroscope_sigma
                    .recip(),
                tilt,
                false,
            )?;
            self.preintegration.terminal = Some((self.segment_and_tau(time)?.0, batch));
        }
        Ok(())
    }

    fn rebuild_imu_interval(&mut self, index: i64) -> Result<()> {
        let start = Time::from_nanos(self.origin.as_nanos() + index * INTERVAL_NS);
        let end = Time::from_nanos(start.as_nanos() + INTERVAL_NS);
        let before = self
            .preintegration
            .samples
            .range(..=start)
            .next_back()
            .map_or(start, |(&t, _)| t);
        let after = self
            .preintegration
            .samples
            .range(end..)
            .next()
            .map_or(end, |(&t, _)| t);
        let (bias_keys, _) = self.ensure_biases(start)?;
        let cached = self
            .preintegration
            .intervals
            .get(&index)
            .filter(|i| i.reusable);
        let reference_biases = if let Some(cached) = cached {
            cached.reference_biases.clone()
        } else {
            [
                self.graph.get(bias_keys[0])?.clone(),
                self.graph.get(bias_keys[1])?.clone(),
            ]
        };
        let p = &self.parameters.imu_preintegration;
        let noise = ImuNoise {
            gyroscope: p.gyroscope_noise_density.powi(2),
            accelerometer: self
                .parameters
                .accelerometer
                .as_ref()
                .map_or(0.0, |a| a.noise_density.powi(2)),
            integration: p.integration_noise_density.powi(2),
        };
        let mut pieces = cached.map_or_else(Vec::new, |i| i.pieces.clone());
        let resume = pieces.last().map_or(start, |(_, end, _)| *end);
        let mut gap_boundaries = Vec::new();
        let mut samples = self.preintegration.samples.range(before..=after).peekable();
        while let Some((&a, sample)) = samples.next() {
            let Some((&b, _)) = samples.peek().copied() else {
                break;
            };
            if b.duration_since(a) > MAX_IMU_ATTITUDE_GAP {
                if a >= start && a < end {
                    gap_boundaries.push((a, sample.gyro));
                }
                continue;
            }
            let left = a.max(resume);
            let right = b.min(end);
            if right <= left {
                continue;
            }
            let force = self.parameters.accelerometer.as_ref().and_then(|p| {
                let force = (sample.force.inner - p.bias.inner).component_mul(&p.scale);
                force
                    .iter()
                    .all(|v| v.is_finite())
                    .then(|| Vector3::wrap(force))
            });
            let acceleration = force.is_some();
            if !pieces
                .last()
                .is_some_and(|(_, last, p)| *last == left && p.acceleration == acceleration)
            {
                pieces.push((
                    left,
                    left,
                    ImuPreintegrator::new(reference_biases.clone(), acceleration),
                ));
            }
            let (_, last, integrated) = pieces.last_mut().unwrap();
            // Midpoint quadrature preserves the independent linearly varying bias.
            let tau = bias_segment_and_tau(self.origin, left).1
                + (right.as_nanos() - left.as_nanos()) as f64 * 0.5 / BIAS_KNOT_SPACING_NS as f64;
            integrated
                .integrate(
                    sample.gyro,
                    force,
                    right.duration_since(left).as_secs_f64(),
                    tau,
                    noise,
                )
                .wrap_err_with(|| {
                    format!("IMU propagation in interval {index}, {left:?}..{right:?}")
                })?;
            *last = right;
        }
        // Prepare all numerical work before replacing graph factors.
        let segment = index * INTERVAL_NS / KNOT_SPACING_NS;
        let controls = self.ensure_segment(segment)?;
        let position = self
            .parameters
            .accelerometer
            .as_ref()
            .map_or(Vector3::zeros(), |p| p.position);
        let segment_start = self.origin.as_nanos() + segment * KNOT_SPACING_NS;
        let mut factors = Vec::new();
        let mut covered = 0.0;
        let mut last = None;
        for (a, b, integrated) in &pieces {
            covered += integrated.delta.duration;
            last = Some(*b);
            factors.push(PreintegratedImu {
                controls,
                biases: bias_keys,
                duration: seconds_per_knot(),
                start_tau: (a.as_nanos() - segment_start) as f64 / KNOT_SPACING_NS as f64,
                end_tau: (b.as_nanos() - segment_start) as f64 / KNOT_SPACING_NS as f64,
                information: integrated.information().wrap_err_with(|| {
                    format!("IMU covariance whitening in interval {index}, {a:?}..{b:?}")
                })?,
                delta: integrated.delta.clone(),
                gravity_compensation: Vector3::wrap(nalgebra::Vector3::new(0.0, 0.0, 9.81)),
                position,
            });
        }
        if let Some(old) = self.preintegration.intervals.remove(&index) {
            for key in old.factors {
                self.graph.remove_factor(key)?;
            }
            for boundary in old.boundaries {
                self.graph.remove_factor(boundary)?;
            }
        }
        let factors = factors
            .into_iter()
            .map(|factor| self.graph.add_factor(factor))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut boundaries = Vec::new();
        if let Some(time) = last {
            boundaries.push(self.add_imu_boundary(
                time,
                Vector3::zeros(),
                0.0,
                (covered / 0.1).sqrt() / self.parameters.imu_preintegration.tilt_sigma,
                true,
            )?);
        }
        for (time, gyro) in gap_boundaries {
            // No held span consumes this gyro reading. Keep its instantaneous
            // evidence instead of fabricating motion across the missing data.
            boundaries.push(
                self.add_imu_boundary(
                    time,
                    gyro,
                    self.parameters
                        .imu_preintegration
                        .terminal_gyroscope_sigma
                        .recip(),
                    if last == Some(time) { 0.0 } else { 10.0 },
                    false,
                )?,
            );
        }
        self.preintegration.intervals.insert(
            index,
            Interval {
                factors,
                boundaries,
                reference_biases,
                end: last,
                pieces,
                reusable: true,
            },
        );
        Ok(())
    }

    fn add_imu_boundary(
        &mut self,
        time: Time,
        gyro: Vector3<Robot, f64>,
        gyro_root: f64,
        tilt_root: f64,
        right_endpoint: bool,
    ) -> Result<FactorKey<ImuKinematics>> {
        // Put exact right-boundary observations on the interval's own controls
        // so their information is marginalized, rather than dropped, with it.
        let query = if right_endpoint && time > self.origin {
            Time::from_nanos(time.as_nanos() - 1)
        } else {
            time
        };
        let (segment, _) = self.segment_and_tau(query)?;
        let tau = (time.as_nanos() - self.origin.as_nanos() - segment * KNOT_SPACING_NS) as f64
            / KNOT_SPACING_NS as f64;
        let controls = self.ensure_segment(segment)?;
        let (biases, bias_tau) = self.ensure_biases(query)?;
        let bias_tau =
            bias_tau + (time.as_nanos() - query.as_nanos()) as f64 / BIAS_KNOT_SPACING_NS as f64;
        let measured_up = self
            .attitude_at(time)
            .filter(|_| tilt_root != 0.0)
            .map(|a| a.rotation::<Robot>().inverse() * Vector3::<ImuReference, f64>::z_axis());
        Ok(self.graph.add_factor(ImuKinematics {
            controls,
            biases,
            duration: seconds_per_knot(),
            gyroscope_information_root: Matrix3::identity() * gyro_root,
            tilt_information_root: Matrix3::identity() * tilt_root,
            tau,
            bias_tau,
            angular_velocity: gyro,
            measured_up,
        })?)
    }

    pub(super) fn retire_preintegration(&mut self, oldest: i64) {
        let first = oldest * KNOT_SPACING_NS / INTERVAL_NS;
        // Endpoint and boundary factors were removed by marginalization.
        self.preintegration
            .intervals
            .retain(|&index, _| index >= first);
        if self
            .preintegration
            .terminal
            .as_ref()
            .is_some_and(|(segment, _)| *segment < oldest)
        {
            self.preintegration.terminal = None;
        }
        let start = Time::from_nanos(self.origin.as_nanos() + oldest * KNOT_SPACING_NS);
        let before = self
            .preintegration
            .samples
            .range(..=start)
            .next_back()
            .map_or(start, |(&t, _)| t);
        self.preintegration
            .samples
            .retain(|&time, _| time >= before);
        self.preintegration.dirty.retain(|&index| index >= first);
    }
}

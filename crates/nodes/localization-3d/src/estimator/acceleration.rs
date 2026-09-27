use coordinate_systems::{ImuReference, Robot};
use linear_algebra::{Orientation3, Vector3};
use localization_fagra::factors::{ForceBias, LeverArmSample};
use nalgebra::Matrix3;
use ros_z::time::Time;
use types::localization::MAX_IMU_ATTITUDE_GAP;

use crate::parameters::AccelerometerParameters;

#[derive(Clone, Default)]
pub(super) struct AccelerationIntegrator {
    previous: Option<Sample>,
    start: Option<Time>,
    integral: Vector3<ImuReference, f64>,
    bias_weights: [Matrix3<f64>; 2],
    lever_samples: Vec<LeverArmSample>,
}

#[derive(Clone)]
struct Sample {
    time: Time,
    gyro: Vector3<Robot, f64>,
    force: Option<Vector3<ImuReference, f64>>,
    rotation: Matrix3<f64>,
}

pub(super) struct AveragedAcceleration {
    pub time: Time,
    pub force: Vector3<ImuReference, f64>,
    pub information_root: f64,
    pub bias: ForceBias,
}

impl AccelerationIntegrator {
    pub(super) fn observe(
        &mut self,
        time: Time,
        attitude: Orientation3<ImuReference, f64>,
        gyro: Vector3<Robot, f64>,
        raw: Vector3<Robot, f64>,
        parameters: &AccelerometerParameters,
        origin: Time,
    ) -> Vec<AveragedAcceleration> {
        // Late samples still enter gyro/attitude fitting, but must not integrate time twice.
        if self
            .previous
            .as_ref()
            .is_some_and(|previous| time <= previous.time)
        {
            return Vec::new();
        }
        if !raw.inner.iter().all(|v| v.is_finite()) {
            *self = Self::default();
            return Vec::new();
        }
        if self
            .previous
            .as_ref()
            .is_some_and(|previous| time.duration_since(previous.time) > MAX_IMU_ATTITUDE_GAP)
        {
            *self = Self::default();
        }
        let mut force = (raw.inner - parameters.bias.inner).component_mul(&parameters.scale);
        let offset = parameters.position.inner;
        let angular_acceleration = self.previous.as_ref().map(|previous| {
            (gyro.inner - previous.gyro.inner) / time.duration_since(previous.time).as_secs_f64()
        });
        let force = if offset.norm_squared() == 0.0 || angular_acceleration.is_some() {
            if let Some(alpha) = angular_acceleration {
                force -= alpha.cross(&offset) + gyro.inner.cross(&gyro.inner.cross(&offset));
            }
            force
                .iter()
                .all(|v| v.is_finite())
                .then(|| attitude.rotation::<Robot>() * Vector3::wrap(force))
        } else {
            None
        };
        let sample = Sample {
            time,
            gyro,
            force,
            rotation: attitude.inner.to_rotation_matrix().into_inner(),
        };
        let current = sample.clone();
        let previous = self.previous.replace(sample);
        let Some((previous, a, b)) =
            previous.and_then(|previous| previous.force.zip(force).map(|(a, b)| (previous, a, b)))
        else {
            self.start = Some(time);
            self.integral = Vector3::zeros();
            self.bias_weights = [Matrix3::zeros(); 2];
            self.lever_samples.clear();
            return Vec::new();
        };
        let mut result = Vec::new();
        let mut left = previous.time;
        self.start.get_or_insert(left);
        while left < time {
            let (segment, _) = super::bias::bias_segment_and_tau(origin, left);
            let boundary = Time::from_nanos(
                origin.as_nanos() + (segment + 1) * super::bias::BIAS_KNOT_SPACING_NS,
            );
            let right = time.min(boundary);
            let weight = right.duration_since(left).as_secs_f64() * 0.5;
            for stamp in [left, right] {
                let fraction = stamp.duration_since(previous.time).as_secs_f64()
                    / time.duration_since(previous.time).as_secs_f64();
                let rotation = previous.rotation * (1.0 - fraction) + current.rotation * fraction;
                let tau = (stamp.as_nanos()
                    - origin.as_nanos()
                    - segment * super::bias::BIAS_KNOT_SPACING_NS) as f64
                    / super::bias::BIAS_KNOT_SPACING_NS as f64;
                self.integral += (a * (1.0 - fraction) + b * fraction) * weight;
                self.bias_weights[0] += rotation * (weight * (1.0 - tau));
                self.bias_weights[1] += rotation * (weight * tau);
                if parameters.position.norm_squared() > 0.0 {
                    self.lever_samples.push(LeverArmSample {
                        weight: rotation * weight,
                        tau,
                        angular_velocity: previous.gyro * (1.0 - fraction) + gyro * fraction,
                    });
                }
            }
            let start = self.start.unwrap();
            let elapsed = right.duration_since(start);
            if right == boundary || elapsed >= parameters.averaging_interval {
                let seconds = elapsed.as_secs_f64();
                let mut samples = std::mem::take(&mut self.lever_samples);
                for sample in &mut samples {
                    sample.weight /= seconds;
                }
                let mean = AveragedAcceleration {
                    time: start + elapsed / 2,
                    force: self.integral / seconds,
                    information_root: seconds.sqrt() / parameters.noise_density,
                    bias: ForceBias {
                        accelerometer_weights: self.bias_weights.map(|w| w / seconds),
                        position: parameters.position,
                        duration: super::bias::BIAS_KNOT_SPACING_NS as f64 * 1e-9,
                        lever_samples: samples,
                    },
                };
                self.start = Some(right);
                self.integral = Vector3::zeros();
                self.bias_weights = [Matrix3::zeros(); 2];
                if (mean.force.inner * mean.information_root)
                    .norm_squared()
                    .is_finite()
                {
                    result.push(mean);
                }
            }
            left = right;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibrated_lever_arm_does_not_create_body_acceleration() {
        let parameters = AccelerometerParameters {
            bias: Vector3::wrap(nalgebra::vector![0.2, -0.1, 0.05]),
            scale: nalgebra::vector![1.02, 0.98, 1.01],
            position: Vector3::wrap(nalgebra::vector![0.12, -0.03, 0.08]),
            ..Default::default()
        };
        let mut integrator = AccelerationIntegrator::default();
        let mut averages = 0;
        for index in 0..=30 {
            let t = index as f64 * 0.002;
            let gyro = nalgebra::vector![0.0, 0.0, 0.5 + 2.0 * t];
            let alpha = nalgebra::vector![0.0, 0.0, 2.0];
            let offset = parameters.position.inner;
            let sensor_force = nalgebra::vector![0.0, 0.0, 9.81]
                + alpha.cross(&offset)
                + gyro.cross(&gyro.cross(&offset));
            let raw = sensor_force.component_div(&parameters.scale) + parameters.bias.inner;
            for mean in integrator.observe(
                Time::from_nanos(index * 2_000_000),
                Orientation3::from_euler_angles(0.0, 0.0, 0.5 * t + t * t),
                Vector3::wrap(gyro),
                Vector3::wrap(raw),
                &parameters,
                Time::from_nanos(0),
            ) {
                assert!((mean.force.inner - nalgebra::vector![0.0, 0.0, 9.81]).norm() < 1e-10);
                averages += 1;
            }
        }
        assert!(averages >= 4);
    }

    #[test]
    fn interpolation_survives_averaging_and_a_bias_boundary_with_lever_arm() {
        use localization_fagra::variables::ImuBias;
        let parameters = AccelerometerParameters {
            position: Vector3::wrap(nalgebra::vector![0.12, -0.03, 0.08]),
            ..Default::default()
        };
        let bias_at = |t: f64| ImuBias {
            gyroscope: Vector3::wrap(nalgebra::vector![
                0.01 + 0.002 * t,
                -0.02 + 0.001 * t,
                0.03 - 0.001 * t
            ]),
            accelerometer: Vector3::wrap(nalgebra::vector![
                0.03 + 0.001 * t,
                -0.04 + 0.002 * t,
                0.01
            ]),
        };
        let omega = nalgebra::vector![0.0, 0.0, 0.4];
        let mut integrator = AccelerationIntegrator::default();
        let mut before = false;
        let mut after = false;
        for i in 0..30 {
            let stamp = 4_979_000_000 + i * 3_000_000;
            let t = stamp as f64 * 1e-9;
            let bias = bias_at(t);
            let raw = nalgebra::vector![0.0, 0.0, 9.81]
                + omega.cross(&omega.cross(&parameters.position.inner))
                + bias.accelerometer.inner;
            for mean in integrator.observe(
                Time::from_nanos(stamp),
                Orientation3::from_euler_angles(0.0, 0.0, 0.4 * t),
                Vector3::wrap(omega + bias.gyroscope.inner),
                Vector3::wrap(raw),
                &parameters,
                Time::from_nanos(0),
            ) {
                let start =
                    (mean.time.as_nanos() / super::super::bias::BIAS_KNOT_SPACING_NS) as f64 * 5.0;
                let biases = [bias_at(start), bias_at(start + 5.0)];
                let (correction, _) = mean.bias.evaluate([&biases[0], &biases[1]]).unwrap();
                assert!(
                    (mean.force.inner - correction - nalgebra::vector![0.0, 0.0, 9.81]).norm()
                        < 1e-6
                );
                before |= mean.time.as_nanos() < 5_000_000_000;
                after |= mean.time.as_nanos() > 5_000_000_000;
            }
        }
        assert!(before && after);
    }

    #[test]
    fn averaging_preserves_impulses_and_zero_specific_force() {
        for impulse in [false, true] {
            let parameters = AccelerometerParameters::default();
            let mut integrator = AccelerationIntegrator::default();
            let mut mean = None;
            for index in 0..=5 {
                let force = if impulse {
                    9.81 + if index == 2 { 100.0 } else { 0.0 }
                } else {
                    0.0
                };
                mean = integrator
                    .observe(
                        Time::from_nanos(index * 2_000_000),
                        Orientation3::default(),
                        Vector3::zeros(),
                        Vector3::wrap(nalgebra::vector![0.0, 0.0, force]),
                        &parameters,
                        Time::from_nanos(0),
                    )
                    .into_iter()
                    .next()
                    .or(mean);
            }
            let mean = mean.unwrap();
            assert_eq!(mean.time, Time::from_nanos(5_000_000));
            assert!((mean.force.z() - if impulse { 29.81 } else { 0.0 }).abs() < 1e-10);
            assert!(
                integrator
                    .observe(
                        Time::from_nanos(8_000_000),
                        Orientation3::default(),
                        Vector3::zeros(),
                        Vector3::zeros(),
                        &parameters,
                        Time::from_nanos(0)
                    )
                    .is_empty()
            );
            assert!(
                integrator
                    .observe(
                        Time::from_nanos(1_000_000_000),
                        Orientation3::default(),
                        Vector3::zeros(),
                        Vector3::zeros(),
                        &parameters,
                        Time::from_nanos(0)
                    )
                    .is_empty()
            );
        }
    }
}

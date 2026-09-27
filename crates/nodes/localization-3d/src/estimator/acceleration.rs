use coordinate_systems::{ImuReference, Robot};
use linear_algebra::{Orientation3, Vector3};
use ros_z::time::Time;
use types::localization::MAX_IMU_ATTITUDE_GAP;

use crate::parameters::AccelerometerParameters;

#[derive(Clone, Default)]
pub(super) struct AccelerationIntegrator {
    previous: Option<Sample>,
    start: Option<Time>,
    integral: Vector3<ImuReference, f64>,
}

#[derive(Clone)]
struct Sample {
    time: Time,
    gyro: Vector3<Robot, f64>,
    force: Option<Vector3<ImuReference, f64>>,
}

pub(super) struct AveragedAcceleration {
    pub time: Time,
    pub force: Vector3<ImuReference, f64>,
    pub information_root: f64,
}

impl AccelerationIntegrator {
    pub(super) fn observe(
        &mut self,
        time: Time,
        attitude: Orientation3<ImuReference, f64>,
        gyro: Vector3<Robot, f64>,
        raw: Vector3<Robot, f64>,
        parameters: &AccelerometerParameters,
    ) -> Option<AveragedAcceleration> {
        // Late samples still enter gyro/attitude fitting, but must not integrate time twice.
        if self
            .previous
            .as_ref()
            .is_some_and(|previous| time <= previous.time)
        {
            return None;
        }
        if !raw.inner.iter().all(|v| v.is_finite()) {
            *self = Self::default();
            return None;
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
        let sample = Sample { time, gyro, force };
        let previous = self.previous.replace(sample);
        let Some((previous, a, b)) =
            previous.and_then(|previous| previous.force.zip(force).map(|(a, b)| (previous, a, b)))
        else {
            self.start = Some(time);
            self.integral = Vector3::zeros();
            return None;
        };
        let dt = time.duration_since(previous.time).as_secs_f64();
        self.integral += (a + b) * (0.5 * dt);
        let start = self.start.get_or_insert(previous.time);
        let elapsed = time.duration_since(*start);
        if elapsed < parameters.averaging_interval {
            return None;
        }
        let mean = AveragedAcceleration {
            time: *start + elapsed / 2,
            force: self.integral / elapsed.as_secs_f64(),
            information_root: elapsed.as_secs_f64().sqrt() / parameters.noise_density,
        };
        self.start = Some(time);
        self.integral = Vector3::zeros();
        (mean.force.inner * mean.information_root)
            .norm_squared()
            .is_finite()
            .then_some(mean)
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
            if let Some(mean) = integrator.observe(
                Time::from_nanos(index * 2_000_000),
                Orientation3::from_euler_angles(0.0, 0.0, 0.5 * t + t * t),
                Vector3::wrap(gyro),
                Vector3::wrap(raw),
                &parameters,
            ) {
                assert!((mean.force.inner - nalgebra::vector![0.0, 0.0, 9.81]).norm() < 1e-10);
                averages += 1;
            }
        }
        assert!(averages >= 4);
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
                    )
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
                        &parameters
                    )
                    .is_none()
            );
            assert!(
                integrator
                    .observe(
                        Time::from_nanos(1_000_000_000),
                        Orientation3::default(),
                        Vector3::zeros(),
                        Vector3::zeros(),
                        &parameters
                    )
                    .is_none()
            );
        }
    }
}

use std::time::Duration;

use ros_z::Message;
use serde::{Deserialize, Serialize};

/// Restart-required timing and interpolation policy. Durations are seconds/nanoseconds.
#[derive(Clone, Debug, Deserialize, Serialize, Message, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct TimingParameters {
    pub trajectory_spacing: Duration,
    pub bias_spacing: Duration,
    pub optimization_window: Duration,
}

impl Default for TimingParameters {
    fn default() -> Self {
        Self {
            trajectory_spacing: Duration::from_millis(200),
            bias_spacing: Duration::from_secs(5),
            optimization_window: Duration::from_secs(2),
        }
    }
}

impl TimingParameters {
    pub(crate) fn knot_ns(&self) -> i64 {
        self.trajectory_spacing.as_nanos() as i64
    }
    pub(crate) fn bias_ns(&self) -> i64 {
        self.bias_spacing.as_nanos() as i64
    }

    pub fn window_ns(&self) -> i64 {
        self.optimization_window.as_nanos() as i64
    }
}

/// Restart-required factor weights and numerical sampling policy.
#[derive(Clone, Debug, Deserialize, Serialize, Message, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ModelParameters {
    /// Intrinsic prior standard deviation, pixels.
    pub intrinsic_prior_sigma: f64,
    /// Gauge standard deviations, metres and radians respectively.
    pub anchor_xy_sigma: f64,
    pub anchor_yaw_sigma: f64,
    /// Rotational process spectral density, rad²/s.
    pub rotation_process_variance: f64,
    /// Variance multiplier for motion priors spanning missing data.
    pub gap_uncertainty_multiplier: f64,
    pub prediction_gap_segments: i64,
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self {
            intrinsic_prior_sigma: 0.001,
            anchor_xy_sigma: 0.001,
            anchor_yaw_sigma: 0.001,
            rotation_process_variance: 0.01,
            gap_uncertainty_multiplier: 10.0,
            prediction_gap_segments: 5,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Message, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ImuBiasParameters {
    pub accelerometer_initial_sigma: f64,
    pub gyroscope_initial_sigma: f64,
    /// Bias random-walk standard deviation per sqrt(second).
    pub accelerometer_random_walk: f64,
    pub gyroscope_random_walk: f64,
}

impl Default for ImuBiasParameters {
    fn default() -> Self {
        Self {
            accelerometer_initial_sigma: 1.0,
            gyroscope_initial_sigma: 0.02,
            accelerometer_random_walk: 0.002,
            gyroscope_random_walk: 0.0002,
        }
    }
}

/// Runtime parameters for the 3D localization node.
#[derive(Clone, Debug, Deserialize, Serialize, Message, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Localization3dParameters {
    #[serde(default)]
    pub timing: TimingParameters,
    #[serde(default)]
    pub model: ModelParameters,
    #[serde(default)]
    pub imu_bias: ImuBiasParameters,
    /// Broad initialization distributions, not measured standing height or zero velocity.
    pub initial_height_sigma: f64,
    pub initial_velocity_sigma: f64,
    /// Translational white-noise-on-acceleration spectral density.
    pub accelerometer_process_noise_variance: f64,
}

impl Localization3dParameters {
    pub fn validate(&self) -> std::result::Result<(), String> {
        let t = &self.timing;
        for duration in [t.trajectory_spacing, t.bias_spacing, t.optimization_window] {
            if duration.is_zero() || duration.as_nanos() > i64::MAX as u128 {
                return Err(
                    "timing durations must be positive and representable in signed nanoseconds"
                        .into(),
                );
            }
        }
        if t.bias_ns() % t.knot_ns() != 0
            || t.optimization_window < t.trajectory_spacing.saturating_mul(2)
        {
            return Err("bias spacing must be a multiple of trajectory spacing, trajectory spacing a multiple of the IMU interval; window must cover at least two trajectory segments".into());
        }
        let m = &self.model;

        for value in [
            m.intrinsic_prior_sigma,
            m.anchor_xy_sigma,
            m.anchor_yaw_sigma,
            m.rotation_process_variance,
            m.gap_uncertainty_multiplier,
        ] {
            if !valid_scale(value) || !(value * value).is_finite() {
                return Err("model, visual and solver scales must be finite and positive with representable squares".into());
            }
        }

        if m.prediction_gap_segments <= 0 || m.gap_uncertainty_multiplier < 1.0 {
            return Err(
                "invalid model sampling, visual association limits or solver iteration limits"
                    .into(),
            );
        }

        for value in [
            self.imu_bias.accelerometer_initial_sigma,
            self.imu_bias.gyroscope_initial_sigma,
            self.imu_bias.accelerometer_random_walk,
            self.imu_bias.gyroscope_random_walk,
        ] {
            if !valid_scale(value) || !(value * value).is_finite() {
                return Err("IMU bias uncertainties must be finite and positive".into());
            }
        }

        for value in [self.initial_height_sigma, self.initial_velocity_sigma] {
            if !valid_scale(value) || !(value * value).is_finite() {
                return Err(
                    "initial uncertainties and recovery height gate must be finite and positive"
                        .into(),
                );
            }
        }

        if !valid_scale(self.accelerometer_process_noise_variance) {
            return Err("accelerometer_process_noise_variance must be finite and > 0".to_string());
        }

        Ok(())
    }
}

fn valid_scale(value: f64) -> bool {
    value.is_finite() && value >= f64::MIN_POSITIVE.sqrt()
}

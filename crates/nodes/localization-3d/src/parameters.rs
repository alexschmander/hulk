use std::time::Duration;

use coordinate_systems::Robot;
use linear_algebra::Vector3;
use ros_z::Message;
use serde::{Deserialize, Serialize};

const MAX_TRACKING_TIMEOUT: Duration = Duration::from_secs(24 * 60 * 60);

#[derive(Clone, Debug, Deserialize, Serialize, Message, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ImuPreintegrationParameters {
    /// Continuous gyro noise density, rad/s sqrt(s).
    pub gyroscope_noise_density: f64,
    /// Position integration/model-error density, m/sqrt(s), as in GTSAM.
    pub integration_noise_density: f64,
    /// SDK up-direction sigma for a complete 100 ms interval.
    pub tilt_sigma: f64,
    /// Instantaneous, not integrated, newest gyro measurement sigma (rad/s).
    pub terminal_gyroscope_sigma: f64,
    /// Reintegrate between solves beyond these reference-bias changes.
    pub gyroscope_reintegration_threshold: f64,
    pub accelerometer_reintegration_threshold: f64,
}

impl Default for ImuPreintegrationParameters {
    fn default() -> Self {
        Self {
            gyroscope_noise_density: (2e-5_f64).sqrt(),
            integration_noise_density: 1e-4,
            tilt_sigma: 0.02,
            terminal_gyroscope_sigma: 0.1,
            gyroscope_reintegration_threshold: 0.01,
            accelerometer_reintegration_threshold: 0.1,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Message)]
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

#[derive(Clone, Debug, Deserialize, Serialize, Message, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AccelerometerParameters {
    /// SDK specific force is already expressed in Robot axes, in m/s².
    pub bias: Vector3<Robot, f64>,
    /// Per-axis multiplicative calibration, applied after subtracting bias.
    pub scale: nalgebra::Vector3<f64>,
    /// Robot origin to IMU, expressed in Robot, in metres.
    pub position: Vector3<Robot, f64>,
    /// Specific-force white-noise density in m/s² sqrt(s), including model error.
    pub noise_density: f64,
}

impl Default for AccelerometerParameters {
    fn default() -> Self {
        Self {
            bias: Vector3::zeros(),
            scale: nalgebra::Vector3::repeat(1.0),
            position: Vector3::zeros(),
            noise_density: 0.3,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Message)]
#[serde(deny_unknown_fields)]
pub struct KinematicOdometryNoise {
    /// Forward/lateral displacement noise floor, in metres.
    pub position_sigma: nalgebra::Vector2<f64>,
    /// Forward/lateral displacement variance growth, in m²/s.
    pub translation_variance_per_second: nalgebra::Vector2<f64>,
}

impl Default for KinematicOdometryNoise {
    fn default() -> Self {
        Self {
            position_sigma: nalgebra::Vector2::repeat(0.005),
            translation_variance_per_second: nalgebra::Vector2::repeat(0.001),
        }
    }
}

/// Runtime parameters for the 3D localization node.
#[derive(Clone, Debug, Deserialize, Serialize, Message)]
#[serde(deny_unknown_fields)]
pub struct Localization3dParameters {
    #[serde(default)]
    pub kinematic_odometry_noise: Option<KinematicOdometryNoise>,
    pub accelerometer: Option<AccelerometerParameters>,
    #[serde(default)]
    pub imu_bias: ImuBiasParameters,
    #[serde(default)]
    pub imu_preintegration: ImuPreintegrationParameters,
    /// Broad initialization distributions, not measured standing height or zero velocity.
    pub initial_height_sigma: f64,
    pub initial_velocity_sigma: f64,
    /// Squared normalized innovation allowed between old and candidate height predictions.
    pub recovery_height_gate: f64,
    /// Maximum discrepancy between estimated and measured up directions, in radians.
    pub max_tilt_error: f64,
    /// Translational white-noise-on-acceleration spectral density.
    pub accelerometer_process_noise_variance: f64,
    /// Pixel-noise variance used to set isotropic angular noise for accepted visual
    /// associations: sigma_theta = sqrt(variance) / sqrt(fx * fy), using the frame's
    /// fixed calibration. This is a near-axis approximation, not a pixel likelihood.
    pub visual_feature_noise_variance: f64,
    /// Soft field containment sigma in meters outside field plus border strip.
    pub field_containment_sigma: f64,
    /// Maximum field-heading innovation against propagated IMU heading, in radians (< pi/2).
    pub max_heading_error: f64,
    /// Maximum correction of the IMU-to-field reference during visual tracking, in rad/s.
    pub max_heading_reference_drift_per_second: f64,
    /// Time without a converged aligned backend result before localization is declared lost.
    #[serde(default = "default_tracking_timeout")]
    pub tracking_timeout: Duration,
    /// Time without a newly acknowledged, valid visual reprojection before tracking is lost.
    #[serde(default = "default_tracking_timeout")]
    pub visual_tracking_timeout: Duration,
}

fn default_tracking_timeout() -> Duration {
    Duration::from_secs(2)
}

impl Localization3dParameters {
    pub(crate) fn validate(&self) -> std::result::Result<(), String> {
        let p = &self.imu_preintegration;
        for value in [
            p.gyroscope_noise_density,
            p.integration_noise_density,
            p.tilt_sigma,
            p.terminal_gyroscope_sigma,
            p.gyroscope_reintegration_threshold,
            p.accelerometer_reintegration_threshold,
        ] {
            if !valid_scale(value) || !(value * value).is_finite() {
                return Err(
                    "IMU preintegration noise and thresholds must be finite and positive".into(),
                );
            }
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
        if !self.max_heading_error.is_finite()
            || self.max_heading_error <= 0.0
            || self.max_heading_error >= std::f64::consts::FRAC_PI_2
            || !self.max_heading_reference_drift_per_second.is_finite()
            || self.max_heading_reference_drift_per_second < 0.0
        {
            return Err(
                "heading error must be in (0, pi/2), and heading drift finite and >= 0".into(),
            );
        }
        if let Some(noise) = &self.kinematic_odometry_noise
            && !noise
                .position_sigma
                .iter()
                .chain(noise.translation_variance_per_second.iter())
                .all(|v| valid_scale(*v) && (*v * *v).is_finite())
        {
            return Err("kinematic odometry noise must be finite and > 0".into());
        }
        for value in [
            self.initial_height_sigma,
            self.initial_velocity_sigma,
            self.recovery_height_gate,
        ] {
            if !valid_scale(value) || !(value * value).is_finite() {
                return Err(
                    "initial uncertainties and recovery height gate must be finite and positive"
                        .into(),
                );
            }
        }
        if !self.max_tilt_error.is_finite()
            || self.max_tilt_error <= 0.0
            || self.max_tilt_error >= std::f64::consts::FRAC_PI_2
        {
            return Err("tilt error must be in (0, pi/2)".into());
        }
        if let Some(accel) = &self.accelerometer
            && (!accel
                .bias
                .inner
                .iter()
                .chain(accel.position.inner.iter())
                .all(|v| v.is_finite())
                || !accel
                    .scale
                    .iter()
                    .all(|v| valid_scale(*v) && (v * v).is_finite())
                || !valid_scale(accel.noise_density)
                || !(accel.noise_density * accel.noise_density).is_finite())
        {
            return Err("invalid accelerometer calibration or noise density".into());
        }
        if !valid_scale(self.accelerometer_process_noise_variance) {
            return Err("accelerometer_process_noise_variance must be finite and > 0".to_string());
        }
        if !valid_scale(self.visual_feature_noise_variance) {
            return Err("visual_feature_noise_variance must be finite and > 0".to_string());
        }
        if !valid_scale(self.field_containment_sigma) {
            return Err("field_containment_sigma must be finite and > 0".to_string());
        }
        for (name, timeout) in [
            ("tracking_timeout", self.tracking_timeout),
            ("visual_tracking_timeout", self.visual_tracking_timeout),
        ] {
            if timeout.is_zero() || timeout > MAX_TRACKING_TIMEOUT {
                return Err(format!("{name} must be > 0 and <= 24 hours"));
            }
        }
        Ok(())
    }
}

fn valid_scale(value: f64) -> bool {
    value.is_finite() && value >= f64::MIN_POSITIVE.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_rejects_unusable_whitening_and_timer_ranges() {
        let parameters = Localization3dParameters {
            imu_preintegration: Default::default(),
            imu_bias: Default::default(),
            kinematic_odometry_noise: Some(Default::default()),
            accelerometer: None,
            initial_height_sigma: 1.0,
            initial_velocity_sigma: 5.0,
            recovery_height_gate: 9.0,
            max_tilt_error: 20.0_f64.to_radians(),
            accelerometer_process_noise_variance: 10.0,
            visual_feature_noise_variance: 10000.0,
            field_containment_sigma: 1.0,
            max_heading_error: 20.0_f64.to_radians(),
            max_heading_reference_drift_per_second: 0.5_f64.to_radians(),
            tracking_timeout: Duration::from_secs(2),
            visual_tracking_timeout: Duration::from_secs(2),
        };
        assert!(parameters.validate().is_ok());
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut invalid = parameters.clone();
            invalid.imu_bias.accelerometer_initial_sigma = value;
            assert!(invalid.validate().is_err());
            invalid = parameters.clone();
            invalid.imu_bias.gyroscope_random_walk = value;
            assert!(invalid.validate().is_err());
        }
        let mut calibrated = parameters.clone();
        calibrated.accelerometer = Some(AccelerometerParameters::default());
        assert!(calibrated.validate().is_ok());
        calibrated.accelerometer.as_mut().unwrap().scale.x = 0.0;
        assert!(calibrated.validate().is_err());
        calibrated.accelerometer = Some(AccelerometerParameters {
            noise_density: f64::NAN,
            ..Default::default()
        });
        assert!(calibrated.validate().is_err());
        for error in [0.0, f64::NAN, std::f64::consts::FRAC_PI_2] {
            let mut invalid = parameters.clone();
            invalid.max_heading_error = error;
            assert!(invalid.validate().is_err());
        }
        for drift in [-1.0, f64::INFINITY, f64::NAN] {
            let mut invalid = parameters.clone();
            invalid.max_heading_reference_drift_per_second = drift;
            assert!(invalid.validate().is_err());
        }
        let mut invalid = parameters.clone();
        invalid.accelerometer_process_noise_variance = 1.0e-250;
        assert!(invalid.validate().is_err());
        invalid = parameters.clone();
        invalid.visual_feature_noise_variance = f64::INFINITY;
        assert!(invalid.validate().is_err());
        invalid = parameters.clone();
        invalid.field_containment_sigma = 1.0e-250;
        assert!(invalid.validate().is_err());
        for timeout in [Duration::ZERO, Duration::MAX] {
            invalid = parameters.clone();
            invalid.tracking_timeout = timeout;
            assert!(invalid.validate().is_err());
            invalid = parameters.clone();
            invalid.visual_tracking_timeout = timeout;
            assert!(invalid.validate().is_err());
        }
    }
}

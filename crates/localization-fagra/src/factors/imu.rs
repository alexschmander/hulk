use coordinate_systems::{Local, Robot};
use fagra::{
    BlockId, EvaluationError, Factor, FactorBatch, FactorSelection, JacobianBlock,
    LinearizationSink, StateKey, StateStore,
};
use linear_algebra::Vector3;
use nalgebra::{Matrix3, RealField, SMatrix, SVector, UnitQuaternion};

use super::common;
use crate::variables::{ImuBias, PoseControl};

/// One trapezoidal quadrature term for gyro-dependent lever-arm correction.
#[derive(Clone, Debug)]
pub struct LeverArmSample<R: RealField + Copy = f64> {
    /// Weighted rotation from sample Robot axes to the averaged measurement axes.
    pub weight: Matrix3<R>,
    pub tau: R,
    pub angular_velocity: Vector3<Robot, R>,
}

/// Bias dependence retained while averaging calibrated specific force.
#[derive(Clone, Debug)]
pub struct ForceBias<R: RealField + Copy = f64> {
    pub accelerometer_weights: [Matrix3<R>; 2],
    pub position: Vector3<Robot, R>,
    /// Separation of the coarse bias knots in seconds, not the pose-knot spacing.
    pub duration: R,
    /// Empty for a sensor at the Robot origin.
    pub lever_samples: Vec<LeverArmSample<R>>,
}

impl<R: RealField + Copy> ForceBias<R> {
    pub fn at_time(tau: R) -> Self {
        Self {
            accelerometer_weights: [
                Matrix3::identity() * (R::one() - tau),
                Matrix3::identity() * tau,
            ],
            position: Vector3::zeros(),
            duration: R::one(),
            lever_samples: Vec::new(),
        }
    }

    /// Added to predicted force: averaged accelerometer bias and the difference
    /// between true and raw-gyro lever-arm acceleration. Includes bias time slope.
    pub fn evaluate(
        &self,
        biases: [&ImuBias<R>; 2],
    ) -> Result<(nalgebra::Vector3<R>, [SMatrix<R, 3, 6>; 2]), EvaluationError> {
        common::finite(self.position.inner.iter())?;
        let inverse_duration = common::positive(self.duration)?;
        let mut correction = nalgebra::Vector3::zeros();
        let mut jacobians = [SMatrix::<R, 3, 6>::zeros(); 2];
        for i in 0..2 {
            common::finite(self.accelerometer_weights[i].iter())?;
            correction += self.accelerometer_weights[i] * biases[i].accelerometer.inner;
            jacobians[i]
                .fixed_columns_mut::<3>(3)
                .copy_from(&self.accelerometer_weights[i]);
        }
        let r = self.position.inner;
        let slope = (biases[1].gyroscope.inner - biases[0].gyroscope.inner) * inverse_duration;
        let slope_jacobian = r.cross_matrix() * inverse_duration;
        for sample in &self.lever_samples {
            validate_bias_tau(sample.tau)?;
            common::finite(
                sample
                    .weight
                    .iter()
                    .chain(sample.angular_velocity.inner.iter()),
            )?;
            let weights = [R::one() - sample.tau, sample.tau];
            let raw = sample.angular_velocity.inner;
            let omega = raw
                - biases[0].gyroscope.inner * weights[0]
                - biases[1].gyroscope.inner * weights[1];
            correction += sample.weight
                * (omega.cross(&omega.cross(&r)) - raw.cross(&raw.cross(&r)) + r.cross(&slope));
            let derivative = omega * r.transpose() + Matrix3::identity() * omega.dot(&r)
                - r * omega.transpose() * (R::one() + R::one());
            for i in 0..2 {
                let slope_sign = if i == 0 { -R::one() } else { R::one() };
                let block =
                    sample.weight * (-derivative * weights[i] + slope_jacobian * slope_sign);
                let old = jacobians[i].fixed_columns::<3>(0).into_owned();
                jacobians[i]
                    .fixed_columns_mut::<3>(0)
                    .copy_from(&(old + block));
            }
        }
        common::finite(correction.iter())?;
        Ok((correction, jacobians))
    }
}

fn validate_bias_tau<R: RealField + Copy>(tau: R) -> Result<(), EvaluationError> {
    if tau.is_finite() && tau >= R::zero() && tau <= R::one() {
        Ok(())
    } else {
        Err(EvaluationError::InvalidEvaluation)
    }
}

/// One IMU sample evaluated on an interval's trajectory.
/// Measurements are expressed in Robot axes. Fixed calibration and raw-gyro
/// lever-arm correction happen upstream; force_bias retains their bias dependence.
#[derive(Clone, Debug)]
pub struct ImuObservation<R: RealField + Copy = f64> {
    pub tau: R,
    /// Linear interpolation fraction on the independent, coarse bias interval.
    pub bias_tau: R,
    /// Angular velocity in rad/s.
    pub angular_velocity: Vector3<Robot, R>,
    /// SDK attitude's up direction, sharing the gyro's spline preparation.
    pub measured_up: Option<Vector3<Robot, R>>,
    /// Accelerometer specific force in m/s², not the robot's translational
    /// acceleration. Absent when accelerometer fitting is disabled.
    pub specific_force: Option<Vector3<Robot, R>>,
    /// Per-observation whitening, preserving each average's actual duration and
    /// insertion-time noise setting. Used only when specific force is present.
    pub accelerometer_information_root: Matrix3<R>,
    /// None for a point observation; Some retains averaging/lever-arm dependence.
    pub force_bias: Option<ForceBias<R>>,
}

/// Batched gyroscope, optional SDK tilt, and accelerometer residuals sharing spline preparation.
///
/// With robot-to-local rotation `R` and local trajectory position `p`, raw
/// residuals are body angular velocity plus interpolated gyro bias minus measured
/// angular velocity, and `Rᵀ * (p̈ + gravity_compensation) + bias_correction - specific_force`.
/// Bias uses two independent coarse knots, never the pose spline. Each sample contributes
/// gyroscope rows (unless their root is zero), optional tilt rows, and optional
/// accelerometer rows within one factor scope. Noise roots whiten robot-axis errors.
#[derive(Clone, Debug)]
pub struct ImuKinematics<R: RealField + Copy = f64> {
    pub controls: [StateKey<PoseControl<R>>; 4],
    pub biases: [StateKey<ImuBias<R>>; 2],
    pub duration: R,
    /// Local-frame acceleration added to trajectory acceleration before rotating
    /// into the robot frame; at rest this predicts the accelerometer reading.
    pub gravity_compensation: Vector3<Local, R>,
    pub gyroscope_information_root: Matrix3<R>,
    pub tilt_information_root: Matrix3<R>,
}

impl<R: RealField + Copy, S: StateStore<PoseControl<R>> + StateStore<ImuBias<R>>> FactorBatch<S>
    for ImuKinematics<R>
{
    type Scalar = R;
    type Factor = ImuObservation<R>;

    fn visit_variables(&self, _factor: &Self::Factor, mut visitor: impl FnMut(BlockId)) {
        for key in self.controls {
            visitor(key.block_id());
        }
        for key in self.biases {
            visitor(key.block_id());
        }
    }

    fn cost(
        &self,
        states: &S,
        factors: FactorSelection<'_, Self::Factor>,
    ) -> Result<R, EvaluationError> {
        if factors.is_empty() {
            return Ok(R::zero());
        }
        self.validate()?;
        let spline = common::spline(states, &self.controls, self.duration)?;
        let biases = [states.get(self.biases[0])?, states.get(self.biases[1])?];
        let observe_gyro = self
            .gyroscope_information_root
            .iter()
            .any(|v| *v != R::zero());
        let mut cost = R::zero();
        for (_, observation) in factors {
            validate_bias_tau(observation.bias_tau)?;
            let gyro_bias = biases[0].gyroscope.inner * (R::one() - observation.bias_tau)
                + biases[1].gyroscope.inner * observation.bias_tau;
            common::finite(gyro_bias.iter())?;
            common::finite(observation.angular_velocity.inner.iter())?;
            let (k, pose) =
                if observation.specific_force.is_some() || observation.measured_up.is_some() {
                    let (pose, k) = spline.pose_and_kinematics(observation.tau)?;
                    (k, Some(pose))
                } else {
                    (spline.kinematics(observation.tau)?, None)
                };
            if observe_gyro {
                cost += common::cost(
                    &(self.gyroscope_information_root
                        * (k.angular_velocity.inner + gyro_bias
                            - observation.angular_velocity.inner)),
                )?;
            }
            if let Some((measured, pose)) = observation.measured_up.zip(pose.as_ref()) {
                cost += common::cost(&tilt(
                    &pose.inner.rotation,
                    &measured,
                    &self.tilt_information_root,
                )?)?;
            }
            if let Some((measured, pose)) = observation.specific_force.zip(pose) {
                common::finite(measured.inner.iter())?;
                common::finite(observation.accelerometer_information_root.iter())?;
                let rotation = pose.inner.rotation;
                let prediction = rotation.inverse()
                    * (k.linear_acceleration.inner + self.gravity_compensation.inner);
                let point_bias = ForceBias::at_time(observation.bias_tau);
                let (correction, _) = observation
                    .force_bias
                    .as_ref()
                    .unwrap_or(&point_bias)
                    .evaluate(biases)?;
                cost += common::cost(
                    &(observation.accelerometer_information_root
                        * (prediction + correction - measured.inner)),
                )?;
            }
        }
        common::checked_cost(cost)
    }

    fn linearize<L: LinearizationSink<Scalar = R>>(
        &self,
        states: &S,
        factors: FactorSelection<'_, Self::Factor>,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        if factors.is_empty() {
            return Ok(());
        }
        self.validate()?;
        let spline = common::spline(states, &self.controls, self.duration)?;
        let linearized = spline.linearize()?;
        let biases = [states.get(self.biases[0])?, states.get(self.biases[1])?];
        let observe_gyro = self
            .gyroscope_information_root
            .iter()
            .any(|v| *v != R::zero());
        for (id, observation) in factors {
            validate_bias_tau(observation.bias_tau)?;
            let weights = [R::one() - observation.bias_tau, observation.bias_tau];
            let gyro_bias =
                biases[0].gyroscope.inner * weights[0] + biases[1].gyroscope.inner * weights[1];
            common::finite(gyro_bias.iter())?;
            common::finite(observation.angular_velocity.inner.iter())?;
            let (k, pose) =
                if observation.specific_force.is_some() || observation.measured_up.is_some() {
                    let (pose, k) = linearized.pose_and_kinematics(observation.tau)?;
                    (k, Some(pose))
                } else {
                    (linearized.kinematics(observation.tau)?, None)
                };
            sink.factor(id, |sink| {
                if observe_gyro {
                    let residual = self.gyroscope_information_root
                        * (k.kinematics.angular_velocity.inner + gyro_bias
                            - observation.angular_velocity.inner);
                    let jacobians = k
                        .angular_velocity_jacobians
                        .map(|j| self.gyroscope_information_root * j);
                    let gyro_jacobians = weights.map(|weight| {
                        let mut j = SMatrix::<R, 3, 6>::zeros();
                        j.fixed_columns_mut::<3>(0)
                            .copy_from(&(self.gyroscope_information_root * weight));
                        j
                    });
                    self.emit(sink, &residual, &jacobians, &gyro_jacobians)?;
                }
                if let Some((measured, pose)) = observation.measured_up.zip(pose.as_ref()) {
                    let residual = tilt(
                        &pose.pose.inner.rotation,
                        &measured,
                        &self.tilt_information_root,
                    )?;
                    let derivative =
                        tilt_jacobian(&pose.pose.inner.rotation, &self.tilt_information_root);
                    let jacobians = pose
                        .jacobians
                        .each_ref()
                        .map(|j| derivative * j.fixed_rows::<3>(0));
                    common::emit(sink, &self.controls, &residual, &jacobians)?;
                }
                if let Some((measured, pose)) = observation.specific_force.zip(pose) {
                    common::finite(measured.inner.iter())?;
                    common::finite(observation.accelerometer_information_root.iter())?;
                    let inverse = pose
                        .pose
                        .inner
                        .rotation
                        .to_rotation_matrix()
                        .inverse()
                        .into_inner();
                    let prediction = inverse
                        * (k.kinematics.linear_acceleration.inner
                            + self.gravity_compensation.inner);
                    let point_bias = ForceBias::at_time(observation.bias_tau);
                    let (correction, bias_jacobians) = observation
                        .force_bias
                        .as_ref()
                        .unwrap_or(&point_bias)
                        .evaluate(biases)?;
                    let jacobians = std::array::from_fn(|i| {
                        observation.accelerometer_information_root
                            * (prediction.cross_matrix() * pose.jacobians[i].fixed_rows::<3>(0)
                                + inverse * k.linear_acceleration_jacobians[i])
                    });
                    self.emit(
                        sink,
                        &(observation.accelerometer_information_root
                            * (prediction + correction - measured.inner)),
                        &jacobians,
                        &bias_jacobians.map(|j| observation.accelerometer_information_root * j),
                    )?;
                }
                Ok(())
            })?;
        }
        Ok(())
    }
}

impl<R: RealField + Copy> ImuKinematics<R> {
    fn emit<L: LinearizationSink<Scalar = R>>(
        &self,
        sink: &mut L,
        residual: &nalgebra::Vector3<R>,
        pose: &[SMatrix<R, 3, 6>; 4],
        bias: &[SMatrix<R, 3, 6>; 2],
    ) -> Result<(), EvaluationError> {
        common::emit_blocks(
            sink,
            residual,
            [
                JacobianBlock::new(self.controls[0], &pose[0]),
                JacobianBlock::new(self.controls[1], &pose[1]),
                JacobianBlock::new(self.controls[2], &pose[2]),
                JacobianBlock::new(self.controls[3], &pose[3]),
                JacobianBlock::new(self.biases[0], &bias[0]),
                JacobianBlock::new(self.biases[1], &bias[1]),
            ],
        )
    }
    fn validate(&self) -> Result<(), EvaluationError> {
        common::finite(
            self.gravity_compensation
                .inner
                .iter()
                .chain(self.gyroscope_information_root.iter())
                .chain(self.tilt_information_root.iter()),
        )
    }
}

fn tilt<R: RealField + Copy>(
    rotation: &UnitQuaternion<R>,
    measured: &Vector3<Robot, R>,
    root: &Matrix3<R>,
) -> Result<nalgebra::Vector3<R>, EvaluationError> {
    common::validate_up(&measured.inner)?;
    common::finite(root.iter())?;
    Ok(root * (common::up(rotation) - measured.inner))
}

fn tilt_jacobian<R: RealField + Copy>(
    rotation: &UnitQuaternion<R>,
    root: &Matrix3<R>,
) -> Matrix3<R> {
    root * common::up(rotation).cross_matrix()
}

/// Up-direction constraint on the evaluated spline, independent of absolute IMU yaw.
///
/// Residual: `W * (Rᵀ * [0, 0, 1] - measured_up)`, where `R` maps Robot to Local.
/// All three components distinguish upright from inverted attitudes while imposing
/// only two rotational constraints. The exact antipode is a stationary maximum.
#[derive(Clone, Debug)]
pub struct RollPitchPrior<R: RealField + Copy = f64> {
    pub controls: [StateKey<PoseControl<R>>; 4],
    pub duration: R,
    pub tau: R,
    /// Finite unit up direction from IMU attitude, expressed in the robot frame.
    pub measured_up: Vector3<Robot, R>,
    /// Whitens dimensionless robot-frame up-vector errors, not Euler angles.
    /// Angular measurement covariance must be mapped into these coordinates upstream.
    pub information_root: Matrix3<R>,
}

impl<R: RealField + Copy, S: StateStore<PoseControl<R>>> Factor<S> for RollPitchPrior<R> {
    type Scalar = R;

    fn visit_variables(&self, mut visitor: impl FnMut(BlockId)) {
        for key in self.controls {
            visitor(key.block_id());
        }
    }

    fn cost(&self, states: &S) -> Result<R, EvaluationError> {
        let pose = common::spline(states, &self.controls, self.duration)?.pose(self.tau)?;
        common::cost(&tilt(
            &pose.inner.rotation,
            &self.measured_up,
            &self.information_root,
        )?)
    }

    fn linearize<L: LinearizationSink<Scalar = R>>(
        &self,
        states: &S,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        let spline = common::spline(states, &self.controls, self.duration)?;
        let pose = spline.linearize()?.pose(self.tau)?;
        let residual = tilt(
            &pose.pose.inner.rotation,
            &self.measured_up,
            &self.information_root,
        )?;
        let h = tilt_jacobian(&pose.pose.inner.rotation, &self.information_root);
        let jacobians = pose.jacobians.map(|j| h * j.fixed_rows::<3>(0));
        common::emit(sink, &self.controls, &residual, &jacobians)
    }
}

/// Scalar change in evaluated yaw from segment start to `end_tau`, without an absolute yaw anchor.
///
/// Residual: `W * wrap(heading(R(end_tau)) - heading(R(0)) - measured_yaw_change)`.
/// Both rotations are evaluated on the spline, not read from control poses.
/// Heading is `atan2(R[1, 0], R[0, 0])`, not the z component of an SO(3) log.
/// Heading is undefined when the robot x axis is vertical; such evaluations must
/// fail. Wrapped-angle derivatives apply away from the principal branch cut.
/// Use `end_tau = 1` for a completed interval. A temporary current-interval
/// constraint must be removed before marginalization when its permanent replacement arrives.
#[derive(Clone, Debug)]
pub struct RelativeYaw<R: RealField + Copy = f64> {
    pub controls: [StateKey<PoseControl<R>>; 4],
    pub duration: R,
    /// Normalized observation time in [0, 1].
    pub end_tau: R,
    /// Wrapped end-minus-start yaw, in radians.
    pub measured_yaw_change: R,
    /// Whitens the yaw difference in radians. For independent attitude samples,
    /// difference variance is the sum of their yaw variances.
    pub information_root: R,
}

impl<R: RealField + Copy, S: StateStore<PoseControl<R>>> Factor<S> for RelativeYaw<R> {
    type Scalar = R;

    fn visit_variables(&self, mut visitor: impl FnMut(BlockId)) {
        for key in self.controls {
            visitor(key.block_id());
        }
    }

    fn cost(&self, states: &S) -> Result<R, EvaluationError> {
        common::positive(self.information_root)?;
        let spline = common::spline(states, &self.controls, self.duration)?;
        let error = common::yaw_error(
            &spline.pose(R::zero())?.inner.rotation,
            &spline.pose(self.end_tau)?.inner.rotation,
            self.measured_yaw_change,
        )?;
        common::cost(&SVector::<R, 1>::new(self.information_root * error))
    }

    fn linearize<L: LinearizationSink<Scalar = R>>(
        &self,
        states: &S,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        common::positive(self.information_root)?;
        let spline = common::spline(states, &self.controls, self.duration)?;
        let linearized = spline.linearize()?;
        let a = linearized.pose(R::zero())?;
        let b = linearized.pose(self.end_tau)?;
        let error = common::yaw_error(
            &a.pose.inner.rotation,
            &b.pose.inner.rotation,
            self.measured_yaw_change,
        )?;
        let ja = common::heading_jacobian(&a.pose.inner.rotation)? * self.information_root;
        let jb = common::heading_jacobian(&b.pose.inner.rotation)? * self.information_root;
        let jacobians = std::array::from_fn(|i| {
            jb * b.jacobians[i].fixed_rows::<3>(0) - ja * a.jacobians[i].fixed_rows::<3>(0)
        });
        common::emit(
            sink,
            &self.controls,
            &SVector::<R, 1>::new(self.information_root * error),
            &jacobians,
        )
    }
}

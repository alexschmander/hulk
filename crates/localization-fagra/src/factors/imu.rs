use coordinate_systems::{Local, Robot};
use fagra::{
    BlockId, EvaluationError, Factor, FactorBatch, FactorSelection, LinearizationSink, StateKey,
    StateStore,
};
use linear_algebra::Vector3;
use nalgebra::{Matrix2, Matrix3, RealField, SMatrix, SVector, UnitQuaternion};

use super::common;
use crate::variables::PoseControl;

/// One IMU sample evaluated on an interval's trajectory.
/// Measurements are expressed in robot axes and referred to the robot origin;
/// sensor extrinsic and lever-arm corrections, if needed, happen upstream.
#[derive(Clone, Debug)]
pub struct ImuObservation<R: RealField + Copy = f64> {
    pub tau: R,
    /// Angular velocity in rad/s.
    pub angular_velocity: Vector3<Robot, R>,
    /// Accelerometer specific force in m/s², not the robot's translational
    /// acceleration. Absent when accelerometer fitting is disabled.
    pub specific_force: Option<Vector3<Robot, R>>,
}

/// Batched gyroscope and optional accelerometer residuals sharing spline preparation.
///
/// With robot-to-local rotation `R` and local trajectory position `p`, raw
/// residuals are body angular velocity minus the measured angular velocity, and
/// `Rᵀ * (p̈ + gravity_compensation) - specific_force`. Each sample contributes
/// three gyroscope rows and, when specific force is present, three accelerometer
/// rows within the same factor scope. Noise roots whiten these robot-axis errors.
#[derive(Clone, Debug)]
pub struct ImuKinematics<R: RealField + Copy = f64> {
    pub controls: [StateKey<PoseControl<R>>; 4],
    pub duration: R,
    /// Local-frame acceleration added to trajectory acceleration before rotating
    /// into the robot frame; at rest this predicts the accelerometer reading.
    pub gravity_compensation: Vector3<Local, R>,
    pub gyroscope_information_root: Matrix3<R>,
    pub accelerometer_information_root: Matrix3<R>,
}

impl<R: RealField + Copy, S: StateStore<PoseControl<R>>> FactorBatch<S> for ImuKinematics<R> {
    type Scalar = R;
    type Factor = ImuObservation<R>;

    fn visit_variables(&self, _factor: &Self::Factor, mut visitor: impl FnMut(BlockId)) {
        for key in self.controls {
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
        let mut cost = R::zero();
        for (_, observation) in factors {
            common::finite(observation.angular_velocity.inner.iter())?;
            let (k, pose) = if observation.specific_force.is_some() {
                let (pose, k) = spline.pose_and_kinematics(observation.tau)?;
                (k, Some(pose))
            } else {
                (spline.kinematics(observation.tau)?, None)
            };
            cost += common::cost(
                &(self.gyroscope_information_root
                    * (k.angular_velocity.inner - observation.angular_velocity.inner)),
            )?;
            if let Some((measured, pose)) = observation.specific_force.zip(pose) {
                common::finite(measured.inner.iter())?;
                let rotation = pose.inner.rotation;
                let prediction = rotation.inverse()
                    * (k.linear_acceleration.inner + self.gravity_compensation.inner);
                cost += common::cost(
                    &(self.accelerometer_information_root * (prediction - measured.inner)),
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
        for (id, observation) in factors {
            common::finite(observation.angular_velocity.inner.iter())?;
            let (k, pose) = if observation.specific_force.is_some() {
                let (pose, k) = linearized.pose_and_kinematics(observation.tau)?;
                (k, Some(pose))
            } else {
                (linearized.kinematics(observation.tau)?, None)
            };
            let residual = self.gyroscope_information_root
                * (k.kinematics.angular_velocity.inner - observation.angular_velocity.inner);
            let jacobians = k
                .angular_velocity_jacobians
                .map(|j| self.gyroscope_information_root * j);
            sink.factor(id, |sink| {
                common::emit(sink, &self.controls, &residual, &jacobians)?;
                if let Some((measured, pose)) = observation.specific_force.zip(pose) {
                    common::finite(measured.inner.iter())?;
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
                    let jacobians = std::array::from_fn(|i| {
                        self.accelerometer_information_root
                            * (prediction.cross_matrix() * pose.jacobians[i].fixed_rows::<3>(0)
                                + inverse * k.linear_acceleration_jacobians[i])
                    });
                    common::emit(
                        sink,
                        &self.controls,
                        &(self.accelerometer_information_root * (prediction - measured.inner)),
                        &jacobians,
                    )?;
                }
                Ok(())
            })?;
        }
        Ok(())
    }
}

impl<R: RealField + Copy> ImuKinematics<R> {
    fn validate(&self) -> Result<(), EvaluationError> {
        common::finite(
            self.gravity_compensation
                .inner
                .iter()
                .chain(self.gyroscope_information_root.iter())
                .chain(self.accelerometer_information_root.iter()),
        )
    }
}

fn tilt<R: RealField + Copy>(
    rotation: &UnitQuaternion<R>,
    measured: &Vector3<Robot, R>,
    root: &Matrix2<R>,
) -> Result<nalgebra::Vector2<R>, EvaluationError> {
    common::validate_up(&measured.inner)?;
    common::finite(root.iter())?;
    Ok(root * (common::up(rotation) - measured.inner).fixed_rows::<2>(0))
}

fn tilt_jacobian<R: RealField + Copy>(
    rotation: &UnitQuaternion<R>,
    root: &Matrix2<R>,
) -> SMatrix<R, 2, 3> {
    root * common::up(rotation).cross_matrix().fixed_rows::<2>(0)
}

/// Two-dimensional tilt constraint on the evaluated spline, independent of absolute IMU yaw.
///
/// Residual: `W * (Rᵀ * [0, 0, 1] - measured_up).xy`, where `R` maps robot
/// to local coordinates. This near-upright model cannot distinguish upright
/// from exactly inverted poses using the two horizontal components alone.
#[derive(Clone, Debug)]
pub struct RollPitchPrior<R: RealField + Copy = f64> {
    pub controls: [StateKey<PoseControl<R>>; 4],
    pub duration: R,
    pub tau: R,
    /// Finite unit up direction from IMU attitude, expressed in the robot frame.
    pub measured_up: Vector3<Robot, R>,
    /// Whitens dimensionless robot-frame up-vector x/y errors, not Euler angles.
    /// Angular measurement covariance must be mapped into these coordinates upstream.
    pub information_root: Matrix2<R>,
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

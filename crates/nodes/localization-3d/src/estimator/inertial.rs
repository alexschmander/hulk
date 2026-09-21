use super::{Estimator, seconds_per_knot};
use booster::ImuState;
use color_eyre::Result;
use kinematics::robot_kinematics::RobotKinematics;
use linear_algebra::{Point3, Vector3};
use localization_fagra::factors::{
    FootGround, FootObservation, ImuKinematics, ImuObservation, RollPitchPrior,
};
use nalgebra::{Matrix2, Matrix3, UnitQuaternion};
use ros_z::time::Time;
use types::time_wrapper::TimeWrapper;

impl Estimator {
    pub(crate) fn ingest_imu(&mut self, time: Time, imu: ImuState) -> Result<bool> {
        if !imu
            .roll_pitch_yaw
            .inner
            .iter()
            .chain(imu.angular_velocity.inner.iter())
            .all(|v| v.is_finite())
        {
            tracing::warn!(?time, "discarding nonfinite IMU measurement");
            return Ok(false);
        }
        let Some((segment, tau)) = self.check_time(time, "IMU")? else {
            return Ok(false);
        };
        let controls = self.ensure_segment(segment)?;
        let batch = *self.imu_batches.entry(segment).or_insert_with(|| {
            self.graph.add_batch(ImuKinematics {
                controls,
                duration: seconds_per_knot(),
                gravity_compensation: Vector3::wrap(nalgebra::Vector3::new(0.0, 0.0, 9.81)),
                gyroscope_information_root: Matrix3::identity() * 10.0,
                accelerometer_information_root: Matrix3::identity(),
            })
        });
        self.graph.add_factor_to(
            batch,
            ImuObservation {
                tau,
                angular_velocity: Vector3::wrap(imu.angular_velocity.inner.cast()),
                specific_force: None,
            },
        )?;
        let rpy = imu.roll_pitch_yaw.inner.cast::<f64>();
        let orientation = UnitQuaternion::from_euler_angles(rpy.x, rpy.y, rpy.z);
        self.attitudes.insert(time, orientation);
        self.graph.add_factor(RollPitchPrior {
            controls,
            duration: seconds_per_knot(),
            tau,
            measured_up: Vector3::wrap(orientation.inverse() * nalgebra::Vector3::z()),
            information_root: Matrix2::identity() * 10.0,
        })?;
        self.commit_time(time);
        *self.measurements.entry(segment).or_default() += 1;
        Ok(true)
    }

    pub(crate) fn ingest_kinematics(
        &mut self,
        sample: TimeWrapper<RobotKinematics>,
    ) -> Result<bool> {
        let left = sample.inner.left_leg.sole_to_robot.inner.translation.vector;
        let right = sample
            .inner
            .right_leg
            .sole_to_robot
            .inner
            .translation
            .vector;
        if !left.iter().chain(right.iter()).all(|v| v.is_finite()) {
            tracing::warn!(time = ?sample.time, "discarding nonfinite foot measurement");
            return Ok(false);
        }
        let Some((segment, tau)) = self.check_time(sample.time, "kinematics")? else {
            return Ok(false);
        };
        let controls = self.ensure_segment(segment)?;
        let batch = *self.foot_batches.entry(segment).or_insert_with(|| {
            self.graph.add_batch(FootGround {
                controls,
                duration: seconds_per_knot(),
                sigma: 1.0e-2,
            })
        });
        self.graph.add_factor_to(
            batch,
            FootObservation {
                tau,
                left_sole: Point3::wrap(left.cast().into()),
                right_sole: Point3::wrap(right.cast().into()),
            },
        )?;
        self.commit_time(sample.time);
        *self.measurements.entry(segment).or_default() += 1;
        Ok(true)
    }
}

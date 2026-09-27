use super::{Estimator, recovery::MotionRecord, seconds_per_knot};
use booster::ImuState;
use color_eyre::Result;
use coordinate_systems::{ImuReference, Robot};
use kinematics::robot_kinematics::RobotKinematics;
use linear_algebra::{Orientation3, Point3, Vector3};
use localization_fagra::factors::{FootGround, FootObservation, ImuKinematics, ImuObservation};
use nalgebra::Matrix3;
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
        if self.check_time(time, "IMU")?.is_none() {
            return Ok(false);
        }
        let rpy = imu.roll_pitch_yaw.inner.cast::<f64>();
        let attitude = Orientation3::from_euler_angles(rpy.x, rpy.y, rpy.z);
        let angular_velocity = Vector3::wrap(imu.angular_velocity.inner.cast());
        self.accept_motion(MotionRecord::Imu {
            time,
            angular_velocity,
            attitude,
        })?;
        if let Some(parameters) = &self.parameters.accelerometer {
            let means = self.acceleration.observe(
                time,
                attitude,
                angular_velocity,
                Vector3::wrap(imu.linear_acceleration.inner.cast()),
                parameters,
                self.origin,
            );
            for mut mean in means {
                let Some(attitude) = self.attitude_at(mean.time) else {
                    continue;
                };
                let inverse = attitude.inner.inverse().to_rotation_matrix().into_inner();
                mean.bias.accelerometer_weights =
                    mean.bias.accelerometer_weights.map(|w| inverse * w);
                for sample in &mut mean.bias.lever_samples {
                    sample.weight = inverse * sample.weight;
                }
                self.accept_motion(MotionRecord::Acceleration {
                    time: mean.time,
                    force: attitude.rotation::<Robot>().inverse() * mean.force,
                    information_root: mean.information_root,
                    bias: mean.bias,
                })?;
            }
        }
        Ok(true)
    }

    pub(super) fn insert_acceleration(
        &mut self,
        time: Time,
        force: Vector3<Robot, f64>,
        root: f64,
        bias: localization_fagra::factors::ForceBias,
    ) -> Result<()> {
        let (segment, tau) = self.segment_and_tau(time)?;
        let controls = self.ensure_segment(segment)?;
        let (biases, bias_tau) = self.ensure_biases(time)?;
        let batch = *self.acceleration_batches.entry(segment).or_insert_with(|| {
            self.graph.add_batch(ImuKinematics {
                controls,
                biases,
                duration: seconds_per_knot(),
                gravity_compensation: Vector3::wrap(nalgebra::Vector3::new(0.0, 0.0, 9.81)),
                gyroscope_information_root: Matrix3::zeros(),
                tilt_information_root: Matrix3::zeros(),
            })
        });
        self.graph.add_factor_to(
            batch,
            ImuObservation {
                tau,
                bias_tau,
                angular_velocity: Vector3::zeros(),
                measured_up: None,
                specific_force: Some(force),
                accelerometer_information_root: Matrix3::identity() * root,
                force_bias: Some(bias),
            },
        )?;
        *self.measurements.entry(segment).or_default() += 1;
        Ok(())
    }

    pub(super) fn insert_imu(
        &mut self,
        time: Time,
        angular_velocity: Vector3<Robot, f64>,
        orientation: Orientation3<ImuReference, f64>,
    ) -> Result<()> {
        let (segment, tau) = self.segment_and_tau(time)?;
        let controls = self.ensure_segment(segment)?;
        let (biases, bias_tau) = self.ensure_biases(time)?;
        let batch = *self.imu_batches.entry(segment).or_insert_with(|| {
            self.graph.add_batch(ImuKinematics {
                controls,
                biases,
                duration: seconds_per_knot(),
                gravity_compensation: Vector3::wrap(nalgebra::Vector3::new(0.0, 0.0, 9.81)),
                gyroscope_information_root: Matrix3::identity() * 10.0,
                tilt_information_root: Matrix3::identity() * 10.0,
            })
        });
        self.graph.add_factor_to(
            batch,
            ImuObservation {
                tau,
                bias_tau,
                angular_velocity,
                measured_up: Some(Vector3::wrap(
                    orientation.inner.inverse() * nalgebra::Vector3::z(),
                )),
                specific_force: None,
                accelerometer_information_root: Matrix3::zeros(),
                force_bias: None,
            },
        )?;
        self.attitudes.insert(time, orientation);
        *self.measurements.entry(segment).or_default() += 1;
        Ok(())
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
        if self.check_time(sample.time, "kinematics")?.is_none() {
            return Ok(false);
        }
        self.accept_motion(MotionRecord::Feet {
            time: sample.time,
            left: Point3::wrap(left.cast().into()),
            right: Point3::wrap(right.cast().into()),
        })
    }

    pub(super) fn insert_feet(
        &mut self,
        time: Time,
        left: Point3<Robot, f64>,
        right: Point3<Robot, f64>,
    ) -> Result<()> {
        let (segment, tau) = self.segment_and_tau(time)?;
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
                left_sole: left,
                right_sole: right,
            },
        )?;
        *self.measurements.entry(segment).or_default() += 1;
        Ok(())
    }
}

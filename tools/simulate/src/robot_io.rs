//! Robot-facing measurements and serial joint control, independent of rendering and ROS-Z.
use booster::{CommandType, ImuState, LowCommand, LowState, MotorState};
use color_eyre::eyre::{Result, ensure, eyre};
use coordinate_systems::{Ground, Robot};
use linear_algebra::{Isometry3, vector};
use mujoco_rs::prelude::{MjData, MjModel, MjtObj};
use nalgebra::{Matrix3, Quaternion, Translation3, UnitQuaternion};
use projection::camera_matrix::CameraMatrix;

// Booster serial order, also used by Joints::into_iter(). MJCF names differ from Rust names:
// shoulder_yaw = Elbow_Pitch, elbow = Elbow_Yaw, ankle_up/down = Ankle_Pitch/Roll.
pub(crate) const JOINTS: [&str; 22] = [
    "AAHead_yaw",
    "Head_pitch",
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
];

pub(crate) type Data = MjData<Box<MjModel>>;

pub struct RobotBinding {
    joints: Vec<(usize, usize, usize)>, // position address, velocity address, actuator
    trunk: usize,
    head: usize,
    feet: [usize; 2],
    camera: usize,
    bodies: [usize; 22],
    orientation: usize,
    gyro: usize,
    acceleration: usize,
}

#[derive(Clone)]
pub struct Observation {
    pub robot_to_world: nalgebra::Isometry3<f32>,
    pub camera_housing: usize,
    pub camera_to_world: nalgebra::Isometry3<f32>,
    pub ground_to_world: nalgebra::Isometry3<f32>,
    pub kinematics: kinematics::robot_kinematics::RobotKinematics,
    pub support: Option<types::support_foot::SupportFootState>,
    pub low_state: LowState,
    pub ground_to_robot: Isometry3<Ground, Robot>,
    pub camera_matrix: CameraMatrix,
}

impl Observation {
    pub fn stationary(&mut self) {
        for motor in &mut self.low_state.motor_state_serial {
            motor.velocity = 0.0;
            motor.acceleration = 0.0;
        }
        self.low_state.imu_state.angular_velocity = vector![0.0, 0.0, 0.0];
        let gravity = self.robot_to_world.rotation.inverse() * nalgebra::vector![0.0, 0.0, 9.81];
        self.low_state.imu_state.linear_acceleration = linear_algebra::Vector3::wrap(gravity);
    }
}

impl RobotBinding {
    pub fn ground_to_world(&self, data: &Data) -> nalgebra::Isometry3<f32> {
        let [left, right] = self.feet.map(|id| data.xpos()[id]);
        ground_pose(body_pose(data, self.trunk), left, right)
    }

    pub fn new(data: &Data, prefix: &str) -> Result<Self> {
        let model = data.model();
        let id = |kind, name: &str| {
            model
                .name_to_id(kind, &format!("{prefix}{name}"))
                .ok_or_else(|| eyre!("missing MuJoCo object {prefix}{name}"))
        };
        let mut joints = Vec::with_capacity(22);
        for name in JOINTS {
            let joint = id(MjtObj::mjOBJ_JOINT, name)?;
            let actuator = id(MjtObj::mjOBJ_ACTUATOR, name)?;
            ensure!(
                model.actuator_trnid()[actuator][0] == joint as i32,
                "actuator {name} targets a different joint"
            );
            joints.push((
                model.jnt_qposadr()[joint] as usize,
                model.jnt_dofadr()[joint] as usize,
                actuator,
            ));
        }
        let sensor = |name| -> Result<usize> {
            Ok(model.sensor_adr()[id(MjtObj::mjOBJ_SENSOR, name)?] as usize)
        };
        Ok(Self {
            bodies: [
                id(MjtObj::mjOBJ_BODY, "Head_1")?,
                id(MjtObj::mjOBJ_BODY, "Head_2")?,
                id(MjtObj::mjOBJ_BODY, "Left_Arm_1")?,
                id(MjtObj::mjOBJ_BODY, "Left_Arm_2")?,
                id(MjtObj::mjOBJ_BODY, "Left_Arm_3")?,
                id(MjtObj::mjOBJ_BODY, "left_hand_link")?,
                id(MjtObj::mjOBJ_BODY, "Right_Arm_1")?,
                id(MjtObj::mjOBJ_BODY, "Right_Arm_2")?,
                id(MjtObj::mjOBJ_BODY, "Right_Arm_3")?,
                id(MjtObj::mjOBJ_BODY, "right_hand_link")?,
                id(MjtObj::mjOBJ_BODY, "Left_Hip_Pitch")?,
                id(MjtObj::mjOBJ_BODY, "Left_Hip_Roll")?,
                id(MjtObj::mjOBJ_BODY, "Left_Hip_Yaw")?,
                id(MjtObj::mjOBJ_BODY, "Left_Shank")?,
                id(MjtObj::mjOBJ_BODY, "Left_Ankle_Cross")?,
                id(MjtObj::mjOBJ_BODY, "left_foot_link")?,
                id(MjtObj::mjOBJ_BODY, "Right_Hip_Pitch")?,
                id(MjtObj::mjOBJ_BODY, "Right_Hip_Roll")?,
                id(MjtObj::mjOBJ_BODY, "Right_Hip_Yaw")?,
                id(MjtObj::mjOBJ_BODY, "Right_Shank")?,
                id(MjtObj::mjOBJ_BODY, "Right_Ankle_Cross")?,
                id(MjtObj::mjOBJ_BODY, "right_foot_link")?,
            ],
            joints,
            trunk: id(MjtObj::mjOBJ_BODY, "Trunk")?,
            head: id(MjtObj::mjOBJ_BODY, "Head_2")?,
            feet: [
                id(MjtObj::mjOBJ_BODY, "left_foot_link")?,
                id(MjtObj::mjOBJ_BODY, "right_foot_link")?,
            ],
            camera: id(MjtObj::mjOBJ_CAMERA, "camera")?,
            orientation: sensor("orientation")?,
            gyro: sensor("angular-velocity")?,
            acceleration: sensor("accelerometer")?,
        })
    }

    pub fn validate_command(command: &LowCommand) -> Result<()> {
        ensure!(
            command.command_type == CommandType::Serial,
            "only serial joint commands are supported"
        );
        ensure!(
            command.motor_commands.len() == 22,
            "expected 22 motor commands"
        );
        for motor in &command.motor_commands {
            ensure!(
                motor.command_type == CommandType::Serial,
                "motor command must use serial coordinates"
            );
            ensure!(
                [
                    motor.position,
                    motor.velocity,
                    motor.torque,
                    motor.kp,
                    motor.kd,
                    motor.weight
                ]
                .iter()
                .all(|v| v.is_finite()),
                "non-finite motor command"
            );
            ensure!(motor.kp >= 0.0 && motor.kd >= 0.0, "negative motor gains");
        }
        Ok(())
    }

    pub fn apply(&self, data: &mut Data, command: Option<&LowCommand>) {
        for (index, &(q, v, actuator)) in self.joints.iter().enumerate() {
            let torque = command.map_or(0.0, |command| {
                let motor = &command.motor_commands[index];
                motor.torque as f64
                    + motor.kp as f64 * (motor.position as f64 - data.qpos()[q])
                    + motor.kd as f64 * (motor.velocity as f64 - data.qvel()[v])
            });
            let [minimum, maximum] = data.model().actuator_ctrlrange()[actuator];
            data.ctrl_mut()[actuator] = torque.clamp(minimum, maximum);
        }
    }

    pub fn joint_positions(&self, data: &Data) -> [f32; 22] {
        std::array::from_fn(|index| data.qpos()[self.joints[index].0] as f32)
    }

    pub fn apply_damping(&self, data: &mut Data, kd: f32) {
        for &(_, velocity, actuator) in &self.joints {
            let torque = -f64::from(kd) * data.qvel()[velocity];
            let [minimum, maximum] = data.model().actuator_ctrlrange()[actuator];
            data.ctrl_mut()[actuator] = torque.clamp(minimum, maximum);
        }
    }

    pub fn reset_joints(&self, data: &mut Data) {
        for (&(q, v, actuator), position) in
            self.joints.iter().zip(crate::simulated_sdk::PREPARE_POSE)
        {
            data.qpos_mut()[q] = f64::from(position);
            data.qvel_mut()[v] = 0.0;
            data.ctrl_mut()[actuator] = 0.0;
        }
        data.qacc_warmstart_mut().fill(0.0);
        data.forward();
    }

    pub fn observe(&self, data: &Data) -> Observation {
        let sensors = data.sensordata();
        let orientation = quaternion(&sensors[self.orientation..self.orientation + 4]);
        let (roll, pitch, yaw) = orientation.euler_angles();
        let serial = self
            .joints
            .iter()
            .map(|&(q, v, _)| MotorState {
                command_type: CommandType::Serial,
                position: data.qpos()[q] as f32,
                velocity: data.qvel()[v] as f32,
                acceleration: data.qacc()[v] as f32,
                torque: data.qfrc_actuator()[v] as f32,
                // MuJoCo has no motor temperature or packet loss model.
                ..Default::default()
            })
            .collect();
        let low_state = LowState {
            imu_state: ImuState {
                roll_pitch_yaw: vector![roll, pitch, yaw],
                angular_velocity: vector![
                    sensors[self.gyro] as f32,
                    sensors[self.gyro + 1] as f32,
                    sensors[self.gyro + 2] as f32
                ],
                linear_acceleration: vector![
                    sensors[self.acceleration] as f32,
                    sensors[self.acceleration + 1] as f32,
                    sensors[self.acceleration + 2] as f32
                ],
            },
            motor_state_serial: serial,
            // The MJCF models serial ankles; it does not model the parallel linkage motors.
            motor_state_parallel: Vec::new(),
        };
        let robot_to_world = body_pose(data, self.trunk);
        let head_to_world = body_pose(data, self.head);
        let [left, right] = self.feet.map(|id| data.xpos()[id]);
        let ground_to_world = ground_pose(robot_to_world, left, right);
        let ground_to_robot = Isometry3::wrap(robot_to_world.inverse() * ground_to_world);
        // MuJoCo camera: right, up, backwards. Projection camera: right, down, forwards.
        let camera_to_world = nalgebra::Isometry3::from_parts(
            Translation3::from(nalgebra::Vector3::from(
                data.cam_xpos()[self.camera].map(|v| v as f32),
            )),
            UnitQuaternion::from_matrix(&Matrix3::from_row_slice(
                &data.cam_xmat()[self.camera].map(|v| v as f32),
            )) * UnitQuaternion::from_euler_angles(std::f32::consts::PI, 0.0, 0.0),
        );
        let [width, height] = data.model().cam_resolution()[self.camera].map(|v| v as f32);
        let fovy = (data.model().cam_fovy()[self.camera] as f32).to_radians();
        let focal = height * 0.5 / (fovy * 0.5).tan();
        let camera_matrix = CameraMatrix::from_normalized_focal_and_center(
            nalgebra::vector![focal / width, focal / height],
            nalgebra::point![0.5, 0.5],
            vector![width, height],
            ground_to_robot,
            Isometry3::wrap(head_to_world.inverse() * robot_to_world),
            Isometry3::wrap(camera_to_world.inverse() * head_to_world),
        );
        let relative = self
            .bodies
            .map(|id| robot_to_world.inverse() * body_pose(data, id));
        let mut kinematics = kinematics::robot_kinematics::RobotKinematics::default();
        kinematics.head.neck_to_robot = Isometry3::wrap(relative[0]);
        kinematics.head.head_to_robot = Isometry3::wrap(relative[1]);
        kinematics.left_arm.inner_shoulder_to_robot = Isometry3::wrap(relative[2]);
        kinematics.left_arm.outer_shoulder_to_robot = Isometry3::wrap(relative[3]);
        kinematics.left_arm.upper_arm_to_robot = Isometry3::wrap(relative[4]);
        kinematics.left_arm.forearm_to_robot = Isometry3::wrap(relative[5]);
        kinematics.right_arm.inner_shoulder_to_robot = Isometry3::wrap(relative[6]);
        kinematics.right_arm.outer_shoulder_to_robot = Isometry3::wrap(relative[7]);
        kinematics.right_arm.upper_arm_to_robot = Isometry3::wrap(relative[8]);
        kinematics.right_arm.forearm_to_robot = Isometry3::wrap(relative[9]);
        kinematics.left_leg.pelvis_to_robot = Isometry3::wrap(relative[10]);
        kinematics.left_leg.hip_to_robot = Isometry3::wrap(relative[11]);
        kinematics.left_leg.thigh_to_robot = Isometry3::wrap(relative[12]);
        kinematics.left_leg.tibia_to_robot = Isometry3::wrap(relative[13]);
        kinematics.left_leg.ankle_to_robot = Isometry3::wrap(relative[14]);
        kinematics.left_leg.foot_to_robot = Isometry3::wrap(relative[15]);
        kinematics.right_leg.pelvis_to_robot = Isometry3::wrap(relative[16]);
        kinematics.right_leg.hip_to_robot = Isometry3::wrap(relative[17]);
        kinematics.right_leg.thigh_to_robot = Isometry3::wrap(relative[18]);
        kinematics.right_leg.tibia_to_robot = Isometry3::wrap(relative[19]);
        kinematics.right_leg.ankle_to_robot = Isometry3::wrap(relative[20]);
        kinematics.right_leg.foot_to_robot = Isometry3::wrap(relative[21]);
        kinematics.left_leg.sole_to_robot = Isometry3::wrap(
            relative[15]
                * nalgebra::Isometry3::from(nalgebra::Translation3::from(
                    kinematics::robot_dimensions::RobotDimensions::LEFT_FOOT_TO_LEFT_SOLE.inner,
                )),
        );
        kinematics.right_leg.sole_to_robot = Isometry3::wrap(
            relative[21]
                * nalgebra::Isometry3::from(nalgebra::Translation3::from(
                    kinematics::robot_dimensions::RobotDimensions::RIGHT_FOOT_TO_RIGHT_SOLE.inner,
                )),
        );
        let mut contacts = [false; 2];
        for (index, contact) in data.contact().iter().enumerate() {
            if contact.geom.iter().any(|&id| id < 0) || data.contact_force(index)[0] <= 1.0 {
                continue;
            }
            let bodies = contact
                .geom
                .map(|id| data.model().geom_bodyid()[id as usize] as usize);
            for (foot, &body) in self.feet.iter().enumerate() {
                contacts[foot] |= bodies.contains(&body) && bodies.contains(&0);
            }
        }
        use types::support_foot::SupportFootState;
        let support = match contacts {
            [true, true] => Some(SupportFootState::Both),
            [true, false] => Some(SupportFootState::Left),
            [false, true] => Some(SupportFootState::Right),
            _ => None,
        };
        Observation {
            robot_to_world,
            camera_housing: self.head,
            camera_to_world,
            ground_to_world,
            kinematics,
            support,
            low_state,
            ground_to_robot,
            camera_matrix,
        }
    }
}

fn quaternion(q: &[f64]) -> UnitQuaternion<f32> {
    UnitQuaternion::new_normalize(Quaternion::new(
        q[0] as f32,
        q[1] as f32,
        q[2] as f32,
        q[3] as f32,
    ))
}

fn body_pose(data: &Data, id: usize) -> nalgebra::Isometry3<f32> {
    nalgebra::Isometry3::from_parts(
        Translation3::from(nalgebra::Vector3::from(data.xpos()[id].map(|v| v as f32))),
        quaternion(&data.xquat()[id]),
    )
}

fn ground_pose(
    robot_to_world: nalgebra::Isometry3<f32>,
    left: [f64; 3],
    right: [f64; 3],
) -> nalgebra::Isometry3<f32> {
    // Ground is robot-relative, centred between the feet on the field plane, with robot yaw.
    let yaw = robot_to_world.rotation.euler_angles().2;
    nalgebra::Isometry3::from_parts(
        Translation3::new(
            ((left[0] + right[0]) * 0.5) as f32,
            ((left[1] + right[1]) * 0.5) as f32,
            0.0,
        ),
        UnitQuaternion::from_euler_angles(0.0, 0.0, yaw),
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use booster::JointsMotorState;
    use mujoco_rs::prelude::MjSpec;

    pub(crate) fn model() -> Data {
        let mut spec =
            MjSpec::from_xml(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/k1_robot.xml")).unwrap();
        let model = spec.compile().unwrap();
        let mut data = MjData::new(Box::new(model));
        data.forward();
        data
    }

    #[test]
    fn low_state_uses_measured_serial_order_and_imu() {
        let mut data = model();
        let binding = RobotBinding::new(&data, "").unwrap();
        for (i, &(q, v, _)) in binding.joints.iter().enumerate() {
            data.qpos_mut()[q] = i as f64 * 0.01;
            data.qvel_mut()[v] = -(i as f64) * 0.02;
        }
        data.forward();
        let observation = binding.observe(&data);
        assert_eq!(observation.low_state.motor_state_serial.len(), 22);
        assert!(observation.low_state.motor_state_parallel.is_empty());
        for (i, state) in observation.low_state.motor_state_serial.iter().enumerate() {
            assert!((state.position - i as f32 * 0.01).abs() < 1e-6);
            assert!((state.velocity + i as f32 * 0.02).abs() < 1e-6);
        }
        let motors = observation.low_state.serial_motor_states().unwrap();
        assert!((motors.left_arm.shoulder_yaw.position - 0.04).abs() < 1e-6);
        assert!((motors.left_leg.ankle_up.position - 0.14).abs() < 1e-6);
        let imu = observation.low_state.imu_state;
        assert!(imu.roll_pitch_yaw.inner.iter().all(|v| v.is_finite()));
        assert_eq!(
            imu.angular_velocity.inner.as_slice(),
            &data.sensordata()[binding.gyro..binding.gyro + 3]
                .iter()
                .map(|v| *v as f32)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn serial_pd_and_feedforward_are_clamped_and_drive_physics() {
        let mut data = model();
        let binding = RobotBinding::new(&data, "").unwrap();
        let mut command = LowCommand {
            command_type: CommandType::Serial,
            motor_commands: vec![booster::MotorCommand::default(); 22],
        };
        let (q, v, actuator) = binding.joints[0];
        data.qpos_mut()[q] = 0.1;
        data.qvel_mut()[v] = -0.2;
        command.motor_commands[0] = booster::MotorCommand {
            command_type: CommandType::Serial,
            position: 0.2,
            velocity: 0.3,
            torque: 1.0,
            kp: 10.0,
            kd: 2.0,
            weight: 1.0,
        };
        RobotBinding::validate_command(&command).unwrap();
        binding.apply(&mut data, Some(&command));
        assert!((data.ctrl()[actuator] - 3.0).abs() < 1e-6);
        data.step();
        assert_ne!(data.qvel()[v], -0.2);
        command.motor_commands[0].torque = 1000.0;
        binding.apply(&mut data, Some(&command));
        assert_eq!(data.ctrl()[actuator], 6.0);
        binding.apply(&mut data, None);
        assert_eq!(data.ctrl()[actuator], 0.0);
    }

    #[test]
    fn physical_kinematics_match_production_joint_and_sole_frames() {
        let mut data = model();
        let binding = RobotBinding::new(&data, "").unwrap();
        binding.reset_joints(&mut data);
        let sample = binding.observe(&data);
        let joints = sample.low_state.serial_motor_states().unwrap().positions();
        let pairs = [
            (
                sample.kinematics.head.head_to_robot.inner,
                kinematics::forward::head_to_robot(&joints.head).inner,
            ),
            (
                sample.kinematics.left_leg.sole_to_robot.inner,
                kinematics::forward::left_sole_to_robot(&joints.left_leg).inner,
            ),
            (
                sample.kinematics.right_leg.sole_to_robot.inner,
                kinematics::forward::right_sole_to_robot(&joints.right_leg).inner,
            ),
        ];
        for (physical, computed) in pairs {
            assert!((physical.translation.vector - computed.translation.vector).norm() < 1e-5);
            assert!(physical.rotation.angle_to(&computed.rotation) < 1e-5);
        }
        let mut frozen = sample.clone();
        frozen.stationary();
        assert_eq!(
            frozen.low_state.imu_state.angular_velocity,
            vector![0.0, 0.0, 0.0]
        );
        assert!(
            frozen
                .low_state
                .motor_state_serial
                .iter()
                .all(|motor| motor.velocity == 0.0)
        );
        assert_eq!(
            frozen.low_state.motor_state_serial[0].position,
            sample.low_state.motor_state_serial[0].position
        );
    }

    #[test]
    fn rejects_wrong_coordinates_lengths_and_nonfinite_commands() {
        let mut command = LowCommand::default();
        assert!(RobotBinding::validate_command(&command).is_err());
        command.motor_commands = vec![booster::MotorCommand::default(); 22];
        assert!(RobotBinding::validate_command(&command).is_ok());
        command.command_type = CommandType::Parallel;
        assert!(RobotBinding::validate_command(&command).is_err());
        command.command_type = CommandType::Serial;
        command.motor_commands[3].position = f32::NAN;
        assert!(RobotBinding::validate_command(&command).is_err());
    }

    #[test]
    fn camera_projects_its_forward_axis_to_model_image_center() {
        let data = model();
        let binding = RobotBinding::new(&data, "").unwrap();
        let observation = binding.observe(&data);
        let camera = observation.camera_matrix;
        assert_eq!(camera.image_size.inner.as_slice(), &[640.0, 544.0]);
        assert!((camera.field_of_view.y.to_degrees() - 94.0).abs() < 1e-4);
        // Compare the physical MJCF mount with production kinematics and the base
        // camera_matrix_calculator mounting pitch, which is expressed in degrees.
        let mount = kinematics::forward::head_to_camera((-0.212_581_1_f32).to_radians());
        assert!(
            (camera.head_to_camera.inner.translation.vector - mount.inner.translation.vector)
                .norm()
                < 1e-5
        );
        assert!(
            camera
                .head_to_camera
                .inner
                .rotation
                .angle_to(&mount.inner.rotation)
                < 1e-5
        );
        let expected = body_pose(&data, binding.head).inverse() * body_pose(&data, binding.trunk);
        assert!(
            (camera.robot_to_head.inner.translation.vector - expected.translation.vector).norm()
                < 1e-5
        );
        assert!(
            camera
                .robot_to_head
                .inner
                .rotation
                .angle_to(&expected.rotation)
                < 1e-5
        );
        let camera_position = nalgebra::Point3::from(nalgebra::Vector3::from(
            data.cam_xpos()[binding.camera].map(|v| v as f32),
        ));
        let rotation = Matrix3::from_row_slice(&data.cam_xmat()[binding.camera].map(|v| v as f32));
        let ahead_in_world = camera_position - rotation.column(2) * 2.0;
        let head_point = body_pose(&data, binding.head).inverse() * ahead_in_world;
        let camera_point = camera.head_to_camera.inner * head_point;
        assert!((camera_point.z - 2.0).abs() < 1e-5);
        let pixel = camera
            .intrinsics
            .project(linear_algebra::Vector3::wrap(camera_point.coords));
        assert!((pixel.x() - 320.0).abs() < 1e-3);
        assert!((pixel.y() - 272.0).abs() < 1e-3);
    }

    #[test]
    fn ground_is_relative_to_robot_yaw_and_feet_on_field_plane() {
        let robot = nalgebra::Isometry3::from_parts(
            Translation3::new(4.0, 3.0, 0.7),
            UnitQuaternion::from_euler_angles(0.2, -0.1, 1.0),
        );
        let ground = ground_pose(robot, [4.1, 3.2, 0.04], [3.9, 2.8, 0.02]);
        assert_eq!(ground.translation.vector, nalgebra::vector![4.0, 3.0, 0.0]);
        assert!((ground.rotation.euler_angles().2 - 1.0).abs() < 1e-6);
        let transformed_robot = ground.inverse() * robot;
        assert!(transformed_robot.translation.x.abs() < 1e-6);
        assert!((transformed_robot.translation.z - 0.7).abs() < 1e-6);
        assert!(transformed_robot.rotation.euler_angles().2.abs() < 1e-6);
    }
}

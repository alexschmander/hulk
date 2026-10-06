//! Simulator-side replacement for the manufacturer's mode and joint controller.
use booster::{LowCommand, RobotMode, RpcReqMsg, RpcRespMsg};
use color_eyre::{Result, eyre::eyre};
use mujoco_rs::prelude::{MjData, MjModel};
use ros_z::prelude::Context;
use tokio::{runtime::Handle, sync::watch, task::JoinHandle};

use crate::robot_io::RobotBinding;

#[derive(Clone, Debug)]
pub struct Control {
    pub mode: RobotMode,
    pub command: Option<LowCommand>,
    generation: u64,
}

impl Default for Control {
    fn default() -> Self {
        Self {
            mode: RobotMode::Damping,
            command: None,
            generation: 0,
        }
    }
}

impl Control {
    fn change_mode(&mut self, mode: RobotMode) {
        if self.mode != mode {
            self.mode = mode;
            self.generation += 1;
            // Custom resumes only after a new joint packet. In particular, an old
            // walking target must not survive Damping -> Prepare -> Custom.
            self.command = None;
        }
    }
}

pub async fn start(
    runtime: &Handle,
    context: &Context,
) -> Result<(watch::Receiver<Control>, JoinHandle<()>)> {
    // Subscribe before starting hardware_interface so its first mode request is not lost.
    let requests = context
        .session()
        .declare_subscriber("rt/LocoApiTopicReq")
        .await
        .map_err(|e| eyre!("{e}"))?;
    let joints = context
        .session()
        .declare_subscriber("rt/joint_ctrl")
        .await
        .map_err(|e| eyre!("{e}"))?;
    let session = context.session().clone();
    let (updates, control) = watch::channel(Control::default());
    let task = runtime.spawn(async move {
        let mut control = Control::default();
        loop {
            tokio::select! {
                sample = requests.recv_async() => {
                    let Ok(sample) = sample else { break };
                    let request = match cdr::deserialize::<RpcReqMsg>(&sample.payload().to_bytes()) {
                        Ok(request) => request,
                        Err(error) => {
                            bevy::log::warn!("invalid simulator SDK request: {error}");
                            continue;
                        }
                    };
                    let mode = requested_mode(&request);
                    if let Some(mode) = mode {
                        if control.mode != mode {
                            // Drop packets queued under the previous owner, including
                            // hardware_interface's packet sent just before Custom.
                            while let Ok(Some(_)) = joints.try_recv() {}
                        }
                        control.change_mode(mode);
                        // Publish the new ownership before acknowledging the mode change.
                        updates.send_replace(control.clone());
                    }
                    let response = RpcRespMsg {
                        uuid: request.uuid,
                        header: serde_json::json!({"status": if mode.is_some() { 0 } else { 1 }}).to_string(),
                        body: if mode.is_some() { String::new() } else { "Unsupported simulator SDK request".into() },
                    };
                    let result = async {
                        let bytes = cdr::serialize::<_, _, cdr::CdrLe>(&response, cdr::Infinite)?;
                        session.put("rt/LocoApiTopicResp", bytes).await.map_err(|e| eyre!("{e}"))?;
                        Ok::<_, color_eyre::Report>(())
                    }.await;
                    if let Err(error) = result {
                        bevy::log::warn!("simulator SDK response failed: {error:#}");
                    }
                }
                sample = joints.recv_async() => {
                    let Ok(sample) = sample else { break };
                    if control.mode != RobotMode::Custom {
                        continue;
                    }
                    match cdr::deserialize::<LowCommand>(&sample.payload().to_bytes())
                        .map_err(|e| eyre!("{e}"))
                        .and_then(|command| {
                            RobotBinding::validate_command(&command)?;
                            Ok(command)
                        }) {
                        Ok(command) => {
                            control.command = Some(command);
                            updates.send_replace(control.clone());
                        }
                        Err(error) => bevy::log::warn!("invalid rt/joint_ctrl command: {error:#}"),
                    }
                }
            }
        }
    });
    Ok((control, task))
}

fn requested_mode(request: &RpcReqMsg) -> Option<RobotMode> {
    let header: serde_json::Value = serde_json::from_str(&request.header).ok()?;
    let body: serde_json::Value = serde_json::from_str(&request.body).ok()?;
    if header["api_id"].as_i64()? != 2000 {
        return None;
    }
    match body["mode"].as_i64()? {
        0 => Some(RobotMode::Damping),
        1 => Some(RobotMode::Prepare),
        3 => Some(RobotMode::Custom),
        _ => None,
    }
}

// Simulator approximations, not Booster's proprietary gains or preparation trajectory.
const DAMPING_KD: f32 = 1.0;
const PREPARE_SECONDS: f64 = 2.0;
// Serial order: head, left arm, right arm, left leg, right leg. Arms down,
// knees slightly bent, feet level. The floating base is never pinned or teleported.
const PREPARE_POSE: [f32; 22] = [
    0.0, 0.0, 0.0, -1.35, 0.0, -0.5, 0.0, 1.35, 0.0, 0.5, -0.2, 0.0, 0.0, 0.4, -0.2, 0.0, -0.2,
    0.0, 0.0, 0.4, -0.2, 0.0,
];

#[derive(Default)]
pub struct Controller {
    prepare: Option<Prepare>,
}

struct Prepare {
    generation: u64,
    start_time: f64,
    start_positions: [f32; 22],
}

impl Controller {
    pub fn apply(
        &mut self,
        robot: &RobotBinding,
        data: &mut MjData<Box<MjModel>>,
        control: &Control,
    ) {
        match control.mode {
            RobotMode::Prepare => {
                let prepare = self.prepare.get_or_insert_with(|| Prepare {
                    generation: control.generation,
                    start_time: data.time(),
                    start_positions: robot.joint_positions(data),
                });
                if prepare.generation != control.generation {
                    *prepare = Prepare {
                        generation: control.generation,
                        start_time: data.time(),
                        start_positions: robot.joint_positions(data),
                    };
                }
                let progress =
                    ((data.time() - prepare.start_time) / PREPARE_SECONDS).clamp(0.0, 1.0) as f32;
                let blend = progress * progress * (3.0 - 2.0 * progress);
                let command = LowCommand {
                    motor_commands: PREPARE_POSE
                        .iter()
                        .enumerate()
                        .map(|(index, &target)| {
                            let (kp, kd) = match index {
                                0..2 => (10.0, 1.2),
                                2..10 => (20.0, 1.5),
                                _ => (80.0, 4.0),
                            };
                            booster::MotorCommand {
                                position: prepare.start_positions[index]
                                    + blend * (target - prepare.start_positions[index]),
                                kp,
                                kd,
                                weight: 1.0,
                                ..Default::default()
                            }
                        })
                        .collect(),
                    ..Default::default()
                };
                robot.apply(data, Some(&command));
            }
            RobotMode::Custom if control.command.is_some() => {
                self.prepare = None;
                robot.apply(data, control.command.as_ref());
            }
            _ => {
                self.prepare = None;
                // Also damp while waiting for the first Custom packet.
                robot.apply_damping(data, DAMPING_KD);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mujoco_rs::prelude::{MjData, MjSpec, MjtGeom, MjtObj};
    use ros_z::prelude::ContextBuilder;
    use std::time::Duration;

    async fn request(context: &Context, api: i32, mode: i32) -> bool {
        let responses = context
            .session()
            .declare_subscriber("rt/LocoApiTopicResp")
            .await
            .unwrap();
        let request = RpcReqMsg {
            uuid: "mode-test".into(),
            header: serde_json::json!({"api_id": api}).to_string(),
            body: serde_json::json!({"mode": mode}).to_string(),
        };
        let bytes = cdr::serialize::<_, _, cdr::CdrLe>(&request, cdr::Infinite).unwrap();
        context
            .session()
            .put("rt/LocoApiTopicReq", bytes)
            .await
            .unwrap();
        let response = tokio::time::timeout(Duration::from_secs(2), responses.recv_async())
            .await
            .unwrap()
            .unwrap();
        let response: RpcRespMsg = cdr::deserialize(&response.payload().to_bytes()).unwrap();
        assert_eq!(response.uuid, request.uuid);
        let header: serde_json::Value = serde_json::from_str(&response.header).unwrap();
        header["status"] == 0
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn sdk_modes_take_ownership_and_discard_old_custom_targets() {
        let context = ContextBuilder::default()
            .with_namespace("/sdk_test")
            .with_mode("peer")
            .disable_multicast_scouting()
            .with_connect_endpoints(std::iter::empty::<&str>())
            .with_listen_endpoints(std::iter::empty::<&str>())
            .build()
            .await
            .unwrap();
        let (mut control, task) = start(&Handle::current(), &context).await.unwrap();
        assert_eq!(control.borrow().mode, RobotMode::Damping);
        assert!(request(&context, 2000, 3).await);
        assert_eq!(control.borrow_and_update().mode, RobotMode::Custom);
        let command = LowCommand {
            motor_commands: vec![
                booster::MotorCommand {
                    position: 0.5,
                    kp: 40.0,
                    ..Default::default()
                };
                22
            ],
            ..Default::default()
        };
        let bytes = cdr::serialize::<_, _, cdr::CdrLe>(&command, cdr::Infinite).unwrap();
        context.session().put("rt/joint_ctrl", bytes).await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), control.changed())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            control.borrow().command.as_ref().unwrap().motor_commands[0].kp,
            40.0
        );
        // RPC retries must not clear an accepted packet or restart a Prepare trajectory.
        let generation = control.borrow().generation;
        assert!(request(&context, 2000, 3).await);
        assert_eq!(control.borrow().generation, generation);
        assert!(control.borrow().command.is_some());
        assert!(request(&context, 2000, 0).await);
        assert_eq!(control.borrow().mode, RobotMode::Damping);
        assert!(control.borrow().command.is_none());
        assert!(request(&context, 2000, 1).await);
        assert_eq!(control.borrow().mode, RobotMode::Prepare);
        let generation = control.borrow().generation;
        assert!(!request(&context, 2000, 2).await);
        assert!(!request(&context, 2001, 0).await);
        assert!(request(&context, 2000, 1).await);
        assert_eq!(control.borrow().generation, generation);
        assert!(request(&context, 2000, 3).await);
        assert!(
            control.borrow().command.is_none(),
            "must wait for a fresh Custom packet"
        );
        task.abort();
        let _ = task.await;
        context.shutdown().unwrap();
    }

    fn model() -> MjData<Box<mujoco_rs::prelude::MjModel>> {
        let mut spec =
            MjSpec::from_xml(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/k1_robot.xml")).unwrap();
        spec.world_body_mut()
            .add_geom()
            .with_type(MjtGeom::mjGEOM_PLANE)
            .with_size([0.0, 0.0, 0.01]);
        let mut data = MjData::new(Box::new(spec.compile().unwrap()));
        data.forward();
        data
    }

    #[test]
    fn damping_releases_position_targets_and_opposes_joint_velocity() {
        let mut data = model();
        let robot = RobotBinding::new(&data, "").unwrap();
        let mut controller = Controller::default();
        let mut control = Control::default();
        let joint = data
            .model()
            .name_to_id(MjtObj::mjOBJ_JOINT, "AAHead_yaw")
            .unwrap();
        let q = data.model().jnt_qposadr()[joint] as usize;
        let v = data.model().jnt_dofadr()[joint] as usize;
        let actuator = data
            .model()
            .name_to_id(MjtObj::mjOBJ_ACTUATOR, "AAHead_yaw")
            .unwrap();
        data.qpos_mut()[q] = 0.5;
        control.change_mode(RobotMode::Custom);
        control.command = Some(LowCommand {
            motor_commands: vec![
                booster::MotorCommand {
                    kp: 10.0,
                    ..Default::default()
                };
                22
            ],
            ..Default::default()
        });
        controller.apply(&robot, &mut data, &control);
        assert_eq!(data.ctrl()[actuator], -5.0);
        control.change_mode(RobotMode::Damping);
        controller.apply(&robot, &mut data, &control);
        assert!(data.ctrl().iter().all(|&torque| torque == 0.0));
        for velocity in [-2.0, 2.0] {
            data.qvel_mut()[v] = velocity;
            controller.apply(&robot, &mut data, &control);
            assert!((data.ctrl()[actuator] + velocity).abs() < 1e-6);
        }
        control.change_mode(RobotMode::Custom);
        controller.apply(&robot, &mut data, &control);
        assert_eq!(
            data.ctrl()[actuator],
            -2.0,
            "Custom without a new packet still damps"
        );
    }

    #[test]
    fn prepare_moves_measured_joints_and_uses_simulation_time() {
        let mut data = model();
        let robot = RobotBinding::new(&data, "").unwrap();
        let mut controller = Controller::default();
        let mut control = Control::default();
        let joint = data
            .model()
            .name_to_id(MjtObj::mjOBJ_JOINT, "AAHead_yaw")
            .unwrap();
        let q = data.model().jnt_qposadr()[joint] as usize;
        data.qpos_mut()[q] = 0.6;
        data.forward();
        control.change_mode(RobotMode::Prepare);
        controller.apply(&robot, &mut data, &control);
        // Starting from measured positions avoids a position-target discontinuity.
        assert!(data.ctrl().iter().all(|torque| torque.abs() < 1e-5));
        let initial = data.ctrl().to_vec();
        controller.apply(&robot, &mut data, &control);
        assert_eq!(
            data.ctrl(),
            initial,
            "paused time must not advance interpolation"
        );
        while data.time() < 10.0 {
            controller.apply(&robot, &mut data, &control);
            data.step();
        }
        data.forward();
        assert!(
            data.qpos()[2] > 0.5,
            "Prepare should hold the initially upright robot above the floor"
        );
        assert!(
            data.qpos()[3].abs() > 0.98,
            "Prepare should keep the initially upright robot standing"
        );
        let positions = robot.joint_positions(&data);
        assert!(
            positions[0].abs() < 0.05,
            "Prepare must physically move the head toward neutral"
        );
        assert!(
            (positions[3] - PREPARE_POSE[3]).abs() < 0.2,
            "Prepare must lower the arm"
        );
        // A fresh Prepare after another mode must recapture the measured pose,
        // even when the render/physics thread did not see the intermediate mode.
        control.change_mode(RobotMode::Damping);
        control.change_mode(RobotMode::Prepare);
        data.qvel_mut().fill(0.0);
        controller.apply(&robot, &mut data, &control);
        assert!(data.ctrl().iter().all(|torque| torque.abs() < 1e-5));
    }
}

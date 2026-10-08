//! Simulator-side replacement for the manufacturer's mode, joint and LED controllers.
use booster::{LedColor, LowCommand, RobotMode, RpcReqMsg, RpcRespMsg};
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
    /// None means no active LED override; firmware-owned colors are not simulated.
    pub led: Option<LedColor>,
}

impl Default for Control {
    fn default() -> Self {
        Self {
            mode: RobotMode::Damping,
            command: None,
            generation: 0,
            led: None,
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
    let lights = context
        .session()
        .declare_subscriber("rt/LightControlApiTopicReq")
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
                            log::warn!("invalid simulator SDK request: {error}");
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
                    respond(&session, "rt/LocoApiTopicResp", request.uuid, mode.is_some()).await;
                }
                sample = lights.recv_async() => {
                    let Ok(sample) = sample else { break };
                    let request = match cdr::deserialize::<RpcReqMsg>(&sample.payload().to_bytes()) {
                        Ok(request) => request,
                        Err(error) => {
                            log::warn!("invalid simulator LED request: {error}");
                            continue;
                        }
                    };
                    let led = requested_led(&request);
                    if let Ok(color) = led.as_ref() {
                        control.led = *color;
                        updates.send_replace(control.clone());
                    }
                    respond(&session, "rt/LightControlApiTopicResp", request.uuid, led.is_ok()).await;
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
                        Err(error) => log::warn!("invalid rt/joint_ctrl command: {error:#}"),
                    }
                }
            }
        }
    });
    Ok((control, task))
}

async fn respond(session: &zenoh::Session, topic: &str, uuid: String, accepted: bool) {
    let response = RpcRespMsg {
        uuid,
        header: serde_json::json!({"status": if accepted { 0 } else { 1 }}).to_string(),
        body: if accepted {
            String::new()
        } else {
            "Unsupported simulator SDK request".into()
        },
    };
    let result = async {
        let bytes = cdr::serialize::<_, _, cdr::CdrLe>(&response, cdr::Infinite)?;
        session.put(topic, bytes).await.map_err(|e| eyre!("{e}"))?;
        Ok::<_, color_eyre::Report>(())
    }
    .await;
    if let Err(error) = result {
        log::warn!("simulator SDK response failed: {error:#}");
    }
}

fn requested_led(request: &RpcReqMsg) -> Result<Option<LedColor>> {
    let header: serde_json::Value = serde_json::from_str(&request.header)?;
    match header["api_id"].as_i64() {
        Some(2000) => Ok(Some(serde_json::from_str(&request.body)?)),
        Some(2001) => Ok(None),
        _ => Err(eyre!("unsupported LED API")),
    }
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
// Serial order: head, left arm, right arm, left leg, right leg. Arm targets match
// the FastGetUp policy offsets; the ankle targets keep the simulated feet level.
// The floating base is never pinned or teleported.
pub(crate) const PREPARE_POSE: [f32; 22] = [
    0.0, 0.0, 0.0, -1.4, -0.4, 0.0, 0.0, 1.4, 0.4, 0.0, -0.2, 0.0, 0.0, 0.4, -0.2, 0.0, -0.2, 0.0,
    0.0, 0.4, -0.2, 0.0,
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
                                // The light arm joints need lower explicit damping at
                                // the 2 ms physics timestep to avoid torque oscillation.
                                2..10 => (20.0, 0.5),
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
    use ros_z::{parameter::NodeParametersExt, prelude::ContextBuilder};
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
    async fn global_zenoh_prefix_isolates_sdk_requests_and_replies() {
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("tcp/{}", port.local_addr().unwrap());
        drop(port);
        let router = ContextBuilder::default()
            .with_mode("router")
            .disable_multicast_scouting()
            .with_connect_endpoints(std::iter::empty::<&str>())
            .with_listen_endpoints([&endpoint])
            .build()
            .await
            .unwrap();
        let mut contexts = Vec::new();
        for number in ["42", "43"] {
            contexts.push(
                ContextBuilder::default()
                    .with_mode("client")
                    .with_json("namespace", number)
                    .with_router_endpoint(&endpoint)
                    .unwrap()
                    .disable_multicast_scouting()
                    .build()
                    .await
                    .unwrap(),
            );
        }
        // Identical ROS topic names and discovery also remain isolated by scope.
        let node42 = contexts[0].create_node("scope_test").build().await.unwrap();
        let node43 = contexts[1].create_node("scope_test").build().await.unwrap();
        let sub42 = node42
            .subscriber::<String>("same_topic")
            .build()
            .await
            .unwrap();
        let sub43 = node43
            .subscriber::<String>("same_topic")
            .build()
            .await
            .unwrap();
        let pub42 = node42
            .publisher::<String>("same_topic")
            .build()
            .await
            .unwrap();
        let pub43 = node43
            .publisher::<String>("same_topic")
            .build()
            .await
            .unwrap();
        pub42.publish(&"robot 42".into()).await.unwrap();
        pub43.publish(&"robot 43".into()).await.unwrap();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), sub42.recv())
                .await
                .unwrap()
                .unwrap(),
            "robot 42"
        );
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), sub43.recv())
                .await
                .unwrap()
                .unwrap(),
            "robot 43"
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(100), sub42.recv())
                .await
                .is_err()
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(100), sub43.recv())
                .await
                .is_err()
        );
        let (robot42, task42) = start(&Handle::current(), &contexts[0]).await.unwrap();
        let (robot43, task43) = start(&Handle::current(), &contexts[1]).await.unwrap();
        let global_replies = router
            .session()
            .declare_subscriber("42/rt/LocoApiTopicResp")
            .await
            .unwrap();
        // The application still uses rt/...; the router sees 42/rt/....
        assert!(request(&contexts[0], 2000, 3).await);
        assert_eq!(robot42.borrow().mode, RobotMode::Custom);
        assert_eq!(robot43.borrow().mode, RobotMode::Damping);
        let reply = tokio::time::timeout(Duration::from_secs(2), global_replies.recv_async())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(reply.key_expr().as_str(), "42/rt/LocoApiTopicResp");
        assert!(request(&contexts[1], 2000, 1).await);
        assert_eq!(robot42.borrow().mode, RobotMode::Custom);
        assert_eq!(robot43.borrow().mode, RobotMode::Prepare);
        let light = hardware_interface::LightClient::new(contexts[0].session())
            .await
            .unwrap();
        let light_replies = router
            .session()
            .declare_subscriber("42/rt/LightControlApiTopicResp")
            .await
            .unwrap();
        let color = LedColor {
            r: 12,
            g: 34,
            b: 56,
        };
        light
            .set_led_light_color(color, Duration::from_secs(2))
            .await
            .unwrap();
        assert_eq!(robot42.borrow().led, Some(color));
        assert_eq!(robot43.borrow().led, None);
        let reply = tokio::time::timeout(Duration::from_secs(2), light_replies.recv_async())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(reply.key_expr().as_str(), "42/rt/LightControlApiTopicResp");
        // Reject unsupported APIs and malformed/out-of-range colors without changing the LED.
        let rpc = hardware_interface::ZenohRpcClient::new(
            contexts[0].session(),
            "rt/LightControlApiTopic",
        )
        .await
        .unwrap();
        for (api, body) in [
            (9999, ""),
            (2000, "invalid"),
            (2000, r#"{"r":256,"g":0,"b":0}"#),
        ] {
            let error = rpc
                .call(api, body, Duration::from_secs(2))
                .await
                .unwrap_err();
            assert!(error.to_string().contains("status 1"), "{error}");
            assert_eq!(robot42.borrow().led, Some(color));
        }
        light
            .stop_led_light_control(Duration::from_secs(2))
            .await
            .unwrap();
        assert_eq!(robot42.borrow().led, None);
        assert_eq!(robot42.borrow().mode, RobotMode::Custom);
        drop(rpc);
        drop(light);
        // Production uses the same isolation helper, even on an unscoped router.
        let simulation = crate::robotics::scoped_transport(
            ContextBuilder::default()
                .with_router_endpoint(&endpoint)
                .unwrap()
                .disable_multicast_scouting(),
        )
        .build()
        .await
        .unwrap();
        let unscoped = router.session().declare_subscriber("rt/**").await.unwrap();
        let scoped = router
            .session()
            .declare_subscriber(format!("{}/rt/**", crate::ZENOH_NAMESPACE))
            .await
            .unwrap();
        let (_, simulation_task) = start(&Handle::current(), &simulation).await.unwrap();
        assert!(request(&simulation, 2000, 3).await);
        assert!(
            tokio::time::timeout(Duration::from_secs(2), scoped.recv_async())
                .await
                .is_ok()
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(100), unscoped.recv_async())
                .await
                .is_err(),
            "simulation leaked raw Booster traffic"
        );
        simulation_task.abort();
        let _ = simulation_task.await;
        simulation.shutdown().unwrap();
        task42.abort();
        task43.abort();
        let _ = task42.await;
        let _ = task43.await;
        for context in contexts {
            context.shutdown().unwrap();
        }
        router.shutdown().unwrap();
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

    #[tokio::test(flavor = "multi_thread")]
    async fn led_handler_reaches_simulated_hardware_and_blinks() {
        use led_handler::DefaultLEDColors;
        use ros_z::{prelude::QosProfile, qos::QosDurability};
        use std::sync::Arc;
        use tokio_util::task::AbortOnDropHandle;
        use types::primary_state::PrimaryState;

        let context = Arc::new(
            ContextBuilder::default()
                .with_namespace("/led_test")
                .with_mode("peer")
                .disable_multicast_scouting()
                .with_connect_endpoints(std::iter::empty::<&str>())
                .with_listen_endpoints(std::iter::empty::<&str>())
                .with_parameter_layers([std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../etc/parameters/base")])
                .build()
                .await
                .unwrap(),
        );
        let (mut control, sdk) = start(&Handle::current(), &context).await.unwrap();
        let sdk = AbortOnDropHandle::new(sdk);
        let node = context.create_node("test_input").build().await.unwrap();
        let primary = node
            .publisher::<PrimaryState>("primary_state")
            .qos(QosProfile {
                durability: QosDurability::TransientLocal,
                ..Default::default()
            })
            .build()
            .await
            .unwrap();
        let commands = node
            .publisher::<hardware_interface::LedCommand>("commands/led_command")
            .build()
            .await
            .unwrap();
        let hardware =
            AbortOnDropHandle::new(tokio::spawn(hardware_interface::run_boxed(context.clone())));
        // Wait for the real hardware subscriber before the LED handler's initial publication.
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                commands
                    .publish(&hardware_interface::LedCommand::SetParam { r: 1, g: 2, b: 3 })
                    .await
                    .unwrap();
                if tokio::time::timeout(
                    Duration::from_millis(20),
                    control.wait_for(|control| control.led.is_some()),
                )
                .await
                .is_ok()
                {
                    break;
                }
            }
        })
        .await
        .unwrap();
        let handler = AbortOnDropHandle::new(tokio::spawn(led_handler::run_boxed(context.clone())));
        for (state, color) in [
            (PrimaryState::Initial, LedColor::MAGENTA),
            (PrimaryState::Damping, LedColor::BLUE),
            (PrimaryState::Prepare, LedColor::YELLOW),
            (PrimaryState::Playing, LedColor::GREEN),
            (PrimaryState::Penalized, LedColor::RED),
        ] {
            primary.publish(&state).await.unwrap();
            tokio::time::timeout(
                Duration::from_secs(5),
                control.wait_for(|control| control.led == Some(color)),
            )
            .await
            .unwrap()
            .unwrap();
        }
        primary.publish(&PrimaryState::Stop).await.unwrap();
        for color in [LedColor::BLACK, LedColor::RED] {
            tokio::time::timeout(
                Duration::from_secs(2),
                control.wait_for(|control| control.led == Some(color)),
            )
            .await
            .unwrap()
            .unwrap();
        }
        handler.abort();
        let _ = handler.await;
        commands
            .publish(&hardware_interface::LedCommand::Stop)
            .await
            .unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            control.wait_for(|control| control.led.is_none()),
        )
        .await
        .unwrap()
        .unwrap();
        hardware.abort();
        let _ = hardware.await;
        sdk.abort();
        let _ = sdk.await;
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
            "Prepare must move the arm to its readiness pose"
        );
        // A fresh Prepare after another mode must recapture the measured pose,
        // even when the render/physics thread did not see the intermediate mode.
        control.change_mode(RobotMode::Damping);
        control.change_mode(RobotMode::Prepare);
        data.qvel_mut().fill(0.0);
        controller.apply(&robot, &mut data, &control);
        assert!(data.ctrl().iter().all(|torque| torque.abs() < 1e-5));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn prepare_makes_a_fallen_robot_ready_for_the_real_detector() {
        let calibration = tempfile::tempdir().unwrap();
        crate::robotics::write_calibration(calibration.path()).unwrap();
        let context = ContextBuilder::default()
            .with_namespace("/prepare_readiness_test")
            .with_mode("peer")
            .disable_multicast_scouting()
            .with_connect_endpoints(std::iter::empty::<&str>())
            .with_listen_endpoints(std::iter::empty::<&str>())
            .with_parameter_layers([
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../etc/parameters/base"),
                calibration.path().to_owned(),
            ])
            .build()
            .await
            .unwrap();
        let node = context.create_node("parameters").build().await.unwrap();
        let parameters = node
            .bind_parameter_as::<fall_detection::Parameters>("fall_detection")
            .unwrap();
        let parameters = parameters.snapshot();
        let inference_node = context
            .create_node("policy_parameters")
            .build()
            .await
            .unwrap();
        let inference = inference_node
            .bind_parameter_as::<motion_inference::config::Parameters>("motion_inference")
            .unwrap();
        let offsets: Vec<_> = motion_inference::config::Policy::FastGetUp
            .offset(inference.snapshot().typed())
            .into_iter()
            .collect();
        assert_eq!(
            &PREPARE_POSE[2..10],
            &offsets[2..10],
            "Prepare arms must match FastGetUp offsets"
        );
        let readiness: Vec<_> = parameters.typed().stand_up_pose.into_iter().collect();
        assert_eq!(
            &readiness[2..10],
            &offsets[2..10],
            "fall readiness must accept the same arm pose"
        );
        for side in [-1.0, 1.0] {
            let mut data = model();
            data.qpos_mut()[2] = 0.25;
            data.qpos_mut()[3..7].copy_from_slice(&[
                std::f64::consts::FRAC_1_SQRT_2,
                0.0,
                side * std::f64::consts::FRAC_1_SQRT_2,
                0.0,
            ]);
            // Zero arm positions start outside the FastGetUp readiness pose.
            data.forward();
            let robot = RobotBinding::new(&data, "").unwrap();
            let mut controller = Controller::default();
            let mut control = Control::default();
            control.change_mode(RobotMode::Prepare);
            let mut detector = fall_detection::Detector::default();
            let mut last = None;
            while data.time() < 6.0 {
                controller.apply(&robot, &mut data, &control);
                data.step();
                data.forward();
                let observation = fall_detection::Observation::from_low_state(
                    &robot.observe(&data).low_state,
                    ros_z::time::Time::from_nanos((data.time() * 1e9).round() as i64),
                )
                .unwrap();
                last = detector
                    .update(observation, false, parameters.typed())
                    .or(last);
            }
            assert_eq!(
                last.unwrap().posture,
                types::fall_detection::Posture::Fallen {
                    ready_for_standup: true
                },
                "Prepare must reach the configured readiness pose; observation: {:?}",
                fall_detection::Observation::from_low_state(
                    &robot.observe(&data).low_state,
                    ros_z::time::Time::zero()
                )
                .unwrap()
            );
        }
        context.shutdown().unwrap();
    }
}

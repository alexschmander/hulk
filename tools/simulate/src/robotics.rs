//! Runs unmodified robot nodes against simulated sensors and the Booster transport.
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use bevy::prelude::Resource;
use color_eyre::Result;
use ros_z::{
    parameter::NodeParameters,
    prelude::*,
    time::{Clock, Time},
};
use tokio::{runtime::Handle, sync::watch, task::JoinSet};
use types::{
    filtered_game_controller_state::FilteredGameControllerState, motion_command::MotionCommand,
};

use crate::{
    observations::Frame,
    parameters::SimulatorParameters,
    profiles::{ControllerSource, Profile},
    robot_io::Observation,
};
#[cfg(test)]
use types::time_wrapper::TimeWrapper;

#[derive(Clone)]
pub struct Configuration {
    pub parameter_root: PathBuf,
    pub model_directory: PathBuf,
    pub router: Option<String>,
    pub namespace: String,
    pub location: Option<String>,
    pub robot_count: u8,
    pub opponent_count: u8,
    pub field_configuration: Option<crate::FieldConfiguration>,
    pub profile: Profile,
    pub controller: ControllerSource,
}

#[derive(Resource, Clone)]
pub struct Robotics(Arc<RobotStack>);
impl std::ops::Deref for Robotics {
    type Target = RobotStack;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
pub(crate) struct RobotSettings {
    pub default_away: bool,
    pub player: hsl_network_messages::PlayerNumber,
    pub scope: String,
    pub clock: Clock,
    pub ports: crate::network::Ports,
}
impl Robotics {
    pub(crate) async fn new_robot(
        runtime: Handle,
        configuration: Configuration,
        settings: RobotSettings,
    ) -> Result<Self> {
        Ok(Self(Arc::new(
            RobotStack::new(runtime, configuration, Some(settings)).await?,
        )))
    }
    #[cfg(test)]
    pub async fn new(runtime: Handle, configuration: Configuration) -> Result<Self> {
        Ok(Self(Arc::new(
            RobotStack::new(runtime, configuration, None).await?,
        )))
    }
}

pub struct RobotStack {
    runtime: Handle,
    context: Arc<Context>,
    _node: Arc<Node>,
    controller_context: Option<Arc<Context>>,
    // The set owns every task, including the SDK receiver, through shutdown and failed startup.
    tasks: Mutex<JoinSet<(&'static str, Result<()>)>>,
    _overrides: tempfile::TempDir,
    pub parameters: NodeParameters<SimulatorParameters>,
    pub field_dimensions: types::field_dimensions::FieldDimensions,
    pub profile: Profile,
    controller_unavailable: Mutex<bool>,
    whistle: watch::Sender<Option<Time>>,
    frames: watch::Sender<Option<Frame>>,
    body: crate::reference::BodyReferences,
    readiness: crate::readiness::Readiness,
    pub primary: ros_z::cache::Cache<types::primary_state::PrimaryState>,
    pub inference: ros_z::cache::Cache<motion_inference::node::Status>,
    pub safe_pose: ros_z::cache::Cache<bool>,
    #[cfg(test)]
    pub field: ros_z::cache::Cache<types::field_dimensions::FieldDimensions>,
    default_away: bool,
    game: Arc<ros_z::cache::Cache<FilteredGameControllerState>>,
    motion: ros_z::cache::Cache<MotionCommand>,
    emergency: ros_z::cache::Cache<()>,
    commands: watch::Receiver<crate::simulated_sdk::Control>,
    failure: Mutex<Option<String>>,
}

impl RobotStack {
    async fn new(
        runtime: Handle,
        configuration: Configuration,
        settings: Option<RobotSettings>,
    ) -> Result<Self> {
        let default_away = settings
            .as_ref()
            .is_some_and(|settings| settings.default_away);
        let overrides = tempfile::tempdir()?;
        // Twix can write this last layer without changing the robot parameter files.
        std::fs::write(
            overrides.path().join("behavior_node.json5"),
            r#"{control: {injected_motion_command: null}}"#,
        )?;
        std::fs::create_dir(overrides.path().join("live"))?;
        write_calibration(overrides.path())?;
        std::fs::write(
            overrides.path().join("motion_inference.json5"),
            serde_json::to_vec(
                &serde_json::json!({"neural_networks_folder": configuration.model_directory.canonicalize()?}),
            )?,
        )?;
        let mut global = serde_json::Map::new();
        if let Some(field) = &configuration.field_configuration {
            field
                .validate()
                .map_err(|error| color_eyre::eyre::eyre!(error))?;
            global.insert(
                "field_dimensions".into(),
                serde_json::to_value(field.dimensions)?,
            );
        }
        if let Some(settings) = &settings {
            global.insert(
                "player_number".into(),
                serde_json::to_value(settings.player)?,
            );
            std::fs::write(
                overrides.path().join("message_receiver.json5"),
                serde_json::to_vec(&serde_json::json!({
                    "ports": {"game_controller_state": settings.ports.state, "game_controller_return": settings.ports.returns,
                        "hsl": settings.ports.team, "hsl_broadcast_address": {"octets": [127,255,255,255]}}
                }))?,
            )?;
        }
        std::fs::write(
            overrides.path().join("global.json5"),
            serde_json::to_vec(&global)?,
        )?;
        let scope = settings
            .as_ref()
            .map_or(crate::ZENOH_NAMESPACE, |settings| settings.scope.as_str());
        let mut layers = vec![
            configuration.parameter_root.join("base"),
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("parameters"),
        ];
        if let Some(location) = &configuration.location {
            layers.push(configuration.parameter_root.join("location").join(location));
        }
        layers.extend([overrides.path().to_owned(), overrides.path().join("live")]);
        let mut builder = ContextBuilder::default()
            .with_clock(settings.as_ref().map_or_else(
                || Clock::logical(Clock::wallclock().now()),
                |settings| settings.clock.clone(),
            ))
            .with_namespace(&configuration.namespace)
            .with_parameter_layers(layers);
        if let Some(router) = &configuration.router {
            builder = builder.with_router_endpoint(router)?;
        }
        // The physics worker advances this clock; the external GameController keeps wall time.
        let context = Arc::new(builder.with_json("namespace", scope).build().await?);
        // ROS-Z applies environment overrides after builder options. Reject a
        // conflicting override before starting any node that can send SDK traffic.
        color_eyre::eyre::ensure!(
            context
                .session()
                .config()
                .get("namespace")
                .is_ok_and(
                    |value| serde_json::from_str::<String>(&value).ok().as_deref() == Some(scope)
                ),
            "ZENOH_CONFIG_OVERRIDE conflicts with the simulator transport namespace"
        );
        // The unchanged behavior node checks gamepad freshness against SystemTime.
        // Host inputs therefore retain wall-clock source timestamps, just like an
        // external controller, while the robotics context uses simulation time.
        let controller_context = if configuration.controller == ControllerSource::Local {
            let mut builder = ContextBuilder::default().with_namespace(&configuration.namespace);
            if let Some(router) = &configuration.router {
                builder = builder.with_router_endpoint(router)?;
            }
            Some(Arc::new(
                builder.with_json("namespace", scope).build().await?,
            ))
        } else {
            None
        };
        let node = Arc::new(context.create_node("simulator").build().await?);
        let parameters = node.bind_parameter_as::<SimulatorParameters>("simulator")?;
        parameters.add_validation_hook(SimulatorParameters::validate)?;
        if let Some(field) = &configuration.field_configuration {
            let height = field.goal_height;
            parameters.add_validation_hook(move |parameters| {
                if parameters.goal_height != height {
                    return Err("Goal height is shared by all robots; change the location parameters and restart".into());
                }
                Ok(())
            })?;
        }
        let inference = node
            .subscriber(motion_inference::node::STATUS_TOPIC)
            .qos(QosProfile {
                durability: ros_z::qos::QosDurability::TransientLocal,
                ..Default::default()
            })
            .cache(1)
            .build()
            .await?;
        let primary = node
            .subscriber("primary_state")
            .qos(QosProfile {
                durability: ros_z::qos::QosDurability::TransientLocal,
                ..Default::default()
            })
            .cache(1)
            .build()
            .await?;
        let safe_pose = node.subscriber("is_safe_pose").cache(1).build().await?;
        let initial_field = node
            .subscriber::<types::field_dimensions::FieldDimensions>("field_dimensions")
            .qos(QosProfile {
                durability: ros_z::qos::QosDurability::TransientLocal,
                ..Default::default()
            })
            .build()
            .await?;
        #[cfg(test)]
        let field = node
            .subscriber("field_dimensions")
            .qos(QosProfile {
                durability: ros_z::qos::QosDurability::TransientLocal,
                ..Default::default()
            })
            .cache(1)
            .build()
            .await?;
        let body = crate::reference::BodyReferences::new(&node, configuration.profile).await?;
        let behavior =
            crate::behavior_inputs::BehaviorInputs::new(&node, configuration.profile).await?;
        let camera_inputs = crate::observations::CameraInputs::new(
            &node,
            configuration.profile,
            parameters.clone(),
        )
        .await?;
        let readiness = crate::readiness::Readiness::new(&node, configuration.profile).await?;
        let game = Arc::new(
            node.subscriber::<FilteredGameControllerState>("filtered_game_controller_state")
                .cache(1)
                .build()
                .await?,
        );
        let motion = node
            .subscriber("behavior/motion_command")
            .cache(1)
            .build()
            .await?;
        let emergency = node
            .subscriber("motion/emergency_stop")
            .cache(1)
            .build()
            .await?;
        let mut tasks = JoinSet::new();
        let (commands, sdk_task) = crate::simulated_sdk::start(&runtime, &context).await?;
        // Abort the receiver if startup fails or the panel is closed.
        let sdk_task = tokio_util::task::AbortOnDropHandle::new(sdk_task);
        tasks.spawn(async move {
            (
                "booster_sdk",
                async {
                    sdk_task.await?;
                    Ok(())
                }
                .await,
            )
        });
        for spec in configuration.profile.nodes(configuration.controller) {
            let node_context = if spec.name == "controller_handler" {
                controller_context
                    .as_ref()
                    .expect("local controller context")
            } else {
                &context
            };
            let future = (spec.run)(node_context.clone());
            tasks.spawn(async move { (spec.name, future.await) });
        }
        let (whistle, pulse) = watch::channel(None);
        let whistle_context = context.clone();
        tasks.spawn(async move {
            (
                "simulator_whistle",
                crate::whistle::run(&whistle_context, pulse).await,
            )
        });
        let (frames, mut receiver) = watch::channel::<Option<Frame>>(None);
        let game_cache = game.clone();
        tasks.spawn(async move {
            let result = async {
                let mut camera_inputs = camera_inputs;
                loop {
                    receiver.changed().await?;
                    let frame = receiver.borrow_and_update().clone();
                    let Some(frame) = frame else {
                        continue;
                    };
                    let side = game_cache
                        .get_latest()
                        .map(|game| game.global_field_side)
                        .unwrap_or(if default_away {
                            types::field_dimensions::GlobalFieldSide::Away
                        } else {
                            types::field_dimensions::GlobalFieldSide::Home
                        });
                    behavior
                        .publish(
                            frame.sample.ground_to_world,
                            frame
                                .balls
                                .first()
                                .map(|ball| (ball.position, ball.velocity)),
                            &frame.field,
                            side,
                            &frame.robots,
                            frame.time,
                        )
                        .await?;
                    camera_inputs.publish(&frame).await?;
                }
            }
            .await;
            ("simulator_observations", result)
        });
        let field_dimensions =
            tokio::time::timeout(std::time::Duration::from_secs(3), initial_field.recv()).await??;
        crate::parameters::validate_field_dimensions(&field_dimensions)
            .map_err(|error| color_eyre::eyre::eyre!(error))?;
        if settings.is_some() {
            let expected = serde_json::to_value(field_dimensions)?;
            tasks.spawn(async move {
                let result: Result<()> = async {
                    loop {
                        let actual = initial_field.recv().await?;
                        color_eyre::eyre::ensure!(serde_json::to_value(actual)? == expected,
                            "Field dimensions changed while running; stop the simulator, edit the location parameters and restart");
                    }
                }.await;
                ("shared_field_guard", result)
            });
        }
        Ok(Self {
            runtime,
            context,
            _node: node,
            controller_context,
            tasks: Mutex::new(tasks),
            _overrides: overrides,
            parameters,
            field_dimensions,
            profile: configuration.profile,
            default_away,
            controller_unavailable: Mutex::new(false),
            whistle,
            frames,
            body,
            readiness,
            primary,
            inference,
            safe_pose,
            #[cfg(test)]
            field,
            game,
            motion,
            emergency,
            commands,
            failure: Mutex::new(None),
        })
    }

    pub fn poll(&self) -> Option<String> {
        let mut tasks = self.tasks.lock().unwrap();
        let mut failure = self.failure.lock().unwrap();
        if failure.is_some() {
            return failure.clone();
        }
        while let Some(result) = tasks.try_join_next() {
            if matches!(&result, Ok(("controller_handler", Ok(())))) {
                *self.controller_unavailable.lock().unwrap() = true;
                continue;
            }
            *failure = Some(match result {
                Ok((name, Ok(()))) => format!("Simulator node {name} exited"),
                Ok((name, Err(error))) => format!("Simulator node {name} failed: {error:#}"),
                Err(error) => format!("Simulator task failed: {error}"),
            });
            tasks.abort_all();
            break;
        }
        failure.clone()
    }

    pub fn status(&self) -> String {
        if let Some(failure) = &*self.failure.lock().unwrap() {
            return failure.clone();
        }
        if self.emergency.get_latest().is_some() {
            return "Emergency stop. Restart the simulator to clear it.".into();
        }
        if !self.ready() {
            return format!(
                "{} · {}",
                self.profile.label(),
                self.readiness.status(self.now())
            );
        }
        let game = match self.game.latest_stamp() {
            None => "Waiting for HSL Game Controller",
            Some(stamp) if self.now().duration_since(stamp) > std::time::Duration::from_secs(3) => {
                "HSL Game Controller state is stale"
            }
            Some(_) => "HSL Game Controller connected",
        };
        format!(
            "{} · {} · {}{}",
            self.profile.label(),
            self.readiness.status(self.now()),
            game,
            if *self.controller_unavailable.lock().unwrap() {
                " · Local gamepad unavailable"
            } else {
                ""
            }
        )
    }

    pub fn actuator_control(&self) -> crate::simulated_sdk::Control {
        self.commands.borrow().clone()
    }

    pub fn led_color(&self) -> Option<booster::LedColor> {
        self.commands.borrow().led
    }

    pub fn away(&self) -> bool {
        self.game.get_latest().map_or(self.default_away, |game| {
            game.global_field_side == types::field_dimensions::GlobalFieldSide::Away
        })
    }

    pub fn active_motion(&self) -> MotionCommand {
        self.motion
            .get_latest()
            .map(|value| value.as_ref().clone())
            .unwrap_or_default()
    }

    pub fn button_event(&self, button: i32, event: booster::ButtonEventType) -> Result<()> {
        self.runtime.block_on(crate::buttons::publish(
            self.context.session(),
            button,
            event,
        ))
    }

    #[cfg(test)]
    pub(crate) fn node(&self) -> &Node {
        &self._node
    }

    pub fn now(&self) -> Time {
        self.context.clock().now()
    }

    #[cfg(test)]
    pub fn advance_time(&self, period: std::time::Duration) -> Result<()> {
        self.context.clock().advance(period)?;
        Ok(())
    }

    pub fn whistle(&self) {
        self.whistle
            .send_replace(Some(self.now() + crate::whistle::PULSE_DURATION));
    }

    pub fn ready(&self) -> bool {
        self.readiness.ready(self.now())
    }

    pub fn publish_observation(&self, observation: &Observation, time: Time) -> Result<()> {
        self.runtime.block_on(async {
            self.body.publish(observation, time).await?;
            crate::sensors::publish(self.context.session(), &observation.low_state, time).await
        })
    }

    pub fn publish_frame(&self, frame: Frame) {
        self.frames.send_replace(Some(frame));
    }
}

impl Drop for RobotStack {
    fn drop(&mut self) {
        let tasks = self.tasks.get_mut().unwrap();
        tasks.abort_all();
        let mut shutdown = || {
            self.runtime
                .block_on(async { while tasks.join_next().await.is_some() {} })
        };
        if tokio::runtime::Handle::try_current().is_ok() {
            tokio::task::block_in_place(shutdown);
        } else {
            shutdown();
        }
        if let Some(context) = &self.controller_context
            && let Err(error) = context.shutdown()
        {
            log::error!("Simulator gamepad shutdown: {error:#}");
        }
        if let Err(error) = self.context.shutdown() {
            log::error!("Simulator shutdown: {error:#}");
        }
    }
}

#[cfg(test)]
fn spawn_network(context: &Arc<Context>, tasks: &mut JoinSet<Result<()>>) {
    for spec in crate::profiles::NETWORK {
        tasks.spawn((spec.run)(context.clone()));
    }
}

#[cfg(test)]
#[path = "network_tests.rs"]
mod network_tests;

pub(crate) fn write_calibration(root: &std::path::Path) -> Result<()> {
    // Calibrate the unmodified checks against our SDK Prepare pose, not a recorded robot.
    let pose: kinematics::joints::Joints = crate::simulated_sdk::PREPARE_POSE.into_iter().collect();
    let motors = pose.map(|position| booster::MotorState {
        position,
        ..Default::default()
    });
    std::fs::write(
        root.join("safe_pose_checker.json5"),
        serde_json::to_vec(&serde_json::json!({
            "prep_mode_serial_motor_states": motors,
            "prep_mode_imu_state": { "rpy": [0,0,0], "gyro": [0,0,0], "acc": [0,0,9.81] }
        }))?,
    )?;
    std::fs::write(
        root.join("fall_detection.json5"),
        serde_json::to_vec(&serde_json::json!({"stand_up_pose": pose}))?,
    )?;
    Ok(())
}

#[cfg(test)]
pub(crate) fn scoped_transport(builder: ContextBuilder) -> ContextBuilder {
    builder.with_json("namespace", crate::ZENOH_NAMESPACE)
}

//! Runs unmodified robot nodes against simulated sensors and the Booster transport.
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use bevy::prelude::Resource;
use booster::LowState;
use color_eyre::Result;
use coordinate_systems::{Ground, Robot};
use linear_algebra::Isometry3;
use projection::camera_matrix::CameraMatrix;
use ros_z::{parameter::NodeParameters, prelude::*, time::Time};
use tokio::{runtime::Handle, sync::watch, task::JoinSet};
use types::{
    filtered_game_controller_state::FilteredGameControllerState, motion_command::MotionCommand,
    time_wrapper::TimeWrapper,
};

use crate::{parameters::SimulatorParameters, robot_io::Observation};

#[derive(Clone)]
pub struct Configuration {
    pub parameter_root: PathBuf,
    pub model_directory: PathBuf,
    pub router: Option<String>,
    pub namespace: String,
}

#[derive(Resource, Clone)]
pub struct Robotics(Arc<RobotStack>);
impl std::ops::Deref for Robotics {
    type Target = RobotStack;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Robotics {
    pub async fn new(runtime: Handle, configuration: Configuration) -> Result<Self> {
        Ok(Self(Arc::new(
            RobotStack::new(runtime, configuration).await?,
        )))
    }
}

pub struct RobotStack {
    runtime: Handle,
    context: Arc<Context>,
    _node: Arc<Node>,
    // The set owns every task, including the SDK receiver, through shutdown and failed startup.
    tasks: Mutex<JoinSet<Result<()>>>,
    _overrides: tempfile::TempDir,
    pub parameters: NodeParameters<SimulatorParameters>,
    pub field_dimensions: types::field_dimensions::FieldDimensions,
    low_state: Publisher<LowState>,
    imu: Publisher<booster::ImuState>,
    pub primary: ros_z::cache::Cache<types::primary_state::PrimaryState>,
    pub safe_pose: ros_z::cache::Cache<bool>,
    pub field: ros_z::cache::Cache<types::field_dimensions::FieldDimensions>,
    serial: Publisher<kinematics::joints::Joints<booster::MotorState>>,
    camera: Publisher<TimeWrapper<CameraMatrix>>,
    ground: Publisher<TimeWrapper<Option<Isometry3<Ground, Robot>>>>,
    behavior: crate::behavior_inputs::BehaviorInputs,
    game: ros_z::cache::Cache<FilteredGameControllerState>,
    motion: ros_z::cache::Cache<MotionCommand>,
    emergency: ros_z::cache::Cache<()>,
    commands: watch::Receiver<crate::simulated_sdk::Control>,
    failure: Mutex<Option<String>>,
}

impl RobotStack {
    async fn new(runtime: Handle, configuration: Configuration) -> Result<Self> {
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
        let mut builder = ContextBuilder::default()
            .with_namespace(&configuration.namespace)
            .with_parameter_layers([
                configuration.parameter_root.join("base"),
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("parameters"),
                overrides.path().to_owned(),
                overrides.path().join("live"),
            ]);
        if let Some(router) = &configuration.router {
            builder = builder.with_router_endpoint(router)?;
        }
        // Wall time matches the unchanged cache timestamps and the external Game Controller.
        let context = Arc::new(scoped_transport(builder).build().await?);
        // ROS-Z applies environment overrides after builder options. Reject a
        // conflicting override before starting any node that can send SDK traffic.
        color_eyre::eyre::ensure!(
            context
                .session()
                .config()
                .get("namespace")
                .is_ok_and(
                    |value| serde_json::from_str::<String>(&value).ok().as_deref()
                        == Some(crate::ZENOH_NAMESPACE)
                ),
            "ZENOH_CONFIG_OVERRIDE conflicts with the simulator transport namespace"
        );
        let node = Arc::new(context.create_node("simulator").build().await?);
        let parameters = node.bind_parameter_as::<SimulatorParameters>("simulator")?;
        parameters.add_validation_hook(SimulatorParameters::validate)?;
        let latest = QosProfile {
            history: ros_z::qos::QosHistory::from_depth(1),
            ..Default::default()
        };
        let low_state = node
            .publisher("inputs/low_state")
            .qos(latest)
            .build()
            .await?;
        let imu = node.publisher("inputs/imu_state").build().await?;
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
        let field = node
            .subscriber("field_dimensions")
            .qos(QosProfile {
                durability: ros_z::qos::QosDurability::TransientLocal,
                ..Default::default()
            })
            .cache(1)
            .build()
            .await?;
        let serial = node
            .publisher("inputs/serial_motor_states")
            .qos(latest)
            .build()
            .await?;
        let camera = node.publisher("camera_matrix").qos(latest).build().await?;
        let ground = node
            .publisher("ground_to_robot")
            .qos(latest)
            .build()
            .await?;
        let behavior = crate::behavior_inputs::BehaviorInputs::new(&node).await?;
        let game = node
            .subscriber("filtered_game_controller_state")
            .cache(1)
            .build()
            .await?;
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
            sdk_task.await?;
            Ok(())
        });
        macro_rules! run { ($($node:path),* $(,)?) => { $(tasks.spawn($node(context.clone()));)* }; }
        run!(
            behavior_node::run_boxed,
            ball_state_composer::run_boxed,
            rule_obstacle_composer::run_boxed,
            motion::run_boxed,
            global_parameter_provider::run_boxed,
            head_motion::node::run_boxed,
            motion_inference::run_boxed,
            hardware_interface::run_boxed,
            fall_detection::run_boxed,
            safe_pose_checker::run_boxed,
            button_event_bridge::run_boxed,
            button_event_handler::run_boxed,
        );
        spawn_network(&context, &mut tasks);
        let field_dimensions =
            tokio::time::timeout(std::time::Duration::from_secs(3), initial_field.recv()).await??;
        crate::parameters::validate_field_dimensions(&field_dimensions)
            .map_err(|error| color_eyre::eyre::eyre!(error))?;
        Ok(Self {
            runtime,
            context,
            _node: node,
            tasks: Mutex::new(tasks),
            _overrides: overrides,
            parameters,
            field_dimensions,
            low_state,
            imu,
            primary,
            safe_pose,
            field,
            serial,
            camera,
            ground,
            behavior,
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
        if let Some(result) = tasks.try_join_next() {
            *failure = Some(match result {
                Ok(Ok(())) => "A simulator node exited".into(),
                Ok(Err(error)) => format!("Simulator node failed: {error:#}"),
                Err(error) => format!("Simulator task failed: {error}"),
            });
            tasks.abort_all();
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
        match self.game.latest_stamp() {
            None => "Waiting for HSL Game Controller",
            Some(stamp) if self.now().duration_since(stamp) > std::time::Duration::from_secs(3) => {
                "HSL Game Controller state is stale"
            }
            Some(_) => "HSL Game Controller connected",
        }
        .into()
    }

    pub fn actuator_control(&self) -> crate::simulated_sdk::Control {
        self.commands.borrow().clone()
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

    pub fn now(&self) -> Time {
        self.context.clock().now()
    }

    pub fn publish_observation(&self, observation: &Observation) -> Result<()> {
        let time = self.now();
        self.runtime.block_on(async {
            self.serial
                .publish(&observation.low_state.serial_motor_states()?)
                .await?;
            self.imu.publish(&observation.low_state.imu_state).await?;
            self.low_state.publish(&observation.low_state).await?;
            self.camera
                .publish(&TimeWrapper {
                    time,
                    inner: observation.camera_matrix.clone(),
                })
                .await?;
            self.ground
                .publish(&TimeWrapper {
                    time,
                    inner: Some(observation.ground_to_robot),
                })
                .await?;
            Ok(())
        })
    }

    pub fn publish_world(
        &self,
        ground: nalgebra::Isometry3<f32>,
        ball: Option<([f64; 3], [f64; 3])>,
    ) -> Result<()> {
        let side = self
            .game
            .get_latest()
            .map(|game| game.global_field_side)
            .unwrap_or(types::field_dimensions::GlobalFieldSide::Home);
        self.runtime.block_on(
            self.behavior
                .publish(ground, ball, Vec::new(), side, self.now()),
        )
    }
}

impl Drop for RobotStack {
    fn drop(&mut self) {
        let tasks = self.tasks.get_mut().unwrap();
        tasks.abort_all();
        self.runtime
            .block_on(async { while tasks.join_next().await.is_some() {} });
        if let Err(error) = self.context.shutdown() {
            log::error!("Simulator shutdown: {error:#}");
        }
    }
}

// Keep this list shared by production and the UDP integration test.
fn spawn_network(context: &Arc<Context>, tasks: &mut JoinSet<Result<()>>) {
    tasks.spawn(message_handler::run_boxed(context.clone()));
    tasks.spawn(message_filter::run_boxed(context.clone()));
    tasks.spawn(game_controller_filter::run_boxed(context.clone()));
    tasks.spawn(game_controller_state_filter::run_boxed(context.clone()));
    tasks.spawn(primary_state_filter::run_boxed(context.clone()));
    tasks.spawn(player_states_receiver::run_boxed(context.clone()));
    tasks.spawn(team_ball_filter::run_boxed(context.clone()));
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

pub(crate) fn scoped_transport(builder: ContextBuilder) -> ContextBuilder {
    builder.with_json("namespace", crate::ZENOH_NAMESPACE)
}

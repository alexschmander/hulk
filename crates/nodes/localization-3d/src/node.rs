use std::{
    future::{Future, pending},
    pin::Pin,
    sync::Arc,
    time::Duration,
};

use booster::ImuState;
use color_eyre::{
    Result,
    eyre::{Context as _, bail},
};
use coordinate_systems::{Field, Robot};
use linear_algebra::Isometry3;
use localization_factrs::initialize;
use projection::{camera_matrix::CameraMatrix, intrinsic::Intrinsic};
use ros_z::{
    cache::Cache,
    context::Context,
    parameter::NodeParametersExt,
    qos::{QosDurability, QosProfile, QosReliability},
};
use tokio::{select, task::JoinSet};
use types::{
    field_dimensions::FieldDimensions,
    localization::{LOCALIZATION_STATE_3D_TOPIC, LocalizationState3D},
    primary_state::PrimaryState,
    time_wrapper::TimeWrapper,
    visual_localization::{
        ASSOCIATION_GEOMETRY_TOPIC, AssociationGeometry, LOCALIZATION_POSE_3D_TOPIC,
        VISUAL_LOCALIZATION_TOPIC, VisualLocalizationFrame,
    },
    visual_odometry::{VisualOdometer, VisualOdometryDelta as VisualOdometryDeltaMessage},
};

use crate::{
    backend_task::spawn_backend_task,
    camera::{fresh_camera_matrix, intrinsic_from_camera_intrinsics},
    diagnostics::SolveDiagnostics,
    event_handlers::{
        AcceptanceWindow, accept_backend_result, handle_odometer_discontinuity,
        handle_visual_odometer, handle_visual_odometry, lose_track,
        tracking_deadline as next_tracking_deadline,
    },
    ingest::ingest_foot_heights,
    live_odometry::LiveVisualOdometryLocalization,
    parameters::{
        Localization3dParameters, backend_configuration_from_parameters_and_field_dimensions,
    },
    pose::{initial_robot_to_local_from_imu, initial_state_from_camera_matrix_and_imu},
    publish::LocalizationPublishers,
    visual_localization::{GlobalVisualLockTracker, handle_visual_localization_frame},
};

const VISUAL_ODOMETER_TOPIC: &str = "visual_odometry/current_left_camera_to_visual_odometer";

#[derive(Clone, Copy, PartialEq, Eq)]
enum IngestionPhase {
    Damping,
    AwaitingInitialImu,
    Running,
}

pub fn run_boxed(ctx: Arc<Context>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
    Box::pin(run(ctx))
}

pub async fn run(ctx: Arc<Context>) -> Result<()> {
    let node = ctx.create_node("localization3d").build().await?;
    let parameters = node.bind_parameter_as::<Localization3dParameters>("localization3d")?;
    parameters.add_validation_hook(Localization3dParameters::validate)?;
    let mut parameter_updates = parameters.subscribe();
    let imu_subscriber = node
        .subscriber::<ImuState>("inputs/imu_state")
        .build()
        .await?;
    let camera_matrix_cache = node
        .subscriber::<TimeWrapper<CameraMatrix>>("camera_matrix")
        .cache(128)
        .with_stamp(|message| message.time)
        .build()
        .await?;
    let field_dimensions_cache = node
        .subscriber::<FieldDimensions>("field_dimensions")
        .qos(QosProfile {
            durability: QosDurability::TransientLocal,
            ..Default::default()
        })
        .cache(1)
        .build()
        .await?;
    let visual_localization_subscriber = node
        .subscriber::<TimeWrapper<VisualLocalizationFrame>>(VISUAL_LOCALIZATION_TOPIC)
        .build()
        .await?;
    let visual_odometry_subscriber = node
        .subscriber::<VisualOdometryDeltaMessage>(
            "visual_odometry/current_left_camera_to_previous_left_camera",
        )
        .build()
        .await?;
    let visual_odometer_cache = node
        .subscriber::<VisualOdometer>(VISUAL_ODOMETER_TOPIC)
        .cache(128)
        .with_stamp(|message| message.time)
        .build()
        .await?;
    let visual_odometer_subscriber = node
        .subscriber::<VisualOdometer>(VISUAL_ODOMETER_TOPIC)
        .build()
        .await?;
    let robot_kinematics_subscriber = node
        .subscriber::<TimeWrapper<kinematics::robot_kinematics::RobotKinematics>>(
            "robot_kinematics",
        )
        .build()
        .await?;
    let robot_kinematics_cache = node
        .subscriber::<TimeWrapper<kinematics::robot_kinematics::RobotKinematics>>(
            "robot_kinematics",
        )
        .cache(128)
        .with_stamp(|message| message.time)
        .build()
        .await?;
    let primary_state_subscriber = node
        .subscriber::<PrimaryState>("primary_state")
        .qos(QosProfile {
            durability: QosDurability::TransientLocal,
            ..Default::default()
        })
        .build()
        .await?;

    let localization_publisher = node
        .publisher::<Option<Isometry3<Field, Robot>>>("localization")
        .build()
        .await?;
    let pose_3d_publisher = node
        .publisher::<TimeWrapper<Option<Isometry3<Field, Robot>>>>(LOCALIZATION_POSE_3D_TOPIC)
        .build()
        .await?;
    let state_3d_publisher = node
        .publisher::<LocalizationState3D>(LOCALIZATION_STATE_3D_TOPIC)
        .qos(QosProfile {
            durability: QosDurability::TransientLocal,
            ..Default::default()
        })
        .build()
        .await?;
    let association_geometry_publisher = node
        .publisher::<TimeWrapper<AssociationGeometry>>(ASSOCIATION_GEOMETRY_TOPIC)
        .build()
        .await?;
    let calibrated_intrinsics_publisher = node
        .publisher::<Intrinsic>("debug/calibrated_intrinsics")
        .build()
        .await?;
    let solve_diagnostics_publisher = node
        .publisher::<TimeWrapper<SolveDiagnostics>>("debug/solve_diagnostics")
        .qos(QosProfile {
            reliability: QosReliability::BestEffort,
            ..Default::default()
        })
        .build()
        .await?;

    let field_dimensions = wait_for_field_dimensions(&field_dimensions_cache).await;
    let (first_imu, initial_state, initial_robot_to_local) = loop {
        let imu = imu_subscriber.recv_with_metadata().await?;
        if let Some((initial_state, pose)) = prepare_initial_state(
            imu.source_time,
            &imu.message,
            &camera_matrix_cache,
            &robot_kinematics_cache,
        ) {
            break (imu, initial_state, pose);
        }
    };
    let first_imu_time = first_imu.source_time;
    let parameter_snapshot = parameter_updates.borrow_and_update().clone();
    let localization_parameters = parameter_snapshot.typed();
    let mut tracking_timeout = localization_parameters.tracking_timeout;
    let mut visual_tracking_timeout = localization_parameters.visual_tracking_timeout;
    let (mut frontend, backend) = initialize(
        backend_configuration_from_parameters_and_field_dimensions(
            localization_parameters,
            &field_dimensions,
        ),
        initial_state,
    );
    frontend.ingest_imu(first_imu_time.to_wallclock(), first_imu.message)?;
    // Dropping ingestion closes mailbox senders; idle publishing workers then exit.
    let mut output_tasks = JoinSet::new();
    let mut backend_handle = std::pin::pin!(spawn_backend_task(
        backend,
        solve_diagnostics_publisher,
        &mut output_tasks,
    ));
    let mut live = LiveVisualOdometryLocalization::default();
    live.set_initial(first_imu_time, initial_robot_to_local);
    let mut visual_lock = GlobalVisualLockTracker::default();
    visual_lock.invalidate(first_imu_time);
    let mut epoch = 0_u64;
    let mut epoch_start = first_imu_time;
    let mut tracking_state = LocalizationState3D::Startup;
    let mut tracking_deadline = None;
    let mut phase = IngestionPhase::Damping;
    let publishers = LocalizationPublishers::new(
        localization_publisher,
        pose_3d_publisher,
        state_3d_publisher,
        association_geometry_publisher,
        calibrated_intrinsics_publisher,
    )
    .spawn(&mut output_tasks);
    if let Some(geometry) = live.association_geometry(tracking_state, epoch) {
        publishers.publish_outputs(geometry);
    }

    loop {
        select! {
            changed = parameter_updates.changed() => {
                changed.wrap_err("localization parameter subscription closed")?;
                let snapshot = parameter_updates.borrow_and_update().clone();
                let updated = snapshot.typed();
                frontend.update_configuration(backend_configuration_from_parameters_and_field_dimensions(
                    updated, &field_dimensions,
                ))?;
                tracking_timeout = updated.tracking_timeout;
                visual_tracking_timeout = updated.visual_tracking_timeout;
                tracking_deadline = next_tracking_deadline(
                    tracking_state, &visual_lock, tracking_timeout, visual_tracking_timeout,
                );
            }
            _ = async {
                match tracking_deadline {
                    Some(deadline) => node.clock().sleep_until(deadline).await,
                    None => pending().await,
                }
            } => {
                tracking_deadline = None;
                tracking_state = lose_track(tracking_state, &mut visual_lock, node.clock().now());
                if let Some(geometry) = live.association_geometry(tracking_state, epoch) {
                    publishers.publish_outputs(geometry);
                }
            }
            primary_state = primary_state_subscriber.recv() => {
                let now_damping = primary_state? == PrimaryState::Damping;
                if now_damping && phase != IngestionPhase::Damping {
                    phase = IngestionPhase::Damping;
                    tracking_deadline = None;
                    tracking_state = LocalizationState3D::Startup;
                    visual_lock.invalidate(node.clock().now());
                    if let Some(mut geometry) = live.association_geometry(tracking_state, epoch) {
                        geometry.inner.local_to_field = None;
                        publishers.publish_outputs(geometry);
                    }
                    live.clear();
                } else if !now_damping && phase == IngestionPhase::Damping {
                    phase = IngestionPhase::AwaitingInitialImu;
                    epoch_start = node.clock().now();
                }
            }
            imu = imu_subscriber.recv_with_metadata() => {
                let imu = imu?;
                if phase == IngestionPhase::Damping { continue; }
                if imu.source_time < epoch_start || imu.source_time > node.clock().now() { continue; }
                if phase == IngestionPhase::AwaitingInitialImu {
                    let time = imu.source_time;
                    let Some((initial_state, robot_to_local)) = prepare_initial_state(
                        time, &imu.message, &camera_matrix_cache, &robot_kinematics_cache,
                    ) else { continue };
                    epoch = epoch.wrapping_add(1);
                    frontend.reset(time.to_wallclock(), epoch, initial_state)?;
                    frontend.ingest_imu(time.to_wallclock(), imu.message)?;
                    epoch_start = time;
                    live.set_initial(time, robot_to_local);
                    visual_lock.invalidate(time);
                    tracking_state = LocalizationState3D::Startup;
                    phase = IngestionPhase::Running;
                    if let Some(geometry) = live.association_geometry(tracking_state, epoch) {
                        publishers.publish_outputs(geometry);
                    }
                } else {
                    frontend.ingest_imu(imu.source_time.to_wallclock(), imu.message)?;
                }
            }
            frame = visual_localization_subscriber.recv() => {
                if phase != IngestionPhase::Running { continue; }
                let frame = frame?;
                let now = node.clock().now();
                handle_visual_localization_frame(&mut frontend, &mut visual_lock, epoch, tracking_state, now, visual_tracking_timeout, frame)?;
            }
            odometry = visual_odometry_subscriber.recv() => {
                if phase != IngestionPhase::Running { continue; }
                let odometry = odometry?;
                if odometry.previous_time < epoch_start || odometry.current_time > node.clock().now() { continue; }
                handle_visual_odometry(&mut frontend, odometry, &camera_matrix_cache)?;
            }
            odometer = visual_odometer_subscriber.recv() => {
                if phase != IngestionPhase::Running { continue; }
                let odometer = odometer?;
                if odometer.time < epoch_start || odometer.time > node.clock().now() { continue; }
                handle_odometer_discontinuity(&live, &mut tracking_state, &mut visual_lock, &odometer, node.clock().now());
                handle_visual_odometer(&mut live, tracking_state, epoch, odometer, &visual_odometer_cache, &camera_matrix_cache, &publishers);
                tracking_deadline = next_tracking_deadline(
                    tracking_state, &visual_lock, tracking_timeout, visual_tracking_timeout,
                );
            }
            kinematics = robot_kinematics_subscriber.recv() => {
                if phase != IngestionPhase::Running { continue; }
                let kinematics = kinematics?;
                if kinematics.time < epoch_start || kinematics.time > node.clock().now() { continue; }
                ingest_foot_heights(&mut frontend, kinematics)?;
            }
            result = &mut backend_handle => {
                result.wrap_err("failed to join")?.wrap_err("solver failed")?;
                bail!("solver stopped unexpectedly")
            }
            result = output_tasks.join_next() => {
                result.expect("output workers remain running").wrap_err("failed to join output worker")??;
                bail!("output worker stopped unexpectedly")
            }
            result = frontend.wait_for_optimization_result() => {
                result?;
                if phase != IngestionPhase::Running { let _ = frontend.last_optimization_result(); continue; }
                let window = AcceptanceWindow {
                    epoch, epoch_start, now: node.clock().now(),
                    tracking_timeout, visual_tracking_timeout,
                };
                if let Some(result) = frontend.last_optimization_result()
                    && let Some(acceptance) = accept_backend_result(
                        &result, tracking_state, &mut visual_lock, window,
                    )
                {
                    publishers.publish_intrinsics(intrinsic_from_camera_intrinsics(&result.camera_intrinsics));
                    let changed_state = acceptance.state != tracking_state;
                    tracking_state = acceptance.state;
                    if let Some(correction) = acceptance.correction {
                        live.reset(correction, &visual_odometer_cache, &camera_matrix_cache);
                    }
                    if (changed_state || acceptance.correction.is_some())
                        && let Some(geometry) = live.association_geometry(tracking_state, epoch)
                    {
                        publishers.publish_outputs(geometry);
                    }
                }
                tracking_deadline = next_tracking_deadline(
                    tracking_state, &visual_lock, tracking_timeout, visual_tracking_timeout,
                );
            }
        }
    }
}

fn prepare_initial_state(
    time: ros_z::time::Time,
    imu: &ImuState,
    cameras: &Cache<TimeWrapper<CameraMatrix>>,
    kinematics: &Cache<TimeWrapper<kinematics::robot_kinematics::RobotKinematics>>,
) -> Option<(
    localization_factrs::InitialState,
    Isometry3<Robot, coordinate_systems::Local>,
)> {
    let camera = fresh_camera_matrix(cameras, time)?;
    let kinematics = kinematics.get_nearest(time)?;
    if time.abs_diff(kinematics.time) > Duration::from_millis(100) {
        return None;
    }
    Some((
        initial_state_from_camera_matrix_and_imu(&camera.inner, imu, &kinematics.inner),
        initial_robot_to_local_from_imu(imu, &kinematics.inner),
    ))
}

async fn wait_for_field_dimensions(cache: &Cache<FieldDimensions>) -> FieldDimensions {
    loop {
        if let Some(value) = cache.get_latest() {
            return *value;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_starts_without_a_global_visual_lock() {
        let tracker = GlobalVisualLockTracker::default();
        assert_eq!(tracker.status(), crate::GlobalVisualLock::Unlocked);
        assert!(!tracker.has_backend_result());
    }
}

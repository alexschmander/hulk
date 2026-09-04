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
    qos::{QosDurability, QosProfile},
};
use tokio::select;
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
    diagnostics::SolveDiagnostics,
    event_handlers::{
        handle_optimization_result, handle_visual_odometer, handle_visual_odometry, lose_track,
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

pub fn run_boxed(ctx: Arc<Context>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
    Box::pin(run(ctx))
}

pub async fn run(ctx: Arc<Context>) -> Result<()> {
    let node = ctx.create_node("localization3d").build().await?;
    let parameters = node.bind_parameter_as::<Localization3dParameters>("localization3d")?;
    parameters.add_validation_hook(Localization3dParameters::validate)?;
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
        .build()
        .await?;

    let field_dimensions = wait_for_field_dimensions(&field_dimensions_cache).await;
    let camera_matrix = wait_for_camera_matrix(&camera_matrix_cache).await;
    let first_imu = imu_subscriber.recv_with_metadata().await?;
    let first_imu_time = first_imu.source_time;
    let initial_state =
        initial_state_from_camera_matrix_and_imu(&camera_matrix.inner, &first_imu.message);
    let initial_robot_to_local = initial_robot_to_local_from_imu(&first_imu.message);
    let localization_parameters = parameters.snapshot().typed().clone();
    let tracking_timeout = localization_parameters.tracking_timeout;
    let (mut frontend, backend) = initialize(
        backend_configuration_from_parameters_and_field_dimensions(
            &localization_parameters,
            &field_dimensions,
        ),
        initial_state,
    );
    frontend.ingest_imu(first_imu_time.to_wallclock(), first_imu.message)?;
    let mut backend_handle =
        std::pin::pin!(spawn_backend_task(backend, solve_diagnostics_publisher));
    let mut live = LiveVisualOdometryLocalization::default();
    live.set_initial(initial_robot_to_local);
    let mut visual_lock = GlobalVisualLockTracker::default();
    let mut epoch = 0_u64;
    let mut epoch_start = first_imu_time;
    let mut tracking_state = LocalizationState3D::Startup;
    let mut tracking_deadline = None;
    let mut damping = true;
    let mut reset_after_damping = true;
    let publishers = LocalizationPublishers::new(
        &localization_publisher,
        &pose_3d_publisher,
        &state_3d_publisher,
        &association_geometry_publisher,
    );
    publishers
        .publish_outputs(
            first_imu_time,
            epoch,
            initial_robot_to_local,
            None,
            None,
            tracking_state,
            types::visual_localization::AssociationPoseHintSource::StartupPrior,
        )
        .await?;

    loop {
        select! {
            biased;
            _ = async {
                match tracking_deadline {
                    Some(deadline) => node.clock().sleep_until(deadline).await,
                    None => pending().await,
                }
            } => {
                let deadline = tracking_deadline.take().expect("deadline branch requires a deadline");
                tracking_state = lose_track(tracking_state);
                if let Some((robot_to_local, local_to_field)) = live.latest() {
                    publishers.publish_outputs(
                        deadline, epoch, robot_to_local, local_to_field, None, tracking_state,
                        types::visual_localization::AssociationPoseHintSource::LiveVisualOdometry,
                    ).await?;
                }
            }
            primary_state = primary_state_subscriber.recv() => {
                let now_damping = primary_state? == PrimaryState::Damping;
                if now_damping && !damping {
                    let last_pose = live.latest();
                    damping = true;
                    reset_after_damping = true;
                    tracking_deadline = None;
                    tracking_state = LocalizationState3D::Startup;
                    live.clear();
                    visual_lock.reset_for_damping();
                    if let Some((robot_to_local, local_to_field)) = last_pose {
                        publishers.publish_outputs(
                            node.clock().now(), epoch, robot_to_local, local_to_field, None, tracking_state,
                            types::visual_localization::AssociationPoseHintSource::LiveVisualOdometry,
                        ).await?;
                    }
                } else if !now_damping && damping {
                    damping = false;
                }
            }
            imu = imu_subscriber.recv_with_metadata() => {
                let imu = imu?;
                if damping { continue; }
                if reset_after_damping {
                    let Some(camera) = camera_matrix_cache.get_latest() else { continue };
                    let time = imu.source_time;
                    let robot_to_local = initial_robot_to_local_from_imu(&imu.message);
                    epoch = epoch.wrapping_add(1);
                    frontend.reset(time.to_wallclock(), epoch, initial_state_from_camera_matrix_and_imu(&camera.inner, &imu.message))?;
                    frontend.ingest_imu(time.to_wallclock(), imu.message)?;
                    epoch_start = time;
                    live.set_initial(robot_to_local);
                    visual_lock.reset_for_damping();
                    tracking_state = LocalizationState3D::Startup;
                    reset_after_damping = false;
                    publishers.publish_outputs(
                        time, epoch, robot_to_local, None, None, tracking_state,
                        types::visual_localization::AssociationPoseHintSource::StartupPrior,
                    ).await?;
                } else {
                    frontend.ingest_imu(imu.source_time.to_wallclock(), imu.message)?;
                }
            }
            frame = visual_localization_subscriber.recv() => {
                if damping || reset_after_damping { continue; }
                handle_visual_localization_frame(&mut frontend, &mut visual_lock, epoch, frame?)?;
            }
            odometry = visual_odometry_subscriber.recv() => {
                if damping || reset_after_damping { continue; }
                handle_visual_odometry(&mut frontend, odometry?, &camera_matrix_cache)?;
            }
            odometer = visual_odometer_subscriber.recv() => {
                if damping || reset_after_damping { continue; }
                handle_visual_odometer(&mut live, tracking_state, epoch, odometer?, &visual_odometer_cache, &camera_matrix_cache, publishers).await?;
            }
            kinematics = robot_kinematics_subscriber.recv() => {
                if damping || reset_after_damping { continue; }
                ingest_foot_heights(&mut frontend, kinematics?)?;
            }
            result = &mut backend_handle => {
                result.wrap_err("failed to join")?.wrap_err("solver failed")?;
                bail!("solver stopped unexpectedly")
            }
            result = frontend.wait_for_optimization_result() => {
                result?;
                if damping || reset_after_damping { let _ = frontend.last_optimization_result(); continue; }
                if let Some(time) = handle_optimization_result(
                    &mut frontend, &mut live, &mut visual_lock, epoch, epoch_start,
                    node.clock().now(), tracking_timeout, &mut tracking_state,
                    &visual_odometer_cache, &camera_matrix_cache, publishers, &calibrated_intrinsics_publisher,
                ).await? {
                    tracking_deadline = Some(time.saturating_add(tracking_timeout));
                }
            }
        }
    }
}

async fn wait_for_camera_matrix(
    cache: &Cache<TimeWrapper<CameraMatrix>>,
) -> Arc<TimeWrapper<CameraMatrix>> {
    loop {
        if let Some(value) = cache.get_latest() {
            return value;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
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

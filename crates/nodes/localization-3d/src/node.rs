use crate::{
    Localization, Localization3dParameters, SolveDiagnostics,
    camera::fresh_camera_matrix,
    inputs::{Inputs, Measurement},
    pose::initial_robot_to_local_from_imu,
};
use color_eyre::{Result, eyre::Context as _};
use ros_z::{
    context::Context,
    parameter::NodeParametersExt,
    qos::{QosDurability, QosHistory, QosProfile},
};
use std::{
    future::{Future, pending},
    num::NonZeroUsize,
    pin::Pin,
    sync::Arc,
    time::Duration,
};
use types::{
    field_dimensions::FieldDimensions,
    localization::{
        LOCALIZATION_ESTIMATE_TOPIC, LOCALIZATION_STATUS_TOPIC, LocalizationEstimate,
        LocalizationState, LocalizationStatus,
    },
    primary_state::PrimaryState,
};

pub fn run_boxed(ctx: Arc<Context>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
    Box::pin(run(ctx))
}

pub async fn run(ctx: Arc<Context>) -> Result<()> {
    let node = ctx.create_node("localization3d").build().await?;
    let parameters = node.bind_parameter_as::<Localization3dParameters>("localization3d")?;
    parameters.add_validation_hook(Localization3dParameters::validate)?;
    let mut updates = parameters.subscribe();
    let inputs = Inputs::new(&node).await?;
    let latched = QosProfile {
        durability: QosDurability::TransientLocal,
        history: QosHistory::KeepLast(NonZeroUsize::MIN),
        ..Default::default()
    };
    let primary = node
        .subscriber::<PrimaryState>("primary_state")
        .qos(latched)
        .build()
        .await?;
    let field = node
        .subscriber::<FieldDimensions>("field_dimensions")
        .qos(latched)
        .cache(1)
        .build()
        .await?;
    let estimates = node
        .publisher::<LocalizationEstimate>(LOCALIZATION_ESTIMATE_TOPIC)
        .build()
        .await?;
    let statuses = node
        .publisher::<LocalizationStatus>(LOCALIZATION_STATUS_TOPIC)
        .qos(latched)
        .build()
        .await?;
    let diagnostics = node
        .publisher::<SolveDiagnostics>("debug/solve_diagnostics")
        .build()
        .await?;
    let mut status = LocalizationStatus {
        time: node.clock().now(),
        epoch: 0,
        state: LocalizationState::Startup,
    };
    statuses.publish(&status).await?;
    let mut active: Option<Localization> = None;
    let mut damping = true;
    let mut epoch_start = status.time;
    loop {
        let deadline = active.as_ref().and_then(Localization::deadline);
        let first = tokio::select! {
            biased;
            v = primary.recv() => {
                let next_damping = v? == PrimaryState::Damping;
                if damping && !next_damping {
                    epoch_start = node.clock().now();
                    active = None;
                    status = LocalizationStatus { time: epoch_start, epoch: status.epoch.wrapping_add(1), state: LocalizationState::Startup };
                    statuses.publish(&status).await?;
                }
                damping = next_damping;
                continue;
            }
            changed = updates.changed() => {
                changed.wrap_err("localization parameters closed")?;
                let snapshot = updates.borrow_and_update().clone();
                if let Some(active) = active.as_mut() { active.set_parameters(snapshot.typed())?; }
                continue;
            }
            _ = async { match deadline { Some(t) => node.clock().sleep_until(t).await, None => pending().await } } => {
                if let Some(active) = active.as_mut() {
                    active.advance_time(node.clock().now());
                    if status != active.status() { status = active.status(); statuses.publish(&status).await?; }
                }
                continue;
            }
            sample = inputs.recv() => sample?,
        };
        let samples = inputs.drain(first).await?;
        let now = node.clock().now();
        let mut inserted = false;
        for sample in samples {
            let time = sample.time();
            if time < epoch_start || time > now {
                tracing::warn!(
                    ?time,
                    "discarding measurement outside current epoch or in the future"
                );
                continue;
            }
            if active
                .as_ref()
                .is_some_and(|localization| localization.has_measurement_gap(time))
            {
                tracing::warn!(
                    ?time,
                    "measurement gap exceeds window; resetting localization epoch"
                );
                active = None;
                epoch_start = time;
                status = LocalizationStatus {
                    time,
                    epoch: status.epoch.wrapping_add(1),
                    state: LocalizationState::Startup,
                };
                statuses.publish(&status).await?;
            }
            if active.is_none() {
                let Measurement::Imu(_, imu) = &sample else {
                    continue;
                };
                let Some(camera) = fresh_camera_matrix(&inputs.cameras, time) else {
                    continue;
                };
                let Some(kinematics) = inputs.robot_kinematics.get_nearest(time) else {
                    continue;
                };
                if kinematics.time.abs_diff(time) > Duration::from_millis(100) {
                    continue;
                }
                let Some(field) = field.get_latest() else {
                    continue;
                };
                let initialized = Localization::new(
                    time,
                    status.epoch,
                    updates.borrow().typed(),
                    &field,
                    &camera.inner,
                    initial_robot_to_local_from_imu(imu, &kinematics.inner),
                );
                match initialized {
                    Ok(localization) => active = Some(localization),
                    Err(error) => {
                        tracing::warn!(%error, "discarding invalid initialization sample");
                        continue;
                    }
                }
                status = active.as_ref().unwrap().status();
                statuses.publish(&status).await?;
            }
            inserted |= inputs.ingest(active.as_mut().unwrap(), sample)?;
        }
        if !inserted {
            continue;
        }
        let localization = active.as_mut().unwrap();
        let output = tokio::task::block_in_place(|| localization.solve(node.clock().now()));
        // Advance lifecycle after the potentially expensive solve, before publishing.
        localization.advance_time(node.clock().now());
        if status != localization.status() {
            status = localization.status();
            statuses.publish(&status).await?;
        }
        if let Some(estimate) = output.estimate {
            estimates.publish(&estimate).await?;
        }
        if let Some(error) = &output.diagnostics.failure {
            tracing::warn!(%error, "localization solve failed");
        }
        diagnostics.publish(&output.diagnostics).await?;
    }
}

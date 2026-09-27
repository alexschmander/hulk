use std::{boxed::Box, future::Future, pin::Pin, sync::Arc, time::Duration};

use color_eyre::Result;
use coordinate_systems::{Field, Ground, Robot};
use linear_algebra::{IntoTransform, Isometry2, Isometry3};
use ros_z::{
    cache::Cache,
    context::Context,
    qos::{QosDurability, QosProfile},
    time::Time,
};
use types::{
    localization::{
        LOCALIZATION_ESTIMATE_TOPIC, LOCALIZATION_STATUS_TOPIC, LocalizationEstimate,
        LocalizationStatus, ground_to_field_from_field_to_robot,
    },
    time_wrapper::TimeWrapper,
};

pub fn run_boxed(ctx: Arc<Context>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
    Box::pin(run(ctx))
}

const MAX_ROBOT_TO_GROUND_TIME_DISTANCE: Duration = Duration::from_millis(100);

pub async fn run(ctx: Arc<Context>) -> Result<()> {
    let node = ctx.create_node("localization2d").build().await?;

    let localization_subscriber = node
        .subscriber::<LocalizationEstimate>(LOCALIZATION_ESTIMATE_TOPIC)
        .build()
        .await?;
    let robot_to_ground_cache = node
        .subscriber::<TimeWrapper<Option<Isometry3<Robot, Ground>>>>("robot_to_ground")
        .cache(128)
        .with_stamp(|wrapper| wrapper.time)
        .build()
        .await?;
    let status = node
        .subscriber::<LocalizationStatus>(LOCALIZATION_STATUS_TOPIC)
        .qos(QosProfile {
            durability: QosDurability::TransientLocal,
            ..Default::default()
        })
        .cache(1)
        .build()
        .await?;

    let ground_to_field_publisher = node
        .publisher::<Isometry2<Ground, Field>>("ground_to_field")
        .build()
        .await?;

    loop {
        let localization = localization_subscriber.recv().await?;
        if status.get_latest().is_none_or(|status| {
            status.epoch != localization.epoch || status.generation != localization.generation
        }) {
            continue;
        }
        let time = localization.time;
        let Some(robot_to_field) = localization.robot_to_field else {
            continue;
        };

        let Some(robot_to_ground) = fresh_robot_to_ground_at(&robot_to_ground_cache, time) else {
            continue;
        };

        ground_to_field_publisher
            .publish(&ground_to_field_from_field_to_robot(
                robot_to_field
                    .pose
                    .inner
                    .cast()
                    .inverse()
                    .framed_transform(),
                robot_to_ground,
            ))
            .await?;
    }
}

fn fresh_robot_to_ground_at(
    robot_to_ground_cache: &Cache<TimeWrapper<Option<Isometry3<Robot, Ground>>>>,
    time: Time,
) -> Option<Isometry3<Robot, Ground>> {
    robot_to_ground_cache
        .get_nearest_with_stamp(time)
        .filter(|(stamp, _)| time_distance(*stamp, time) <= MAX_ROBOT_TO_GROUND_TIME_DISTANCE)
        .and_then(|(_, transform)| transform.inner)
}

fn time_distance(a: Time, b: Time) -> Duration {
    Duration::from_nanos(a.as_nanos().abs_diff(b.as_nanos()))
}

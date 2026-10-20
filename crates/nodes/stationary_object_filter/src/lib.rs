//! Local map of stationary objects: goal posts and field features.
//!
//! Tracks live in the continuous odometry frame as static Bernoulli tracks with a soft class
//! belief. Their uncertainty grows only with the drift of odometry.

use std::{future::Future, pin::Pin, sync::Arc, time::Duration};

use color_eyre::Result;
use coordinate_systems::{Field, Ground, Odometry};
use linear_algebra::{Isometry2, Pose2};
use projection::camera_matrix::CameraMatrix;
use ros_z::{cache::Cache, prelude::*, qos::QosDurability, time::Time};
use ros_z_streams::CreateFutureMapBuilder;
use tokio::task::block_in_place;
use types::{
    object_detection::{Object, RobocupObjectLabel},
    obstacles::Obstacle,
    primary_state::PrimaryState,
    time_wrapper::TimeWrapper,
    tracked_objects::{StationaryTrack, TrackedObjects},
};

pub use crate::{
    filter::StationaryObjectFilter,
    parameters::Parameters,
    processor::{Inputs, Outputs, Processor},
};
use object_tracking::multi_bernoulli::ScanReport;

mod filter;
mod parameters;
mod processor;

struct CachedInputs {
    camera_matrix: Cache<TimeWrapper<CameraMatrix>>,
    primary_state: Cache<PrimaryState>,
}

impl Inputs for CachedInputs {
    fn primary_state(&self) -> Option<PrimaryState> {
        self.primary_state.get_latest().map(|state| *state)
    }

    fn camera_matrix_nearest(&self, time: Time) -> Option<TimeWrapper<CameraMatrix>> {
        self.camera_matrix
            .get_nearest(time)
            .map(|camera_matrix| camera_matrix.as_ref().clone())
    }
}

pub fn run_boxed(ctx: Arc<Context>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
    Box::pin(run(ctx))
}

pub async fn run(ctx: Arc<Context>) -> Result<()> {
    let node = ctx.create_node("stationary_object_filter").build().await?;

    let parameters = node.bind_parameter_as::<Parameters>("stationary_object_filter")?;
    let camera_matrix_cache = node
        .subscriber::<TimeWrapper<CameraMatrix>>("camera_matrix")
        .cache(200)
        .with_stamp(|wrapper: &TimeWrapper<CameraMatrix>| wrapper.time)
        .build()
        .await?;
    // Localization stamps its estimates with the time of their image, which only the source time
    // of a publication carries.
    let ground_to_field_subscriber = node
        .subscriber::<Isometry2<Ground, Field>>("ground_to_field")
        .build()
        .await?;
    let primary_state_cache = node
        .subscriber::<PrimaryState>("primary_state")
        .qos(QosProfile {
            durability: QosDurability::TransientLocal,
            ..Default::default()
        })
        .cache(1)
        .build()
        .await?;
    let mut future_map = node
        .create_future_map_builder()
        .create_future_subscriber::<Pose2<Odometry>>("inputs/odometry", Duration::from_millis(1))
        .await?
        .create_future_subscriber::<TimeWrapper<Vec<Object<RobocupObjectLabel>>>>(
            "detected_objects",
            Duration::from_millis(25),
        )
        .await?
        .build();

    let tracks_pub = node
        .publisher::<TrackedObjects<StationaryTrack>>("stationary_object_filter/tracks")
        .build()
        .await?;
    let obstacles_pub = node
        .publisher::<Vec<Obstacle>>("stationary_object_filter/obstacles")
        .build()
        .await?;
    let scan_report_pub = node
        .publisher::<ScanReport>("stationary_object_filter/scan_report")
        .build()
        .await?;

    let inputs = CachedInputs {
        camera_matrix: camera_matrix_cache,
        primary_state: primary_state_cache,
    };
    let mut processor = Processor::default();

    loop {
        tokio::select! {
            ground_to_field = ground_to_field_subscriber.recv_with_metadata() => {
                let ground_to_field = ground_to_field?;
                processor.receive_ground_to_field(ground_to_field.source_time, ground_to_field.message);
            }
            item = future_map.recv() => {
                let item = item?;
                let parameters_snapshot = parameters.snapshot();
                let parameters = parameters_snapshot.typed();

                let outputs =
                    block_in_place(|| processor.process(item.persistent, &inputs, parameters));
                let Some(Outputs {
                    reports,
                    tracks,
                    obstacles,
                }) = outputs
                else {
                    continue;
                };
                for report in &reports {
                    scan_report_pub.publish(report).await?;
                }
                if let Some(tracks) = tracks {
                    tracks_pub.publish(&tracks).await?;
                }
                obstacles_pub.publish(&obstacles).await?;
            }
        }
    }
}

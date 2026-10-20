//! Tracks robots and humans as labeled multi-Bernoulli tracks in the odometry frame.
//!
//! Robots are published as opponents unless team messages associate them with a teammate. Team
//! messages enter through our localization with a covariance covering both localizations and can
//! start remote-only tracks for teammates the camera has not seen.

use std::{future::Future, pin::Pin, sync::Arc, time::Duration};

use color_eyre::Result;
use coordinate_systems::{Field, Ground, Odometry};
use hsl_network_messages::PlayerNumber;
use linear_algebra::{Isometry2, Pose2};
use object_tracking::multi_bernoulli::ScanReport;
use projection::camera_matrix::CameraMatrix;
use ros_z::{cache::Cache, prelude::*, qos::QosDurability, time::Time};
use ros_z_streams::CreateFutureMapBuilder;
use tokio::task::block_in_place;
use types::{
    game_controller_state::GameControllerState,
    object_detection::{Object, RobocupObjectLabel},
    obstacles::Obstacle,
    players::Players,
    primary_state::PrimaryState,
    time_wrapper::TimeWrapper,
    tracked_objects::{HumanoidTrack, TrackedObjects},
    world_state::PlayerState,
};

pub use crate::{
    filter::{HumanoidFilter, team_report},
    parameters::Parameters,
    processor::{Inputs, Outputs, Processor},
};

mod filter;
mod parameters;
mod processor;
#[cfg(test)]
mod tests;

struct CachedInputs {
    camera_matrix: Cache<TimeWrapper<CameraMatrix>>,
    player_number: Cache<PlayerNumber>,
    primary_state: Cache<PrimaryState>,
    game_controller_state: Cache<Option<GameControllerState>>,
}

impl Inputs for CachedInputs {
    fn primary_state(&self) -> Option<PrimaryState> {
        self.primary_state.get_latest().map(|state| *state)
    }

    fn game_controller_state(&self) -> Option<GameControllerState> {
        self.game_controller_state
            .get_latest()
            .and_then(|state| state.as_ref().clone())
    }

    fn player_number(&self) -> Option<PlayerNumber> {
        self.player_number.get_latest().map(|number| *number)
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
    let node = ctx.create_node("humanoid_filter").build().await?;

    let parameters = node.bind_parameter_as::<Parameters>("humanoid_filter")?;
    let transient_local = || QosProfile {
        durability: QosDurability::TransientLocal,
        ..Default::default()
    };
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
    let player_number_cache = node
        .subscriber::<PlayerNumber>("player_number")
        .qos(transient_local())
        .cache(1)
        .build()
        .await?;
    let primary_state_cache = node
        .subscriber::<PrimaryState>("primary_state")
        .qos(transient_local())
        .cache(1)
        .build()
        .await?;
    let game_controller_state_cache = node
        .subscriber::<Option<GameControllerState>>("game_controller_state")
        .cache(1)
        .build()
        .await?;
    let player_states_subscriber = node
        .subscriber::<Players<Option<TimeWrapper<PlayerState>>>>("player_states")
        .qos(transient_local())
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
        .publisher::<TrackedObjects<HumanoidTrack>>("humanoid_filter/tracks")
        .build()
        .await?;
    let obstacles_pub = node
        .publisher::<Vec<Obstacle>>("humanoid_filter/obstacles")
        .build()
        .await?;
    let scan_report_pub = node
        .publisher::<ScanReport>("humanoid_filter/scan_report")
        .build()
        .await?;

    let inputs = CachedInputs {
        camera_matrix: camera_matrix_cache,
        player_number: player_number_cache,
        primary_state: primary_state_cache,
        game_controller_state: game_controller_state_cache,
    };
    let mut processor = Processor::default();

    loop {
        tokio::select! {
            ground_to_field = ground_to_field_subscriber.recv_with_metadata() => {
                let ground_to_field = ground_to_field?;
                processor.receive_ground_to_field(ground_to_field.source_time, ground_to_field.message);
            }
            player_states = player_states_subscriber.recv() => {
                let player_states = player_states?;
                processor.receive_player_states(&player_states, inputs.player_number());
            }
            item = future_map.recv() => {
                let item = item?;
                let parameters_snapshot = parameters.snapshot();
                let parameters = parameters_snapshot.typed();

                let outputs =
                    block_in_place(|| processor.process(item.persistent, &inputs, parameters));
                let Some(Outputs { reports, tracks, obstacles }) = outputs else {
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

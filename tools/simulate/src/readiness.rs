//! Startup is driven by outputs from the selected pipeline, after physics starts.
use crate::profiles::Profile;
use color_eyre::Result;
use coordinate_systems::{Ground, Odometry, Robot};
use linear_algebra::{Isometry3, Pose2};
use ros_z::{Message, prelude::*, time::Time};
use std::time::Duration;
use types::{ball_position::BallPosition, obstacles::Obstacle, time_wrapper::TimeWrapper};

struct Probe {
    name: &'static str,
    ready: Box<dyn Fn(Time) -> bool + Send + Sync>,
}
impl Probe {
    async fn new<T: Message>(
        node: &Node,
        name: &'static str,
        valid: impl Fn(&T) -> bool + Send + Sync + 'static,
    ) -> Result<Self> {
        let cache = node.subscriber::<T>(name).cache(1).build().await?;
        Ok(Self {
            name,
            ready: Box::new(move |time| {
                cache
                    .latest_stamp()
                    .is_some_and(|stamp| time.duration_since(stamp) < Duration::from_secs(2))
                    && cache.get_latest().is_some_and(|value| valid(&value))
            }),
        })
    }
}
pub(crate) struct Readiness {
    required: Vec<Probe>,
    pose: Option<Probe>,
}
impl Readiness {
    pub async fn new(node: &Node, profile: Profile) -> Result<Self> {
        let mut required = vec![
            Probe::new::<booster::LowState>(node, "inputs/low_state", |_| true).await?,
            Probe::new::<types::motion_command::MotionCommand>(
                node,
                "behavior/motion_command",
                |_| true,
            )
            .await?,
            Probe::new::<types::filtered_whistle::FilteredWhistle>(
                node,
                "filtered_whistle",
                |_| true,
            )
            .await?,
        ];
        if profile.filtering() {
            required.extend([
                Probe::new::<Option<BallPosition<Ground>>>(
                    node,
                    "ball_filter/ball_position",
                    |_| true,
                )
                .await?,
                Probe::new::<Option<BallPosition<Ground>>>(
                    node,
                    "visual_kick/ball_position",
                    |_| true,
                )
                .await?,
                Probe::new::<Vec<Obstacle>>(node, "obstacles", |_| true).await?,
            ]);
        }
        if profile.body_state() {
            required.extend([
                Probe::new::<TimeWrapper<Option<Isometry3<Robot, Ground>>>>(
                    node,
                    "robot_to_ground",
                    |v| v.inner.is_some(),
                )
                .await?,
                Probe::new::<TimeWrapper<projection::camera_matrix::CameraMatrix>>(
                    node,
                    "camera_matrix",
                    |_| true,
                )
                .await?,
                Probe::new::<Pose2<Odometry>>(node, "inputs/odometry", |_| true).await?,
            ]);
        }
        let pose = if profile.localization() {
            required.push(
                Probe::new::<TimeWrapper<types::visual_localization::VisualLocalizationFrame>>(
                    node,
                    "field_mark_association/visual_localization_local",
                    |_| true,
                )
                .await?,
            );
            Some(
                Probe::new::<types::localization::LocalizationEstimate>(
                    node,
                    "localization/estimate",
                    |estimate| estimate.robot_to_field.is_some(),
                )
                .await?,
            )
        } else {
            None
        };
        Ok(Self { required, pose })
    }
    pub fn ready(&self, now: Time) -> bool {
        self.required.iter().all(|probe| (probe.ready)(now))
    }
    pub fn status(&self, now: Time) -> String {
        let missing: Vec<_> = self
            .required
            .iter()
            .filter(|probe| !(probe.ready)(now))
            .map(|probe| probe.name)
            .collect();
        if !missing.is_empty() {
            return format!("Waiting for {}", missing.join(", "));
        }
        if self.pose.as_ref().is_some_and(|pose| !(pose.ready)(now)) {
            return "Localization acquiring".into();
        }
        if self.pose.is_some() {
            "Localization acquired".into()
        } else {
            "Ready".into()
        }
    }
}

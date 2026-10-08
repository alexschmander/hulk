//! Truth remains observable even when a real node owns the production topic.
use color_eyre::Result;
use ros_z::{Message, prelude::*, time::Time};
use ros_z_streams::{AnnouncingPublisher, CreateAnnouncingPublisher};

pub(crate) struct Reference<T: Message> {
    truth: Publisher<T>,
    production: Option<Publisher<T>>,
}
impl<T: Message> Reference<T> {
    pub async fn new(node: &Node, topic: &str, substitute: bool) -> Result<Self> {
        Ok(Self {
            truth: node
                .publisher(&format!("ground_truth/{topic}"))
                .build()
                .await?,
            production: if substitute {
                Some(node.publisher(topic).build().await?)
            } else {
                None
            },
        })
    }
    pub async fn publish(&self, value: &T, time: Time) -> Result<()> {
        self.publish_observed(value, value, time).await
    }
    pub async fn publish_observed(&self, truth: &T, observed: &T, time: Time) -> Result<()> {
        self.truth.publish_with_source_time(truth, time).await?;
        if let Some(publisher) = &self.production {
            publisher.publish_with_source_time(observed, time).await?;
        }
        Ok(())
    }
}

pub(crate) struct AnnouncedReference<T: Message> {
    truth: Reference<T>,
    production: Option<AnnouncingPublisher<T>>,
}
impl<T: Message> AnnouncedReference<T> {
    pub async fn new(node: &Node, topic: &str, substitute: bool) -> Result<Self> {
        Ok(Self {
            truth: Reference::new(node, topic, false).await?,
            production: if substitute {
                Some(node.announcing_publisher(topic).await?)
            } else {
                None
            },
        })
    }
    pub async fn publish(&self, value: &T, time: Time) -> Result<()> {
        self.publish_observed(value, value, time).await
    }
    pub async fn publish_observed(&self, truth: &T, observed: &T, time: Time) -> Result<()> {
        if let Some(publisher) = &self.production {
            publisher.announce(time).await?.publish(observed).await?;
        }
        self.truth.publish(truth, time).await
    }
}

use crate::{profiles::Profile, robot_io::Observation};
use coordinate_systems::{Ground, Odometry, Robot};
use kinematics::robot_kinematics::RobotKinematics;
use linear_algebra::{Isometry3, Pose2, point};
use projection::camera_matrix::CameraMatrix;
use types::{
    camera_geometry::CameraGeometry, support_foot::SupportFootState, time_wrapper::TimeWrapper,
};

pub(crate) struct BodyReferences {
    camera: Reference<TimeWrapper<CameraMatrix>>,
    camera_geometry: Reference<TimeWrapper<CameraGeometry>>,
    ground: Reference<TimeWrapper<Option<Isometry3<Ground, Robot>>>>,
    robot: Reference<TimeWrapper<Option<Isometry3<Robot, Ground>>>>,
    kinematics: Reference<TimeWrapper<RobotKinematics>>,
    support: Reference<TimeWrapper<Option<SupportFootState>>>,
    odometry: AnnouncedReference<Pose2<Odometry>>,
    odometry_origin: std::sync::Mutex<Option<nalgebra::Isometry2<f32>>>,
}
impl BodyReferences {
    pub async fn new(node: &Node, profile: Profile) -> Result<Self> {
        Ok(Self {
            odometry_origin: std::sync::Mutex::new(None),
            camera: Reference::new(node, "camera_matrix", !profile.body_state()).await?,
            camera_geometry: Reference::new(node, "camera_geometry", !profile.body_state()).await?,
            ground: Reference::new(node, "ground_to_robot", !profile.body_state()).await?,
            robot: Reference::new(node, "robot_to_ground", !profile.body_state()).await?,
            kinematics: Reference::new(node, "robot_kinematics", !profile.body_state()).await?,
            support: Reference::new(node, "support_foot_state", !profile.body_state()).await?,
            odometry: AnnouncedReference::new(node, "inputs/odometry", !profile.body_state())
                .await?,
        })
    }
    pub async fn publish(&self, sample: &Observation, time: Time) -> Result<()> {
        self.camera_geometry
            .publish(
                &TimeWrapper {
                    time,
                    inner: CameraGeometry::from(&sample.camera_matrix),
                },
                time,
            )
            .await?;
        self.camera
            .publish(
                &TimeWrapper {
                    time,
                    inner: sample.camera_matrix.clone(),
                },
                time,
            )
            .await?;
        self.ground
            .publish(
                &TimeWrapper {
                    time,
                    inner: Some(sample.ground_to_robot),
                },
                time,
            )
            .await?;
        self.robot
            .publish(
                &TimeWrapper {
                    time,
                    inner: Some(sample.ground_to_robot.inverse()),
                },
                time,
            )
            .await?;
        self.kinematics
            .publish(
                &TimeWrapper {
                    time,
                    inner: sample.kinematics.clone(),
                },
                time,
            )
            .await?;
        self.support
            .publish(
                &TimeWrapper {
                    time,
                    inner: sample.support,
                },
                time,
            )
            .await?;
        let world = sample.ground_to_world;
        let world = nalgebra::Isometry2::new(
            nalgebra::vector![world.translation.x, world.translation.y],
            world.rotation.euler_angles().2,
        );
        let pose = {
            let mut origin = self.odometry_origin.lock().unwrap();
            origin.get_or_insert(world).inverse() * world
        };
        self.odometry
            .publish(
                &Pose2::new(
                    point![pose.translation.x, pose.translation.y],
                    pose.rotation.angle(),
                ),
                time,
            )
            .await?;
        Ok(())
    }
}

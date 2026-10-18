//! Truth remains observable even when a real node owns the production topic.
use color_eyre::Result;
use ros_z::{Message, prelude::*, time::Time};
use ros_z_streams::{AnnouncingPublisher, CreateAnnouncingPublisher};

use crate::profiles::Profile;

pub(crate) struct Reference<T: Message> {
    truth: Publisher<T>,
    demand: crate::demand::Demand,
    production: Option<Publisher<T>>,
}

#[cfg(test)]
mod demand_tests {
    use super::*;
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };

    static SERIALIZATIONS: AtomicUsize = AtomicUsize::new(0);

    #[derive(serde::Deserialize, ros_z::Message)]
    struct Counted {
        value: u64,
    }
    impl serde::Serialize for Counted {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            SERIALIZATIONS.fetch_add(1, Ordering::Relaxed);
            serde::Serialize::serialize(&self.value, serializer)
        }
    }

    async fn context() -> ros_z::context::Context {
        ContextBuilder::default()
            .with_namespace("/truth_demand")
            .disable_multicast_scouting()
            .with_connect_endpoints(std::iter::empty::<&str>())
            .with_listen_endpoints(std::iter::empty::<&str>())
            .build()
            .await
            .unwrap()
    }

    async fn matching(demand: &crate::demand::Demand, expected: bool) {
        tokio::time::timeout(Duration::from_secs(3), async {
            while demand.active() != expected {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unobserved_truth_does_not_encode_but_production_values_and_times_remain_live() {
        let context = context().await;
        let node = context.create_node("production").build().await.unwrap();
        let reference = Reference::<Counted>::with_production(&node, "value", true)
            .await
            .unwrap();
        let production = node.subscriber::<Counted>("value").build().await.unwrap();
        assert!(!reference.demand.active());
        let unobserved = Reference::<Counted>::with_production(&node, "idle", false)
            .await
            .unwrap();
        SERIALIZATIONS.store(0, Ordering::Relaxed);
        unobserved
            .publish(&Counted { value: 0 }, Time::from_nanos(0))
            .await
            .unwrap();
        assert_eq!(SERIALIZATIONS.load(Ordering::Relaxed), 0);
        let time = Time::from_nanos(1234);
        reference
            .publish_observed(&Counted { value: 7 }, &Counted { value: 9 }, time)
            .await
            .unwrap();
        let received =
            tokio::time::timeout(Duration::from_secs(3), production.recv_with_metadata())
                .await
                .unwrap()
                .unwrap();
        assert_eq!(received.message.value, 9);
        assert_eq!(received.source_time, time);
        assert!(SERIALIZATIONS.load(Ordering::Relaxed) > 0);
        context.shutdown().unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn truth_starts_and_stops_for_late_typed_and_raw_subscribers() {
        let context = context().await;
        let node = context.create_node("late_observer").build().await.unwrap();
        let reference = Reference::<u64>::with_production(&node, "late", false)
            .await
            .unwrap();
        assert!(!reference.needed());
        let subscriber = node
            .subscriber::<u64>("ground_truth/late")
            .build()
            .await
            .unwrap();
        matching(&reference.demand, true).await;
        reference.publish(&42, Time::from_nanos(42)).await.unwrap();
        let received =
            tokio::time::timeout(Duration::from_secs(3), subscriber.recv_with_metadata())
                .await
                .unwrap()
                .unwrap();
        assert_eq!(received.message, 42);
        assert_eq!(received.source_time, Time::from_nanos(42));
        drop(subscriber);
        matching(&reference.demand, false).await;
        let raw = node
            .session()
            .declare_subscriber("rt/truth_demand/ground_truth/**")
            .await
            .unwrap();
        matching(&reference.demand, true).await;
        reference.publish(&43, Time::from_nanos(43)).await.unwrap();
        let sample = tokio::time::timeout(Duration::from_secs(3), raw.recv_async())
            .await
            .unwrap()
            .unwrap();
        assert!(!sample.payload().is_empty());
        drop(raw);
        matching(&reference.demand, false).await;
        context.shutdown().unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recorder_wildcard_present_at_start_always_receives_truth() {
        let context = context().await;
        let node = context.create_node("recorded").build().await.unwrap();
        let recorder = node
            .session()
            .declare_subscriber("rt/truth_demand/**")
            .await
            .unwrap();
        let reference = Reference::<u64>::with_production(&node, "recorded", false)
            .await
            .unwrap();
        assert!(reference.demand.active());
        reference.publish(&99, Time::from_nanos(99)).await.unwrap();
        let sample = tokio::time::timeout(Duration::from_secs(3), recorder.recv_async())
            .await
            .unwrap()
            .unwrap();
        assert!(
            sample
                .key_expr()
                .as_str()
                .contains("ground_truth/recorded/")
        );
        context.shutdown().unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn scoped_remote_observers_start_and_stop_truth_through_a_router() {
        let router = ContextBuilder::default()
            .with_mode("router")
            .disable_multicast_scouting()
            .with_connect_endpoints(std::iter::empty::<&str>())
            .with_listen_endpoints(["tcp/127.0.0.1:0"])
            .build()
            .await
            .unwrap();
        let endpoint = router.session().info().locators().await[0].to_string();
        let connect = || {
            ContextBuilder::default()
                .with_namespace("/truth_demand")
                .with_router_endpoint(&endpoint)
                .unwrap()
                .disable_multicast_scouting()
                .with_json("namespace", "truth_demand_scope")
                .build()
        };
        let producer = connect().await.unwrap();
        let observer = connect().await.unwrap();
        let publisher = producer.create_node("producer").build().await.unwrap();
        let subscriber = observer.create_node("observer").build().await.unwrap();
        let reference = Reference::<u64>::with_production(&publisher, "remote", false)
            .await
            .unwrap();
        assert!(!reference.needed());
        let typed = subscriber
            .subscriber::<u64>("ground_truth/remote")
            .build()
            .await
            .unwrap();
        matching(&reference.demand, true).await;
        reference.publish(&71, Time::from_nanos(71)).await.unwrap();
        let received = tokio::time::timeout(Duration::from_secs(3), typed.recv_with_metadata())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(received.message, 71);
        assert_eq!(received.source_time, Time::from_nanos(71));
        drop(typed);
        matching(&reference.demand, false).await;
        let raw = observer
            .session()
            .declare_subscriber("rt/truth_demand/ground_truth/**")
            .await
            .unwrap();
        matching(&reference.demand, true).await;
        reference.publish(&72, Time::from_nanos(72)).await.unwrap();
        assert!(
            !tokio::time::timeout(Duration::from_secs(3), raw.recv_async())
                .await
                .unwrap()
                .unwrap()
                .payload()
                .is_empty()
        );
        drop(raw);
        matching(&reference.demand, false).await;
        observer.shutdown().unwrap();
        producer.shutdown().unwrap();
        router.shutdown().unwrap();
    }
}
impl<T: Message> Reference<T> {
    /// The profile decides whether the production topic is published too.
    pub async fn new(node: &Node, topic: &str, profile: Profile) -> Result<Self> {
        Self::with_production(node, topic, profile.publishes(topic)).await
    }
    async fn with_production(node: &Node, topic: &str, substitute: bool) -> Result<Self> {
        let truth = node
            .publisher(&format!("ground_truth/{topic}"))
            .build()
            .await?;
        let demand = crate::demand::Demand::new(node, &truth).await?;
        Ok(Self {
            truth,
            demand,
            production: if substitute {
                Some(node.publisher(topic).build().await?)
            } else {
                None
            },
        })
    }
    pub fn needed(&self) -> bool {
        self.production.is_some() || self.demand.active()
    }
    pub async fn publish(&self, value: &T, time: Time) -> Result<()> {
        self.publish_observed(value, value, time).await
    }
    pub async fn publish_observed(&self, truth: &T, observed: &T, time: Time) -> Result<()> {
        if self.demand.active() {
            self.truth.publish_with_source_time(truth, time).await?;
        }
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
    pub async fn new(node: &Node, topic: &str, profile: Profile) -> Result<Self> {
        Ok(Self {
            truth: Reference::with_production(node, topic, false).await?,
            production: if profile.publishes(topic) {
                Some(node.announcing_publisher(topic).await?)
            } else {
                None
            },
        })
    }
    pub fn demand(&self) -> crate::demand::Demand {
        self.truth.demand.clone()
    }
    pub fn needed(&self) -> bool {
        self.production.is_some() || self.truth.needed()
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

use crate::robot_io::Observation;
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
            camera: Reference::new(node, "camera_matrix", profile).await?,
            camera_geometry: Reference::new(node, "camera_geometry", profile).await?,
            ground: Reference::new(node, "ground_to_robot", profile).await?,
            robot: Reference::new(node, "robot_to_ground", profile).await?,
            kinematics: Reference::new(node, "robot_kinematics", profile).await?,
            support: Reference::new(node, "support_foot_state", profile).await?,
            odometry: AnnouncedReference::new(node, "inputs/odometry", profile).await?,
        })
    }
    pub async fn publish(&self, sample: &Observation, time: Time) -> Result<()> {
        if self.camera_geometry.needed() {
            self.camera_geometry
                .publish(
                    &TimeWrapper {
                        time,
                        inner: CameraGeometry::from(&sample.camera_matrix),
                    },
                    time,
                )
                .await?;
        }
        if self.camera.needed() {
            self.camera
                .publish(
                    &TimeWrapper {
                        time,
                        inner: sample.camera_matrix.clone(),
                    },
                    time,
                )
                .await?;
        }
        if self.ground.needed() {
            self.ground
                .publish(
                    &TimeWrapper {
                        time,
                        inner: Some(sample.ground_to_robot),
                    },
                    time,
                )
                .await?;
        }
        if self.robot.needed() {
            self.robot
                .publish(
                    &TimeWrapper {
                        time,
                        inner: Some(sample.ground_to_robot.inverse()),
                    },
                    time,
                )
                .await?;
        }
        if self.kinematics.needed() {
            self.kinematics
                .publish(
                    &TimeWrapper {
                        time,
                        inner: sample.kinematics.clone(),
                    },
                    time,
                )
                .await?;
        }
        if self.support.needed() {
            self.support
                .publish(
                    &TimeWrapper {
                        time,
                        inner: sample.support,
                    },
                    time,
                )
                .await?;
        }
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

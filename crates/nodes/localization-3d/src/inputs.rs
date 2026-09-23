use crate::Localization;
use booster::ImuState;
use color_eyre::Result;
use kinematics::robot_kinematics::RobotKinematics;
use ros_z::{
    cache::Cache,
    node::Node,
    pubsub::{QueueOverflowReporting, Subscriber},
    time::Time,
};
use std::num::NonZeroUsize;
use types::camera_geometry::{CAMERA_GEOMETRY_TOPIC, CameraGeometry, camera_geometry_at};
use types::{
    odometry::{KINEMATIC_ODOMETRY_TOPIC, KinematicOdometryDelta},
    time_wrapper::TimeWrapper,
    visual_localization::{VISUAL_LOCALIZATION_TOPIC, VisualLocalizationFrame},
    visual_odometry::VisualOdometer,
};

pub(crate) enum Measurement {
    Imu(Time, ImuState),
    Kinematics(Box<TimeWrapper<RobotKinematics>>),
    Visual(TimeWrapper<VisualLocalizationFrame>),
    Odometry(VisualOdometer),
    KinematicOdometry(KinematicOdometryDelta),
}

impl Measurement {
    pub(crate) fn time(&self) -> Time {
        match self {
            Self::Imu(time, _) => *time,
            Self::Kinematics(v) => v.time,
            Self::Visual(v) => v.time,
            Self::Odometry(v) => v.time,
            Self::KinematicOdometry(v) => v.time,
        }
    }
}

pub(crate) struct Inputs {
    imu: Subscriber<ImuState>,
    kinematics: Subscriber<TimeWrapper<RobotKinematics>>,
    visual: Subscriber<TimeWrapper<VisualLocalizationFrame>>,
    odometry: Subscriber<VisualOdometer>,
    kinematic_odometry: Subscriber<KinematicOdometryDelta>,
    pub(crate) cameras: Cache<TimeWrapper<CameraGeometry>>,
    pub(crate) robot_kinematics: Cache<TimeWrapper<RobotKinematics>>,
}

impl Inputs {
    pub(crate) async fn new(node: &Node) -> Result<Self> {
        Ok(Self {
            kinematic_odometry: node
                .subscriber(KINEMATIC_ODOMETRY_TOPIC)
                .queue_capacity(NonZeroUsize::new(60).unwrap())
                .queue_overflow_reporting(QueueOverflowReporting::Warn)
                .build()
                .await?,
            imu: node
                .subscriber("inputs/imu_state")
                .queue_capacity(NonZeroUsize::new(500).unwrap())
                .queue_overflow_reporting(QueueOverflowReporting::Warn)
                .build()
                .await?,
            kinematics: node
                .subscriber("robot_kinematics")
                .queue_capacity(NonZeroUsize::new(500).unwrap())
                .queue_overflow_reporting(QueueOverflowReporting::Warn)
                .build()
                .await?,
            visual: node
                .subscriber(VISUAL_LOCALIZATION_TOPIC)
                .queue_capacity(NonZeroUsize::new(60).unwrap())
                .queue_overflow_reporting(QueueOverflowReporting::Warn)
                .build()
                .await?,
            odometry: node
                .subscriber("visual_odometry/current_left_camera_to_visual_odometer")
                .queue_capacity(NonZeroUsize::new(60).unwrap())
                .queue_overflow_reporting(QueueOverflowReporting::Warn)
                .build()
                .await?,
            cameras: node
                .subscriber::<TimeWrapper<CameraGeometry>>(CAMERA_GEOMETRY_TOPIC)
                .cache(1500)
                .with_stamp(|v| v.time)
                .build()
                .await?,
            robot_kinematics: node
                .subscriber::<TimeWrapper<RobotKinematics>>("robot_kinematics")
                .cache(1000)
                .with_stamp(|v| v.time)
                .build()
                .await?,
        })
    }

    pub(crate) async fn recv(&self) -> Result<Measurement> {
        Ok(tokio::select! {
            v = self.imu.recv_with_metadata() => { let v = v?; Measurement::Imu(v.source_time, v.message) },
            v = self.kinematics.recv() => Measurement::Kinematics(Box::new(v?)),
            v = self.visual.recv() => Measurement::Visual(v?),
            v = self.odometry.recv() => Measurement::Odometry(v?),
            v = self.kinematic_odometry.recv() => Measurement::KinematicOdometry(v?),
        })
    }

    /// A transient work batch, not another queue. Samples arriving during this
    /// pass or the solve stay in the bounded ROSZ queues until the next pass.
    pub(crate) async fn drain(&self, first: Measurement) -> Result<Vec<Measurement>> {
        let [imu, feet, visual, vo, ko] = [
            self.imu.queued_len(),
            self.kinematics.queued_len(),
            self.visual.queued_len(),
            self.odometry.queued_len(),
            self.kinematic_odometry.queued_len(),
        ];
        let mut samples = Vec::with_capacity(1 + imu + feet + visual + vo + ko);
        samples.push(first);
        for _ in 0..imu {
            let v = self.imu.recv_with_metadata().await?;
            samples.push(Measurement::Imu(v.source_time, v.message));
        }
        for _ in 0..feet {
            samples.push(Measurement::Kinematics(Box::new(
                self.kinematics.recv().await?,
            )));
        }
        for _ in 0..visual {
            samples.push(Measurement::Visual(self.visual.recv().await?));
        }
        for _ in 0..vo {
            samples.push(Measurement::Odometry(self.odometry.recv().await?));
        }
        for _ in 0..ko {
            samples.push(Measurement::KinematicOdometry(
                self.kinematic_odometry.recv().await?,
            ));
        }
        samples.sort_by_key(Measurement::time);
        Ok(samples)
    }

    pub(crate) fn ingest(
        &self,
        localization: &mut Localization,
        sample: Measurement,
    ) -> Result<bool> {
        match sample {
            Measurement::Imu(time, imu) => localization.ingest_imu(time, imu),
            Measurement::Kinematics(v) => localization.ingest_kinematics(*v),
            Measurement::Visual(v) => localization.ingest_visual_localization_frame(v),
            Measurement::KinematicOdometry(v) => localization.ingest_kinematic_odometry(v),
            Measurement::Odometry(v) => {
                let previous = v
                    .delta
                    .as_ref()
                    .and_then(|d| camera_geometry_at(&self.cameras, d.previous_time));
                let current = camera_geometry_at(&self.cameras, v.time);
                localization.ingest_visual_odometry(v, previous.as_ref(), current.as_ref())
            }
        }
    }
}

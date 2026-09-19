use std::future::{Future, ready};

use color_eyre::Result;
use coordinate_systems::{Field, Robot};
use linear_algebra::Isometry3;
use projection::intrinsic::Intrinsic;
use ros_z::pubsub::Publisher;
use tokio::{runtime::Handle, sync::watch, task::JoinSet};
use types::{
    localization::LocalizationState3D, time_wrapper::TimeWrapper,
    visual_localization::AssociationGeometry,
};

use crate::pose::compose_robot_to_field;

pub(crate) fn spawn_publisher(
    tasks: &mut JoinSet<Result<()>>,
    publish: impl Future<Output = Result<()>> + Send + 'static,
) {
    let runtime = Handle::current();
    // Zenoh publication can synchronously block even when awaited. Sender closure wakes idle
    // mailbox workers; active sends cannot be aborted and still rely on the transport timeout.
    tasks.spawn_blocking(move || runtime.block_on(publish));
}

pub(crate) struct LocalizationPublishers {
    localization: Publisher<Option<Isometry3<Field, Robot>>>,
    pose_3d: Publisher<TimeWrapper<Option<Isometry3<Field, Robot>>>>,
    state_3d: Publisher<LocalizationState3D>,
    association_geometry: Publisher<TimeWrapper<AssociationGeometry>>,
    calibrated_intrinsics: Publisher<Intrinsic>,
}

impl LocalizationPublishers {
    pub(crate) fn new(
        localization: Publisher<Option<Isometry3<Field, Robot>>>,
        pose_3d: Publisher<TimeWrapper<Option<Isometry3<Field, Robot>>>>,
        state_3d: Publisher<LocalizationState3D>,
        association_geometry: Publisher<TimeWrapper<AssociationGeometry>>,
        calibrated_intrinsics: Publisher<Intrinsic>,
    ) -> Self {
        Self {
            localization,
            pose_3d,
            state_3d,
            association_geometry,
            calibrated_intrinsics,
        }
    }

    pub(crate) fn spawn(self, tasks: &mut JoinSet<Result<()>>) -> OutputMailbox {
        let (outputs, mut receiver) =
            watch::channel::<Option<TimeWrapper<AssociationGeometry>>>(None);
        let (intrinsics, mut intrinsics_receiver) = watch::channel::<Option<Intrinsic>>(None);
        spawn_publisher(tasks, async move {
            while receiver.changed().await.is_ok() {
                // Never hold a watch borrow across publishing: it would block ingestion writes.
                let Some(geometry) = receiver.borrow_and_update().clone() else {
                    continue;
                };
                let localization = field_to_robot(&geometry.inner);
                self.state_3d.publish(&geometry.inner.state).await?;
                self.association_geometry.publish(&geometry).await?;
                self.localization.publish(&localization).await?;
                self.pose_3d
                    .publish(&TimeWrapper {
                        time: geometry.time,
                        inner: localization,
                    })
                    .await?;
            }
            Ok(())
        });
        spawn_publisher(tasks, async move {
            while intrinsics_receiver.changed().await.is_ok() {
                let Some(intrinsic) = *intrinsics_receiver.borrow_and_update() else {
                    continue;
                };
                self.calibrated_intrinsics
                    .publish_if_subscribed(|| ready(intrinsic))
                    .await?;
            }
            Ok(())
        });
        OutputMailbox {
            outputs,
            intrinsics,
        }
    }
}

pub(crate) fn field_to_robot(geometry: &AssociationGeometry) -> Option<Isometry3<Field, Robot>> {
    if !matches!(geometry.state, LocalizationState3D::Tracking { .. }) {
        return None;
    }
    geometry
        .local_to_field
        .map(|alignment| compose_robot_to_field(geometry.robot_to_local, alignment).inverse())
}

pub(crate) struct OutputMailbox {
    outputs: watch::Sender<Option<TimeWrapper<AssociationGeometry>>>,
    intrinsics: watch::Sender<Option<Intrinsic>>,
}

impl OutputMailbox {
    pub(crate) fn publish_intrinsics(&self, intrinsic: Intrinsic) {
        self.intrinsics.send_replace(Some(intrinsic));
    }

    pub(crate) fn publish_outputs(&self, geometry: TimeWrapper<AssociationGeometry>) {
        // Ingestion supplies authoritative state in arrival order. A correction or loss can
        // carry older geometry; its measurement stamp must not suppress the lifecycle update.
        self.outputs.send_replace(Some(geometry));
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::mpsc, time::Duration};

    use linear_algebra::Isometry2;
    use ros_z::time::Time;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn synchronous_publication_does_not_block_ingestion() {
        let mut tasks = JoinSet::new();
        let (started, publishing) = tokio::sync::oneshot::channel();
        let (release, blocked) = mpsc::channel();
        spawn_publisher(&mut tasks, async move {
            started.send(()).unwrap();
            // A timeout makes a regression fail instead of deadlocking the test runtime.
            blocked.recv_timeout(Duration::from_secs(5))?;
            Ok(())
        });
        publishing.await.unwrap();

        let (outputs, receiver) = watch::channel(None);
        let (intrinsics, _intrinsics_receiver) = watch::channel(None);
        let mailbox = OutputMailbox {
            outputs,
            intrinsics,
        };
        for epoch in 0..1000 {
            mailbox.publish_outputs(geometry(
                Time::from_nanos(epoch as i64),
                epoch,
                LocalizationState3D::Startup,
            ));
            tokio::task::yield_now().await;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
        assert_eq!(receiver.borrow().as_ref().unwrap().inner.epoch, 999);
        release.send(()).unwrap();
        tasks.join_next().await.unwrap().unwrap().unwrap();
    }

    fn geometry(
        time: Time,
        epoch: u64,
        state: LocalizationState3D,
    ) -> TimeWrapper<AssociationGeometry> {
        TimeWrapper {
            time,
            inner: AssociationGeometry {
                epoch,
                state,
                robot_to_local: Isometry3::identity(),
                local_to_field: Some(Isometry2::identity()),
            },
        }
    }

    #[test]
    fn stalled_output_consumer_keeps_only_latest_coherent_snapshot() {
        let (outputs, mut receiver) = watch::channel(None);
        let (intrinsics, _intrinsics_receiver) = watch::channel(None);
        let mailbox = OutputMailbox {
            outputs,
            intrinsics,
        };
        for epoch in 0..1000 {
            mailbox.publish_outputs(geometry(
                Time::from_nanos(epoch as i64),
                epoch,
                LocalizationState3D::Startup,
            ));
        }
        let output = receiver.borrow_and_update().clone().unwrap();
        assert_eq!(output.inner.epoch, 999);
        assert_eq!(output.time, Time::from_nanos(999));
        assert_eq!(output.inner.state, LocalizationState3D::Startup);
        assert!(field_to_robot(&output.inner).is_none());
        assert!(!receiver.has_changed().unwrap());
        mailbox.publish_outputs(geometry(
            Time::from_nanos(998),
            999,
            LocalizationState3D::Startup,
        ));
        assert!(receiver.has_changed().unwrap());
        assert_eq!(
            receiver.borrow().as_ref().unwrap().time,
            Time::from_nanos(998)
        );
    }

    #[test]
    fn loss_with_older_geometry_is_observable_without_refreshing_its_measurement_stamp() {
        let (outputs, mut receiver) = watch::channel(None);
        let (intrinsics, _intrinsics_receiver) = watch::channel(None);
        let mailbox = OutputMailbox {
            outputs,
            intrinsics,
        };
        let estimate = types::localization::LocalizationEstimate3D {
            robot_to_field: Isometry3::identity(),
            covariance: nalgebra::SMatrix::identity(),
        };
        let solve_time = Time::from_nanos(100);
        let tracking = LocalizationState3D::Tracking {
            estimate,
            last_successful_solve: solve_time,
        };
        mailbox.publish_outputs(geometry(Time::from_nanos(200), 0, tracking));
        assert_eq!(
            field_to_robot(&receiver.borrow_and_update().as_ref().unwrap().inner),
            Some(Isometry3::identity())
        );
        let lost = crate::event_handlers::lose_track(
            tracking,
            &mut crate::visual_localization::GlobalVisualLockTracker::default(),
            Time::from_nanos(300),
        );
        mailbox.publish_outputs(geometry(solve_time, 0, lost));
        assert!(receiver.has_changed().unwrap());
        let snapshot = receiver.borrow_and_update().clone().unwrap();
        assert_eq!(snapshot.time, solve_time);
        assert_eq!(snapshot.inner.state, lost);
        assert!(field_to_robot(&snapshot.inner).is_none());
        assert_eq!(snapshot.inner.local_to_field, Some(Isometry2::identity()));
    }
}

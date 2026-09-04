use color_eyre::Result;
use coordinate_systems::{Field, Local, Robot};
use linear_algebra::{Isometry2, Isometry3};
use ros_z::{pubsub::Publisher, time::Time};
use types::{
    localization::LocalizationState3D,
    time_wrapper::TimeWrapper,
    visual_localization::{AssociationGeometry, AssociationPoseHintSource},
};

#[derive(Clone, Copy)]
pub(crate) struct LocalizationPublishers<'a> {
    localization: &'a Publisher<Option<Isometry3<Field, Robot>>>,
    pose_3d: &'a Publisher<TimeWrapper<Option<Isometry3<Field, Robot>>>>,
    state_3d: &'a Publisher<LocalizationState3D>,
    association_geometry: &'a Publisher<TimeWrapper<AssociationGeometry>>,
}

impl<'a> LocalizationPublishers<'a> {
    pub(crate) fn new(
        localization: &'a Publisher<Option<Isometry3<Field, Robot>>>,
        pose_3d: &'a Publisher<TimeWrapper<Option<Isometry3<Field, Robot>>>>,
        state_3d: &'a Publisher<LocalizationState3D>,
        association_geometry: &'a Publisher<TimeWrapper<AssociationGeometry>>,
    ) -> Self {
        Self {
            localization,
            pose_3d,
            state_3d,
            association_geometry,
        }
    }

    pub(crate) async fn publish_outputs(
        self,
        time: Time,
        epoch: u64,
        robot_to_local: Isometry3<Robot, Local>,
        local_to_field: Option<Isometry2<Local, Field>>,
        localization: Option<Isometry3<Field, Robot>>,
        state: LocalizationState3D,
        source: AssociationPoseHintSource,
    ) -> Result<()> {
        self.localization.publish(&localization).await?;
        self.pose_3d
            .publish(&TimeWrapper {
                time,
                inner: localization,
            })
            .await?;
        self.state_3d.publish(&state).await?;
        self.association_geometry
            .publish(&TimeWrapper {
                time,
                inner: AssociationGeometry {
                    epoch,
                    robot_to_local,
                    local_to_field,
                    source,
                },
            })
            .await?;
        Ok(())
    }
}

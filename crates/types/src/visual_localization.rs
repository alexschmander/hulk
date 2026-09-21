use coordinate_systems::{Camera, Field, Local, Pixel, Robot};
use linear_algebra::{IntoTransform, Isometry2, Isometry3, Point2, Point3};
use ros_z::Message;
use serde::{Deserialize, Serialize};

pub const VISUAL_LOCALIZATION_TOPIC: &str = "field_mark_association/visual_localization_local";
pub const GLOBAL_LOCALIZATION_DEBUG_TOPIC: &str = "debug/global_localization";
pub const MIN_CERTIFIED_VISUAL_ASSOCIATIONS: usize = 3;
pub const MAX_CERTIFIED_VISUAL_ASSOCIATIONS: usize = 32;

/// Internal association input assembled from estimate and lifecycle messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociationGeometry {
    pub epoch: u64,
    /// The tracking prior stays anchored during loss while local poses continue updating.
    pub state: crate::localization::LocalizationState3D,
    pub robot_to_local: Isometry3<Robot, Local>,
    pub local_to_field: Option<Isometry2<Local, Field>>,
}

impl AssociationGeometry {
    /// Build association input from a coherent estimate and lifecycle snapshot.
    /// An old local frame must never be paired with a new epoch's state.
    pub fn from_estimate(
        estimate: &crate::localization::LocalizationEstimate,
        status: &crate::localization::LocalizationStatus,
        tracking_reference: Option<&crate::localization::LocalizationEstimate>,
    ) -> Option<Self> {
        use crate::localization::{LocalizationEstimate3D, LocalizationState, LocalizationState3D};
        if estimate.epoch != status.epoch {
            return None;
        }
        let field = estimate.robot_to_field.map(|field| LocalizationEstimate3D {
            robot_to_field: field.pose.inner.cast().framed_transform(),
            covariance: field.covariance.cast(),
        });
        let state = match status.state {
            LocalizationState::Startup => LocalizationState3D::Startup,
            LocalizationState::Tracking => LocalizationState3D::Tracking {
                estimate: field?,
                last_successful_solve: estimate.time,
            },
            LocalizationState::LostTrack => LocalizationState3D::LostTrack {
                last_known_estimate: {
                    let prior = tracking_reference
                        .filter(|p| p.epoch == status.epoch)?
                        .robot_to_field?;
                    LocalizationEstimate3D {
                        robot_to_field: prior.pose.inner.cast().framed_transform(),
                        covariance: prior.covariance.cast(),
                    }
                },
                last_successful_solve: tracking_reference?.time,
            },
        };
        let local_to_field = estimate.robot_to_field.map(|field| {
            let alignment = field.pose * estimate.robot_to_local.pose.inverse();
            let (_, _, yaw) = alignment.inner.rotation.euler_angles();
            nalgebra::Isometry2::new(alignment.inner.translation.vector.xy(), yaw)
                .cast()
                .framed_transform()
        });
        Some(Self {
            epoch: estimate.epoch,
            state,
            robot_to_local: estimate.robot_to_local.pose.inner.cast().framed_transform(),
            local_to_field,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Message)]
pub struct VisualLocalizationFrame {
    pub epoch: u64,
    pub robot_to_camera: Isometry3<Robot, Camera>,
    /// Local geometry used for association and localization-owned bootstrap seeding.
    pub robot_to_local: Isometry3<Robot, Local>,
    pub camera_intrinsic: projection::intrinsic::Intrinsic,
    pub associations: Vec<FieldMarkAssociation>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Message)]
pub struct FieldMarkAssociation {
    pub detection: Point2<Pixel>,
    pub field_point: Point3<Field>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Message)]
pub struct GlobalLocalizationDebug {
    pub association_count: usize,
    pub pairwise_distance_rms: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::localization::{
        LocalizationEstimate, LocalizationState, LocalizationState3D, LocalizationStatus,
        PoseEstimate,
    };
    use ros_z::time::Time;

    #[test]
    fn association_rejects_mixed_epochs_and_keeps_recovery_prior_frozen() {
        let prior = LocalizationEstimate {
            time: Time::from_nanos(10),
            epoch: 3,
            robot_to_local: PoseEstimate {
                pose: Isometry3::identity(),
                covariance: nalgebra::SMatrix::identity(),
            },
            robot_to_field: Some(PoseEstimate {
                pose: Isometry3::identity(),
                covariance: nalgebra::SMatrix::identity(),
            }),
        };
        let mut current = prior;
        current.time = Time::from_nanos(30);
        current.robot_to_local.pose.inner.translation.vector.x = 2.0;
        current.robot_to_field.as_mut().unwrap().covariance *= 5.0;
        let mut status = LocalizationStatus {
            time: Time::from_nanos(20),
            epoch: 3,
            state: LocalizationState::LostTrack,
        };
        let geometry = AssociationGeometry::from_estimate(&current, &status, Some(&prior)).unwrap();
        assert_eq!(geometry.robot_to_local.translation().x(), 2.0);
        let LocalizationState3D::LostTrack {
            last_known_estimate,
            last_successful_solve,
        } = geometry.state
        else {
            panic!("expected recovery prior");
        };
        assert_eq!(last_successful_solve, prior.time);
        assert_eq!(
            last_known_estimate.covariance,
            nalgebra::SMatrix::<f32, 6, 6>::identity()
        );
        status.epoch = 4;
        assert!(AssociationGeometry::from_estimate(&current, &status, Some(&prior)).is_none());
    }
}

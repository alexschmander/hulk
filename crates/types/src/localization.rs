use nalgebra::SMatrix;
use ros_z::time::Time;
use ros_z::{Message, MessageSchema, SchemaBuilder, SerdeCdrCodec};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use coordinate_systems::{Field, Ground, Local, Robot};
use linear_algebra::{Isometry2, Isometry3};

use crate::multivariate_normal_distribution::MultivariateNormalDistribution;

pub const LOCALIZATION_ESTIMATE_TOPIC: &str = "localization/estimate";
pub const LOCALIZATION_STATUS_TOPIC: &str = "localization/status";

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub struct PoseEstimate<From, To> {
    pub pose: Isometry3<From, To, f64>,
    /// Right-local tangent covariance, ordered [rotation xyz, translation xyz].
    pub covariance: SMatrix<f64, 6, 6>,
}

impl<From, To> Message for PoseEstimate<From, To>
where
    From: Message + Serialize + DeserializeOwned,
    To: Message + Serialize + DeserializeOwned,
{
    type Codec = SerdeCdrCodec<Self>;

    fn type_name() -> String {
        format!(
            "types::localization::PoseEstimate<{},{}>",
            From::type_name(),
            To::type_name()
        )
    }
}

impl<From, To> MessageSchema for PoseEstimate<From, To>
where
    From: Message + Serialize + DeserializeOwned,
    To: Message + Serialize + DeserializeOwned,
{
    fn build_schema(
        builder: &mut SchemaBuilder,
    ) -> Result<ros_z::__private::ros_z_schema::TypeDef, ros_z::__private::ros_z_schema::SchemaError>
    {
        builder.define_message_struct::<Self>(|fields| {
            fields.field::<Isometry3<From, To, f64>>("pose")?;
            let covariance = fields.shape::<[f64; 36]>()?;
            fields.field_with_shape("covariance", covariance);
            Ok(())
        })
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Message, PartialEq)]
pub struct LocalizationEstimate {
    pub time: Time,
    pub epoch: u64,
    pub robot_to_local: PoseEstimate<Robot, Local>,
    pub robot_to_field: Option<PoseEstimate<Robot, Field>>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Message, PartialEq, Eq)]
pub enum LocalizationState {
    Startup,
    Tracking,
    LostTrack,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Message, PartialEq, Eq)]
pub struct LocalizationStatus {
    pub time: Time,
    pub epoch: u64,
    pub state: LocalizationState,
}

/// Pose prior used by the field-association algorithm (not a node output).
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub struct LocalizationEstimate3D {
    pub robot_to_field: Isometry3<Robot, Field>,
    pub covariance: SMatrix<f32, 6, 6>,
}

/// Association's pose prior and freshness anchor, assembled by its consumer.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub enum LocalizationState3D {
    Startup,
    Tracking {
        estimate: LocalizationEstimate3D,
        last_successful_solve: Time,
    },
    LostTrack {
        last_known_estimate: LocalizationEstimate3D,
        last_successful_solve: Time,
    },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Message)]
pub struct ScoredPose {
    pub state: MultivariateNormalDistribution<3>,
    pub score: f32,
}

pub fn ground_to_field_from_field_to_robot(
    field_to_robot: Isometry3<Field, Robot>,
    robot_to_ground: Isometry3<Robot, Ground>,
) -> Isometry2<Ground, Field> {
    let robot_to_field = field_to_robot.inverse();
    let ground_to_field = robot_to_field * robot_to_ground.inverse();
    let (_, _, field_to_robot_yaw) = field_to_robot.inner.rotation.euler_angles();
    let yaw = -field_to_robot_yaw;
    let translation = ground_to_field.inner.translation.vector;

    Isometry2::wrap(nalgebra::Isometry2::new(
        nalgebra::vector![translation.x, translation.y],
        yaw,
    ))
}

#[cfg(test)]
mod tests {
    use linear_algebra::IntoTransform;

    use super::*;

    #[test]
    fn localization_output_schemas_are_valid() {
        LocalizationEstimate::schema();
        LocalizationStatus::schema();
    }

    #[test]
    fn ground_to_field_from_field_to_robot_flattens_robot_pose() {
        let robot_to_field = nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(1.5, -2.0, 0.4),
            nalgebra::UnitQuaternion::from_euler_angles(0.0, 0.0, 0.7),
        );
        let field_to_robot: Isometry3<Field, Robot> = robot_to_field.inverse().framed_transform();
        let robot_to_ground = Isometry3::identity();

        let ground_to_field = ground_to_field_from_field_to_robot(field_to_robot, robot_to_ground);

        assert!((ground_to_field.translation().x() - 1.5).abs() < 1.0e-6);
        assert!((ground_to_field.translation().y() + 2.0).abs() < 1.0e-6);
        assert!((ground_to_field.orientation().angle() - 0.7).abs() < 1.0e-6);
    }

    #[test]
    fn ground_to_field_from_field_to_robot_ignores_ground_roll_pitch() {
        let robot_to_field = nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(1.5, -2.0, 0.4),
            nalgebra::UnitQuaternion::from_euler_angles(0.0, 0.0, 0.7),
        );
        let field_to_robot: Isometry3<Field, Robot> = robot_to_field.inverse().framed_transform();
        let robot_to_ground: Isometry3<Robot, Ground> = nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.0, 0.0, 0.523),
            nalgebra::UnitQuaternion::from_euler_angles(0.045, 0.047, 0.0),
        )
        .framed_transform();

        let ground_to_field = ground_to_field_from_field_to_robot(field_to_robot, robot_to_ground);

        assert!((ground_to_field.orientation().angle() - 0.7).abs() < 1.0e-6);
    }
}

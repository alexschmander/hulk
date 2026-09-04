use coordinate_systems::{Field, Local, Robot};
use factrs::{
    core::SO3,
    variables::{SE2, SE23},
};
use linear_algebra::{IntoTransform, Isometry2, Isometry3};
use nalgebra::Vector3;

pub fn robot_to_local_to_se23(robot_to_local: Isometry3<Robot, Local>) -> SE23<f64> {
    let rotation = robot_to_local.inner.rotation.quaternion();

    SE23::from_rot_vel_trans(
        SO3::from_xyzw(
            rotation.i as f64,
            rotation.j as f64,
            rotation.k as f64,
            rotation.w as f64,
        ),
        Vector3::zeros(),
        robot_to_local.inner.translation.vector.cast(),
    )
}

pub fn local_to_field_to_se2(local_to_field: Isometry2<Local, Field>) -> SE2<f64> {
    SE2::new(
        local_to_field.inner.rotation.angle() as f64,
        local_to_field.inner.translation.x as f64,
        local_to_field.inner.translation.y as f64,
    )
}

pub fn se2_to_local_to_field(local_to_field: &SE2) -> Isometry2<Local, Field, f64> {
    nalgebra::Isometry2::new(local_to_field.xy().into_owned(), local_to_field.theta())
        .framed_transform()
}

#[cfg(test)]
mod tests {
    use linear_algebra::Orientation3;

    use super::*;

    #[test]
    fn robot_to_local_conversion_preserves_pose_with_zero_velocity() {
        let rotation: Orientation3<Local> = Orientation3::from_euler_angles(0.1, -0.2, 0.3);
        let robot_to_local: Isometry3<Robot, Local> = linear_algebra::Isometry3::from_parts(
            linear_algebra::vector![<Local>, 1.0, 2.0, 0.4],
            rotation,
        );

        let pose = robot_to_local_to_se23(robot_to_local);

        assert!((pose.xyz() - nalgebra::vector![1.0, 2.0, 0.4]).norm() < 1.0e-6);
        assert!((pose.uvw() - nalgebra::vector![0.0, 0.0, 0.0]).norm() < 1.0e-9);
        assert!((pose.rot().w() - rotation.inner.w as f64).abs() < 1.0e-9);
        assert!((pose.rot().x() - rotation.inner.i as f64).abs() < 1.0e-9);
        assert!((pose.rot().y() - rotation.inner.j as f64).abs() < 1.0e-9);
        assert!((pose.rot().z() - rotation.inner.k as f64).abs() < 1.0e-9);
    }
}

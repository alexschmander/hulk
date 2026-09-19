use linear_algebra::{IntoTransform, Isometry3, point};
use nalgebra::{Isometry2, Translation3, UnitComplex, UnitQuaternion, Vector2, Vector3};
use types::localization::{LocalizationEstimate3D, LocalizationState3D};

use super::{
    AssociationFixture, AssociationGeometry, GlobalLocalizerParameters, Time, fixture_geometry,
    robot_to_camera,
};

pub struct FitMetrics {
    pub camera_height: f64,
    pub free_height_scale: f64,
    pub metric_rms: f64,
    pub pixel_rms: f64,
}

/// Test-only prior from the recorded labels, not a production startup pose estimator.
/// Fit translation/yaw with measured height and tilt fixed; scale is diagnostic only.
pub fn expected_geometry(fixture: &AssociationFixture) -> (AssociationGeometry, FitMetrics) {
    let camera = &fixture.camera_matrix;
    let mut geometry = fixture_geometry(camera);
    let camera_to_local = (geometry.robot_to_local * robot_to_camera(camera).inverse())
        .inner
        .cast::<f64>();
    let origin = camera_to_local.translation.vector;
    let local: Vec<_> = fixture
        .expected
        .iter()
        .map(|e| {
            let ray = camera_to_local.rotation
                * camera
                    .intrinsics
                    .bearing(point![e.detection[0], e.detection[1]])
                    .inner
                    .cast::<f64>();
            assert!(origin.z > 0.0 && ray.z < 0.0);
            (origin - ray * origin.z / ray.z).xy()
        })
        .collect();
    let field: Vec<_> = fixture
        .expected
        .iter()
        .map(|e| Vector2::from(e.landmark).cast::<f64>())
        .collect();
    let count = local.len() as f64;
    let local_mean = local.iter().copied().sum::<Vector2<f64>>() / count;
    let field_mean = field.iter().copied().sum::<Vector2<f64>>() / count;
    let mut dot = 0.0;
    let mut cross = 0.0;
    let mut norm = 0.0;
    for (l, f) in local.iter().zip(&field) {
        let l = l - local_mean;
        let f = f - field_mean;
        dot += l.dot(&f);
        cross += l.x * f.y - l.y * f.x;
        norm += l.norm_squared();
    }
    let yaw = cross.atan2(dot);
    let rotation = UnitComplex::new(yaw);
    let translation = field_mean - rotation * local_mean;
    let alignment = Isometry2::from_parts(translation.into(), rotation);
    let metric_rms = (local
        .iter()
        .zip(&field)
        .map(|(l, f)| (rotation * l + translation - f).norm_squared())
        .sum::<f64>()
        / count)
        .sqrt();
    let alignment_3d = nalgebra::Isometry3::from_parts(
        Translation3::new(translation.x, translation.y, 0.0),
        UnitQuaternion::from_euler_angles(0.0, 0.0, yaw),
    );
    let field_to_camera = (alignment_3d * camera_to_local).inverse();
    let pixel_rms = (fixture
        .expected
        .iter()
        .map(|e| {
            let p =
                field_to_camera * nalgebra::point![e.landmark[0] as f64, e.landmark[1] as f64, 0.0];
            assert!(p.z > 0.0);
            let projected = camera
                .intrinsics
                .focals
                .cast::<f64>()
                .component_mul(&(p.coords.xy() / p.z))
                + camera.intrinsics.optical_center.inner.coords.cast::<f64>();
            (projected - Vector2::from(e.detection).cast::<f64>()).norm_squared()
        })
        .sum::<f64>()
        / count)
        .sqrt();
    // Isotropic planar least-squares covariance: 2N observations, three fitted parameters.
    // Centering decouples centroid translation from yaw. Transform their joint uncertainty
    // to the robot's right tangent, retaining the yaw/translation lever-arm correlations.
    assert!(count >= 3.0 && norm > 0.0);
    let variance = metric_rms.powi(2) * count / (2.0 * count - 3.0);
    let centered_covariance = nalgebra::Matrix3::from_diagonal(&nalgebra::vector![
        variance / count,
        variance / count,
        variance / norm,
    ]);
    let robot_to_field = alignment_3d * geometry.robot_to_local.inner.cast::<f64>();
    let field_to_robot = robot_to_field.rotation.inverse();
    let lever = rotation
        * (geometry
            .robot_to_local
            .inner
            .translation
            .vector
            .xy()
            .cast::<f64>()
            - local_mean);
    let mut tangent_jacobian = nalgebra::SMatrix::<f64, 6, 3>::zeros();
    for axis in 0..2 {
        tangent_jacobian
            .fixed_view_mut::<3, 1>(3, axis)
            .copy_from(&(field_to_robot * Vector3::ith(axis, 1.0)));
    }
    tangent_jacobian
        .fixed_view_mut::<3, 1>(0, 2)
        .copy_from(&(field_to_robot * Vector3::z()));
    tangent_jacobian
        .fixed_view_mut::<3, 1>(3, 2)
        .copy_from(&(field_to_robot * Vector3::new(-lever.y, lever.x, 0.0)));
    let covariance = tangent_jacobian * centered_covariance * tangent_jacobian.transpose();
    geometry.local_to_field = Some(alignment.cast::<f32>().framed_transform());
    geometry.state = LocalizationState3D::Tracking {
        estimate: LocalizationEstimate3D {
            robot_to_field: Isometry3::wrap(robot_to_field.cast::<f32>()),
            covariance: covariance.cast::<f32>(),
        },
        last_successful_solve: Time::from_nanos(0),
    };
    (
        geometry,
        FitMetrics {
            camera_height: origin.z,
            free_height_scale: dot.hypot(cross) / norm,
            metric_rms,
            pixel_rms,
        },
    )
}

/// Finite-difference oracle independent of the associator's analytic Jacobians.
/// Returns signed doubled area and its uncertainty gate for every possible seed triangle.
pub fn triangle_margins(
    fixture: &AssociationFixture,
    config: GlobalLocalizerParameters,
) -> Vec<(f64, f64)> {
    let camera = &fixture.camera_matrix;
    let geometry = fixture_geometry(camera);
    let robot_origin = geometry
        .robot_to_local
        .inner
        .translation
        .vector
        .cast::<f64>();
    let camera_to_local = (geometry.robot_to_local * robot_to_camera(camera).inverse())
        .inner
        .cast::<f64>();
    let project = |pixels: [Vector2<f64>; 3], tilt: Vector3<f64>, height: f64| {
        let perturb = UnitQuaternion::from_scaled_axis(tilt);
        let origin = robot_origin
            + perturb * (camera_to_local.translation.vector - robot_origin)
            + Vector3::new(0.0, 0.0, height);
        pixels.map(|pixel| {
            let normalized = (pixel - camera.intrinsics.optical_center.inner.coords.cast::<f64>())
                .component_div(&camera.intrinsics.focals.cast::<f64>());
            let ray =
                perturb * camera_to_local.rotation * Vector3::new(normalized.x, normalized.y, 1.0);
            (origin - ray * origin.z / ray.z).xy()
        })
    };
    let area = |pixels, tilt, height| {
        let p = project(pixels, tilt, height);
        let u = p[1] - p[0];
        let v = p[2] - p[0];
        u.x * v.y - u.y * v.x
    };
    let mut margins = Vec::new();
    for a in 0..fixture.expected.len() {
        for b in a + 1..fixture.expected.len() {
            for c in b + 1..fixture.expected.len() {
                let pixels =
                    [a, b, c].map(|i| Vector2::from(fixture.expected[i].detection).cast::<f64>());
                let mut pixel_variance = 0.0;
                for i in 0..3 {
                    for axis in 0..2 {
                        let mut plus = pixels;
                        let mut minus = pixels;
                        plus[i][axis] += 0.01;
                        minus[i][axis] -= 0.01;
                        let derivative = (area(plus, Vector3::zeros(), 0.0)
                            - area(minus, Vector3::zeros(), 0.0))
                            / 0.02;
                        pixel_variance +=
                            (derivative * config.detection_pixel_sigma as f64).powi(2);
                    }
                }
                let tilt = [Vector3::x(), Vector3::y()].map(|axis| {
                    (area(pixels, axis * 1.0e-5, 0.0) - area(pixels, -axis * 1.0e-5, 0.0)) / 2.0e-5
                });
                let height = (area(pixels, Vector3::zeros(), 1.0e-5)
                    - area(pixels, Vector3::zeros(), -1.0e-5))
                    / 2.0e-5;
                let p = project(pixels, Vector3::zeros(), 0.0);
                let gate = (config.mahalanobis_gate as f64).sqrt()
                    * (pixel_variance.sqrt()
                        + config.imu_tilt_sigma as f64 * tilt[0].hypot(tilt[1])
                        + config.height_sigma as f64 * height.abs())
                    + config.geometric_tolerance as f64
                        * ((p[1] - p[0]).norm() + (p[2] - p[0]).norm());
                margins.push((area(pixels, Vector3::zeros(), 0.0), gate));
            }
        }
    }
    margins
}

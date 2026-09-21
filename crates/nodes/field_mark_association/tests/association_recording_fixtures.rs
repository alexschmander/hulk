use coordinate_systems::{Camera, Local, Robot};
use field_mark_association::{
    AssociationInput, FieldMarkAssociationParameters, GlobalAssociationInput,
    GlobalLocalizerParameters, associate_global_visual_features, associate_visual_features,
    find_detected_visual_features,
};
use linear_algebra::{IntoTransform, Isometry3};
use projection::camera_matrix::CameraMatrix;
use ros_z::time::Time;
use serde::Deserialize;
use types::{
    field_dimensions::FieldDimensions,
    localization::LocalizationState3D,
    object_detection::{Object, RobocupObjectLabel},
    visual_localization::AssociationGeometry,
};

#[path = "support/geometry_oracle.rs"]
mod geometry_oracle;

fn fixtures() -> Vec<AssociationFixture> {
    let fixtures: Vec<_> = serde_json::from_str(include_str!("association_fixtures.json")).unwrap();
    assert!(!fixtures.is_empty());
    fixtures
}

#[derive(Debug, Deserialize)]
struct AssociationFixture {
    name: String,
    camera_matrix: CameraMatrix,
    detections: Vec<Object<RobocupObjectLabel>>,
    expected: Vec<ExpectedAssociation>,
}

#[derive(Debug, Deserialize)]
struct ExpectedAssociation {
    detection: [f32; 2],
    landmark: [f32; 2],
}

#[test]
fn real_recording_startup_rejects_uncertifiable_chirality() {
    for fixture in fixtures() {
        let features = find_detected_visual_features(&fixture.detections);
        assert_eq!(features.supported_feature_count(), fixture.expected.len());
        for config in [
            GlobalLocalizerParameters::default(),
            GlobalLocalizerParameters {
                detection_pixel_sigma: 10.0,
                imu_tilt_sigma: 0.1,
                mahalanobis_gate: 100.0,
                ..Default::default()
            },
        ] {
            let margins = geometry_oracle::triangle_margins(&fixture, config);
            assert_eq!(margins.len(), 10);
            assert!(
                margins
                    .iter()
                    .all(|(area, gate)| area.is_finite() && gate - area.abs() > 0.3)
            );
            assert!(
                associate_global_visual_features(GlobalAssociationInput {
                    visual_features: &features,
                    robot_to_local: robot_to_local(&fixture.camera_matrix),
                    robot_to_camera: robot_to_camera(&fixture.camera_matrix),
                    camera_intrinsic: fixture.camera_matrix.intrinsics,
                    field_dimensions: &FieldDimensions::SPL_2025,
                    parameters: &config,
                })
                .associations
                .is_empty()
            );
        }
    }
}

#[test]
fn real_recording_tracking_fixtures_match_all_expected_landmarks() {
    for fixture in fixtures() {
        let features = find_detected_visual_features(&fixture.detections);
        let (geometry, metrics) = geometry_oracle::expected_geometry(&fixture);
        let LocalizationState3D::Tracking { estimate, .. } = geometry.state else {
            unreachable!("oracle supplies a tracking prior");
        };
        let covariance = estimate.covariance;
        assert!((covariance - covariance.transpose()).norm() < 1.0e-7);
        assert!(covariance.symmetric_eigen().eigenvalues.min() > -1.0e-7);
        let rotation = estimate.robot_to_field.inner.rotation.to_rotation_matrix();
        let field_rotation_covariance =
            rotation.matrix() * covariance.fixed_view::<3, 3>(0, 0) * rotation.matrix().transpose();
        let field_translation_covariance =
            rotation.matrix() * covariance.fixed_view::<3, 3>(3, 3) * rotation.matrix().transpose();
        // The independent fit has planar freedom only, not height or tilt freedom.
        assert!(field_rotation_covariance[(0, 0)].abs() < 1.0e-8);
        assert!(field_rotation_covariance[(1, 1)].abs() < 1.0e-8);
        assert!(field_translation_covariance[(2, 2)].abs() < 1.0e-8);
        assert!((metrics.camera_height - 0.6314).abs() < 0.001);
        assert!((metrics.free_height_scale - 1.0242).abs() < 0.001);
        assert!(metrics.metric_rms < 0.16);
        assert!(metrics.pixel_rms < 6.0);
        let localization = associate_visual_features(
            AssociationInput {
                visual_features: &features,
                robot_to_camera: robot_to_camera(&fixture.camera_matrix),
                geometry: &geometry,
                camera_intrinsic: fixture.camera_matrix.intrinsics,
                field_dimensions: &FieldDimensions::SPL_2025,
                time: Time::from_nanos(0),
            },
            &FieldMarkAssociationParameters::default(),
        );
        let mut actual: Vec<_> = localization
            .associations
            .iter()
            .map(|a| {
                association_key(
                    [a.detection.x(), a.detection.y()],
                    [a.field_point.x(), a.field_point.y()],
                )
            })
            .collect();
        let mut expected: Vec<_> = fixture
            .expected
            .iter()
            .map(|a| association_key(a.detection, a.landmark))
            .collect();
        actual.sort_unstable();
        expected.sort_unstable();
        assert_eq!(actual, expected, "{}", fixture.name);
    }
}

fn association_key(detection: [f32; 2], landmark: [f32; 2]) -> [i32; 4] {
    [detection[0], detection[1], landmark[0], landmark[1]]
        .map(|value| (value * 1000.0).round() as i32)
}

fn robot_to_camera(camera: &CameraMatrix) -> Isometry3<Robot, Camera> {
    camera.head_to_camera * camera.robot_to_head
}

// This historical frame uses Ground as Local: both have the same horizontal plane.
fn robot_to_local(camera: &CameraMatrix) -> Isometry3<Robot, Local> {
    camera.ground_to_robot.inverse().inner.framed_transform()
}

fn fixture_geometry(camera: &CameraMatrix) -> AssociationGeometry {
    AssociationGeometry {
        epoch: 0,
        state: LocalizationState3D::Startup,
        robot_to_local: robot_to_local(camera),
        local_to_field: None,
    }
}

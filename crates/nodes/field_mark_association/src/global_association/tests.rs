use coordinate_systems::{Camera, Field, Robot};
use linear_algebra::{Isometry3, Point2, point};
use nalgebra::{SMatrix, UnitQuaternion, vector};
use projection::intrinsic::Intrinsic;
use ros_z::time::Time;
use types::{
    field_dimensions::{FieldDimensions, Half, Side},
    localization::PoseEstimate,
    visual_localization_next::AssociationGeometry,
};

use super::GlobalAssociationConfig;
use crate::{
    AssociationResult, DetectedVisualFeature, DetectedVisualFeatures, GlobalAssociationInput,
    VisualFeatureClass, associate_global_visual_features, map::LandmarkMap,
};

#[derive(Clone, Copy)]
struct AssociationInput<'a> {
    pub visual_features: &'a DetectedVisualFeatures,
    pub robot_to_camera: Isometry3<Robot, Camera>,
    pub geometry: &'a AssociationGeometry,
    pub camera_intrinsic: Intrinsic,
    pub field_dimensions: &'a FieldDimensions,
}

fn geometry() -> AssociationGeometry {
    AssociationGeometry {
        epoch: 0,
        generation: 0,
        estimate: tracking_estimate(0.0),
        last_successful_solve: Time::from_nanos(1_000_000_000),
    }
}

fn intrinsic() -> Intrinsic {
    Intrinsic::new(vector![500.0, 500.0], point![320.0, 240.0])
}

fn robot_to_camera() -> Isometry3<Robot, Camera> {
    Isometry3::wrap(
        nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.0, 0.0, 0.1),
            UnitQuaternion::from_euler_angles(std::f32::consts::PI, 0.0, 0.0),
        )
        .inverse(),
    )
}

fn input<'a>(
    features: &'a DetectedVisualFeatures,
    geometry: &'a AssociationGeometry,
) -> AssociationInput<'a> {
    AssociationInput {
        visual_features: features,
        robot_to_camera: robot_to_camera(),
        geometry,
        camera_intrinsic: intrinsic(),
        field_dimensions: &FieldDimensions::SPL_2025,
    }
}

fn global_input<'a>(
    input: AssociationInput<'a>,
    parameters: &'a GlobalAssociationConfig,
) -> GlobalAssociationInput<'a> {
    let (roll, pitch, _) = input.geometry.estimate.pose.inner.rotation.euler_angles();
    GlobalAssociationInput {
        visual_features: input.visual_features,
        robot_to_ground: linear_algebra::Rotation3::from_euler_angles(
            roll as f32,
            pitch as f32,
            0.0,
        ),
        robot_to_camera: input.robot_to_camera,
        camera_intrinsic: input.camera_intrinsic,
        field_dimensions: input.field_dimensions,
        parameters,
        heading: None,
    }
}

fn calibrated_parameters() -> GlobalAssociationConfig {
    let mut parameters = GlobalAssociationConfig::default();
    parameters.imu_tilt_sigma = 0.001;
    parameters.height_sigma = 0.001;
    parameters
}

fn project(
    landmarks: impl IntoIterator<Item = (VisualFeatureClass, Point2<Field>)>,
    geometry: &AssociationGeometry,
) -> DetectedVisualFeatures {
    let field_to_camera = robot_to_camera()
        * Isometry3::<Robot, Field>::wrap(geometry.estimate.pose.inner.cast()).inverse();
    let mut features = DetectedVisualFeatures::default();
    for (class, landmark) in landmarks {
        let pixel = intrinsic().project((field_to_camera * landmark.extend(0.0)).coords());
        let feature = DetectedVisualFeature {
            pixel,
            confidence: 0.95,
        };
        match class {
            VisualFeatureClass::GoalPost => &mut features.goalposts,
            VisualFeatureClass::LSpot => &mut features.l_spots,
            VisualFeatureClass::TSpot => &mut features.t_spots,
            VisualFeatureClass::XSpot => &mut features.x_spots,
            VisualFeatureClass::PenaltySpot => &mut features.penalty_spots,
        }
        .push(feature);
    }
    features
}

fn stationary_three() -> [(VisualFeatureClass, Point2<Field>); 3] {
    let field = FieldDimensions::SPL_2025;
    [
        (
            VisualFeatureClass::GoalPost,
            field.goal_post(Half::Opponent, Side::Left),
        ),
        (
            VisualFeatureClass::GoalPost,
            field.goal_post(Half::Opponent, Side::Right),
        ),
        (
            VisualFeatureClass::PenaltySpot,
            field.penalty_spot(Half::Opponent),
        ),
    ]
}

fn key(result: &AssociationResult) -> Vec<([f32; 2], [f32; 2])> {
    result
        .associations
        .iter()
        .map(|a| {
            (
                [a.detection.x(), a.detection.y()],
                [a.field_point.x(), a.field_point.y()],
            )
        })
        .collect()
}

fn tracking_estimate(covariance: f64) -> PoseEstimate<Robot, Field> {
    PoseEstimate {
        pose: Isometry3::wrap(nalgebra::Isometry3::translation(2.0, 0.0, 0.45)),
        covariance: SMatrix::identity() * covariance,
    }
}

#[test]
fn geometric_numerical_guards_are_applied() {
    let geometry = geometry();
    let features = project(stationary_three(), &geometry);
    let baseline = GlobalAssociationConfig::default();
    let associate = |config: &GlobalAssociationConfig| {
        associate_global_visual_features(global_input(input(&features, &geometry), config))
    };
    assert_eq!(associate(&baseline).associations.len(), 3);
    for config in [
        GlobalAssociationConfig {
            min_pair_distance: 1.0e6,
            ..baseline
        },
        GlobalAssociationConfig {
            min_triangle_denominator: 1.0e12,
            ..baseline
        },
    ] {
        config.validate().unwrap();
        assert!(associate(&config).associations.is_empty());
    }
}

#[test]
fn three_stationary_features_certify_without_history_or_pose() {
    let geometry = geometry();
    let features = project(stationary_three(), &geometry);
    let mut parameters = GlobalAssociationConfig::default();
    assert_eq!(parameters.min_inliers, 3);
    let first =
        associate_global_visual_features(global_input(input(&features, &geometry), &parameters));
    assert_eq!(first.associations.len(), 3);
    assert!(first.associations.iter().all(|a| a.field_point.x() < 0.0));
    assert_eq!(first.debug.as_ref().unwrap().association_count, 3);
    assert!(first.debug.as_ref().unwrap().pairwise_distance_rms < 1.0e-4);
    // Global fitting estimates height; tracking's height-noise floor is irrelevant.
    parameters.height_sigma = 0.2;
    assert_eq!(
        key(&first),
        key(&associate_global_visual_features(global_input(
            input(&features, &geometry),
            &parameters
        )))
    );
}

#[test]
fn heading_preserves_orientation_and_budget_exhaustion_still_rejects() {
    let geometry = geometry();
    let features = project(stationary_three(), &geometry);
    let config = GlobalAssociationConfig::default();
    let mut input = global_input(input(&features, &geometry), &config);
    let heading = crate::HeadingConstraint {
        expected: linear_algebra::Orientation2::identity(),
        max_error: 0.3,
    };
    input.heading = Some(heading);
    let result = associate_global_visual_features(input);
    assert_eq!(result.associations.len(), 3);
    assert!(result.associations.iter().all(|a| a.field_point.x() > 0.0));
    let exhausted = GlobalAssociationConfig {
        max_work: 1,
        ..config
    };
    assert!(
        associate_global_visual_features(GlobalAssociationInput {
            parameters: &exhausted,
            ..input
        })
        .associations
        .is_empty()
    );
    for limit in [f64::NAN, 0.0, std::f64::consts::FRAC_PI_2] {
        input.heading = Some(crate::HeadingConstraint {
            max_error: limit,
            ..heading
        });
        assert!(
            associate_global_visual_features(input)
                .associations
                .is_empty()
        );
    }
}

#[test]
fn canonical_landmarks_do_not_select_the_robot_half() {
    for robot_x in [-2.0, 2.0] {
        let mut geometry = geometry();
        geometry.estimate.pose.inner.translation.vector.x = robot_x;
        let features = project(stationary_three(), &geometry);
        let config = calibrated_parameters();
        let result =
            associate_global_visual_features(global_input(input(&features, &geometry), &config));
        assert_eq!(result.associations.len(), 3, "robot x {robot_x}");
        assert!(result.associations.iter().all(|a| a.field_point.x() < 0.0));
        // These correspondences imply a half-turn, so the canonical robot is at -robot_x.
        // In particular, a robot really at -2 yields a representative at +2, not own-half.
        for (association, (_, expected)) in result.associations.iter().zip(stationary_three()) {
            assert!(
                (association.field_point.xy().coords().inner + expected.coords().inner).norm()
                    < 1.0e-5
            );
        }
    }
}

#[test]
fn halfturn_equivalent_triangles_are_not_ambiguous() {
    let geometry = geometry();
    let field = FieldDimensions::SPL_2025;
    // Swapping the two penalties and the circle crossing is exactly a field half-turn.
    let features = project(
        [
            (
                VisualFeatureClass::PenaltySpot,
                field.penalty_spot(Half::Own),
            ),
            (
                VisualFeatureClass::PenaltySpot,
                field.penalty_spot(Half::Opponent),
            ),
            (VisualFeatureClass::XSpot, field.x_crossing(Side::Left)),
        ],
        &geometry,
    );
    let config = calibrated_parameters();
    let result =
        associate_global_visual_features(global_input(input(&features, &geometry), &config));
    assert_eq!(result.associations.len(), 3);
}

#[test]
fn translated_triangle_orbits_are_rejected() {
    let geometry = geometry();
    let field = FieldDimensions {
        penalty_area_length: FieldDimensions::SPL_2025.goal_box_area_length,
        ..FieldDimensions::SPL_2025
    };
    // Translation by -(goal_box_width + penalty_box_width)/2 yields another L,T,T
    // triangle on the same goal line, with identical edge lengths AND chirality.
    let features = project(
        [
            (
                VisualFeatureClass::LSpot,
                field.goal_box_corner(Half::Opponent, Side::Left),
            ),
            (
                VisualFeatureClass::TSpot,
                field.goal_box_goal_line_intersection(Half::Opponent, Side::Left),
            ),
            (
                VisualFeatureClass::TSpot,
                field.penalty_box_goal_line_intersection(Half::Opponent, Side::Left),
            ),
        ],
        &geometry,
    );
    let config = calibrated_parameters();
    let mut input = input(&features, &geometry);
    input.field_dimensions = &field;
    assert!(
        associate_global_visual_features(global_input(input, &config))
            .associations
            .is_empty()
    );
}

#[test]
fn collinear_and_duplicate_seeds_do_not_certify() {
    let geometry = geometry();
    let field = FieldDimensions::SPL_2025;
    let features = project(
        [
            (VisualFeatureClass::XSpot, field.center()),
            (VisualFeatureClass::XSpot, field.x_crossing(Side::Left)),
            (VisualFeatureClass::XSpot, field.x_crossing(Side::Right)),
        ],
        &geometry,
    );
    assert!(
        associate_global_visual_features(global_input(
            input(&features, &geometry),
            &GlobalAssociationConfig::default()
        ))
        .associations
        .is_empty()
    );
    let mut features = project(stationary_three(), &geometry);
    features.goalposts[1] = features.goalposts[0];
    assert!(
        associate_global_visual_features(global_input(
            input(&features, &geometry),
            &GlobalAssociationConfig::default()
        ))
        .associations
        .is_empty()
    );
}

#[test]
fn rich_frame_has_bounded_search_and_budget_exhaustion_rejects() {
    let geometry = geometry();
    let map = LandmarkMap::new(
        &FieldDimensions::SPL_2025,
        GlobalAssociationConfig::default().symmetry_epsilon,
    );
    let features = project(map.landmarks.iter().map(|l| (l.class, l.xy)), &geometry);
    let mut config = calibrated_parameters();
    let result =
        associate_global_visual_features(global_input(input(&features, &geometry), &config));
    assert_eq!(result.associations.len(), map.landmarks.len());
    config.max_work = 1;
    assert!(
        associate_global_visual_features(global_input(input(&features, &geometry), &config))
            .associations
            .is_empty()
    );
}

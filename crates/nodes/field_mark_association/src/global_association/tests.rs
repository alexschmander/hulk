use super::*;
use crate::{DetectedVisualFeature, DetectedVisualFeatures};
use ::types::field_dimensions::{FieldDimensions, Half, Side};
use coordinate_systems::{Camera, Field, Ground, Pixel, Robot};
use linear_algebra::{IntoTransform, Isometry3, Point2, point};
use projection::intrinsic::Intrinsic;

fn camera_intrinsic() -> Intrinsic {
    Intrinsic::new(nalgebra::vector![100.0, 100.0], point![320.0, 240.0])
}

fn robot_to_camera() -> Isometry3<Robot, Camera> {
    nalgebra::Isometry3::translation(0.0, 0.0, 0.5).framed_transform()
}

fn input<'a>(
    visual_features: &'a DetectedVisualFeatures,
    field_dimensions: &'a FieldDimensions,
    config_hint: Option<Isometry3<Robot, Field>>,
) -> GlobalLocalizationInput<'a> {
    GlobalLocalizationInput {
        visual_features,
        field_dimensions,
        ground_to_robot: Isometry3::<Ground, Robot>::identity(),
        robot_to_camera: robot_to_camera(),
        camera_intrinsic: camera_intrinsic(),
        pose_hint: config_hint,
    }
}

fn input_with_camera_height<'a>(
    visual_features: &'a DetectedVisualFeatures,
    field_dimensions: &'a FieldDimensions,
    camera_height: f32,
) -> GlobalLocalizationInput<'a> {
    GlobalLocalizationInput {
        robot_to_camera: nalgebra::Isometry3::translation(0.0, 0.0, camera_height)
            .framed_transform(),
        ..input(visual_features, field_dimensions, None)
    }
}

fn project_point(point: Point2<Field>) -> DetectedVisualFeature {
    let field_to_camera = robot_to_camera() * Isometry3::<Field, Robot>::identity();
    let pixel = camera_intrinsic().project((field_to_camera * point.extend(0.0)).coords());
    DetectedVisualFeature {
        pixel,
        confidence: 0.9,
    }
}

fn project_point_from_pose(
    point: Point2<Field>,
    robot_to_field: Isometry3<Robot, Field>,
) -> DetectedVisualFeature {
    let field_to_camera = robot_to_camera() * robot_to_field.inverse();
    DetectedVisualFeature {
        pixel: camera_intrinsic().project((field_to_camera * point.extend(0.0)).coords()),
        confidence: 0.9,
    }
}

fn synthetic_features_from_pose(
    field: &FieldDimensions,
    robot_to_field: Isometry3<Robot, Field>,
) -> DetectedVisualFeatures {
    DetectedVisualFeatures {
        goalposts: vec![
            project_point_from_pose(field.goal_post(Half::Opponent, Side::Left), robot_to_field),
            project_point_from_pose(field.goal_post(Half::Opponent, Side::Right), robot_to_field),
        ],
        l_spots: vec![project_point_from_pose(
            field.corner(Half::Opponent, Side::Left),
            robot_to_field,
        )],
        t_spots: vec![project_point_from_pose(
            field.t_crossing(Side::Left),
            robot_to_field,
        )],
        x_spots: Vec::new(),
        penalty_spots: vec![project_point_from_pose(
            field.penalty_spot(Half::Opponent),
            robot_to_field,
        )],
    }
}

fn synthetic_features(field: &FieldDimensions) -> DetectedVisualFeatures {
    DetectedVisualFeatures {
        goalposts: vec![
            project_point(field.goal_post(Half::Opponent, Side::Left)),
            project_point(field.goal_post(Half::Opponent, Side::Right)),
        ],
        l_spots: vec![project_point(field.corner(Half::Opponent, Side::Left))],
        x_spots: Vec::new(),
        t_spots: vec![project_point(field.t_crossing(Side::Left))],
        penalty_spots: vec![project_point(field.penalty_spot(Half::Opponent))],
    }
}

fn perturb_pixels(features: &mut DetectedVisualFeatures) {
    for (index, feature) in features
        .goalposts
        .iter_mut()
        .chain(features.l_spots.iter_mut())
        .chain(features.t_spots.iter_mut())
        .chain(features.x_spots.iter_mut())
        .chain(features.penalty_spots.iter_mut())
        .enumerate()
    {
        let dx = if index.is_multiple_of(2) { 1.5 } else { -1.25 };
        let dy = if index.is_multiple_of(3) { -1.75 } else { 1.0 };
        feature.pixel = point![<Pixel>, feature.pixel.x() + dx, feature.pixel.y() + dy];
    }
}

fn add_distinct_l_spots(features: &mut DetectedVisualFeatures, target_count: usize) {
    let template = features.l_spots[0];
    let mut offset_index = features.l_spots.len();
    while features.supported_feature_count() < target_count {
        let mut detection = template;
        detection.pixel = point![<Pixel>,
            template.pixel.x() + 2.0 * offset_index as f32,
            template.pixel.y()
        ];
        features.l_spots.push(detection);
        offset_index += 1;
    }
}

#[test]
fn recovers_mixed_feature_associations_with_static_height_gate() -> Result<(), String> {
    let field = FieldDimensions::SPL_2025;
    let features = synthetic_features(&field);
    let Some(result) = solve(
        input(
            &features,
            &field,
            Some(Isometry3::<Robot, Field>::identity()),
        ),
        GlobalAssociationConfig::default(),
    ) else {
        return Err("synthetic features should localize".to_string());
    };

    assert_eq!(result.associations.len(), 5);
    assert!(result.debug.metric_rms_residual < 1.0e-3);
    Ok(())
}

#[test]
fn recovers_noisy_mixed_feature_assignments() -> Result<(), String> {
    let field = FieldDimensions::SPL_2025;
    let mut features = synthetic_features(&field);
    perturb_pixels(&mut features);

    let result = solve(
        input(
            &features,
            &field,
            Some(Isometry3::<Robot, Field>::identity()),
        ),
        GlobalAssociationConfig::default(),
    )
    .ok_or_else(|| "two-pixel noise should preserve the assignment".to_string())?;

    let expected = [
        field.goal_post(Half::Opponent, Side::Left),
        field.goal_post(Half::Opponent, Side::Right),
        field.corner(Half::Opponent, Side::Left),
        field.t_crossing(Side::Left),
        field.penalty_spot(Half::Opponent),
    ];
    assert_eq!(result.associations.len(), expected.len());
    assert!(expected.into_iter().all(|point| {
        result
            .associations
            .iter()
            .any(|association| (association.field_point.xy() - point).inner.norm() < 1.0e-4)
    }));
    Ok(())
}

#[test]
fn varied_pose_noise_sweep_never_returns_incorrect_associations() -> Result<(), String> {
    let field = FieldDimensions::SPL_2025;
    let expected = [
        field.goal_post(Half::Opponent, Side::Left),
        field.goal_post(Half::Opponent, Side::Right),
        field.corner(Half::Opponent, Side::Left),
        field.t_crossing(Side::Left),
        field.penalty_spot(Half::Opponent),
    ];
    let poses = [
        Isometry3::<Robot, Field>::identity(),
        Isometry3::wrap(nalgebra::Isometry3::new(
            nalgebra::vector![0.8, -0.4, 0.0],
            nalgebra::vector![0.0, 0.0, 0.35],
        )),
        Isometry3::wrap(nalgebra::Isometry3::new(
            nalgebra::vector![-1.1, 0.7, 0.0],
            nalgebra::vector![0.0, 0.0, -0.6],
        )),
    ];

    let mut accepted = 0;
    for pose in poses {
        let mut features = synthetic_features_from_pose(&field, pose);
        perturb_pixels(&mut features);
        let Some(result) = solve(
            input(&features, &field, Some(pose)),
            GlobalAssociationConfig::default(),
        ) else {
            continue;
        };
        accepted += 1;
        if result.associations.len() != expected.len()
            || !expected.into_iter().all(|point| {
                result
                    .associations
                    .iter()
                    .any(|association| (association.field_point.xy() - point).inner.norm() < 1.0e-4)
            })
        {
            return Err(format!("incorrect associations returned for pose {pose:?}"));
        }
    }
    if accepted == 0 {
        return Err("pose sweep did not produce any accepted result".to_string());
    }
    Ok(())
}

#[test]
fn matching_does_not_use_supplied_camera_height() {
    let field = FieldDimensions::SPL_2025;
    let features = synthetic_features(&field);
    assert!(
        solve(
            input_with_camera_height(&features, &field, 0.0),
            GlobalAssociationConfig::default(),
        )
        .is_some()
    );
}

#[test]
fn pose_hint_selects_concrete_symmetry_branch_after_certification() -> Result<(), String> {
    let field = FieldDimensions::SPL_2025;
    let features = synthetic_features(&field);
    let half_turn = Isometry3::<Robot, Field>::wrap(nalgebra::Isometry3::from_parts(
        nalgebra::Translation3::identity(),
        nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Vector3::z_axis(),
            std::f32::consts::PI,
        ),
    ));
    let result = solve(
        input(&features, &field, Some(half_turn)),
        GlobalAssociationConfig::default(),
    )
    .ok_or_else(|| "symmetric hint should only orient a certified result".to_string())?;

    assert!(result.associations.iter().any(|association| {
        (association.field_point.xy() - field.goal_post(Half::Own, Side::Right))
            .inner
            .norm()
            < 1.0e-4
    }));
    Ok(())
}

#[test]
fn pose_hint_does_not_create_fallback_associations() {
    let field = FieldDimensions::SPL_2025;
    let features = DetectedVisualFeatures {
        penalty_spots: vec![project_point(field.penalty_spot(Half::Opponent))],
        ..Default::default()
    };

    assert!(
        solve(
            input(
                &features,
                &field,
                Some(Isometry3::<Robot, Field>::identity()),
            ),
            GlobalAssociationConfig::default(),
        )
        .is_none()
    );
}

#[test]
fn duplicate_clutter_is_left_unmatched() -> Result<(), String> {
    let field = FieldDimensions::SPL_2025;
    let mut features = synthetic_features(&field);
    features.goalposts.push(features.goalposts[0]);
    let result = solve(
        input(
            &features,
            &field,
            Some(Isometry3::<Robot, Field>::identity()),
        ),
        GlobalAssociationConfig::default(),
    )
    .ok_or_else(|| "one duplicate should not prevent localization".to_string())?;

    assert_eq!(result.associations.len(), 5);
    for (index, association) in result.associations.iter().enumerate() {
        assert!(
            result.associations[index + 1..].iter().all(|other| {
                (association.field_point - other.field_point).inner.norm() > 1.0e-4
            })
        );
    }
    Ok(())
}

#[test]
fn rejects_static_height_outside_gate() {
    let field = FieldDimensions::SPL_2025;
    let features = synthetic_features(&field);
    let config = GlobalAssociationConfig {
        height_max: 0.4,
        association_gate: 0.01,
        ..Default::default()
    };

    assert!(
        solve(
            input(
                &features,
                &field,
                Some(Isometry3::<Robot, Field>::identity())
            ),
            config,
        )
        .is_none()
    );
}

#[test]
fn rejects_low_confidence_detections() {
    let field = FieldDimensions::SPL_2025;
    let mut features = synthetic_features(&field);
    for feature in features
        .goalposts
        .iter_mut()
        .chain(features.l_spots.iter_mut())
        .chain(features.t_spots.iter_mut())
        .chain(features.x_spots.iter_mut())
        .chain(features.penalty_spots.iter_mut())
    {
        feature.confidence = 0.1;
    }
    assert!(
        solve(
            input(
                &features,
                &field,
                Some(Isometry3::<Robot, Field>::identity())
            ),
            GlobalAssociationConfig::default(),
        )
        .is_none()
    );
}

#[test]
fn rejects_invalid_confidence_threshold() {
    let parameters = GlobalAssociationConfig {
        confidence_threshold: 1.1,
        ..Default::default()
    };

    assert_eq!(
        parameters.validate(),
        Err("global_localizer.confidence_threshold must be finite and in [0, 1]".to_string())
    );
}

#[test]
fn uniqueness_ratio_must_exceed_one() {
    let parameters = GlobalAssociationConfig {
        score_ratio: 1.0,
        ..Default::default()
    };

    assert_eq!(
        parameters.validate(),
        Err("global_localizer.score_ratio must be finite and > 1".to_string())
    );
}

#[test]
fn repeated_frames_are_associated_without_cross_frame_state() -> Result<(), String> {
    let field = FieldDimensions::SPL_2025;
    let features = synthetic_features(&field);
    let config = GlobalAssociationConfig::default();
    let mut workspace = SolverWorkspace::default();
    let first = workspace
        .solve(input(&features, &field, None), config)
        .ok_or_else(|| "first frame should localize".to_string())?;
    let second = workspace
        .solve(input(&features, &field, None), config)
        .ok_or_else(|| "second frame should localize independently".to_string())?;

    let association_key = |result: &GlobalAssociationResult| {
        result
            .associations
            .iter()
            .map(|association| (association.detection, association.field_point))
            .collect::<Vec<_>>()
    };
    assert_eq!(association_key(&first), association_key(&second));
    Ok(())
}

#[test]
fn detection_overflow_fails_closed_without_truncation() {
    let field = FieldDimensions::SPL_2025;
    let mut features = synthetic_features(&field);
    add_distinct_l_spots(&mut features, GLOBAL_LOCALIZER_MAX_DETECTIONS);
    assert_eq!(
        super::solver::retained_detection_count(
            input(&features, &field, None),
            GlobalAssociationConfig::default(),
        ),
        Some(GLOBAL_LOCALIZER_MAX_DETECTIONS)
    );

    add_distinct_l_spots(&mut features, GLOBAL_LOCALIZER_MAX_DETECTIONS + 1);
    assert_eq!(
        super::solver::retained_detection_count(
            input(&features, &field, None),
            GlobalAssociationConfig::default(),
        ),
        None
    );
}

#[test]
fn proposal_budget_overflow_fails_closed() {
    let field = FieldDimensions::SPL_2025;
    let features = synthetic_features(&field);

    assert_eq!(
        super::solver::proposal_count_with_limit(
            input(&features, &field, None),
            GlobalAssociationConfig::default(),
            1,
        ),
        None
    );
}

#[test]
fn non_finite_pose_hint_is_ignored() -> Result<(), String> {
    let field = FieldDimensions::SPL_2025;
    let features = synthetic_features(&field);
    let expected = solve(
        input(&features, &field, None),
        GlobalAssociationConfig::default(),
    )
    .ok_or_else(|| "synthetic features should localize".to_string())?;
    let mut invalid_hint = Isometry3::<Robot, Field>::identity();
    invalid_hint.inner.translation.vector.x = f32::NAN;
    let actual = solve(
        input(&features, &field, Some(invalid_hint)),
        GlobalAssociationConfig::default(),
    )
    .ok_or_else(|| "invalid hint should not suppress localization".to_string())?;

    assert_eq!(
        expected
            .associations
            .iter()
            .map(|association| association.field_point)
            .collect::<Vec<_>>(),
        actual
            .associations
            .iter()
            .map(|association| association.field_point)
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn non_finite_camera_extrinsic_is_rejected() {
    let field = FieldDimensions::SPL_2025;
    let features = synthetic_features(&field);
    let mut invalid_input = input(&features, &field, None);
    invalid_input.robot_to_camera.inner.translation.vector.z = f32::NAN;

    assert!(solve(invalid_input, GlobalAssociationConfig::default()).is_none());
}

#[test]
#[ignore = "manual rich-frame runtime characterization"]
fn rich_frame_runtime_characterization() {
    const SAMPLE_COUNT: u32 = 20;

    fn measure(
        solver: &mut SolverWorkspace,
        features: &DetectedVisualFeatures,
        field: &FieldDimensions,
    ) -> (std::time::Duration, bool) {
        let pose_hint = Some(Isometry3::<Robot, Field>::identity());
        let config = GlobalAssociationConfig::default();
        let _ = solver.solve(input(features, field, pose_hint), config);
        let start = std::time::Instant::now();
        let mut accepted = false;
        for _ in 0..SAMPLE_COUNT {
            accepted = solver
                .solve(input(features, field, pose_hint), config)
                .is_some();
        }
        (start.elapsed() / SAMPLE_COUNT, accepted)
    }

    let field = FieldDimensions::SPL_2025;
    let mut solver = SolverWorkspace::default();
    let nominal = synthetic_features(&field);
    let (elapsed, accepted) = measure(&mut solver, &nominal, &field);
    eprintln!(
        "nominal association took {:?}, result={}",
        elapsed, accepted
    );

    let mut features = DetectedVisualFeatures::default();
    for (class, point) in super::map::candidate_points(&field) {
        let target = match class {
            VisualFeatureClass::GoalPost => &mut features.goalposts,
            VisualFeatureClass::LSpot => &mut features.l_spots,
            VisualFeatureClass::TSpot => &mut features.t_spots,
            VisualFeatureClass::XSpot => &mut features.x_spots,
            VisualFeatureClass::PenaltySpot => &mut features.penalty_spots,
        };
        let limit = match class {
            VisualFeatureClass::GoalPost => 2,
            VisualFeatureClass::LSpot => 6,
            VisualFeatureClass::TSpot => 4,
            VisualFeatureClass::XSpot => 3,
            VisualFeatureClass::PenaltySpot => 2,
        };
        if target.len() < limit {
            target.push(project_point(point));
        }
    }

    let (elapsed, accepted) = measure(&mut solver, &features, &field);
    eprintln!(
        "rich-frame association took {:?}, result={}",
        elapsed, accepted
    );

    let mut bounded_features = DetectedVisualFeatures {
        x_spots: vec![
            project_point(field.center()),
            project_point(field.x_crossing(Side::Left)),
            project_point(field.x_crossing(Side::Right)),
        ],
        penalty_spots: vec![
            project_point(field.penalty_spot(Half::Opponent)),
            project_point(field.penalty_spot(Half::Own)),
        ],
        ..Default::default()
    };
    let detections_per_class = GLOBAL_LOCALIZER_MAX_DETECTIONS / 2;
    while bounded_features.x_spots.len() < detections_per_class {
        let mut clutter = bounded_features.x_spots[0];
        clutter.confidence = 0.36;
        clutter.pixel = point![<Pixel>,
            clutter.pixel.x() + 2.0 * bounded_features.x_spots.len() as f32,
            clutter.pixel.y()
        ];
        bounded_features.x_spots.push(clutter);
    }
    while bounded_features.penalty_spots.len() < detections_per_class {
        let mut clutter = bounded_features.penalty_spots[0];
        clutter.confidence = 0.36;
        clutter.pixel = point![<Pixel>,
            clutter.pixel.x() + 2.0 * bounded_features.penalty_spots.len() as f32,
            clutter.pixel.y()
        ];
        bounded_features.penalty_spots.push(clutter);
    }
    assert_eq!(
        super::solver::retained_detection_count(
            input(&bounded_features, &field, None),
            GlobalAssociationConfig::default(),
        ),
        Some(GLOBAL_LOCALIZER_MAX_DETECTIONS)
    );
    let (elapsed, accepted) = measure(&mut solver, &bounded_features, &field);
    eprintln!(
        "32-detection association took {:?}, result={}",
        elapsed, accepted
    );
}

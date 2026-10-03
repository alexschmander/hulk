use coordinate_systems::{Field, Pixel, Robot};
use linear_algebra::{Isometry3, Point2, Point3};
use linear_sum_assignment::{AssignmentSolver, Objective};
use nalgebra::{Matrix2, Matrix2x3, Matrix3, Matrix3x6, Matrix6, Vector3};
use ndarray::Array2;
use types::{localization::PoseEstimate, visual_localization_next::FieldMarkAssociation};

use crate::{
    AssociationResult, DetectedVisualFeature, DetectedVisualFeatures,
    FieldMarkAssociationParameters, TrackingAssociationInput as AssociationInput,
    VisualFeatureClass, features::raw_detections, map::LandmarkMap,
};

#[derive(Clone, Copy)]
struct Prediction {
    pixel: Point2<Pixel>,
    covariance: Matrix2<f64>,
}

struct PredictionNoise {
    pose: Matrix6<f64>,
    pixel_variance: f64,
}

pub(crate) fn associate(
    input: AssociationInput<'_>,
    parameters: &FieldMarkAssociationParameters,
) -> Option<AssociationResult> {
    let age = validated_age(input, parameters)?;
    let map = LandmarkMap::new(
        input.field_dimensions,
        parameters.global_localizer.symmetry_epsilon,
    );
    if map
        .landmarks
        .iter()
        .any(|landmark| !landmark.xy.coords().inner.iter().all(|x| x.is_finite()))
    {
        return None;
    }
    let config = parameters.global_localizer;
    let detections = filter_detections(input.visual_features, &map, config)?;
    if detections.len() < config.min_inliers {
        return None;
    }
    let estimate = input.geometry.estimate;
    let noise = PredictionNoise {
        pose: estimate.covariance + process_covariance(estimate.pose, age, parameters),
        pixel_variance: f64::from(config.detection_pixel_sigma).powi(2),
    };
    let predictions = map
        .landmarks
        .iter()
        .map(|point| project_landmark(point.xy.extend(0.0), input, &noise))
        .collect::<Vec<_>>();
    match_predictions(&detections, &map, &predictions, &noise, parameters)
}

fn validated_age(
    input: AssociationInput<'_>,
    parameters: &FieldMarkAssociationParameters,
) -> Option<f32> {
    let tracking = parameters.tracking;
    let estimate = input.geometry.estimate;
    let last_successful_solve = input.geometry.last_successful_solve;
    if !valid_geometry(input, estimate, parameters.global_localizer)
        || !valid_covariance(estimate.covariance, tracking)
    {
        return None;
    }
    if input.time < last_successful_solve {
        return None;
    }
    let age = input.time.duration_since(last_successful_solve);
    if age > tracking.max_age {
        return None;
    }
    Some(age.as_secs_f32())
}

fn valid_geometry(
    input: AssociationInput<'_>,
    estimate: PoseEstimate<Robot, Field>,
    config: crate::GlobalLocalizerParameters,
) -> bool {
    let intrinsic = input.camera_intrinsic;
    [
        estimate.pose.inner.to_homogeneous(),
        input.robot_to_camera.inner.to_homogeneous().cast::<f64>(),
    ]
    .iter()
    .all(|matrix| matrix.iter().all(|x| x.is_finite()))
        && intrinsic.is_valid()
        && input.visual_features.supported_feature_count() <= config.max_input_detections
}

fn valid_covariance(
    covariance: Matrix6<f64>,
    parameters: crate::TrackingAssociationParameters,
) -> bool {
    if !covariance.iter().all(|x| x.is_finite())
        || (covariance - covariance.transpose()).amax()
            > f64::from(parameters.covariance_symmetry_tolerance)
    {
        return false;
    }
    covariance
        .try_symmetric_eigen(f64::EPSILON, parameters.covariance_eigen_max_iterations)
        .is_some_and(|eigen| {
            eigen.eigenvalues.iter().all(|value| {
                value.is_finite() && *value >= -f64::from(parameters.covariance_psd_tolerance)
            })
        })
}

fn process_covariance(
    robot_to_field: Isometry3<Robot, Field, f64>,
    age: f32,
    parameters: &FieldMarkAssociationParameters,
) -> Matrix6<f64> {
    let config = parameters.global_localizer;
    let tracking = parameters.tracking;
    // Horizontal isotropy makes these floors invariant to the Local-to-Field yaw.
    let field_to_robot = robot_to_field.inner.rotation.inverse().to_rotation_matrix();
    let rotation_noise = Matrix3::from_diagonal(&Vector3::new(
        f64::from(config.imu_tilt_sigma).powi(2),
        f64::from(config.imu_tilt_sigma).powi(2),
        (f64::from(age) * f64::from(tracking.yaw_sigma_per_second)).powi(2),
    ));
    let position_variance =
        (f64::from(age) * f64::from(tracking.position_sigma_per_second)).powi(2);
    let translation_noise = Matrix3::from_diagonal(&Vector3::new(
        position_variance,
        position_variance,
        position_variance + f64::from(config.height_sigma).powi(2),
    ));
    let mut current_covariance = Matrix6::zeros();
    current_covariance.fixed_view_mut::<3, 3>(0, 0).copy_from(
        &(field_to_robot.matrix() * rotation_noise * field_to_robot.matrix().transpose()),
    );
    current_covariance.fixed_view_mut::<3, 3>(3, 3).copy_from(
        &(field_to_robot.matrix() * translation_noise * field_to_robot.matrix().transpose()),
    );
    current_covariance
}

fn filter_detections(
    features: &DetectedVisualFeatures,
    map: &LandmarkMap,
    config: crate::GlobalLocalizerParameters,
) -> Option<Vec<(VisualFeatureClass, DetectedVisualFeature)>> {
    let mut detections = raw_detections(features)
        .filter(|(_, feature)| {
            (config.confidence_threshold..=1.0).contains(&feature.confidence)
                && feature.pixel.coords().inner.iter().all(|x| x.is_finite())
        })
        .collect::<Vec<_>>();
    detections.sort_by(|(a, x), (b, y)| {
        (y.confidence * map.rarity_weight(*b)).total_cmp(&(x.confidence * map.rarity_weight(*a)))
    });
    let mut retained: Vec<(VisualFeatureClass, DetectedVisualFeature)> = Vec::new();
    for (class, detection) in detections {
        if retained.iter().any(|(other_class, other)| {
            *other_class == class
                && (other.pixel - detection.pixel).inner.norm() <= config.duplicate_pixel_distance
        }) {
            continue;
        }
        if retained.len() == config.max_retained_detections {
            return None;
        }
        retained.push((class, detection));
    }
    Some(retained)
}

fn match_predictions(
    detections: &[(VisualFeatureClass, DetectedVisualFeature)],
    map: &LandmarkMap,
    predictions: &[Option<Prediction>],
    _noise: &PredictionNoise,
    parameters: &FieldMarkAssociationParameters,
) -> Option<AssociationResult> {
    let log_likelihoods = prediction_log_likelihoods(
        detections,
        map,
        predictions,
        parameters.global_localizer.mahalanobis_gate,
    )?;
    let pairs = match detections.len() {
        3..=5 => None,
        _ => {
            // ponytail: more than five features retain marginal assignment, not a joint likelihood.
            // Extend joint search only with a measured real-time bound. Row normalization bounds weights.
            let mut benefits = log_likelihoods;
            let m = map.landmarks.len();
            for (row, (_, detection)) in detections.iter().enumerate() {
                let maximum = benefits
                    .row(row)
                    .iter()
                    .take(m)
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                for column in 0..benefits.ncols() {
                    let likelihood = benefits[(row, column)];
                    benefits[(row, column)] = if column < m && likelihood.is_finite() {
                        detection.confidence * (likelihood - maximum).exp()
                    } else {
                        0.0
                    };
                }
            }
            unique_assignment(
                &mut benefits,
                map.landmarks.len(),
                parameters.tracking.score_ratio,
            )
        }
    }?;
    certify(detections, map, predictions, &pairs, parameters)
}

fn prediction_log_likelihoods(
    detections: &[(VisualFeatureClass, DetectedVisualFeature)],
    map: &LandmarkMap,
    predictions: &[Option<Prediction>],
    mahalanobis_gate: f32,
) -> Option<Array2<f32>> {
    let n = detections.len();
    let m = map.landmarks.len();
    // The extra n columns become zero-benefit unassignment slots after normalization.
    let mut log_likelihoods = Array2::from_elem((n, m + n), f32::NEG_INFINITY);
    for (landmark, prediction) in predictions.iter().enumerate() {
        let Some(prediction) = prediction else {
            continue;
        };
        let cholesky = prediction.covariance.cholesky()?;
        let log_determinant = 2.0 * cholesky.l().diagonal().iter().map(|x| x.ln()).sum::<f64>();
        let information = cholesky.inverse();
        for (row, (_, detection)) in detections
            .iter()
            .enumerate()
            .filter(|(_, (class, _))| *class == map.landmarks[landmark].class)
        {
            let error = (prediction.pixel - detection.pixel).inner.cast::<f64>();
            let mahalanobis = error.dot(&(information * error));
            // The hard output distance must not hide a statistically plausible rival.
            if (0.0..f64::from(mahalanobis_gate)).contains(&mahalanobis) {
                log_likelihoods[(row, landmark)] = (-0.5 * (mahalanobis + log_determinant)) as f32;
            }
        }
    }
    Some(log_likelihoods)
}

fn certify(
    detections: &[(VisualFeatureClass, DetectedVisualFeature)],
    map: &LandmarkMap,
    predictions: &[Option<Prediction>],
    pairs: &[(usize, usize)],
    parameters: &FieldMarkAssociationParameters,
) -> Option<AssociationResult> {
    let config = parameters.global_localizer;
    let tracking = parameters.tracking;
    if pairs.len() < config.min_inliers
        || pairs.iter().any(|&(row, column)| {
            predictions[column].is_none_or(|prediction| {
                (prediction.pixel - detections[row].1.pixel)
                    .inner
                    .norm_squared()
                    > tracking.max_pixel_distance.powi(2)
            })
        })
    {
        return None;
    }
    Some(AssociationResult {
        associations: pairs
            .iter()
            .map(|&(row, column)| FieldMarkAssociation {
                detection: detections[row].1.pixel,
                field_point: map.landmarks[column].xy.extend(0.0),
            })
            .collect(),
        source: types::visual_localization_next::VisualAssociationSource::Tracking,
        // GlobalLocalizationDebug has a metric residual, not an image-space residual.
        debug: None,
    })
}

fn project_landmark(
    point: Point3<Field>,
    input: AssociationInput<'_>,
    noise: &PredictionNoise,
) -> Option<Prediction> {
    let point_robot = Isometry3::<Robot, Field>::wrap(input.geometry.estimate.pose.inner.cast())
        .inverse()
        * point;
    let camera = input.robot_to_camera * point_robot;
    if camera.z() <= 0.0 || !camera.coords().inner.iter().all(|x| x.is_finite()) {
        return None;
    }
    let intrinsic = input.camera_intrinsic;
    let pixel = intrinsic.project(camera.coords());
    let projection = Matrix2x3::new(
        intrinsic.focals.x / camera.z(),
        0.0,
        -intrinsic.focals.x * camera.x() / camera.z().powi(2),
        0.0,
        intrinsic.focals.y / camera.z(),
        -intrinsic.focals.y * camera.y() / camera.z().powi(2),
    );
    let inverse_point = |p: Vector3<f32>| {
        Matrix3x6::from_columns(&[
            p.cross(&Vector3::x()),
            p.cross(&Vector3::y()),
            p.cross(&Vector3::z()),
            -Vector3::x(),
            -Vector3::y(),
            -Vector3::z(),
        ])
    };
    // Right-local pose perturbation: J_camera = R_rc [skew(point_robot), -I].
    let jacobian = projection
        * input
            .robot_to_camera
            .inner
            .rotation
            .to_rotation_matrix()
            .matrix()
        * inverse_point(point_robot.coords().inner);
    let jacobian = jacobian.cast::<f64>();
    let covariance =
        jacobian * noise.pose * jacobian.transpose() + Matrix2::identity() * noise.pixel_variance;
    (pixel.coords().inner.iter().all(|x| x.is_finite()) && covariance.iter().all(|x| x.is_finite()))
        .then_some(Prediction { pixel, covariance })
}

fn unique_assignment(
    benefits: &mut Array2<f32>,
    landmarks: usize,
    score_ratio: f32,
) -> Option<Vec<(usize, usize)>> {
    let mut assignment = AssignmentSolver::new(benefits.dim());
    let columns = assignment
        .solve(benefits.view(), Objective::Maximize)
        .ok()?;
    let pairs = columns
        .iter()
        .enumerate()
        .filter_map(|(row, column)| {
            let column = (*column)?;
            (column < landmarks && benefits[(row, column)] > 0.0).then_some((row, column))
        })
        .collect::<Vec<_>>();
    let score = pairs.iter().map(|&(r, c)| benefits[(r, c)]).sum::<f32>();
    // Every distinct assignment omits at least one winning edge. This checks all alternatives
    // with one additional bounded assignment solve per winning edge, without fitting candidate poses.
    for &(row, column) in &pairs {
        let saved = benefits[(row, column)];
        // With one zero-benefit dummy per row, zeroing this edge has the same
        // optimal score as forbidding it: its row can always use a free dummy.
        benefits[(row, column)] = 0.0;
        let alternative = assignment
            .solve(benefits.view(), Objective::Maximize)
            .ok()?;
        let alternative_score = alternative
            .iter()
            .enumerate()
            .filter_map(|(r, c)| c.map(|c| benefits[(r, c)]))
            .sum::<f32>();
        benefits[(row, column)] = saved;
        if score - alternative_score <= saved * (1.0 - 1.0 / score_ratio) {
            return None;
        }
    }
    Some(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covariance_numerical_tolerances_are_applied() {
        let mut parameters = crate::TrackingAssociationParameters::default();
        let mut covariance = Matrix6::identity();
        covariance[(0, 1)] = 5.0e-5;
        assert!(!valid_covariance(covariance, parameters));
        parameters.covariance_symmetry_tolerance = 1.0e-4;
        assert!(valid_covariance(covariance, parameters));
        covariance = Matrix6::identity();
        covariance[(0, 0)] = -5.0e-6;
        assert!(!valid_covariance(covariance, parameters));
        parameters.covariance_psd_tolerance = 1.0e-5;
        assert!(valid_covariance(covariance, parameters));
    }

    #[test]
    fn retained_detection_limit_can_exceed_the_old_ceiling() {
        let map = LandmarkMap::new(
            &types::field_dimensions::FieldDimensions::SPL_2025,
            crate::GlobalLocalizerParameters::default().symmetry_epsilon,
        );
        let features = DetectedVisualFeatures {
            penalty_spots: (0..33)
                .map(|index| DetectedVisualFeature {
                    pixel: linear_algebra::point![index as f32 * 10.0, 100.0],
                    confidence: 1.0,
                })
                .collect(),
            ..Default::default()
        };
        let mut config = crate::GlobalLocalizerParameters::default();
        assert!(filter_detections(&features, &map, config).is_none());
        config.max_retained_detections = 33;
        config.validate().unwrap();
        assert_eq!(
            filter_detections(&features, &map, config).unwrap().len(),
            33
        );
    }
    use linear_algebra::point;
    use ndarray::array;

    #[test]
    fn single_prior_covariance_matches_finite_differences_and_coherent_two_pose_model() {
        use nalgebra::{Matrix2x6, Translation3, UnitQuaternion};
        use types::{
            field_dimensions::FieldDimensions, visual_localization_next::AssociationGeometry,
        };

        let pose = |translation: [f32; 3], angles: [f32; 3]| {
            nalgebra::Isometry3::from_parts(
                Translation3::from(Vector3::from(translation)),
                UnitQuaternion::from_euler_angles(angles[0], angles[1], angles[2]),
            )
        };
        let local = pose([-1.0, 0.5, 0.7], [0.1, -0.15, 0.6]);
        let alignment = pose([0.7, -0.2, 0.0], [0.0, 0.0, 0.5]);
        let field_pose = alignment * local;
        let factor = Matrix6::from_fn(|r, c| {
            if r == c {
                0.02 + r as f32 * 0.005
            } else if c < r {
                0.01 * (r + c + 1) as f32 / 6.0
            } else {
                0.0
            }
        });
        let covariance = factor * factor.transpose();
        let estimate = PoseEstimate {
            pose: Isometry3::wrap(field_pose.cast()),
            covariance: covariance.cast(),
        };
        let geometry = AssociationGeometry {
            generation: 0,
            epoch: 0,
            estimate,
            last_successful_solve: ros_z::time::Time::zero(),
        };
        let features = crate::DetectedVisualFeatures::default();
        let input = AssociationInput {
            visual_features: &features,
            robot_to_camera: Isometry3::wrap(pose([0.15, 0.07, 0.08], [3.0, 0.05, -0.1]).inverse()),
            geometry: &geometry,
            camera_intrinsic: projection::intrinsic::Intrinsic::new(
                nalgebra::vector![430.0, 520.0],
                point![320.0, 240.0],
            ),
            field_dimensions: &FieldDimensions::SPL_2025,
            time: ros_z::time::Time::zero(),
        };
        let point = point![0.5, -0.3, 0.0];
        let parameters = FieldMarkAssociationParameters::default();
        let process = process_covariance(estimate.pose, 0.7, &parameters);
        let noise = PredictionNoise {
            pose: covariance.cast::<f64>() + process,
            pixel_variance: 2.3_f64.powi(2),
        };
        let projected = project_landmark(point, input, &noise).unwrap();
        let pixel = projected.pixel;
        let actual = projected.covariance;
        let project = |pose: nalgebra::Isometry3<f64>| {
            let p = input.robot_to_camera.inner.cast::<f64>()
                * pose.inverse()
                * point.inner.cast::<f64>();
            nalgebra::vector![430.0 * p.x / p.z + 320.0, 520.0 * p.y / p.z + 240.0]
        };
        let jacobian = Matrix2x6::<f64>::from_columns(&std::array::from_fn::<_, 6, _>(|axis| {
            let sample = |step| {
                let mut tangent = Vector3::<f64>::zeros();
                tangent[axis % 3] = step;
                let perturbation = if axis < 3 {
                    nalgebra::Isometry3::rotation(tangent)
                } else {
                    nalgebra::Isometry3::translation(tangent.x, tangent.y, tangent.z)
                };
                project(field_pose.cast::<f64>() * perturbation)
            };
            (sample(1.0e-5) - sample(-1.0e-5)) / 2.0e-5
        }));
        let expected = jacobian * noise.pose * jacobian.transpose()
            + Matrix2::identity() * noise.pixel_variance;
        assert!(
            (pixel.coords().inner - project(field_pose.cast::<f64>()).cast::<f32>()).norm() < 0.001
        );
        assert!(
            (actual - expected).norm() < expected.norm() * 0.0002,
            "actual {actual}, expected {expected}"
        );
        let without_correlations =
            jacobian * Matrix6::from_diagonal(&noise.pose.diagonal()) * jacobian.transpose()
                + Matrix2::identity() * noise.pixel_variance;
        assert!((actual - without_correlations).norm() > expected.norm() * 0.01);
    }

    #[test]
    fn assignment_is_one_to_one_and_rejects_near_tied_rematching() {
        let mut benefits = array![
            [0.9, 0.8, 0.0, 0.0, 0.0, 0.0],
            [0.89, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
        ];
        let pairs = unique_assignment(&mut benefits, 3, 1.05).unwrap();
        assert_eq!(pairs, vec![(0, 1), (1, 0), (2, 2)]);
        benefits[(1, 1)] = 0.8;
        assert!(unique_assignment(&mut benefits, 3, 1.05).is_none());
        benefits[(0, 1)] = 0.0;
        assert_eq!(
            unique_assignment(&mut benefits, 3, 1.05).unwrap(),
            vec![(0, 0), (1, 1), (2, 2)]
        );
    }
}

use ::types::field_dimensions::FieldDimensions;
use coordinate_systems::{Camera, Field, Ground, Pixel, Robot};
use linear_algebra::{Isometry3, Point2, point};
use linear_sum_assignment::{AssignmentSolver, Objective};
use nalgebra::{Matrix2, Similarity2, Translation3, UnitQuaternion, Vector2, Vector3};
use ndarray::{ArrayView2, s};
use projection::intrinsic::Intrinsic;
use smallvec::SmallVec;
use types::visual_localization::{FieldMarkAssociation, GlobalLocalizationDebug};

use crate::{DetectedVisualFeature, DetectedVisualFeatures};

use super::{
    FEATURE_CLASSES, GLOBAL_LOCALIZER_MAX_DETECTIONS, GlobalAssociationConfig, VisualFeatureClass,
    config::{GLOBAL_LOCALIZER_MAX_INPUT_DETECTIONS, GLOBAL_LOCALIZER_MAX_PROPOSALS},
    map::{FIELD_LANDMARK_COUNT, LandmarkMap, MAX_LANDMARKS_PER_CLASS},
};

const RAY_EPSILON: f32 = 1.0e-4;
const TILT_JACOBIAN_STEP: f32 = 1.0e-3;
const COVARIANCE_FLOOR: f32 = 1.0e-8;
const DUPLICATE_PIXEL_DISTANCE: f32 = 1.0;
const MAX_ASSIGNMENT_REFIT_ITERATIONS: usize = 4;
const MAX_ASSIGNMENT_COLUMNS: usize = MAX_LANDMARKS_PER_CLASS + GLOBAL_LOCALIZER_MAX_DETECTIONS;
const INLINE_MATCH_CAPACITY: usize = 8;
type MatchSet = SmallVec<[Match; INLINE_MATCH_CAPACITY]>;
type AssociationKey = SmallVec<[(usize, usize); INLINE_MATCH_CAPACITY]>;

#[derive(Clone, Debug)]
pub(crate) struct GlobalLocalizationInput<'a> {
    pub visual_features: &'a DetectedVisualFeatures,
    pub field_dimensions: &'a FieldDimensions,
    pub ground_to_robot: Isometry3<Ground, Robot>,
    pub robot_to_camera: Isometry3<Robot, Camera>,
    pub camera_intrinsic: Intrinsic,
    pub pose_hint: Option<Isometry3<Robot, Field>>,
}

#[derive(Clone, Copy, Debug)]
struct Detection {
    id: usize,
    class: VisualFeatureClass,
    pixel: Point2<Pixel>,
    weight: f32,
    unmatched_penalty: f32,
    a: Vector2<f32>,
    covariance: Matrix2<f32>,
}

#[derive(Clone, Copy, Debug)]
struct Match {
    detection: usize,
    landmark: usize,
    residual: f32,
    mahalanobis: f32,
}

const EMPTY_MATCH: Match = Match {
    detection: 0,
    landmark: 0,
    residual: 0.0,
    mahalanobis: 0.0,
};

struct AssignmentScratch {
    benefits: [[f32; MAX_ASSIGNMENT_COLUMNS]; GLOBAL_LOCALIZER_MAX_DETECTIONS],
    residuals: [[(f32, f32); MAX_LANDMARKS_PER_CLASS]; GLOBAL_LOCALIZER_MAX_DETECTIONS],
    detection_indices: [usize; GLOBAL_LOCALIZER_MAX_DETECTIONS],
    predicted: [Vector2<f32>; GLOBAL_LOCALIZER_MAX_DETECTIONS],
    metric_residuals: [[f32; FIELD_LANDMARK_COUNT]; GLOBAL_LOCALIZER_MAX_DETECTIONS],
    selected_columns: [usize; GLOBAL_LOCALIZER_MAX_DETECTIONS],
    assignment: AssignmentSolver<f32>,
}

impl Default for AssignmentScratch {
    fn default() -> Self {
        Self {
            benefits: [[0.0; MAX_ASSIGNMENT_COLUMNS]; GLOBAL_LOCALIZER_MAX_DETECTIONS],
            residuals: [[(f32::INFINITY, f32::INFINITY); MAX_LANDMARKS_PER_CLASS];
                GLOBAL_LOCALIZER_MAX_DETECTIONS],
            detection_indices: [0; GLOBAL_LOCALIZER_MAX_DETECTIONS],
            predicted: std::array::from_fn(|_| Vector2::zeros()),
            metric_residuals: [[f32::INFINITY; FIELD_LANDMARK_COUNT];
                GLOBAL_LOCALIZER_MAX_DETECTIONS],
            selected_columns: [0; GLOBAL_LOCALIZER_MAX_DETECTIONS],
            assignment: AssignmentSolver::new((
                GLOBAL_LOCALIZER_MAX_DETECTIONS,
                MAX_ASSIGNMENT_COLUMNS,
            )),
        }
    }
}

#[derive(Clone, Debug)]
struct Candidate {
    transform: Similarity2<f32>,
    matches: MatchSet,
    score: f32,
    rms: f32,
    key: AssociationKey,
    orbit_key: AssociationKey,
}

struct Problem<'a> {
    input: GlobalLocalizationInput<'a>,
    config: GlobalAssociationConfig,
    map: LandmarkMap,
    detections: Vec<Detection>,
}

pub(crate) struct GlobalAssociationResult {
    pub associations: Vec<FieldMarkAssociation>,
    pub debug: GlobalLocalizationDebug,
}

#[cfg(test)]
pub(crate) fn solve(
    input: GlobalLocalizationInput<'_>,
    config: GlobalAssociationConfig,
) -> Option<GlobalAssociationResult> {
    SolverWorkspace::default().solve(input, config)
}

#[derive(Default)]
pub(crate) struct SolverWorkspace {
    proposals: Vec<Similarity2<f32>>,
    assignment_scratch: AssignmentScratch,
}

impl SolverWorkspace {
    pub(crate) fn solve(
        &mut self,
        input: GlobalLocalizationInput<'_>,
        config: GlobalAssociationConfig,
    ) -> Option<GlobalAssociationResult> {
        config.validate().ok()?;
        let problem = Problem::new(input, config)?;
        let candidate = solve_problem(&problem, &mut self.proposals, &mut self.assignment_scratch)?;
        Some(to_public(&problem, &candidate))
    }
}

impl<'a> Problem<'a> {
    fn new(input: GlobalLocalizationInput<'a>, config: GlobalAssociationConfig) -> Option<Self> {
        if !valid_intrinsic(input.camera_intrinsic)
            || !input
                .ground_to_robot
                .inner
                .to_homogeneous()
                .iter()
                .all(|value| value.is_finite())
            || !input
                .robot_to_camera
                .inner
                .to_homogeneous()
                .iter()
                .all(|value| value.is_finite())
            || input.visual_features.supported_feature_count()
                > GLOBAL_LOCALIZER_MAX_INPUT_DETECTIONS
        {
            return None;
        }
        let map = LandmarkMap::new(input.field_dimensions);
        let detections = preprocess(&input, &map, config)?;
        (detections.len() >= config.min_inliers.max(2)).then_some(Self {
            input,
            config,
            map,
            detections,
        })
    }
}

fn preprocess(
    input: &GlobalLocalizationInput<'_>,
    map: &LandmarkMap,
    config: GlobalAssociationConfig,
) -> Option<Vec<Detection>> {
    let ground_to_camera = input.robot_to_camera * input.ground_to_robot;
    let camera_to_ground = ground_to_camera.inverse();
    let camera_to_ground_rotation = camera_to_ground.inner.rotation;
    let camera_origin_z = camera_to_ground.inner.translation.vector.z;
    let mut detections = raw_detections(input.visual_features)
        .enumerate()
        .filter_map(|(id, (class, feature))| {
            if !map.has_class(class)
                || !feature.confidence.is_finite()
                || feature.confidence < config.confidence_threshold
                || feature.confidence > 1.0
            {
                return None;
            }
            let bearing = input.camera_intrinsic.bearing(feature.pixel).inner;
            let ray = camera_to_ground_rotation * bearing;
            if camera_origin_z.abs() > RAY_EPSILON && camera_origin_z * ray.z >= 0.0 {
                return None;
            }
            let a = normalized_ground_direction(ray)?;
            let covariance = normalized_ray_covariance(
                input.camera_intrinsic,
                feature.pixel,
                camera_to_ground_rotation,
                config,
            )?;
            let weight = feature.confidence * map.rarity_weight(class);
            Some(Detection {
                id,
                class,
                pixel: feature.pixel,
                weight,
                unmatched_penalty: if feature.confidence >= 0.5 {
                    0.5 * weight
                } else {
                    0.0
                },
                a,
                covariance,
            })
        })
        .collect::<Vec<_>>();

    detections.sort_by(|left, right| {
        right
            .weight
            .total_cmp(&left.weight)
            .then_with(|| left.id.cmp(&right.id))
    });
    let mut retained_count = 0;
    for candidate_index in 0..detections.len() {
        let detection = detections[candidate_index];
        if detections[..retained_count].iter().any(|other| {
            other.class == detection.class
                && (other.pixel - detection.pixel).inner.norm() <= DUPLICATE_PIXEL_DISTANCE
        }) {
            continue;
        }
        if retained_count == GLOBAL_LOCALIZER_MAX_DETECTIONS {
            return None;
        }
        detections[retained_count] = detection;
        retained_count += 1;
    }
    detections.truncate(retained_count);
    Some(detections)
}

#[cfg(test)]
pub(super) fn retained_detection_count(
    input: GlobalLocalizationInput<'_>,
    config: GlobalAssociationConfig,
) -> Option<usize> {
    let map = LandmarkMap::new(input.field_dimensions);
    preprocess(&input, &map, config).map(|detections| detections.len())
}

fn solve_problem(
    problem: &Problem<'_>,
    proposals: &mut Vec<Similarity2<f32>>,
    assignment_scratch: &mut AssignmentScratch,
) -> Option<Candidate> {
    proposals.clear();
    generate_proposals(problem, proposals, GLOBAL_LOCALIZER_MAX_PROPOSALS)?;

    let mut best = None;
    let mut runner_up = None;
    for &proposal in proposals.iter() {
        if let Some(candidate) = evaluate(problem, proposal, assignment_scratch) {
            retain_orbit_candidate(&mut best, &mut runner_up, candidate);
        }
    }

    let best = select_unique_orbit(best, runner_up, problem.config.score_ratio)?;
    select_symmetry_representative(problem, best)
}

fn generate_proposals(
    problem: &Problem<'_>,
    proposals: &mut Vec<Similarity2<f32>>,
    limit: usize,
) -> Option<()> {
    for first in 0..problem.detections.len() {
        for second in first + 1..problem.detections.len() {
            let source = problem.detections[second].a - problem.detections[first].a;
            let source_baseline = source.norm();
            if !source_baseline.is_finite()
                || source_baseline < problem.config.min_detection_baseline
            {
                continue;
            }
            let first_landmarks = problem
                .map
                .landmarks_for_class(problem.detections[first].class);
            let second_landmarks = problem
                .map
                .landmarks_for_class(problem.detections[second].class);
            for &first_landmark in first_landmarks {
                for &second_landmark in second_landmarks {
                    if first_landmark == second_landmark {
                        continue;
                    }
                    let target = (problem.map.landmarks[second_landmark].xy
                        - problem.map.landmarks[first_landmark].xy)
                        .inner;
                    let target_baseline = target.norm();
                    if !target_baseline.is_finite()
                        || target_baseline < problem.config.min_map_baseline
                    {
                        continue;
                    }
                    let scale = target_baseline / source_baseline;
                    if !plausible_scale(scale, problem.config) {
                        continue;
                    }
                    let yaw = target.y.atan2(target.x) - source.y.atan2(source.x);
                    let rotation = nalgebra::UnitComplex::new(yaw);
                    let translation = problem.map.landmarks[first_landmark].xy.coords().inner
                        - scale * (rotation * problem.detections[first].a);
                    let transform = Similarity2::new(translation, yaw, scale);
                    if proposals.len() == limit {
                        return None;
                    }
                    proposals.push(transform);
                }
            }
        }
    }
    Some(())
}

#[cfg(test)]
pub(super) fn proposal_count_with_limit(
    input: GlobalLocalizationInput<'_>,
    config: GlobalAssociationConfig,
    limit: usize,
) -> Option<usize> {
    config.validate().ok()?;
    let problem = Problem::new(input, config)?;
    let mut proposals = Vec::new();
    generate_proposals(&problem, &mut proposals, limit)?;
    Some(proposals.len())
}

fn retain_orbit_candidate(
    best: &mut Option<Candidate>,
    runner_up: &mut Option<Candidate>,
    candidate: Candidate,
) {
    if let Some(current) = best
        .as_ref()
        .filter(|current| current.orbit_key == candidate.orbit_key)
    {
        if better(&candidate, current) {
            *best = Some(candidate);
        }
        return;
    }
    if let Some(current) = runner_up
        .as_ref()
        .filter(|current| current.orbit_key == candidate.orbit_key)
    {
        if better(&candidate, current) {
            *runner_up = Some(candidate);
            if matches!((best.as_ref(), runner_up.as_ref()), (Some(best), Some(runner_up)) if better(runner_up, best))
            {
                std::mem::swap(best, runner_up);
            }
        }
        return;
    }
    if best
        .as_ref()
        .is_none_or(|current| better(&candidate, current))
    {
        *runner_up = best.replace(candidate);
    } else if runner_up
        .as_ref()
        .is_none_or(|current| better(&candidate, current))
    {
        *runner_up = Some(candidate);
    }
}

fn select_unique_orbit(
    best: Option<Candidate>,
    runner_up: Option<Candidate>,
    score_ratio: f32,
) -> Option<Candidate> {
    let best = best?;
    if runner_up
        .as_ref()
        .is_some_and(|runner_up| best.score / runner_up.score < score_ratio)
    {
        None
    } else {
        Some(best)
    }
}

fn select_symmetry_representative(problem: &Problem<'_>, best: Candidate) -> Option<Candidate> {
    let symmetric_key = symmetric_key(&best.key, &problem.map);
    let Some(hint) = problem.input.pose_hint.filter(is_finite_pose) else {
        return Some(if symmetric_key < best.key {
            symmetric_candidate(&best, &problem.map)
        } else {
            best
        });
    };

    let symmetric = symmetric_candidate(&best, &problem.map);
    if yaw_error(robot_to_field(problem, &symmetric), hint)
        .abs()
        .total_cmp(&yaw_error(robot_to_field(problem, &best), hint).abs())
        .then_with(|| symmetric.key.cmp(&best.key))
        .is_lt()
    {
        Some(symmetric)
    } else {
        Some(best)
    }
}

fn evaluate(
    problem: &Problem<'_>,
    proposal: Similarity2<f32>,
    assignment_scratch: &mut AssignmentScratch,
) -> Option<Candidate> {
    let mut matches = [EMPTY_MATCH; GLOBAL_LOCALIZER_MAX_DETECTIONS];
    let mut match_count = assign(
        problem,
        proposal,
        problem.config.association_gate,
        assignment_scratch,
        &mut matches,
    )?;
    if match_count < problem.config.min_inliers {
        return None;
    }

    for _ in 0..MAX_ASSIGNMENT_REFIT_ITERATIONS {
        let transform = fit_similarity(problem, &matches[..match_count])?;
        if !plausible_scale(transform.scaling(), problem.config) {
            return None;
        }
        let mut reassigned = [EMPTY_MATCH; GLOBAL_LOCALIZER_MAX_DETECTIONS];
        let reassigned_count = assign(
            problem,
            transform,
            problem.config.association_gate,
            assignment_scratch,
            &mut reassigned,
        )?;
        if reassigned_count < problem.config.min_inliers {
            return None;
        }
        if same_assignment(&matches[..match_count], &reassigned[..reassigned_count]) {
            return candidate(problem, transform, &reassigned[..reassigned_count]);
        }
        matches = reassigned;
        match_count = reassigned_count;
    }
    None
}

fn same_assignment(left: &[Match], right: &[Match]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.detection == right.detection && left.landmark == right.landmark
        })
}

fn assign(
    problem: &Problem<'_>,
    transform: Similarity2<f32>,
    gate: f32,
    scratch: &mut AssignmentScratch,
    matches: &mut [Match; GLOBAL_LOCALIZER_MAX_DETECTIONS],
) -> Option<usize> {
    if !prepare_metric_support(problem, transform, gate, scratch) {
        return Some(0);
    }
    let mut match_count = 0;
    for class in FEATURE_CLASSES {
        let landmark_ids = problem.map.landmarks_for_class(class);
        if landmark_ids.is_empty() {
            continue;
        }
        debug_assert!(landmark_ids.len() <= MAX_LANDMARKS_PER_CLASS);
        let mut active_row_count = 0;
        for (detection_index, &detection) in problem.detections.iter().enumerate() {
            if detection.class != class {
                continue;
            }
            let row = active_row_count;
            scratch.benefits[row][..landmark_ids.len()].fill(0.0);
            scratch.residuals[row][..landmark_ids.len()].fill((f32::INFINITY, f32::INFINITY));
            let predicted = scratch.predicted[detection_index];
            let detection_weight = detection.weight;
            let unmatched_penalty = detection.unmatched_penalty;
            let mut has_beneficial_landmark = false;
            let mut information = None;
            for (column, &landmark_id) in landmark_ids.iter().enumerate() {
                let residual = scratch.metric_residuals[detection_index][landmark_id];
                if !residual.is_finite() {
                    continue;
                }
                let error = problem.map.landmarks[landmark_id].xy.coords().inner - predicted;
                let information = match information {
                    Some(information) => information,
                    None => {
                        let Some(computed) = transformed_information(transform, detection) else {
                            break;
                        };
                        information = Some(computed);
                        computed
                    }
                };
                let Some(mahalanobis) = mahalanobis_distance_squared(information, error) else {
                    continue;
                };
                if mahalanobis <= problem.config.mahalanobis_gate {
                    let benefit = assignment_benefit(
                        detection_weight,
                        unmatched_penalty,
                        mahalanobis,
                        problem.config.mahalanobis_gate,
                    );
                    if benefit > 0.0 {
                        scratch.residuals[row][column] = (residual, mahalanobis);
                        scratch.benefits[row][column] = benefit;
                        has_beneficial_landmark = true;
                    }
                }
            }
            if has_beneficial_landmark {
                scratch.detection_indices[row] = detection_index;
                active_row_count += 1;
            }
        }
        if active_row_count == 0 {
            continue;
        }
        let column_count = landmark_ids.len() + active_row_count;
        for row in &mut scratch.benefits[..active_row_count] {
            row[landmark_ids.len()..column_count].fill(0.0);
        }
        if active_row_count == 1 {
            let mut best_column = 0;
            for column in 1..column_count {
                if scratch.benefits[0][column] > scratch.benefits[0][best_column] {
                    best_column = column;
                }
            }
            scratch.selected_columns[0] = best_column;
        } else {
            let benefits = ArrayView2::from_shape(
                (GLOBAL_LOCALIZER_MAX_DETECTIONS, MAX_ASSIGNMENT_COLUMNS),
                scratch.benefits.as_flattened(),
            )
            .ok()?;
            let assignments = scratch
                .assignment
                .solve(
                    benefits.slice(s![..active_row_count, ..column_count]),
                    Objective::Maximize,
                )
                .ok()?;
            for (output, column) in scratch.selected_columns.iter_mut().zip(assignments) {
                *output = (*column)?;
            }
        }
        for row in 0..active_row_count {
            let column = scratch.selected_columns[row];
            if column >= landmark_ids.len() || scratch.benefits[row][column] <= 0.0 {
                continue;
            }
            matches[match_count] = Match {
                detection: scratch.detection_indices[row],
                landmark: landmark_ids[column],
                residual: scratch.residuals[row][column].0,
                mahalanobis: scratch.residuals[row][column].1,
            };
            match_count += 1;
        }
    }
    Some(match_count)
}

fn prepare_metric_support(
    problem: &Problem<'_>,
    transform: Similarity2<f32>,
    gate: f32,
    scratch: &mut AssignmentScratch,
) -> bool {
    let mut supported_by_class = [0; FEATURE_CLASSES.len()];
    let gate_squared = gate * gate;
    for (detection_index, detection) in problem.detections.iter().enumerate() {
        let predicted = transform * nalgebra::Point2::from(detection.a);
        scratch.predicted[detection_index] = predicted.coords;
        scratch.metric_residuals[detection_index].fill(f32::INFINITY);
        let mut supported = false;
        for &landmark_id in problem.map.landmarks_for_class(detection.class) {
            let residual_squared = (problem.map.landmarks[landmark_id].xy.coords().inner
                - predicted.coords)
                .norm_squared();
            if residual_squared.is_finite() && residual_squared <= gate_squared {
                scratch.metric_residuals[detection_index][landmark_id] = residual_squared.sqrt();
                supported = true;
            }
        }
        if supported {
            supported_by_class[detection.class.index()] += 1;
        }
    }
    FEATURE_CLASSES
        .into_iter()
        .map(|class| {
            supported_by_class[class.index()].min(problem.map.landmarks_for_class(class).len())
        })
        .sum::<usize>()
        >= problem.config.min_inliers
}

fn fit_similarity(problem: &Problem<'_>, matches: &[Match]) -> Option<Similarity2<f32>> {
    let weight_sum = matches
        .iter()
        .map(|matched| problem.detections[matched.detection].weight)
        .sum::<f32>();
    if weight_sum <= 0.0 {
        return None;
    }
    let source_center = matches
        .iter()
        .map(|matched| {
            problem.detections[matched.detection].weight * problem.detections[matched.detection].a
        })
        .sum::<Vector2<f32>>()
        / weight_sum;
    let target_center = matches
        .iter()
        .map(|matched| {
            problem.detections[matched.detection].weight
                * problem.map.landmarks[matched.landmark].xy.coords().inner
        })
        .sum::<Vector2<f32>>()
        / weight_sum;
    let mut sin = 0.0;
    let mut cos = 0.0;
    let mut variance = 0.0;
    for matched in matches {
        let weight = problem.detections[matched.detection].weight;
        let source = problem.detections[matched.detection].a - source_center;
        let target = problem.map.landmarks[matched.landmark].xy.coords().inner - target_center;
        sin += weight * (source.x * target.y - source.y * target.x);
        cos += weight * source.dot(&target);
        variance += weight * source.norm_squared();
    }
    if variance <= 1.0e-8 || sin.abs() + cos.abs() <= 1.0e-8 {
        return None;
    }
    let yaw = sin.atan2(cos);
    let rotation = nalgebra::UnitComplex::new(yaw);
    let numerator = matches
        .iter()
        .map(|matched| {
            let weight = problem.detections[matched.detection].weight;
            let source = problem.detections[matched.detection].a - source_center;
            let target = problem.map.landmarks[matched.landmark].xy.coords().inner - target_center;
            weight * (rotation * source).dot(&target)
        })
        .sum::<f32>();
    let scale = numerator / variance;
    let translation = target_center - scale * (rotation * source_center);
    (scale.is_finite() && translation.iter().all(|value| value.is_finite()))
        .then(|| Similarity2::new(translation, yaw, scale))
}

fn candidate(
    problem: &Problem<'_>,
    transform: Similarity2<f32>,
    matches: &[Match],
) -> Option<Candidate> {
    let total_squared = matches
        .iter()
        .map(|matched| matched.residual.powi(2))
        .sum::<f32>();
    let rms = (total_squared / matches.len() as f32).sqrt();
    let matched_weight = matches
        .iter()
        .map(|matched| problem.detections[matched.detection].weight)
        .sum::<f32>();
    let unmatched = problem
        .detections
        .iter()
        .enumerate()
        .filter(|(index, _)| !matches.iter().any(|matched| matched.detection == *index))
        .map(|(_, detection)| detection.unmatched_penalty)
        .sum::<f32>();
    let normalized_error = matches
        .iter()
        .map(|matched| {
            problem.detections[matched.detection].weight * matched.mahalanobis
                / problem.config.mahalanobis_gate
        })
        .sum::<f32>();
    let score = matched_weight - normalized_error - unmatched;
    if !score.is_finite() || score <= 0.0 || rms > problem.config.rms_threshold {
        return None;
    }
    let key = association_key(problem, matches);
    let symmetric = symmetric_key(&key, &problem.map);
    let orbit_key = key.clone().min(symmetric);
    let mut matches = MatchSet::from_slice(matches);
    matches.sort_by_key(|matched| problem.detections[matched.detection].id);
    Some(Candidate {
        transform,
        matches,
        score,
        rms,
        key,
        orbit_key,
    })
}

fn association_key(problem: &Problem<'_>, matches: &[Match]) -> AssociationKey {
    let mut key = matches
        .iter()
        .map(|matched| (problem.detections[matched.detection].id, matched.landmark))
        .collect::<AssociationKey>();
    key.sort_unstable();
    key
}

fn symmetric_key(key: &AssociationKey, map: &LandmarkMap) -> AssociationKey {
    let mut symmetric = key.clone();
    for (_, landmark_id) in &mut symmetric {
        *landmark_id = map.symmetric_id(*landmark_id);
    }
    symmetric.sort_unstable();
    symmetric
}

fn symmetric_candidate(candidate: &Candidate, map: &LandmarkMap) -> Candidate {
    let mut symmetric = candidate.clone();
    for matched in &mut symmetric.matches {
        matched.landmark = map.symmetric_id(matched.landmark);
    }
    symmetric.key = symmetric_key(&candidate.key, map);
    symmetric.transform = Similarity2::new(
        -candidate.transform.isometry.translation.vector,
        candidate.transform.isometry.rotation.angle() + std::f32::consts::PI,
        candidate.transform.scaling(),
    );
    symmetric
}

fn compare(left: &Candidate, right: &Candidate) -> std::cmp::Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| right.matches.len().cmp(&left.matches.len()))
        .then_with(|| left.rms.total_cmp(&right.rms))
        .then_with(|| left.key.cmp(&right.key))
}

fn better(left: &Candidate, right: &Candidate) -> bool {
    compare(left, right).is_lt()
}

fn assignment_benefit(
    detection_weight: f32,
    unmatched_penalty: f32,
    mahalanobis: f32,
    mahalanobis_gate: f32,
) -> f32 {
    detection_weight * (1.0 - mahalanobis / mahalanobis_gate) + unmatched_penalty
}

fn plausible_scale(scale: f32, config: GlobalAssociationConfig) -> bool {
    scale.is_finite() && scale >= config.height_min && scale <= config.height_max
}

fn raw_detections(
    features: &DetectedVisualFeatures,
) -> impl Iterator<Item = (VisualFeatureClass, DetectedVisualFeature)> + '_ {
    [
        (VisualFeatureClass::GoalPost, features.goalposts.as_slice()),
        (VisualFeatureClass::LSpot, features.l_spots.as_slice()),
        (VisualFeatureClass::TSpot, features.t_spots.as_slice()),
        (VisualFeatureClass::XSpot, features.x_spots.as_slice()),
        (
            VisualFeatureClass::PenaltySpot,
            features.penalty_spots.as_slice(),
        ),
    ]
    .into_iter()
    .flat_map(|(class, detections)| detections.iter().copied().map(move |item| (class, item)))
}

fn valid_intrinsic(intrinsic: Intrinsic) -> bool {
    intrinsic.focals.x.is_finite()
        && intrinsic.focals.y.is_finite()
        && intrinsic.optical_center.x().is_finite()
        && intrinsic.optical_center.y().is_finite()
        && intrinsic.focals.x.abs() > 1.0e-6
        && intrinsic.focals.y.abs() > 1.0e-6
}

fn normalized_ray(
    intrinsic: Intrinsic,
    pixel: Point2<Pixel>,
    camera_to_ground_rotation: UnitQuaternion<f32>,
) -> Option<Vector2<f32>> {
    let bearing = intrinsic.bearing(pixel).inner;
    normalized_ground_direction(camera_to_ground_rotation * bearing)
}

fn normalized_ground_direction(ray: Vector3<f32>) -> Option<Vector2<f32>> {
    if !ray.iter().all(|value| value.is_finite()) || ray.z.abs() <= RAY_EPSILON {
        return None;
    }
    let normalized = Vector2::new(ray.x / ray.z.abs(), ray.y / ray.z.abs());
    normalized
        .iter()
        .all(|value| value.is_finite())
        .then_some(normalized)
}

fn normalized_ray_covariance(
    intrinsic: Intrinsic,
    pixel: Point2<Pixel>,
    camera_to_ground_rotation: UnitQuaternion<f32>,
    config: GlobalAssociationConfig,
) -> Option<Matrix2<f32>> {
    let pixel_x = normalized_ray(
        intrinsic,
        point![<Pixel>, pixel.x() + 1.0, pixel.y()],
        camera_to_ground_rotation,
    )? - normalized_ray(
        intrinsic,
        point![<Pixel>, pixel.x() - 1.0, pixel.y()],
        camera_to_ground_rotation,
    )?;
    let pixel_y = normalized_ray(
        intrinsic,
        point![<Pixel>, pixel.x(), pixel.y() + 1.0],
        camera_to_ground_rotation,
    )? - normalized_ray(
        intrinsic,
        point![<Pixel>, pixel.x(), pixel.y() - 1.0],
        camera_to_ground_rotation,
    )?;
    let pixel_jacobian = Matrix2::from_columns(&[pixel_x / 2.0, pixel_y / 2.0]);

    let bearing = intrinsic.bearing(pixel).inner;
    let ray = camera_to_ground_rotation * bearing;
    let roll = finite_tilt_difference(ray, Vector3::x_axis())?;
    let pitch = finite_tilt_difference(ray, Vector3::y_axis())?;
    let tilt_jacobian = Matrix2::from_columns(&[roll, pitch]);

    let covariance =
        config.detection_pixel_sigma.powi(2) * pixel_jacobian * pixel_jacobian.transpose()
            + config.imu_tilt_sigma.powi(2) * tilt_jacobian * tilt_jacobian.transpose()
            + Matrix2::identity() * COVARIANCE_FLOOR;
    covariance
        .iter()
        .all(|value| value.is_finite())
        .then_some(covariance)
}

fn finite_tilt_difference(
    ray: Vector3<f32>,
    axis: nalgebra::Unit<Vector3<f32>>,
) -> Option<Vector2<f32>> {
    let positive = UnitQuaternion::from_axis_angle(&axis, TILT_JACOBIAN_STEP) * ray;
    let negative = UnitQuaternion::from_axis_angle(&axis, -TILT_JACOBIAN_STEP) * ray;
    Some(
        (normalized_ground_direction(positive)? - normalized_ground_direction(negative)?)
            / (2.0 * TILT_JACOBIAN_STEP),
    )
}

fn transformed_information(
    transform: Similarity2<f32>,
    detection: Detection,
) -> Option<Matrix2<f32>> {
    let rotation = transform.isometry.rotation.to_rotation_matrix();
    let covariance = transform.scaling().powi(2)
        * rotation.matrix()
        * detection.covariance
        * rotation.matrix().transpose();
    covariance.try_inverse()
}

fn mahalanobis_distance_squared(information: Matrix2<f32>, error: Vector2<f32>) -> Option<f32> {
    let value = error.dot(&(information * error));
    (value.is_finite() && value >= 0.0).then_some(value)
}

fn to_public(problem: &Problem<'_>, candidate: &Candidate) -> GlobalAssociationResult {
    GlobalAssociationResult {
        associations: candidate
            .matches
            .iter()
            .map(|matched| FieldMarkAssociation {
                detection: problem.detections[matched.detection].pixel,
                field_point: problem.map.landmarks[matched.landmark].xy.extend(0.0),
            })
            .collect(),
        debug: GlobalLocalizationDebug {
            inliers: candidate.matches.len(),
            candidate_score: candidate.score,
            metric_rms_residual: candidate.rms,
        },
    }
}

// The internal proposal is converted to a pose only to select a certified symmetry representative.
fn robot_to_field(problem: &Problem<'_>, candidate: &Candidate) -> Isometry3<Robot, Field> {
    let ground_to_camera = problem.input.robot_to_camera * problem.input.ground_to_robot;
    let camera_to_ground = ground_to_camera.inverse();
    let camera_xy = camera_to_ground.inner.translation.vector.xy();
    let yaw = candidate.transform.isometry.rotation.angle();
    let translation = candidate.transform.isometry.translation.vector
        - candidate.transform.isometry.rotation * camera_xy;
    let ground_to_field = Isometry3::<Ground, Field>::wrap(nalgebra::Isometry3::from_parts(
        Translation3::new(translation.x, translation.y, 0.0),
        nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::z_axis(), yaw),
    ));
    ground_to_field * problem.input.ground_to_robot.inverse()
}

fn yaw_error(left: Isometry3<Robot, Field>, right: Isometry3<Robot, Field>) -> f32 {
    let left = left.inner.rotation.euler_angles().2;
    let right = right.inner.rotation.euler_angles().2;
    (left - right + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

fn is_finite_pose(pose: &Isometry3<Robot, Field>) -> bool {
    pose.inner
        .to_homogeneous()
        .iter()
        .all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_goalpost_pair_proposals_differ_by_half_turn() {
        let field = FieldDimensions::SPL_2025;
        let map = LandmarkMap::new(&field);
        let ids = map.landmarks_for_class(VisualFeatureClass::GoalPost);
        let source = Vector2::<f32>::new(0.0, 1.0);
        let forward = (map.landmarks[ids[1]].xy - map.landmarks[ids[0]].xy).inner;
        let reverse = (map.landmarks[ids[0]].xy - map.landmarks[ids[1]].xy).inner;
        let forward_yaw = forward.y.atan2(forward.x) - source.y.atan2(source.x);
        let reverse_yaw = reverse.y.atan2(reverse.x) - source.y.atan2(source.x);
        let difference = (forward_yaw - reverse_yaw).abs();
        assert!((difference - std::f32::consts::PI).abs() < 1.0e-5);
    }

    #[test]
    fn symmetric_key_is_an_involution() {
        let map = LandmarkMap::new(&FieldDimensions::SPL_2025);
        let key = association_key_from_pairs(&[(2, 0), (7, 5), (9, 20)]);
        assert_eq!(symmetric_key(&symmetric_key(&key, &map), &map), key);
    }

    #[test]
    fn non_symmetric_runner_up_fails_closed() {
        let map = LandmarkMap::new(&FieldDimensions::SPL_2025);
        let mut best = None;
        let mut runner_up = None;
        retain_orbit_candidate(
            &mut best,
            &mut runner_up,
            test_candidate(&map, 1.0, &[(0, 0), (1, 1), (2, 2)]),
        );
        retain_orbit_candidate(
            &mut best,
            &mut runner_up,
            test_candidate(&map, 0.99, &[(0, 0), (1, 1), (2, 4)]),
        );

        assert!(select_unique_orbit(best, runner_up, 1.05).is_none());
    }

    #[test]
    fn orbit_winners_are_replaced_without_retaining_other_candidates() {
        let map = LandmarkMap::new(&FieldDimensions::SPL_2025);
        let first_orbit = [(0, 0), (1, 1), (2, 2)];
        let second_orbit = [(0, 0), (1, 1), (2, 4)];
        let mut best = None;
        let mut runner_up = None;

        for candidate in [
            test_candidate(&map, 1.0, &first_orbit),
            test_candidate(&map, 0.9, &second_orbit),
            test_candidate(&map, 1.1, &first_orbit),
            test_candidate(&map, 1.2, &second_orbit),
        ] {
            retain_orbit_candidate(&mut best, &mut runner_up, candidate);
        }

        assert_eq!(best.as_ref().map(|candidate| candidate.score), Some(1.2));
        assert_eq!(
            runner_up.as_ref().map(|candidate| candidate.score),
            Some(1.1)
        );
        assert_ne!(
            best.as_ref().map(|candidate| &candidate.orbit_key),
            runner_up.as_ref().map(|candidate| &candidate.orbit_key)
        );
    }

    #[test]
    fn equal_scoring_distinct_orbits_fail_closed() {
        let map = LandmarkMap::new(&FieldDimensions::SPL_2025);
        let best = test_candidate(&map, 1.0, &[(0, 0), (1, 1), (2, 2)]);
        let runner_up = test_candidate(&map, 1.0, &[(0, 0), (1, 1), (2, 4)]);

        assert!(select_unique_orbit(Some(best), Some(runner_up), 1.05).is_none());
    }

    fn association_key_from_pairs(pairs: &[(usize, usize)]) -> AssociationKey {
        let mut key = AssociationKey::from_slice(pairs);
        key.sort_unstable();
        key
    }

    fn test_candidate(map: &LandmarkMap, score: f32, pairs: &[(usize, usize)]) -> Candidate {
        let key = association_key_from_pairs(pairs);
        Candidate {
            transform: Similarity2::identity(),
            matches: MatchSet::new(),
            score,
            rms: 0.0,
            orbit_key: key.clone().min(symmetric_key(&key, map)),
            key,
        }
    }

    #[test]
    fn assignment_benefit_equals_candidate_score_improvement() {
        let detection_weight = 0.2;
        let unmatched_penalty = 0.1;
        let mahalanobis = 2.0;
        let mahalanobis_gate = 10.0;
        let unmatched_score = -unmatched_penalty;
        let matched_score = detection_weight * (1.0 - mahalanobis / mahalanobis_gate);

        assert_eq!(
            assignment_benefit(
                detection_weight,
                unmatched_penalty,
                mahalanobis,
                mahalanobis_gate,
            ),
            matched_score - unmatched_score
        );
    }
}

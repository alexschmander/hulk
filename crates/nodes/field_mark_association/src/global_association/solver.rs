use coordinate_systems::Pixel;
use linear_algebra::Point2;
use nalgebra::{Matrix2, Matrix2x3, Vector2, Vector3};
use types::visual_localization::{FieldMarkAssociation, GlobalLocalizationDebug};

use crate::{
    AssociationResult, DetectedVisualFeature, GlobalAssociationInput, VisualFeatureClass,
    features::raw_detections, map::LandmarkMap,
};

use super::{
    GLOBAL_LOCALIZER_MAX_DETECTIONS, GlobalAssociationConfig,
    config::{GLOBAL_LOCALIZER_MAX_INPUT_DETECTIONS, SEED_POOL_SIZE},
};

#[derive(Clone, Debug)]
pub(crate) struct Detection {
    pub id: usize,
    pub class: VisualFeatureClass,
    pub pixel: Point2<Pixel>,
    pub confidence: f32,
    pub xy: Vector2<f32>,
    pixel_covariance: Matrix2<f32>,
    tilt_jacobian: Matrix2<f32>,
    height_jacobian: Vector2<f32>,
}

impl Detection {
    /// Independent-source point covariance, used for finiteness and Jacobian checks.
    /// Certification instead uses the shared-error bounds in `invariant_sigma`.
    pub(crate) fn covariance(&self, config: GlobalAssociationConfig) -> Matrix2<f32> {
        self.pixel_covariance
            + config.imu_tilt_sigma.powi(2) * self.tilt_jacobian * self.tilt_jacobian.transpose()
            + config.height_sigma.powi(2) * self.height_jacobian * self.height_jacobian.transpose()
    }
}

/// Project using measured LOCAL height. No scale, translation or yaw is inferred from the map.
pub(crate) fn preprocess(
    input: GlobalAssociationInput<'_>,
) -> Option<(LandmarkMap, Vec<Detection>)> {
    let config = *input.parameters;
    config.validate().ok()?;
    let intrinsic = input.camera_intrinsic;
    if input.visual_features.supported_feature_count() > GLOBAL_LOCALIZER_MAX_INPUT_DETECTIONS
        || !intrinsic.is_valid()
        || !input
            .robot_to_camera
            .inner
            .to_homogeneous()
            .iter()
            .all(|x| x.is_finite())
        || !input
            .robot_to_local
            .inner
            .to_homogeneous()
            .iter()
            .all(|x| x.is_finite())
    {
        return None;
    }
    let map = LandmarkMap::new(input.field_dimensions);
    if map
        .landmarks
        .iter()
        .any(|p| !p.xy.coords().inner.iter().all(|x| x.is_finite()))
    {
        return None;
    }
    let mut detections = Vec::new();
    for (id, (class, feature)) in raw_detections(input.visual_features).enumerate() {
        if !feature.confidence.is_finite()
            || !(config.confidence_threshold..=1.0).contains(&feature.confidence)
        {
            continue;
        }
        if let Some(detection) = project_detection(input, config, id, class, feature) {
            detections.push(detection);
        }
    }
    detections.sort_by(|a, b| {
        (b.confidence * map.rarity_weight(b.class))
            .total_cmp(&(a.confidence * map.rarity_weight(a.class)))
            .then_with(|| a.id.cmp(&b.id))
    });
    let mut retained = Vec::<Detection>::new();
    for detection in detections {
        if retained.iter().any(|other| {
            other.class == detection.class && (other.pixel - detection.pixel).inner.norm() <= 1.0
        }) {
            continue;
        }
        if retained.len() == GLOBAL_LOCALIZER_MAX_DETECTIONS {
            return None;
        }
        retained.push(detection);
    }
    Some((map, retained))
}

fn project_detection(
    input: GlobalAssociationInput<'_>,
    config: GlobalAssociationConfig,
    id: usize,
    class: VisualFeatureClass,
    feature: DetectedVisualFeature,
) -> Option<Detection> {
    let camera_to_local = input.robot_to_local * input.robot_to_camera.inverse();
    let origin = camera_to_local.inner.translation.vector;
    let offset = origin - input.robot_to_local.inner.translation.vector;
    let rotation = camera_to_local.inner.rotation;
    let ray = rotation * input.camera_intrinsic.bearing(feature.pixel).inner;
    let pixel_rays = [
        (rotation * Vector3::x()) / input.camera_intrinsic.focals.x,
        (rotation * Vector3::y()) / input.camera_intrinsic.focals.y,
    ];
    let sigma_gate = config.mahalanobis_gate.sqrt();
    let height_sigma = config.height_sigma + config.imu_tilt_sigma * offset.xy().norm();
    let ray_z_sigma = config.detection_pixel_sigma * pixel_rays[0].z.hypot(pixel_rays[1].z)
        + config.imu_tilt_sigma * ray.xy().norm();
    // Linearized intersections are unusable if uncertainty reaches the horizon or ground.
    if !ray.iter().all(|x| x.is_finite())
        || origin.z <= sigma_gate * height_sigma
        || -ray.z <= (sigma_gate * ray_z_sigma).max(1.0e-4 * ray.norm())
    {
        return None;
    }
    let height_jacobian = -ray.xy() / ray.z;
    let ray_jacobian = Matrix2x3::new(
        -origin.z / ray.z,
        0.0,
        origin.z * ray.x / ray.z.powi(2),
        0.0,
        -origin.z / ray.z,
        origin.z * ray.y / ray.z.powi(2),
    );
    let pixel_jacobian = Matrix2::from_columns(&pixel_rays.map(|ray| ray_jacobian * ray));
    let tilt_jacobian = Matrix2::from_columns(&[Vector3::x(), Vector3::y()].map(|axis| {
        // IMU tilt rotates both the bearing and the camera lever arm about the robot origin.
        let origin_derivative = axis.cross(&offset);
        origin_derivative.xy()
            + height_jacobian * origin_derivative.z
            + ray_jacobian * axis.cross(&ray)
    }));
    let detection = Detection {
        id,
        class,
        pixel: feature.pixel,
        confidence: feature.confidence,
        xy: origin.xy() + origin.z * height_jacobian,
        pixel_covariance: config.detection_pixel_sigma.powi(2)
            * pixel_jacobian
            * pixel_jacobian.transpose()
            + Matrix2::identity() * 1.0e-8,
        tilt_jacobian,
        height_jacobian,
    };
    detection
        .xy
        .iter()
        .chain(detection.covariance(config).iter())
        .all(|x| x.is_finite())
        .then_some(detection)
}

/// Pixel errors are independent; tilt/height errors are shared across the entire frame.
/// Adding the three standard deviations also bounds unknown cross-source correlations.
/// Each detection must appear once, with its gradients combined before propagating pixel noise.
fn invariant_sigma(terms: &[(&Detection, Vector2<f32>)], config: GlobalAssociationConfig) -> f32 {
    let mut pixel_variance = 0.0;
    let mut tilt = Vector2::zeros();
    let mut height = 0.0;
    for (detection, gradient) in terms {
        pixel_variance += gradient.dot(&(detection.pixel_covariance * gradient));
        tilt += detection.tilt_jacobian.transpose() * gradient;
        height += detection.height_jacobian.dot(gradient);
    }
    pixel_variance.max(0.0).sqrt()
        + config.imu_tilt_sigma * tilt.norm()
        + config.height_sigma * height.abs()
}

fn cross(a: Vector2<f32>, b: Vector2<f32>) -> f32 {
    a.x * b.y - a.y * b.x
}

fn chirality(a: &Detection, b: &Detection, c: &Detection, config: GlobalAssociationConfig) -> f32 {
    let u = b.xy - a.xy;
    let v = c.xy - a.xy;
    let perpendicular = |p: Vector2<f32>| Vector2::new(-p.y, p.x);
    let sigma = invariant_sigma(
        &[
            (a, perpendicular(v - u)),
            (b, -perpendicular(v)),
            (c, perpendicular(u)),
        ],
        config,
    );
    let area = cross(u, v);
    let tolerance =
        config.mahalanobis_gate.sqrt() * sigma + config.geometric_tolerance * (u.norm() + v.norm());
    if area.abs() > tolerance {
        area.signum()
    } else {
        0.0
    }
}

#[derive(Clone, Copy)]
struct TriangleConstraint {
    sign: f32,
    // At each vertex a, contrast d(a,b) - d(a,c), with vertices ordered cyclically.
    contrasts: [(f32, f32); 3],
}

impl TriangleConstraint {
    fn new(
        a: &Detection,
        b: &Detection,
        c: &Detection,
        config: GlobalAssociationConfig,
    ) -> Option<Self> {
        let points = [a, b, c];
        let mut contrasts = [(0.0, 0.0); 3];
        for i in 0..3 {
            let [a, b, c] = [points[i], points[(i + 1) % 3], points[(i + 2) % 3]];
            let ab = a.xy - b.xy;
            let ac = a.xy - c.xy;
            let (distance_ab, distance_ac) = (ab.norm(), ac.norm());
            // Coincident vertices have no defined distance gradient; do not invent one.
            if distance_ab == 0.0 || distance_ac == 0.0 {
                return None;
            }
            let (u_ab, u_ac) = (ab / distance_ab, ac / distance_ac);
            let difference = distance_ab - distance_ac;
            let gate = config.mahalanobis_gate.sqrt()
                * invariant_sigma(&[(a, u_ab - u_ac), (b, -u_ab), (c, u_ac)], config)
                + 2.0 * config.geometric_tolerance;
            if !difference.is_finite() || !gate.is_finite() {
                return None;
            }
            contrasts[i] = (difference, gate);
        }
        Some(Self {
            sign: chirality(a, b, c, config),
            contrasts,
        })
    }
}

pub(crate) fn associate(input: GlobalAssociationInput<'_>) -> Option<AssociationResult> {
    let (map, detections) = preprocess(input)?;
    let (pairs, rms) = match_geometry(&map, &detections, *input.parameters)?;
    Some(result(&map, &detections, &pairs, rms))
}

#[derive(Clone, Copy, Default)]
struct DistanceConstraint {
    distance: f32,
    tolerance: f32,
}

fn distance_constraints(
    detections: &[Detection],
    config: GlobalAssociationConfig,
) -> Option<Vec<Vec<DistanceConstraint>>> {
    let n = detections.len();
    let mut distances = vec![vec![DistanceConstraint::default(); n]; n];
    for i in 0..n {
        for j in 0..i {
            let edge = detections[i].xy - detections[j].xy;
            let distance = edge.norm();
            let direction = edge / distance.max(1.0e-6);
            let gate = config.geometric_tolerance
                + config.mahalanobis_gate.sqrt()
                    * invariant_sigma(
                        &[(&detections[i], direction), (&detections[j], -direction)],
                        config,
                    );
            if !distance.is_finite() || !gate.is_finite() {
                return None;
            }
            let constraint = DistanceConstraint {
                distance,
                tolerance: gate,
            };
            distances[i][j] = constraint;
            distances[j][i] = constraint;
        }
    }
    Some(distances)
}

fn match_geometry(
    map: &LandmarkMap,
    detections: &[Detection],
    config: GlobalAssociationConfig,
) -> Option<(Vec<(usize, usize)>, f32)> {
    if detections.len() < config.min_inliers || detections.len() > map.landmarks.len() {
        return None;
    }
    let n = detections.len();
    let distances = distance_constraints(detections, config)?;
    let seed = select_seed(detections, &distances, config)?;
    let order = seed
        .into_iter()
        .chain((0..n).filter(|i| !seed.contains(i)))
        .collect::<Vec<_>>();
    let triangles = order
        .iter()
        .skip(2)
        .map(|&i| {
            TriangleConstraint::new(
                &detections[seed[0]],
                &detections[seed[1]],
                &detections[i],
                config,
            )
        })
        .collect::<Option<Vec<_>>>()?;
    let mut search = Search {
        map,
        detections,
        distances,
        order,
        triangles,
        remaining_work: config.max_work,
        solution: None,
    };
    search.visit(&mut Vec::with_capacity(n))?;
    let pairs = search.solution?;
    let rms = pairwise_distance_rms(&pairs, &search.distances, map);
    Some((pairs, rms))
}

fn pairwise_distance_rms(
    pairs: &[(usize, usize)],
    distances: &[Vec<DistanceConstraint>],
    map: &LandmarkMap,
) -> f32 {
    let mut squared = 0.0;
    let mut count = 0;
    for (i, &(d, m)) in pairs.iter().enumerate() {
        for &(e, l) in &pairs[..i] {
            squared += (distances[d][e].distance
                - (map.landmarks[m].xy - map.landmarks[l].xy).inner.norm())
            .powi(2);
            count += 1;
        }
    }
    (squared / count.max(1) as f32).sqrt()
}

fn select_seed(
    detections: &[Detection],
    distances: &[Vec<DistanceConstraint>],
    config: GlobalAssociationConfig,
) -> Option<[usize; 3]> {
    // ponytail: only eight detections seed the search, and chirality rejects even valid known-height
    // collinear frames; expand seed handling if recall requires it. Every retained detection must match.
    let n = detections.len().min(SEED_POOL_SIZE);
    let mut seed = None;
    let mut best_quality = 0.0;
    for a in 0..n {
        for b in a + 1..n {
            for c in b + 1..n {
                let edges = [distances[a][b], distances[a][c], distances[b][c]];
                if edges
                    .iter()
                    .any(|edge| edge.distance < config.min_detection_baseline)
                    || chirality(&detections[a], &detections[b], &detections[c], config) == 0.0
                {
                    continue;
                }
                let quality = edges
                    .iter()
                    .map(|edge| edge.distance / edge.tolerance)
                    .fold(f32::INFINITY, f32::min)
                    * detections[a]
                        .confidence
                        .min(detections[b].confidence)
                        .min(detections[c].confidence);
                if quality > best_quality {
                    best_quality = quality;
                    seed = Some([a, b, c]);
                }
            }
        }
    }
    seed
}

struct Search<'a> {
    map: &'a LandmarkMap,
    detections: &'a [Detection],
    distances: Vec<Vec<DistanceConstraint>>,
    order: Vec<usize>,
    triangles: Vec<TriangleConstraint>,
    remaining_work: usize,
    solution: Option<Vec<(usize, usize)>>,
}

impl Search<'_> {
    fn spend(&mut self) -> Option<()> {
        self.remaining_work = self.remaining_work.checked_sub(1)?;
        Some(())
    }

    /// Some means traversal completed (possibly without a solution). None aborts
    /// certification on exhaustion or non-uniqueness; a stored solution is then unusable.
    fn visit(&mut self, pairs: &mut Vec<(usize, usize)>) -> Option<()> {
        self.spend()?;
        let depth = pairs.len();
        if depth == self.order.len() {
            let key = canonical_pairs(pairs, self.detections, self.map);
            if let Some(solution) = &self.solution {
                if key != *solution {
                    return None;
                }
            } else {
                self.solution = Some(key);
            }
            return Some(());
        }
        let detection = self.order[depth];
        let landmarks = self
            .map
            .landmarks_for_class(self.detections[detection].class);
        for &landmark in landmarks {
            self.spend()?;
            if self.candidate_compatible(detection, landmark, pairs)? {
                pairs.push((detection, landmark));
                self.visit(pairs)?;
                pairs.pop();
            }
        }
        Some(())
    }

    /// Some(false) prunes one branch; None aborts the entire traversal.
    fn candidate_compatible(
        &mut self,
        detection: usize,
        landmark: usize,
        pairs: &[(usize, usize)],
    ) -> Option<bool> {
        for &(other, other_landmark) in pairs {
            self.spend()?;
            let DistanceConstraint {
                distance,
                tolerance: gate,
            } = self.distances[detection][other];
            let map_distance = (self.map.landmarks[landmark].xy
                - self.map.landmarks[other_landmark].xy)
                .inner
                .norm();
            if landmark == other_landmark || (distance - map_distance).abs() > gate {
                return Some(false);
            }
        }
        let depth = pairs.len();
        if depth >= 2 {
            let triangle = self.triangles[depth - 2];
            let a = self.map.landmarks[pairs[0].1].xy;
            let b = self.map.landmarks[pairs[1].1].xy;
            let c = self.map.landmarks[landmark].xy;
            let same_chirality = cross((b - a).inner, (c - a).inner) * triangle.sign > 0.0;
            if triangle.sign != 0.0 && !same_chirality {
                return Some(false);
            }
            let lengths = [
                (a - b).inner.norm(),
                (b - c).inner.norm(),
                (c - a).inner.norm(),
            ];
            for (i, (difference, gate)) in triangle.contrasts.into_iter().enumerate() {
                self.spend()?;
                let map_difference = lengths[i] - lengths[(i + 2) % 3];
                if (difference - map_difference).abs() > gate {
                    return Some(false);
                }
            }
        }
        Some(true)
    }
}

fn canonical_pairs(
    pairs: &[(usize, usize)],
    detections: &[Detection],
    map: &LandmarkMap,
) -> Vec<(usize, usize)> {
    let mut key = pairs.to_vec();
    key.sort_by_key(|&(d, _)| detections[d].id);
    // Canonical negative landmark coordinate, NOT negative robot position / own-half selection.
    let flip = key
        .iter()
        .find_map(|&(_, m)| {
            let p = map.landmarks[m].xy;
            if p.x() != 0.0 {
                Some(p.x() > 0.0)
            } else if p.y() != 0.0 {
                Some(p.y() > 0.0)
            } else {
                None
            }
        })
        .unwrap_or(false);
    if flip {
        for (_, landmark) in &mut key {
            *landmark = map.symmetric_id(*landmark);
        }
    }
    key
}

fn result(
    map: &LandmarkMap,
    detections: &[Detection],
    pairs: &[(usize, usize)],
    rms: f32,
) -> AssociationResult {
    AssociationResult {
        associations: pairs
            .iter()
            .map(|&(d, m)| FieldMarkAssociation {
                detection: detections[d].pixel,
                field_point: map.landmarks[m].xy.extend(0.0),
            })
            .collect(),
        debug: Some(GlobalLocalizationDebug {
            association_count: pairs.len(),
            pairwise_distance_rms: rms,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linear_algebra::{Isometry3, point};
    use types::field_dimensions::FieldDimensions;

    fn detection(class: VisualFeatureClass, xy: Vector2<f32>, id: usize) -> Detection {
        Detection {
            id,
            class,
            pixel: point![id as f32, 0.0],
            confidence: 1.0,
            xy,
            pixel_covariance: Matrix2::identity() * 1.0e-6,
            tilt_jacobian: Matrix2::zeros(),
            height_jacobian: Vector2::zeros(),
        }
    }

    fn boundary_features() -> crate::DetectedVisualFeatures {
        let feature = |pixel| DetectedVisualFeature {
            pixel,
            confidence: 1.0,
        };
        crate::DetectedVisualFeatures {
            l_spots: vec![
                feature(point![94.36139, 324.65942]),
                feature(point![175.08398, 246.89253]),
            ],
            penalty_spots: vec![feature(point![534.52747, 318.63763])],
            ..Default::default()
        }
    }

    fn boundary_input<'a>(
        features: &'a crate::DetectedVisualFeatures,
        config: &'a GlobalAssociationConfig,
    ) -> GlobalAssociationInput<'a> {
        use nalgebra::{Matrix3, Rotation3, Translation3, UnitQuaternion};

        GlobalAssociationInput {
            visual_features: features,
            robot_to_local: Isometry3::wrap(nalgebra::Isometry3::from_parts(
                Translation3::new(-4.5, 0.0, 0.55),
                UnitQuaternion::from_euler_angles(0.0, 0.2, std::f32::consts::FRAC_PI_6),
            )),
            robot_to_camera: Isometry3::wrap(nalgebra::Isometry3::from_parts(
                Translation3::identity(),
                UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(
                    Matrix3::new(0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0),
                )),
            )),
            camera_intrinsic: projection::intrinsic::Intrinsic::new(
                nalgebra::vector![400.0, 300.0],
                point![320.0, 240.0],
            ),
            field_dimensions: &FieldDimensions::SPL_2025,
            parameters: config,
        }
    }

    #[test]
    fn boundary_triangle_contrasts_reject_the_rival_that_passes_every_edge_and_chirality() {
        let config = GlobalAssociationConfig::default();
        let truth = [point![-3.2, 0.0], point![-3.9, 1.1], point![-2.85, 2.0]];
        let rival = [truth[0], point![-3.9, -1.1], truth[1]];
        for noisy in [false, true] {
            let mut features = boundary_features();
            if !noisy {
                let input = boundary_input(&features, &config);
                let local_to_camera = input.robot_to_camera * input.robot_to_local.inverse();
                let intrinsic = input.camera_intrinsic;
                for (feature, point) in features
                    .penalty_spots
                    .iter_mut()
                    .chain(&mut features.l_spots)
                    .zip(truth)
                {
                    let camera = local_to_camera
                        * point![<coordinate_systems::Local>, point.x(), point.y(), 0.0];
                    feature.pixel = intrinsic.project(camera.coords());
                }
            }
            let input = boundary_input(&features, &config);
            let (map, detections) = preprocess(input).unwrap();
            assert_eq!(detections.len(), 3);
            assert_eq!(detections[0].class, VisualFeatureClass::PenaltySpot);
            let distances = distance_constraints(&detections, config).unwrap();
            let triangle =
                TriangleConstraint::new(&detections[0], &detections[1], &detections[2], config)
                    .unwrap();
            let mut search = Search {
                map: &map,
                detections: &detections,
                distances,
                order: vec![0, 1, 2],
                triangles: vec![triangle],
                remaining_work: config.max_work,
                solution: None,
            };
            for (points, accepted) in [(truth, true), (rival, false)] {
                let columns = points.map(|point| {
                    map.landmarks
                        .iter()
                        .position(|landmark| (landmark.xy - point).inner.norm() < 1.0e-5)
                        .unwrap()
                });
                for i in 0..3 {
                    assert_eq!(detections[i].class, map.landmarks[columns[i]].class);
                    for j in 0..i {
                        let DistanceConstraint {
                            distance,
                            tolerance: gate,
                        } = search.distances[i][j];
                        assert!((distance - (points[i] - points[j]).inner.norm()).abs() <= gate);
                    }
                }
                assert!(
                    cross((points[1] - points[0]).inner, (points[2] - points[0]).inner)
                        * triangle.sign
                        > 0.0
                );
                assert_eq!(
                    search.candidate_compatible(2, columns[2], &[(0, columns[0]), (1, columns[1])]),
                    Some(accepted)
                );
            }
            let (contrast, gate) = triangle.contrasts[2];
            let contrast_error = |points: [Point2<coordinate_systems::Field>; 3]| {
                contrast
                    - ((points[2] - points[0]).inner.norm() - (points[2] - points[1]).inner.norm())
            };
            let sigma = (gate - 2.0 * config.geometric_tolerance) / config.mahalanobis_gate.sqrt();
            assert!(contrast_error(truth).abs() < 0.002);
            assert!((1.54..1.55).contains(&contrast_error(rival)));
            assert!((0.071..0.072).contains(&sigma));
            assert!((0.276..0.279).contains(&gate));
            search.remaining_work = config.max_work;
            search.visit(&mut Vec::new()).unwrap();
            let work = config.max_work - search.remaining_work;
            assert_eq!(work, 181);
            let result = associate(input).unwrap();
            assert_eq!(result.associations.len(), 3);
            for association in result.associations {
                let row = detections
                    .iter()
                    .position(|detection| detection.pixel == association.detection)
                    .unwrap();
                assert!((association.field_point.xy() - truth[row]).inner.norm() < 1.0e-5);
            }
            for budget in [work - 1, work] {
                let bounded = GlobalAssociationConfig {
                    max_work: budget,
                    ..config
                };
                assert_eq!(
                    associate(boundary_input(&features, &bounded)).is_some(),
                    budget == work
                );
            }
        }
    }

    #[test]
    fn contrast_combines_shared_vertex_pixel_gradients_before_covariance() {
        let config = GlobalAssociationConfig::default();
        let mut points = [0.0, -1.0, -2.0]
            .map(|x| detection(VisualFeatureClass::LSpot, Vector2::new(x, 0.0), 0));
        for (id, point) in points.iter_mut().enumerate() {
            point.id = id;
            point.tilt_jacobian = Matrix2::identity() * 100.0;
            point.height_jacobian = Vector2::new(100.0, 100.0);
        }
        points[0].pixel_covariance = Matrix2::identity();
        let triangle = TriangleConstraint::new(&points[0], &points[1], &points[2], config).unwrap();
        assert_eq!(triangle.contrasts[0].0, -1.0);
        // At the common vertex u_ab == u_ac: its pixel gradient is zero, not two independent terms.
        let expected_gate =
            config.mahalanobis_gate.sqrt() * 2.0e-6_f32.sqrt() + 2.0 * config.geometric_tolerance;
        assert!((triangle.contrasts[0].1 - expected_gate).abs() < 1.0e-7);
        points[1].xy = points[0].xy;
        assert!(TriangleConstraint::new(&points[0], &points[1], &points[2], config).is_none());
    }

    #[test]
    fn contrasts_preserve_genuinely_congruent_distinct_orbits() {
        use types::field_dimensions::{Half, Side};

        let field = FieldDimensions {
            penalty_area_length: FieldDimensions::SPL_2025.goal_box_area_length,
            ..FieldDimensions::SPL_2025
        };
        let map = LandmarkMap::new(&field);
        let classes = [
            VisualFeatureClass::LSpot,
            VisualFeatureClass::TSpot,
            VisualFeatureClass::TSpot,
        ];
        let points = [
            field.goal_box_corner(Half::Opponent, Side::Left),
            field.goal_box_goal_line_intersection(Half::Opponent, Side::Left),
            field.penalty_box_goal_line_intersection(Half::Opponent, Side::Left),
        ];
        let rival = [
            field.penalty_box_corner(Half::Opponent, Side::Right),
            field.penalty_box_goal_line_intersection(Half::Opponent, Side::Right),
            field.goal_box_goal_line_intersection(Half::Opponent, Side::Right),
        ];
        let detections =
            std::array::from_fn::<_, 3, _>(|i| detection(classes[i], points[i].coords().inner, i));
        let alternative =
            std::array::from_fn::<_, 3, _>(|i| detection(classes[i], rival[i].coords().inner, i));
        let config = GlobalAssociationConfig::default();
        let triangle =
            TriangleConstraint::new(&detections[0], &detections[1], &detections[2], config)
                .unwrap();
        let other =
            TriangleConstraint::new(&alternative[0], &alternative[1], &alternative[2], config)
                .unwrap();
        assert_eq!(triangle.sign, other.sign);
        assert_ne!(triangle.sign, 0.0);
        for ((value, gate), (other, _)) in triangle.contrasts.into_iter().zip(other.contrasts) {
            assert!((value - other).abs() < gate);
        }
        assert!(match_geometry(&map, &detections, config).is_none());
    }

    #[test]
    #[ignore = "host runtime characterization; run explicitly with --ignored --nocapture"]
    fn boundary_runtime_characterization() {
        use std::{hint::black_box, time::Instant};

        let features = boundary_features();
        let config = GlobalAssociationConfig::default();
        let input = boundary_input(&features, &config);
        let mut samples = Vec::with_capacity(500);
        for iteration in 0..520 {
            let start = Instant::now();
            let result = black_box(associate(black_box(input))).unwrap();
            let elapsed = start.elapsed();
            assert_eq!(result.associations.len(), 3);
            if iteration >= 20 {
                samples.push(elapsed);
            }
        }
        samples.sort_unstable();
        eprintln!(
            "global/boundary: samples=500 p50={:?} p95={:?} p99={:?} max={:?}",
            samples[250], samples[475], samples[495], samples[499]
        );
    }

    #[test]
    fn invariant_covariance_cancels_shared_errors_before_taking_norms() {
        let mut a = detection(VisualFeatureClass::GoalPost, Vector2::zeros(), 0);
        a.tilt_jacobian = Matrix2::identity() * 100.0;
        a.height_jacobian = Vector2::new(100.0, 100.0);
        let mut b = a.clone();
        b.id = 1;
        let config = GlobalAssociationConfig::default();
        let sigma = invariant_sigma(&[(&a, Vector2::x()), (&b, -Vector2::x())], config);
        assert!((sigma - 2.0e-6_f32.sqrt()).abs() < 1.0e-7);
        b.height_jacobian.x += 2.0;
        let sigma = invariant_sigma(&[(&a, Vector2::x()), (&b, -Vector2::x())], config);
        assert!((sigma - (2.0e-6_f32.sqrt() + 2.0 * config.height_sigma)).abs() < 1.0e-6);
    }

    #[test]
    fn exhaustion_after_a_candidate_still_rejects_uncertified_uniqueness() {
        let map = LandmarkMap::new(&FieldDimensions::SPL_2025);
        let detections = [0, 1, 29]
            .into_iter()
            .enumerate()
            .map(|(id, m)| {
                detection(
                    map.landmarks[m].class,
                    map.landmarks[m].xy.coords().inner,
                    id,
                )
            })
            .collect::<Vec<_>>();
        let distances = detections
            .iter()
            .map(|a| {
                detections
                    .iter()
                    .map(|b| DistanceConstraint {
                        distance: (a.xy - b.xy).norm(),
                        tolerance: 0.03,
                    })
                    .collect()
            })
            .collect::<Vec<Vec<_>>>();
        let config = GlobalAssociationConfig::default();
        let triangle =
            TriangleConstraint::new(&detections[0], &detections[1], &detections[2], config)
                .unwrap();
        let mut found_partial = false;
        for budget in 1..200 {
            let mut search = Search {
                map: &map,
                detections: &detections,
                distances: distances.clone(),
                order: vec![0, 1, 2],
                triangles: vec![triangle],
                remaining_work: budget,
                solution: None,
            };
            if search.visit(&mut Vec::new()).is_none() && search.solution.is_some() {
                assert_eq!(search.remaining_work, 0);
                assert!(
                    match_geometry(
                        &map,
                        &detections,
                        GlobalAssociationConfig {
                            max_work: budget,
                            ..config
                        }
                    )
                    .is_none()
                );
                found_partial = true;
                break;
            }
        }
        assert!(found_partial);
        assert_eq!(
            match_geometry(&map, &detections, config).unwrap().0.len(),
            3
        );
        // 11 visits, 28 attempts, 28 pair checks and 6 contrasts, including rejected branches.
        for budget in [72, 73] {
            let result = match_geometry(
                &map,
                &detections,
                GlobalAssociationConfig {
                    max_work: budget,
                    ..config
                },
            );
            assert_eq!(result.is_some(), budget == 73);
        }
    }
}

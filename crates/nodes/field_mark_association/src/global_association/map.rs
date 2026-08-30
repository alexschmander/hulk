use linear_algebra::Point2;
use types::field_dimensions::{FieldDimensions, Half, Side};

use coordinate_systems::Field;

use super::{FEATURE_CLASSES, VisualFeatureClass};

const SYMMETRY_EPSILON: f32 = 1.0e-4;
const FIELD_HALVES: usize = 2;
const FIELD_SIDES: usize = 2;
const L_SPOTS_PER_QUADRANT: usize = 3;
const T_SPOTS_PER_PENALTY_BOX: usize = 2;
const GOALPOST_COUNT: usize = FIELD_HALVES * FIELD_SIDES;
const T_SPOT_COUNT: usize = FIELD_SIDES + FIELD_HALVES * FIELD_SIDES * T_SPOTS_PER_PENALTY_BOX;
const X_SPOT_COUNT: usize = 1 + FIELD_SIDES;
const PENALTY_SPOT_COUNT: usize = FIELD_HALVES;
pub(crate) const MAX_LANDMARKS_PER_CLASS: usize = FIELD_HALVES * FIELD_SIDES * L_SPOTS_PER_QUADRANT;
pub(crate) const FIELD_LANDMARK_COUNT: usize =
    GOALPOST_COUNT + MAX_LANDMARKS_PER_CLASS + T_SPOT_COUNT + X_SPOT_COUNT + PENALTY_SPOT_COUNT;

#[derive(Clone, Debug)]
pub(crate) struct LandmarkMap {
    pub landmarks: Vec<Landmark>,
    landmarks_by_class: [Vec<usize>; FEATURE_CLASSES.len()],
    class_rarity_weight: [f32; FEATURE_CLASSES.len()],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Landmark {
    pub symmetric_id: usize,
    pub class: VisualFeatureClass,
    pub xy: Point2<Field>,
}

impl LandmarkMap {
    pub fn new(field: &FieldDimensions) -> Self {
        let mut landmarks = candidate_landmarks(field);
        fill_symmetric_ids(&mut landmarks);
        let landmarks_by_class = landmarks_by_class(&landmarks);
        assert!(
            landmarks_by_class
                .iter()
                .all(|landmarks| landmarks.len() <= MAX_LANDMARKS_PER_CLASS),
            "generated landmark class exceeds assignment capacity"
        );
        assert_eq!(landmarks.len(), FIELD_LANDMARK_COUNT);
        let class_rarity_weight = class_rarity_weights(&landmarks_by_class);

        Self {
            landmarks,
            landmarks_by_class,
            class_rarity_weight,
        }
    }

    pub fn symmetric_id(&self, landmark_id: usize) -> usize {
        self.landmarks
            .get(landmark_id)
            .map_or(landmark_id, |landmark| landmark.symmetric_id)
    }

    pub fn has_class(&self, class: VisualFeatureClass) -> bool {
        !self.landmarks_for_class(class).is_empty()
    }

    pub fn landmarks_for_class(&self, class: VisualFeatureClass) -> &[usize] {
        self.landmarks_by_class
            .get(class.index())
            .map_or(&[], Vec::as_slice)
    }

    pub fn rarity_weight(&self, class: VisualFeatureClass) -> f32 {
        self.class_rarity_weight
            .get(class.index())
            .copied()
            .unwrap_or(0.0)
    }
}

fn landmarks_by_class(landmarks: &[Landmark]) -> [Vec<usize>; FEATURE_CLASSES.len()] {
    let mut landmarks_by_class = std::array::from_fn(|_| Vec::new());
    for (id, landmark) in landmarks.iter().enumerate() {
        landmarks_by_class[landmark.class.index()].push(id);
    }
    landmarks_by_class
}

fn class_rarity_weights(
    landmarks_by_class: &[Vec<usize>; FEATURE_CLASSES.len()],
) -> [f32; FEATURE_CLASSES.len()] {
    std::array::from_fn(|index| {
        let count = landmarks_by_class[index].len();
        if count == 0 { 0.0 } else { 1.0 / count as f32 }
    })
}

fn candidate_landmarks(field: &FieldDimensions) -> Vec<Landmark> {
    candidate_points(field)
        .into_iter()
        .enumerate()
        .map(|(id, (class, xy))| Landmark {
            symmetric_id: id,
            class,
            xy,
        })
        .collect()
}

pub(crate) fn candidate_points(
    field: &FieldDimensions,
) -> Vec<(VisualFeatureClass, Point2<Field>)> {
    let mut points = Vec::with_capacity(FIELD_LANDMARK_COUNT);
    for half in [Half::Opponent, Half::Own] {
        for side in [Side::Left, Side::Right] {
            points.push((VisualFeatureClass::GoalPost, field.goal_post(half, side)));
        }
    }
    for half in [Half::Opponent, Half::Own] {
        for side in [Side::Left, Side::Right] {
            points.push((VisualFeatureClass::LSpot, field.corner(half, side)));
            points.push((VisualFeatureClass::LSpot, field.goal_box_corner(half, side)));
            points.push((
                VisualFeatureClass::LSpot,
                field.penalty_box_corner(half, side),
            ));
        }
    }
    for side in [Side::Left, Side::Right] {
        points.push((VisualFeatureClass::TSpot, field.t_crossing(side)));
    }
    for half in [Half::Opponent, Half::Own] {
        for side in [Side::Left, Side::Right] {
            points.push((
                VisualFeatureClass::TSpot,
                field.goal_box_goal_line_intersection(half, side),
            ));
            points.push((
                VisualFeatureClass::TSpot,
                field.penalty_box_goal_line_intersection(half, side),
            ));
        }
    }
    points.push((VisualFeatureClass::XSpot, field.center()));
    for side in [Side::Left, Side::Right] {
        points.push((VisualFeatureClass::XSpot, field.x_crossing(side)));
    }
    for half in [Half::Opponent, Half::Own] {
        points.push((VisualFeatureClass::PenaltySpot, field.penalty_spot(half)));
    }
    points
}

fn fill_symmetric_ids(landmarks: &mut [Landmark]) {
    for index in 0..landmarks.len() {
        let landmark = landmarks[index];
        if let Some((partner_id, _)) = landmarks.iter().enumerate().find(|(_, candidate)| {
            candidate.class == landmark.class
                && (candidate.xy.x() + landmark.xy.x()).abs() <= SYMMETRY_EPSILON
                && (candidate.xy.y() + landmark.xy.y()).abs() <= SYMMETRY_EPSILON
        }) {
            landmarks[index].symmetric_id = partner_id;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_sets_match_field_feature_classes() {
        let map = LandmarkMap::new(&FieldDimensions::SPL_2025);

        assert_eq!(
            map.landmarks_for_class(VisualFeatureClass::GoalPost).len(),
            4
        );
        assert_eq!(map.landmarks_for_class(VisualFeatureClass::LSpot).len(), 12);
        assert_eq!(map.landmarks_for_class(VisualFeatureClass::TSpot).len(), 10);
        assert_eq!(map.landmarks_for_class(VisualFeatureClass::XSpot).len(), 3);
        assert_eq!(
            map.landmarks_for_class(VisualFeatureClass::PenaltySpot)
                .len(),
            2
        );
    }
}

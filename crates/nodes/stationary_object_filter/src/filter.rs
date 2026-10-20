use coordinate_systems::{Field, Ground, Odometry};
use linear_algebra::{IntoFramed, Isometry2, Pose2};
use nalgebra::{SVector, Vector2};
use object_tracking::{
    camera::{CameraView, GroundAnchor},
    classification::{ClassBelief, ClassEvidence},
    gaussian::{Gaussian, marginal_sigma_points},
    motion::{MotionModel, Stationary},
    multi_bernoulli::{Birth, Measurement, MultiBernoulli, ScanReport, Track, TrackAttributes},
};
use projection::camera_matrix::CameraMatrix;
use ros_z::time::Time;
use serde::{Deserialize, Serialize};
use types::{
    object_detection::{Object, RobocupObjectLabel},
    obstacles::Obstacle,
    odometry,
    tracked_objects::{StationaryClass, StationaryTrack, TrackStatus, TrackedObjects},
};

use crate::parameters::Parameters;

pub const CLASS_COUNT: usize = StationaryClass::ALL.len();

/// Detector label of a measurement together with its likelihood under each class.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Label {
    pub class: StationaryClass,
    pub evidence: ClassEvidence<CLASS_COUNT>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Attributes {
    pub class: ClassBelief<CLASS_COUNT>,
    pub last_detection: Option<Time>,
    pub detection_probability: f32,
}

impl TrackAttributes<Label> for Attributes {
    fn label_likelihood(&self, label: &Label) -> f32 {
        self.class.label_likelihood(&label.evidence)
    }

    fn update(&mut self, missed_weight: f32, observations: &[(f32, &Label)]) {
        let evidence: Vec<_> = observations
            .iter()
            .map(|(weight, label)| (*weight, &label.evidence))
            .collect();
        self.class.update(missed_weight, &evidence);
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct StationaryObjectFilter {
    pub tracks: MultiBernoulli<2, Attributes>,
    odometry: Option<(Time, Pose2<Odometry>)>,
}

impl StationaryObjectFilter {
    pub fn new() -> Self {
        Self {
            tracks: MultiBernoulli::new(),
            odometry: None,
        }
    }

    pub fn clear(&mut self) {
        self.tracks.tracks.clear();
    }

    pub fn odometry_pose(&self) -> Option<Pose2<Odometry>> {
        self.odometry.map(|(_, pose)| pose)
    }

    /// Advances the map to a new odometry pose, adding the drift of the odometry frame.
    pub fn predict_to_odometry(
        &mut self,
        time: Time,
        pose: Pose2<Odometry>,
        parameters: &Parameters,
    ) {
        if let Some((last_time, last_pose)) = self.odometry {
            let delta = odometry::previous_to_current(last_pose, pose);
            let translation = delta.translation().coords().norm();
            let rotation = delta.orientation().angle();
            let delta_time = time.duration_since(last_time).as_secs_f32();
            let robot = pose.position().inner.coords;
            let drift = parameters.drift;
            self.tracks.predict(
                |_| Stationary.transition(delta_time),
                |_| 1.0,
                |gaussian| {
                    drift.position_noise(gaussian.mean - robot, translation, rotation, delta_time)
                },
            );
        }
        self.odometry = Some((time, pose));
    }

    /// Updates the map with the detections of one image, using the odometry pose of that image.
    ///
    /// Returns `None` if no odometry pose is known yet.
    pub fn update_with_detections(
        &mut self,
        time: Time,
        detected_objects: &[Object<RobocupObjectLabel>],
        camera_matrix: &CameraMatrix,
        parameters: &Parameters,
    ) -> Option<ScanReport> {
        let (_, pose) = self.odometry?;
        let view = CameraView {
            camera_matrix,
            ground_to_odometry: pose.as_transform::<Ground>(),
        };

        let measurements: Vec<Measurement<Label>> = detected_objects
            .iter()
            .filter_map(|object| {
                let class = stationary_class(object.label)?;
                let class_parameters = parameters.classes.get(class);
                if object.bounding_box.confidence < class_parameters.minimum_confidence {
                    return None;
                }
                let anchor = if class == StationaryClass::GoalPost {
                    GroundAnchor::BottomCenter
                } else {
                    GroundAnchor::Center
                };
                let model = view.measure(&object.bounding_box, anchor, &parameters.observation)?;
                let clutter_density = model.clutter_density(
                    class_parameters.clutter_density,
                    class_parameters.bearing_clutter_density,
                );
                Some(Measurement {
                    model,
                    label: Label {
                        class,
                        evidence: label_evidence(class, parameters),
                    },
                    clutter_density,
                })
            })
            .collect();

        let report = self.tracks.update(
            &measurements,
            &parameters.scan,
            |track, gaussian| expected_detection_probability(&view, track, gaussian, parameters),
            |measurement| birth(&view, measurement, parameters),
        );

        for track in &mut self.tracks.tracks {
            let probability =
                expected_detection_probability(&view, track, &track.estimate(), parameters);
            track.attributes.detection_probability = probability;
            if track.last_association_probability >= 0.5 {
                track.attributes.last_detection = Some(time);
            }
        }
        let retention = parameters.retention_without_detection;
        self.tracks.tracks.retain(|track| {
            track
                .attributes
                .last_detection
                .is_some_and(|last| time.duration_since(last) <= retention)
                || track.attributes.last_detection.is_none()
        });
        self.tracks
            .prune(parameters.minimum_existence, parameters.maximum_tracks);
        Some(report)
    }

    pub fn tracked_objects(
        &self,
        now: Time,
        ground_to_field: Option<Isometry2<Ground, Field>>,
        parameters: &Parameters,
    ) -> Option<TrackedObjects<StationaryTrack>> {
        let pose = self.odometry_pose()?;
        let ground_to_odometry = pose.as_transform::<Ground>();
        let odometry_to_ground = ground_to_odometry.inverse();
        let tracks = self
            .tracks
            .tracks
            .iter()
            .map(|track| {
                let state = track.estimate();
                let position = state.mean.framed::<Odometry>().as_point();
                let position_in_ground = odometry_to_ground * position;
                let class = StationaryClass::ALL[track.attributes.class.most_likely()];
                let time_since_detection = track
                    .attributes
                    .last_detection
                    .map(|last| now.duration_since(last));
                StationaryTrack {
                    id: track.id,
                    existence: track.existence,
                    status: status(track.existence, time_since_detection, parameters),
                    class,
                    class_probabilities: track
                        .attributes
                        .class
                        .probabilities
                        .iter()
                        .copied()
                        .collect(),
                    is_obstacle: class.is_obstacle(),
                    detection_probability: track.attributes.detection_probability,
                    time_since_detection,
                    state,
                    position_in_ground,
                    position_in_field: ground_to_field
                        .map(|ground_to_field| ground_to_field * position_in_ground),
                }
            })
            .collect();
        Some(TrackedObjects {
            time: now,
            ground_to_odometry,
            ground_to_field,
            tracks,
        })
    }

    pub fn obstacles(&self, parameters: &Parameters) -> Vec<Obstacle> {
        let Some(pose) = self.odometry_pose() else {
            return Vec::new();
        };
        let odometry_to_ground = pose.as_transform::<Ground>().inverse();
        self.tracks
            .tracks
            .iter()
            .filter(|track| track.existence >= parameters.obstacle_existence)
            .filter(|track| {
                StationaryClass::ALL[track.attributes.class.most_likely()].is_obstacle()
            })
            .map(|track| {
                let position = track.estimate().mean.framed::<Odometry>().as_point();
                Obstacle::goal_post(
                    odometry_to_ground * position,
                    parameters.goal_post_obstacle_radius,
                )
            })
            .collect()
    }
}

pub fn stationary_class(label: RobocupObjectLabel) -> Option<StationaryClass> {
    match label {
        RobocupObjectLabel::GoalPost => Some(StationaryClass::GoalPost),
        RobocupObjectLabel::LSpot => Some(StationaryClass::LSpot),
        RobocupObjectLabel::TSpot => Some(StationaryClass::TSpot),
        RobocupObjectLabel::XSpot => Some(StationaryClass::XSpot),
        RobocupObjectLabel::PenaltySpot => Some(StationaryClass::PenaltySpot),
        RobocupObjectLabel::Ball | RobocupObjectLabel::Robot => None,
    }
}

fn class_index(class: StationaryClass) -> usize {
    StationaryClass::ALL
        .iter()
        .position(|candidate| *candidate == class)
        .expect("all classes are listed")
}

/// `P(label | class)` for a detector label: correct with the true class's label accuracy,
/// otherwise uniformly confused with every other class.
pub fn label_evidence(
    label: StationaryClass,
    parameters: &Parameters,
) -> ClassEvidence<CLASS_COUNT> {
    let label_index = class_index(label);
    let likelihoods = SVector::<f32, CLASS_COUNT>::from_fn(|class, _| {
        let accuracy = parameters
            .classes
            .get(StationaryClass::ALL[class])
            .label_accuracy
            .clamp(0.0, 1.0);
        if class == label_index {
            accuracy
        } else {
            (1.0 - accuracy) / (CLASS_COUNT - 1) as f32
        }
    });
    ClassEvidence { likelihoods }
}

fn class_detection_probability(
    view: &CameraView,
    position: Vector2<f32>,
    class: StationaryClass,
    parameters: &Parameters,
) -> f32 {
    let class_parameters = parameters.classes.get(class);
    let point = position.framed::<Odometry>().as_point();
    if view.ground_distance(point) > class_parameters.maximum_distance {
        return 0.0;
    }
    class_parameters.detection_probability
        * view.visible_fraction(point, class_parameters.height, parameters.visibility_margin)
}

/// Detection probability averaged over the component's position uncertainty and class belief.
fn expected_detection_probability(
    view: &CameraView,
    track: &Track<2, Attributes>,
    gaussian: &Gaussian<2>,
    parameters: &Parameters,
) -> f32 {
    let points = marginal_sigma_points::<2, 2>(gaussian);
    let per_class = SVector::<f32, CLASS_COUNT>::from_fn(|class, _| {
        if track.attributes.class.probabilities[class] < 1e-3 {
            return 0.0;
        }
        points
            .iter()
            .map(|point| {
                class_detection_probability(view, *point, StationaryClass::ALL[class], parameters)
            })
            .sum::<f32>()
            / points.len() as f32
    });
    track.attributes.class.expectation(&per_class)
}

fn birth(
    view: &CameraView,
    measurement: &Measurement<Label>,
    parameters: &Parameters,
) -> Option<Birth<2, Attributes>> {
    let (position, covariance) = measurement.model.ground_position()?;
    let class = measurement.label.class;
    // In the units of the measurement likelihood, like the clutter density.
    let intensity = parameters.classes.get(class).undetected_density
        * measurement.model.square_meters_per_unit()
        * class_detection_probability(view, position, class, parameters);
    Some(Birth {
        gaussian: Gaussian {
            mean: position,
            covariance,
        },
        attributes: Attributes {
            class: ClassBelief::from_evidence(&ClassBelief::uniform(), &measurement.label.evidence),
            last_detection: None,
            detection_probability: 0.0,
        },
        intensity,
    })
}

fn status(
    existence: f32,
    time_since_detection: Option<std::time::Duration>,
    parameters: &Parameters,
) -> TrackStatus {
    if existence < parameters.confirmation_existence {
        TrackStatus::Tentative
    } else if time_since_detection.is_none_or(|age| age > parameters.dormant_after) {
        TrackStatus::Dormant
    } else {
        TrackStatus::Confirmed
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use geometry::rectangle::Rectangle;
    use linear_algebra::point;
    use object_tracking::camera::{GroundAnchor, synthetic::forward_looking_camera};
    use projection::Projection;
    use types::bounding_box::BoundingBox;

    use super::*;

    fn parameters() -> Parameters {
        json5::from_str(include_str!(
            "../../../../etc/parameters/base/stationary_object_filter.json5"
        ))
        .expect("base parameters must deserialize")
    }

    fn detection(
        camera: &CameraMatrix,
        label: RobocupObjectLabel,
        position: linear_algebra::Point2<Ground>,
        anchor: GroundAnchor,
    ) -> Object<RobocupObjectLabel> {
        let pixel = camera.ground_to_pixel(position).unwrap();
        let area = match anchor {
            GroundAnchor::BottomCenter => Rectangle {
                min: point![pixel.x() - 5.0, pixel.y() - 60.0],
                max: point![pixel.x() + 5.0, pixel.y()],
            },
            GroundAnchor::Center => Rectangle {
                min: point![pixel.x() - 8.0, pixel.y() - 4.0],
                max: point![pixel.x() + 8.0, pixel.y() + 4.0],
            },
        };
        Object {
            label,
            bounding_box: BoundingBox {
                area,
                confidence: 0.9,
            },
        }
    }

    fn time(seconds: f32) -> Time {
        Time::from_nanos((seconds * 1e9) as i64)
    }

    #[test]
    fn repeated_goal_post_detections_confirm_one_obstacle() {
        let parameters = parameters();
        let camera = forward_looking_camera(0.8, 0.4);
        let mut filter = StationaryObjectFilter::new();
        let post = point![3.0, 0.5];

        for step in 0..10 {
            let now = time(step as f32 * 0.033);
            filter.predict_to_odometry(now, Pose2::zero(), &parameters);
            filter
                .update_with_detections(
                    now,
                    &[detection(
                        &camera,
                        RobocupObjectLabel::GoalPost,
                        post,
                        GroundAnchor::BottomCenter,
                    )],
                    &camera,
                    &parameters,
                )
                .unwrap();
        }

        let confident: Vec<_> = filter
            .tracks
            .tracks
            .iter()
            .filter(|track| track.existence > 0.5)
            .collect();
        assert_eq!(confident.len(), 1);
        let estimate = confident[0].estimate().mean;
        assert_relative_eq!(estimate, Vector2::new(3.0, 0.5), epsilon = 0.05);
        let obstacles = filter.obstacles(&parameters);
        assert_eq!(obstacles.len(), 1);
    }

    #[test]
    fn out_of_view_landmark_is_kept_and_visible_miss_removes_it() {
        let parameters = parameters();
        let camera = forward_looking_camera(0.8, 0.4);
        let mut filter = StationaryObjectFilter::new();
        let post = point![3.0, 0.5];

        for step in 0..5 {
            let now = time(step as f32 * 0.033);
            filter.predict_to_odometry(now, Pose2::zero(), &parameters);
            filter.update_with_detections(
                now,
                &[detection(
                    &camera,
                    RobocupObjectLabel::GoalPost,
                    post,
                    GroundAnchor::BottomCenter,
                )],
                &camera,
                &parameters,
            );
        }
        let confirmed = filter.tracks.tracks[0].existence;

        // Turned around: the post is behind the robot.
        let turned = Pose2::new(point![0.0, 0.0], std::f32::consts::PI);
        for step in 5..60 {
            let now = time(step as f32 * 0.033);
            filter.predict_to_odometry(now, turned, &parameters);
            filter.update_with_detections(now, &[], &camera, &parameters);
        }
        assert_relative_eq!(filter.tracks.tracks[0].existence, confirmed, epsilon = 1e-6);

        // Looking at it again without seeing it.
        for step in 60..70 {
            let now = time(step as f32 * 0.033);
            filter.predict_to_odometry(now, Pose2::zero(), &parameters);
            filter.update_with_detections(now, &[], &camera, &parameters);
        }
        assert!(filter.tracks.tracks.is_empty());
    }

    #[test]
    fn conflicting_spot_labels_stay_on_one_track() {
        let parameters = parameters();
        let camera = forward_looking_camera(0.8, 0.4);
        let mut filter = StationaryObjectFilter::new();
        let spot = point![2.0, -0.3];
        let labels = [
            RobocupObjectLabel::TSpot,
            RobocupObjectLabel::LSpot,
            RobocupObjectLabel::TSpot,
            RobocupObjectLabel::TSpot,
            RobocupObjectLabel::TSpot,
        ];

        for (step, label) in labels.into_iter().enumerate() {
            let now = time(step as f32 * 0.033);
            filter.predict_to_odometry(now, Pose2::zero(), &parameters);
            filter.update_with_detections(
                now,
                &[detection(&camera, label, spot, GroundAnchor::Center)],
                &camera,
                &parameters,
            );
        }

        let confident: Vec<_> = filter
            .tracks
            .tracks
            .iter()
            .filter(|track| track.existence > 0.5)
            .collect();
        assert_eq!(confident.len(), 1);
        assert_eq!(
            StationaryClass::ALL[confident[0].attributes.class.most_likely()],
            StationaryClass::TSpot
        );
        assert!(filter.obstacles(&parameters).is_empty());
    }
}

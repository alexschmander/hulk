use coordinate_systems::{Field, Ground, Odometry};
use linear_algebra::{IntoFramed, Isometry2, Pose2};
use nalgebra::Vector2;
use object_tracking::{
    camera::{CameraView, GroundAnchor},
    gaussian::{Gaussian, marginal_sigma_points},
    motion::{MotionModel, Stationary},
    multi_bernoulli::{Birth, Measurement, MultiBernoulli, ScanReport, TrackAttributes},
};
use projection::camera_matrix::CameraMatrix;
use ros_z::time::Time;
use serde::{Deserialize, Serialize};
use types::{
    object_detection::{Object, RobocupObjectLabel},
    obstacles::Obstacle,
    odometry,
    tracked_objects::{GoalPostTrack, TrackStatus, TrackedObjects},
};

use crate::parameters::Parameters;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Attributes {
    pub last_detection: Option<Time>,
    pub detection_probability: f32,
}

/// Goal post detections carry no label beyond their class, which the filter only accepts.
impl TrackAttributes<()> for Attributes {
    fn label_likelihood(&self, _label: &()) -> f32 {
        1.0
    }

    fn update(&mut self, _missed_weight: f32, _observations: &[(f32, &())]) {}
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct GoalPostFilter {
    pub tracks: MultiBernoulli<2, Attributes>,
    odometry: Option<(Time, Pose2<Odometry>)>,
}

impl GoalPostFilter {
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
        let measurements: Vec<Measurement<()>> = detected_objects
            .iter()
            .filter(|object| {
                object.label == RobocupObjectLabel::GoalPost
                    && object.bounding_box.confidence >= parameters.minimum_confidence
            })
            .filter_map(|object| {
                let model = view.measure(
                    &object.bounding_box,
                    GroundAnchor::BottomCenter,
                    &parameters.observation,
                )?;
                let clutter_density = model.clutter_density(
                    parameters.clutter_density,
                    parameters.bearing_clutter_density,
                );
                Some(Measurement {
                    model,
                    label: (),
                    clutter_density,
                })
            })
            .collect();

        let report = self.tracks.update(
            &measurements,
            &parameters.scan,
            |_, gaussian| expected_detection_probability(&view, gaussian, parameters),
            |measurement| birth(&view, measurement, parameters),
        );

        for track in &mut self.tracks.tracks {
            let probability = expected_detection_probability(&view, &track.estimate(), parameters);
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
    ) -> Option<TrackedObjects<GoalPostTrack>> {
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
                let time_since_detection = track
                    .attributes
                    .last_detection
                    .map(|last| now.duration_since(last));
                GoalPostTrack {
                    id: track.id,
                    existence: track.existence,
                    status: status(track.existence, time_since_detection, parameters),
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
            .map(|track| {
                let position = track.estimate().mean.framed::<Odometry>().as_point();
                Obstacle::goal_post(odometry_to_ground * position, parameters.obstacle_radius)
            })
            .collect()
    }
}

fn detection_probability(
    view: &CameraView,
    position: Vector2<f32>,
    parameters: &Parameters,
) -> f32 {
    let point = position.framed::<Odometry>().as_point();
    if view.ground_distance(point) > parameters.maximum_distance {
        return 0.0;
    }
    parameters.detection_probability
        * view.visible_fraction(point, parameters.height, parameters.visibility_margin)
}

/// Detection probability averaged over the component's position uncertainty.
fn expected_detection_probability(
    view: &CameraView,
    gaussian: &Gaussian<2>,
    parameters: &Parameters,
) -> f32 {
    let points = marginal_sigma_points::<2, 2>(gaussian);
    points
        .iter()
        .map(|point| detection_probability(view, *point, parameters))
        .sum::<f32>()
        / points.len() as f32
}

fn birth(
    view: &CameraView,
    measurement: &Measurement<()>,
    parameters: &Parameters,
) -> Option<Birth<2, Attributes>> {
    let (position, covariance) = measurement.model.ground_position()?;
    // In the units of the measurement likelihood, like the clutter density.
    let intensity = parameters.undetected_density
        * measurement.model.square_meters_per_unit()
        * detection_probability(view, position, parameters);
    Some(Birth {
        gaussian: Gaussian {
            mean: position,
            covariance,
        },
        attributes: Attributes {
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
    use object_tracking::camera::synthetic::forward_looking_camera;
    use projection::Projection;
    use types::bounding_box::BoundingBox;

    use super::*;

    fn parameters() -> Parameters {
        json5::from_str(include_str!(
            "../../../../etc/parameters/base/goal_post_filter.json5"
        ))
        .expect("base parameters must deserialize")
    }

    fn goal_post_detection(
        camera: &CameraMatrix,
        position: linear_algebra::Point2<Ground>,
    ) -> Object<RobocupObjectLabel> {
        let pixel = camera.ground_to_pixel(position).unwrap();
        let area = Rectangle {
            min: point![pixel.x() - 5.0, pixel.y() - 60.0],
            max: point![pixel.x() + 5.0, pixel.y()],
        };
        Object {
            label: RobocupObjectLabel::GoalPost,
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
        let mut filter = GoalPostFilter::new();
        let post = point![3.0, 0.5];

        for step in 0..10 {
            let now = time(step as f32 * 0.033);
            filter.predict_to_odometry(now, Pose2::zero(), &parameters);
            filter
                .update_with_detections(
                    now,
                    &[goal_post_detection(&camera, post)],
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
        let mut filter = GoalPostFilter::new();
        let post = point![3.0, 0.5];

        for step in 0..5 {
            let now = time(step as f32 * 0.033);
            filter.predict_to_odometry(now, Pose2::zero(), &parameters);
            filter.update_with_detections(
                now,
                &[goal_post_detection(&camera, post)],
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
    fn field_feature_detections_are_ignored() {
        let parameters = parameters();
        let camera = forward_looking_camera(0.8, 0.4);
        let mut filter = GoalPostFilter::new();
        let mut spot = goal_post_detection(&camera, point![2.0, -0.3]);
        spot.label = RobocupObjectLabel::TSpot;

        for step in 0..10 {
            let now = time(step as f32 * 0.033);
            filter.predict_to_odometry(now, Pose2::zero(), &parameters);
            filter.update_with_detections(now, std::slice::from_ref(&spot), &camera, &parameters);
        }

        assert!(filter.tracks.tracks.is_empty());
    }
}

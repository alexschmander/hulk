use std::{f32::consts::PI, time::Duration};

use approx::assert_relative_eq;
use coordinate_systems::{Field, Ground, Odometry};
use geometry::rectangle::Rectangle;
use hsl_network_messages::PlayerNumber;
use linear_algebra::{IntoFramed, Isometry2, Point2, Pose2, point};
use nalgebra::{Matrix4, SVector, Vector2, Vector4};
use object_tracking::{
    camera::synthetic::forward_looking_camera, classification::ClassBelief, gaussian::Gaussian,
};
use projection::{Projection, camera_matrix::CameraMatrix};
use ros_z::time::Time;
use types::{
    bounding_box::BoundingBox,
    object_detection::{Object, RobocupObjectLabel},
    obstacles::ObstacleKind,
    tracked_objects::HumanoidRole,
};

use crate::{
    Parameters,
    filter::{Attributes, HUMAN, HumanoidFilter, team_report},
};

const ROBOT_HEIGHT: f32 = 0.9;
const ROBOT_RADIUS: f32 = 0.2;
const FRAME: f32 = 1.0 / 30.0;

fn parameters() -> Parameters {
    json5::from_str(include_str!(
        "../../../../etc/parameters/base/humanoid_filter.json5"
    ))
    .expect("base parameters must deserialize")
}

fn time(seconds: f32) -> Time {
    Time::from_nanos((seconds * 1e9) as i64)
}

/// Ideal detector: boxes of all robots in view that are not mostly hidden by a closer robot.
fn detect(camera: &CameraMatrix, robots: &[Point2<Ground>]) -> Vec<Object<RobocupObjectLabel>> {
    let camera_position = camera.ground_to_camera.inverse().as_pose().position().xy();
    let bearing = |robot: &Point2<Ground>| {
        let offset = (*robot - camera_position).inner;
        (offset.norm(), offset.y.atan2(offset.x))
    };
    robots
        .iter()
        .filter(|robot| {
            // Hidden when its center lies behind a closer robot.
            let (distance, angle) = bearing(robot);
            !robots.iter().any(|other| {
                let (other_distance, other_angle) = bearing(other);
                other_distance < distance
                    && (other_angle - angle).abs() < (ROBOT_RADIUS / other_distance).asin()
            })
        })
        .filter_map(|robot| {
            let bottom = camera.ground_to_pixel(*robot).ok()?;
            let top = camera.ground_with_z_to_pixel(*robot, ROBOT_HEIGHT).ok()?;
            let size = camera.image_size;
            if bottom.x() < 0.0 || bottom.x() > size.x() || top.y() > size.y() || bottom.y() < 0.0 {
                return None;
            }
            Some(Object {
                label: RobocupObjectLabel::Robot,
                bounding_box: BoundingBox {
                    area: Rectangle {
                        min: point![bottom.x() - 15.0, top.y().max(0.0)],
                        max: point![bottom.x() + 15.0, bottom.y().min(size.y())],
                    },
                    confidence: 0.9,
                },
            })
        })
        .collect()
}

struct Scene {
    filter: HumanoidFilter,
    camera: CameraMatrix,
    parameters: Parameters,
    now: f32,
}

impl Scene {
    fn new() -> Self {
        Self {
            filter: HumanoidFilter::new(),
            camera: forward_looking_camera(0.8, 0.4),
            parameters: parameters(),
            now: 0.0,
        }
    }

    /// Advances one camera frame with the robot at `pose` observing robots at `robots`.
    fn step(&mut self, pose: Pose2<Odometry>, robots: &[Point2<Odometry>]) {
        self.now += FRAME;
        let now = time(self.now);
        let odometry_to_ground = pose.as_transform::<Ground>().inverse();
        let in_ground: Vec<_> = robots
            .iter()
            .map(|robot| odometry_to_ground * *robot)
            .collect();
        let detections = detect(&self.camera, &in_ground);
        self.filter.predict(now, Some(pose), &self.parameters);
        self.filter
            .update_with_detections(now, &detections, &self.camera, Some(6), &self.parameters)
            .expect("odometry is known");
    }

    fn report(&mut self, player_number: PlayerNumber, position: Point2<Field>) {
        let now = time(self.now);
        let pose = self.filter.odometry_pose().expect("odometry is known");
        // Field and odometry frames coincide in these scenes.
        let ground_to_field: Isometry2<Ground, Field> =
            Isometry2::from_parts(pose.position().inner.coords.framed(), pose.angle());
        let report = team_report(
            player_number,
            position,
            Duration::from_millis(100),
            ground_to_field,
            pose.as_transform::<Ground>(),
            &self.parameters,
        );
        self.filter
            .update_with_team_report(now, &report, 4, &self.parameters);
    }

    fn confident_tracks(&self) -> Vec<&object_tracking::multi_bernoulli::Track<4, Attributes>> {
        self.filter
            .tracks
            .tracks
            .iter()
            .filter(|track| track.existence > 0.5)
            .collect()
    }

    fn roles(&self) -> Vec<(Vector2<f32>, HumanoidRole)> {
        self.filter
            .tracked_objects(time(self.now), None, &self.parameters)
            .expect("odometry is known")
            .tracks
            .into_iter()
            .filter(|track| track.existence > 0.5)
            .map(|track| (track.state.mean.fixed_rows::<2>(0).into_owned(), track.role))
            .collect()
    }
}

#[test]
fn walking_robot_is_tracked_with_its_velocity() {
    let mut scene = Scene::new();
    let velocity = Vector2::new(0.0, -0.4);
    let start = Vector2::new(3.0, 0.6);
    for frame in 0..90 {
        let position = start + velocity * (frame as f32 * FRAME);
        scene.step(Pose2::zero(), &[point![position.x, position.y]]);
    }

    let tracks = scene.confident_tracks();
    assert_eq!(tracks.len(), 1);
    let estimate = tracks[0].estimate().mean;
    let truth = start + velocity * (90.0 * FRAME);
    assert_relative_eq!(
        estimate.fixed_rows::<2>(0).into_owned(),
        truth,
        epsilon = 0.1
    );
    assert_relative_eq!(
        estimate.fixed_rows::<2>(2).into_owned(),
        velocity,
        epsilon = 0.15
    );
}

#[test]
fn track_survives_looking_away_and_is_reacquired() {
    let mut scene = Scene::new();
    let robot = point![3.0, 0.3];
    for _ in 0..30 {
        scene.step(Pose2::zero(), &[robot]);
    }
    let id = scene.confident_tracks()[0].id;
    let existence = scene.confident_tracks()[0].existence;

    let turned = Pose2::new(point![0.0, 0.0], PI);
    for _ in 0..90 {
        scene.step(turned, &[robot]);
    }
    let blind = scene.confident_tracks();
    assert_eq!(blind.len(), 1);
    assert_eq!(blind[0].id, id);
    assert_relative_eq!(blind[0].existence, existence, epsilon = 1e-4);

    for _ in 0..10 {
        scene.step(Pose2::zero(), &[robot]);
    }
    let reacquired = scene.confident_tracks();
    assert_eq!(reacquired.len(), 1);
    assert_eq!(reacquired[0].id, id);
}

#[test]
fn robots_are_opponents_until_team_reports_identify_them() {
    let mut scene = Scene::new();
    let teammate = point![3.0, 1.0];
    let opponent = point![3.0, -1.0];
    for _ in 0..20 {
        scene.step(Pose2::zero(), &[teammate, opponent]);
    }
    assert!(
        scene
            .roles()
            .iter()
            .all(|(_, role)| *role == HumanoidRole::Opponent)
    );

    for _ in 0..3 {
        scene.report(PlayerNumber::Three, point![3.1, 0.9]);
        for _ in 0..15 {
            scene.step(Pose2::zero(), &[teammate, opponent]);
        }
    }

    let roles = scene.roles();
    assert_eq!(roles.len(), 2);
    for (position, role) in roles {
        if position.y > 0.0 {
            assert_eq!(role, HumanoidRole::Teammate(PlayerNumber::Three));
        } else {
            assert_eq!(role, HumanoidRole::Opponent);
        }
    }
}

#[test]
fn single_report_does_not_promote_a_teammate() {
    let mut scene = Scene::new();
    let teammate = point![3.0, 0.5];
    for _ in 0..20 {
        scene.step(Pose2::zero(), &[teammate]);
    }
    scene.report(PlayerNumber::Two, point![3.0, 0.5]);
    scene.step(Pose2::zero(), &[teammate]);

    assert_eq!(scene.roles()[0].1, HumanoidRole::Opponent);
}

#[test]
fn unseen_teammate_becomes_a_remote_only_track() {
    let mut scene = Scene::new();
    scene.step(Pose2::zero(), &[]);
    scene.report(PlayerNumber::Four, point![-3.0, 0.0]);
    scene.step(Pose2::zero(), &[]);
    scene.report(PlayerNumber::Four, point![-3.0, 0.1]);
    scene.step(Pose2::zero(), &[]);

    let tracks = scene
        .filter
        .tracked_objects(time(scene.now), None, &scene.parameters)
        .unwrap()
        .tracks;
    let remote: Vec<_> = tracks
        .iter()
        .filter(|track| track.existence > 0.5)
        .collect();
    assert_eq!(remote.len(), 1);
    assert!(remote[0].remote_only);
    assert_eq!(remote[0].role, HumanoidRole::Teammate(PlayerNumber::Four));
    let standard_deviation = remote[0].state.covariance[(0, 0)].sqrt();
    assert!(standard_deviation >= scene.parameters.team.position_standard_deviation_floor - 1e-3);
}

#[test]
fn occluded_robot_is_not_deleted() {
    let mut scene = Scene::new();
    let far = point![4.0, 0.0];
    for _ in 0..20 {
        scene.step(Pose2::zero(), &[point![2.0, 0.8], far]);
    }
    let far_id = scene
        .confident_tracks()
        .into_iter()
        .find(|track| track.estimate().mean.x > 3.0)
        .expect("far robot is tracked")
        .id;

    for _ in 0..60 {
        scene.step(Pose2::zero(), &[point![2.0, 0.0], far]);
    }

    let far_track = scene
        .filter
        .tracks
        .tracks
        .iter()
        .find(|track| track.id == far_id)
        .expect("occluded robot keeps its track");
    assert!(
        far_track.existence > 0.5,
        "existence {}",
        far_track.existence
    );
}

#[test]
fn humans_are_published_as_people() {
    let mut scene = Scene::new();
    scene.step(Pose2::zero(), &[]);
    let mut class = ClassBelief::uniform();
    class.probabilities = SVector::from([0.1, 0.9]);
    scene.filter.tracks.insert(
        0.9,
        Gaussian {
            mean: Vector4::new(2.0, 0.0, 0.0, 0.0),
            covariance: Matrix4::identity() * 0.01,
        },
        Attributes {
            class,
            teammates: Vec::new(),
            published_teammate: None,
            remote_only: false,
            last_detection: Some(time(scene.now)),
            last_team_report: None,
            detection_probability: 0.0,
        },
    );

    let obstacles = scene.filter.obstacles(&scene.parameters);
    assert_eq!(obstacles.len(), 1);
    assert_eq!(obstacles[0].kind, ObstacleKind::Person);
    assert_eq!(scene.roles()[0].1, HumanoidRole::Human);
    assert_eq!(
        scene.filter.tracks.tracks[0].attributes.class.most_likely(),
        HUMAN
    );
}

#[test]
fn duplicate_track_of_a_seen_robot_is_absorbed() {
    let mut scene = Scene::new();
    let robot = point![2.0, 0.3];
    for _ in 0..20 {
        scene.step(Pose2::zero(), &[robot]);
    }
    let id = scene.confident_tracks()[0].id;
    let mut duplicate = scene.confident_tracks()[0].clone();
    duplicate.components[0].gaussian.mean.x -= 0.02;
    scene
        .filter
        .tracks
        .insert(0.9, duplicate.components[0].gaussian, duplicate.attributes);

    for _ in 0..5 {
        scene.step(Pose2::zero(), &[robot]);
    }

    let tracks = scene.confident_tracks();
    assert_eq!(tracks.len(), 1);
    assert_eq!(tracks[0].id, id);
}

#[test]
fn teammate_identity_moves_to_the_track_that_matches_its_reports() {
    let mut scene = Scene::new();
    let first = point![5.0, 0.6];
    let second = point![5.0, -0.6];
    for _ in 0..20 {
        scene.step(Pose2::zero(), &[first, second]);
    }
    for _ in 0..3 {
        scene.report(PlayerNumber::Three, point![5.0, 0.6]);
        for _ in 0..30 {
            scene.step(Pose2::zero(), &[first, second]);
        }
    }
    let teammate_at = |scene: &Scene| {
        scene
            .roles()
            .into_iter()
            .find(|(_, role)| *role == HumanoidRole::Teammate(PlayerNumber::Three))
            .map(|(position, _)| position.y)
    };
    assert!(teammate_at(&scene).is_some_and(|y| y > 0.0));

    // The tracks swapped the robots they follow: the teammate now reports the other position.
    for _ in 0..6 {
        scene.report(PlayerNumber::Three, point![5.0, -0.6]);
        for _ in 0..30 {
            scene.step(Pose2::zero(), &[first, second]);
        }
    }

    assert!(teammate_at(&scene).is_some_and(|y| y < 0.0));
    assert_eq!(scene.confident_tracks().len(), 2);
}

//! Synthetic camera observations, generated from physical scene geometry.
use crate::{
    bevy_mujoco::MujocoWorld,
    profiles::Profile,
    reference::{AnnouncedReference, Reference},
    robot_io::{Data, Observation},
};
use color_eyre::Result;
use coordinate_systems::Pixel;
use geometry::rectangle::Rectangle;
use linear_algebra::{Point2, point};
use nalgebra::{Isometry3, Point3};
use ros_z::{prelude::*, time::Time};
use ros2::{sensor_msgs::camera_info::CameraInfo, std_msgs::header::Header};
use serde::{Deserialize, Serialize};
use types::{
    bounding_box::BoundingBox,
    field_dimensions::{FieldDimensions, Half, Side},
    object_detection::{Object, RobocupObjectLabel as Label},
    time_wrapper::TimeWrapper,
    visual_odometry::{VisualOdometer, VisualOdometryDelta},
};

use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};

#[derive(Clone, Debug, Default, Serialize, Deserialize, ros_z::Message)]
#[serde(default, deny_unknown_fields)]
pub struct ObservationParameters {
    pub pixel_noise_std_dev: f32,
    pub detection_dropout_probability: f32,
    pub seed: u64,
}
impl ObservationParameters {
    pub fn validate(&self) -> std::result::Result<(), String> {
        if !self.pixel_noise_std_dev.is_finite()
            || !(0.0..=1000.0).contains(&self.pixel_noise_std_dev)
        {
            return Err("pixel noise standard deviation must be between 0 and 1000 pixels".into());
        }
        if !(0.0..=1.0).contains(&self.detection_dropout_probability) {
            return Err("detection dropout probability must be between 0 and 1".into());
        }
        Ok(())
    }
}
struct ObservationNoise {
    seed: u64,
    rng: StdRng,
}
impl Default for ObservationNoise {
    fn default() -> Self {
        Self {
            seed: 0,
            rng: StdRng::seed_from_u64(0),
        }
    }
}
impl ObservationNoise {
    fn apply(
        &mut self,
        objects: &[Object<Label>],
        parameters: &ObservationParameters,
    ) -> Vec<Object<Label>> {
        if self.seed != parameters.seed {
            self.seed = parameters.seed;
            self.rng = StdRng::seed_from_u64(self.seed);
        }
        objects
            .iter()
            .copied()
            .filter_map(|mut object| {
                if self.rng.random::<f32>() < parameters.detection_dropout_probability {
                    return None;
                }
                let x: f32 = StandardNormal.sample(&mut self.rng);
                let y: f32 = StandardNormal.sample(&mut self.rng);
                let offset = linear_algebra::vector![
                    x * parameters.pixel_noise_std_dev,
                    y * parameters.pixel_noise_std_dev
                ];
                object.bounding_box.area.min += offset;
                object.bounding_box.area.max += offset;
                Some(object)
            })
            .collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, ros_z::Message)]
pub(crate) struct Ball {
    pub id: u64,
    pub position: [f64; 3],
    pub velocity: [f64; 3],
}

pub(crate) fn balls(world: &MujocoWorld) -> Vec<Ball> {
    let data = world.data();
    world
        .balls
        .iter()
        .filter_map(|entity| {
            let body = data.body(&format!("object_{}_ball", entity.to_bits()))?;
            let joint = data.joint(&format!("object_{}_ball_free_joint", entity.to_bits()))?;
            let p = body.view(data).xpos;
            let v = joint.view(data).qvel;
            Some(Ball {
                id: entity.to_bits(),
                position: [p[0], p[1], p[2]],
                velocity: [v[0], v[1], v[2]],
            })
        })
        .collect()
}

#[derive(Clone)]
pub(crate) struct Frame {
    pub time: Time,
    pub sequence: u64,
    pub epoch: u64,
    pub sample: Observation,
    pub balls: Vec<Ball>,
    pub detections: Vec<Object<Label>>,
    pub field: FieldDimensions,
}

pub(crate) fn detect(
    data: &mut Data,
    sample: &Observation,
    balls: &[Ball],
    field: &FieldDimensions,
) -> Vec<Object<Label>> {
    let mut output = Vec::new();
    let world_to_camera = sample.camera_to_world.inverse();
    let project = |position: Point3<f32>| -> Option<(Point2<Pixel>, f32)> {
        let camera = world_to_camera * position;
        if camera.z <= 0.01 {
            return None;
        }
        let pixel = sample
            .camera_matrix
            .intrinsics
            .project(linear_algebra::Vector3::wrap(camera.coords));
        (pixel.x() >= 0.0
            && pixel.y() >= 0.0
            && pixel.x() < sample.camera_matrix.image_size.x()
            && pixel.y() < sample.camera_matrix.image_size.y())
        .then_some((pixel, camera.z))
    };
    let mut visible = |position: Point3<f32>, tolerance: f32| {
        let origin = sample.camera_to_world.translation.vector.cast::<f64>();
        let offset = position.coords.cast::<f64>() - origin;
        let distance = offset.norm();
        if distance < 0.01 {
            return false;
        }
        let direction = offset / distance;
        // The collision mesh is a solid camera housing with no lens aperture.
        let (_, hit) = data.ray(
            origin.as_ref(),
            direction.as_ref(),
            None,
            true,
            Some(sample.camera_housing),
            None,
        );
        hit < 0.0 || hit + f64::from(tolerance) >= distance
    };
    let bbox = |label, min, max| Object {
        label,
        bounding_box: BoundingBox {
            area: Rectangle { min, max },
            confidence: 1.0,
        },
    };
    for ball in balls {
        let position = Point3::from(ball.position.map(|v| v as f32));
        if let Some((pixel, depth)) = project(position) {
            if !visible(position, field.ball_radius + 0.005) {
                continue;
            }
            let radius = sample.camera_matrix.intrinsics.focals.x * field.ball_radius / depth;
            // Keep the true center even when the silhouette crosses the image edge.
            output.push(bbox(
                Label::Ball,
                point![pixel.x() - radius, pixel.y() - radius],
                point![pixel.x() + radius, pixel.y() + radius],
            ));
        }
    }
    for (label, point) in landmarks(field) {
        let position = nalgebra::point![point.x(), point.y(), 0.0];
        let Some((pixel, depth)) = project(position) else {
            continue;
        };
        if !visible(
            position,
            if label == Label::GoalPost {
                field.goal_post_diameter
            } else {
                0.01
            },
        ) {
            continue;
        }
        if label == Label::GoalPost {
            let top = world_to_camera
                * nalgebra::point![point.x(), point.y(), crate::scene::goal::GOAL_HEIGHT as f32];
            if top.z <= 0.01 {
                continue;
            }
            let top_pixel = sample
                .camera_matrix
                .intrinsics
                .project(linear_algebra::Vector3::wrap(top.coords));
            let radius =
                sample.camera_matrix.intrinsics.focals.x * field.goal_post_diameter / (2.0 * depth);
            // Consumers use bottom-center as the footpoint. Do not move it by clipping.
            output.push(bbox(
                label,
                point![pixel.x() - radius, top_pixel.y().min(pixel.y() - 1.0)],
                point![pixel.x() + radius, pixel.y()],
            ));
        } else {
            let radius =
                (sample.camera_matrix.intrinsics.focals.x * field.line_width / depth).max(1.0);
            output.push(bbox(
                label,
                point![pixel.x() - radius, pixel.y() - radius],
                point![pixel.x() + radius, pixel.y() + radius],
            ));
        }
    }
    output
}

fn landmarks(field: &FieldDimensions) -> Vec<(Label, Point2<coordinate_systems::Field>)> {
    let mut points = vec![(Label::XSpot, field.center())];
    for side in [Side::Left, Side::Right] {
        points.extend([
            (Label::TSpot, field.t_crossing(side)),
            (Label::XSpot, field.x_crossing(side)),
        ]);
        for half in [Half::Own, Half::Opponent] {
            points.extend([
                (Label::GoalPost, field.goal_post(half, side)),
                (Label::LSpot, field.corner(half, side)),
                (Label::LSpot, field.goal_box_corner(half, side)),
                (Label::LSpot, field.penalty_box_corner(half, side)),
                (
                    Label::TSpot,
                    field.goal_box_goal_line_intersection(half, side),
                ),
                (
                    Label::TSpot,
                    field.penalty_box_goal_line_intersection(half, side),
                ),
            ]);
        }
    }
    for half in [Half::Own, Half::Opponent] {
        points.push((Label::PenaltySpot, field.penalty_spot(half)));
    }
    points
}

pub(crate) fn camera_info(sample: &Observation, time: Time) -> CameraInfo {
    let intrinsics = sample.camera_matrix.intrinsics;
    let (fx, fy, cx, cy) = (
        f64::from(intrinsics.focals.x),
        f64::from(intrinsics.focals.y),
        f64::from(intrinsics.optical_center.x()),
        f64::from(intrinsics.optical_center.y()),
    );
    CameraInfo {
        header: Header {
            stamp: time.to_wallclock().into(),
            frame_id: "left_camera_optical".into(),
        },
        width: sample.camera_matrix.image_size.x() as u32,
        height: sample.camera_matrix.image_size.y() as u32,
        distortion_model: "plumb_bob".into(),
        d: vec![0.0; 5],
        k: [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        p: [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        ..Default::default()
    }
}

#[derive(Default)]
struct CameraMotion {
    previous: Option<(u64, Time, Isometry3<f32>)>,
    origin: Isometry3<f32>,
}
impl CameraMotion {
    fn observe(
        &mut self,
        epoch: u64,
        time: Time,
        camera: Isometry3<f32>,
    ) -> (VisualOdometer, Option<VisualOdometryDelta>) {
        let delta = self
            .previous
            .filter(|(old_epoch, old_time, _)| *old_epoch == epoch && *old_time < time)
            .map(|(_, previous_time, previous)| VisualOdometryDelta {
                previous_time,
                current_left_camera_to_previous_left_camera: previous.inverse() * camera,
            });
        if delta.is_none() {
            self.origin = camera;
        }
        self.previous = Some((epoch, time, camera));
        (
            VisualOdometer {
                delta: delta.clone(),
                time,
                epoch,
                current_left_camera_to_visual_odometer: self.origin.inverse() * camera,
            },
            delta,
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, ros_z::Message)]
pub(crate) struct FrameTiming {
    pub captured: Time,
    pub published: Time,
    pub skipped_frames: u64,
}

pub(crate) struct CameraInputs {
    timing: Publisher<FrameTiming>,
    clock: ros_z::time::Clock,
    last_sequence: u64,
    skipped_frames: u64,
    info: Publisher<CameraInfo>,
    balls: Reference<TimeWrapper<Vec<Ball>>>,
    detections: AnnouncedReference<TimeWrapper<Vec<Object<Label>>>>,
    delta: Reference<VisualOdometryDelta>,
    odometer: Reference<VisualOdometer>,
    motion: CameraMotion,
    parameters: ros_z::parameter::NodeParameters<crate::parameters::SimulatorParameters>,
    noise: ObservationNoise,
}
impl CameraInputs {
    pub async fn new(
        node: &Node,
        profile: Profile,
        parameters: ros_z::parameter::NodeParameters<crate::parameters::SimulatorParameters>,
    ) -> Result<Self> {
        Ok(Self {
            timing: node.publisher("diagnostics/camera_frames").build().await?,
            clock: node.clock().clone(),
            last_sequence: 0,
            skipped_frames: 0,
            info: node.publisher("inputs/camera_info").build().await?,
            balls: Reference::new(node, "balls", false).await?,
            detections: AnnouncedReference::new(node, "detected_objects", profile.filtering())
                .await?,
            delta: Reference::new(
                node,
                "visual_odometry/current_left_camera_to_previous_left_camera",
                false,
            )
            .await?,
            odometer: Reference::new(
                node,
                "visual_odometry/current_left_camera_to_visual_odometer",
                profile.localization(),
            )
            .await?,
            motion: CameraMotion::default(),
            parameters,
            noise: ObservationNoise::default(),
        })
    }
    pub async fn publish(&mut self, frame: &Frame) -> Result<()> {
        let time = frame.time;
        self.info
            .publish_with_source_time(&camera_info(&frame.sample, time), time)
            .await?;
        self.balls
            .publish(
                &TimeWrapper {
                    time,
                    inner: frame.balls.clone(),
                },
                time,
            )
            .await?;
        let observed = self.noise.apply(
            &frame.detections,
            &self.parameters.snapshot().typed().observations,
        );
        self.detections
            .publish_observed(
                &TimeWrapper {
                    time,
                    inner: frame.detections.clone(),
                },
                &TimeWrapper {
                    time,
                    inner: observed,
                },
                time,
            )
            .await?;
        let (odometer, delta) =
            self.motion
                .observe(frame.epoch, time, frame.sample.camera_to_world);
        self.odometer.publish(&odometer, time).await?;
        if let Some(delta) = delta {
            self.delta.publish(&delta, time).await?;
        }
        self.skipped_frames += frame.sequence.saturating_sub(self.last_sequence + 1);
        self.last_sequence = frame.sequence;
        self.timing
            .publish(&FrameTiming {
                captured: time,
                published: self.clock.now(),
                skipped_frames: self.skipped_frames,
            })
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robot_io::{RobotBinding, tests::model};

    #[test]
    fn camera_info_and_ball_detection_use_the_physical_optical_frame() {
        let mut data = model();
        let sample = RobotBinding::new(&data, "").unwrap().observe(&data);
        let info = camera_info(&sample, Time::from_nanos(42));
        assert_eq!(
            projection::intrinsic::Intrinsic::from(&info),
            sample.camera_matrix.intrinsics
        );
        let center = sample.camera_to_world * nalgebra::point![0.0, 0.0, 2.0];
        let behind = sample.camera_to_world * nalgebra::point![0.0, 0.0, -2.0];
        let balls = [
            Ball {
                id: 1,
                position: center.coords.cast::<f64>().into(),
                velocity: [0.0; 3],
            },
            Ball {
                id: 2,
                position: behind.coords.cast::<f64>().into(),
                velocity: [0.0; 3],
            },
        ];
        let objects = detect(&mut data, &sample, &balls, &FieldDimensions::SPL_2025);
        let balls: Vec<_> = objects.iter().filter(|o| o.label == Label::Ball).collect();
        assert_eq!(balls.len(), 1);
        let area = balls[0].bounding_box.area;
        assert!(((area.min.x() + area.max.x()) / 2.0 - 320.0).abs() < 0.001);
        assert!(((area.min.y() + area.max.y()) / 2.0 - 272.0).abs() < 0.001);
    }

    #[test]
    fn observation_noise_is_repeatable_and_never_changes_truth() {
        let truth = vec![Object {
            label: Label::Robot,
            bounding_box: BoundingBox {
                area: Rectangle {
                    min: point![100.0, 100.0],
                    max: point![120.0, 150.0],
                },
                confidence: 1.0,
            },
        }];
        let mut a = ObservationNoise::default();
        let mut b = ObservationNoise::default();
        let config = ObservationParameters {
            pixel_noise_std_dev: 2.0,
            seed: 42,
            ..Default::default()
        };
        for _ in 0..20 {
            let aa = a.apply(&truth, &config);
            let bb = b.apply(&truth, &config);
            assert_eq!(aa[0].bounding_box.area, bb[0].bounding_box.area);
            assert!(
                ((aa[0].bounding_box.area.max - aa[0].bounding_box.area.min)
                    - linear_algebra::vector![20.0, 50.0])
                .norm()
                    < 1e-4
            );
        }
        assert_eq!(truth[0].bounding_box.area.min, point![100.0, 100.0]);
        let lost = ObservationParameters {
            detection_dropout_probability: 1.0,
            ..Default::default()
        };
        assert!(a.apply(&truth, &lost).is_empty());
        assert_eq!(
            a.apply(&truth, &ObservationParameters::default())[0]
                .bounding_box
                .area,
            truth[0].bounding_box.area
        );
        assert!(
            ObservationParameters {
                pixel_noise_std_dev: f32::NAN,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            ObservationParameters {
                detection_dropout_probability: -0.1,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn scene_geometry_occludes_balls_and_posts_keep_ground_anchors() {
        use mujoco_rs::prelude::{MjData, MjSpec, MjtGeom};
        let mut data = model();
        let binding = RobotBinding::new(&data, "").unwrap();
        let sample = binding.observe(&data);
        let field = FieldDimensions::SPL_2025;
        let objects = detect(&mut data, &sample, &[], &field);
        for post in objects.iter().filter(|o| o.label == Label::GoalPost) {
            let anchor = point![
                (post.bounding_box.area.min.x() + post.bounding_box.area.max.x()) / 2.0,
                post.bounding_box.area.max.y()
            ];
            assert!([Side::Left, Side::Right].iter().any(|&side| {
                let position = field.goal_post(Half::Opponent, side);
                let camera = sample.camera_to_world.inverse()
                    * nalgebra::point![position.x(), position.y(), 0.0];
                (anchor
                    - sample
                        .camera_matrix
                        .intrinsics
                        .project(linear_algebra::Vector3::wrap(camera.coords)))
                .norm()
                    < 0.001
            }));
        }
        assert_eq!(
            objects
                .iter()
                .filter(|o| o.label == Label::GoalPost)
                .count(),
            2
        );
        let occluder = sample.camera_to_world * nalgebra::point![0.0, 0.0, 1.0];
        let target = sample.camera_to_world * nalgebra::point![0.0, 0.0, 2.0];
        let mut spec =
            MjSpec::from_xml(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/k1_robot.xml")).unwrap();
        spec.world_body_mut()
            .add_geom()
            .with_type(MjtGeom::mjGEOM_BOX)
            .with_size([0.1; 3])
            .with_pos(occluder.coords.cast::<f64>().into());
        let mut data = MjData::new(Box::new(spec.compile().unwrap()));
        data.forward();
        let sample = RobotBinding::new(&data, "").unwrap().observe(&data);
        let ball = Ball {
            id: 1,
            position: target.coords.cast::<f64>().into(),
            velocity: [0.0; 3],
        };
        assert!(
            !detect(&mut data, &sample, &[ball], &field)
                .iter()
                .any(|o| o.label == Label::Ball)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn real_obstacle_filter_consumes_robot_boxes_and_expires_them() {
        use projection::Projection;
        use ros_z_streams::CreateAnnouncingPublisher;
        use std::{sync::Arc, time::Duration};
        use types::obstacles::{Obstacle, ObstacleKind};
        let context = Arc::new(
            ContextBuilder::default()
                .with_mode("peer")
                .disable_multicast_scouting()
                .with_connect_endpoints(std::iter::empty::<&str>())
                .with_listen_endpoints(std::iter::empty::<&str>())
                .with_parameter_layers([std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../etc/parameters/base")])
                .build()
                .await
                .unwrap(),
        );
        let node = context
            .create_node("obstacle_fixture")
            .build()
            .await
            .unwrap();
        let field = node
            .publisher::<FieldDimensions>("field_dimensions")
            .build()
            .await
            .unwrap();
        let camera = node
            .publisher::<TimeWrapper<projection::camera_matrix::CameraMatrix>>("camera_matrix")
            .build()
            .await
            .unwrap();
        let detections = node
            .announcing_publisher::<TimeWrapper<Vec<Object<Label>>>>("detected_objects")
            .await
            .unwrap();
        let odometry = node
            .announcing_publisher::<linear_algebra::Pose2<coordinate_systems::Odometry>>(
                "inputs/odometry",
            )
            .await
            .unwrap();
        let obstacles = node
            .subscriber::<Vec<Obstacle>>("obstacles")
            .cache(1)
            .build()
            .await
            .unwrap();
        let task = tokio::spawn(obstacle_filter::run_boxed(context.clone()));
        let sample = {
            let data = model();
            RobotBinding::new(&data, "").unwrap().observe(&data)
        };
        let position = point![2.0, 0.3];
        let pixel = sample.camera_matrix.ground_to_pixel(position).unwrap();
        let object = Object {
            label: Label::Robot,
            bounding_box: BoundingBox {
                area: Rectangle {
                    min: point![pixel.x() - 20.0, pixel.y() - 80.0],
                    max: point![pixel.x() + 20.0, pixel.y()],
                },
                confidence: 1.0,
            },
        };
        for _ in 0..80 {
            let time = node.clock().now();
            field.publish(&FieldDimensions::SPL_2025).await.unwrap();
            camera
                .publish(&TimeWrapper {
                    time,
                    inner: sample.camera_matrix.clone(),
                })
                .await
                .unwrap();
            odometry
                .announce(time)
                .await
                .unwrap()
                .publish(&linear_algebra::Pose2::new(point![0.0, 0.0], 0.0))
                .await
                .unwrap();
            detections
                .announce(time)
                .await
                .unwrap()
                .publish(&TimeWrapper {
                    time,
                    inner: vec![object],
                })
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(15)).await;
        }
        let output = obstacles.get_latest().expect("no obstacle frames");
        let robot = output
            .iter()
            .find(|o| o.kind == ObstacleKind::Robot)
            .expect("robot fixture not acquired");
        assert!((robot.position - position).norm() < 0.001);
        for _ in 0..45 {
            let time = node.clock().now();
            odometry
                .announce(time)
                .await
                .unwrap()
                .publish(&linear_algebra::Pose2::new(point![0.0, 0.0], 0.0))
                .await
                .unwrap();
            detections
                .announce(time)
                .await
                .unwrap()
                .publish(&TimeWrapper {
                    time,
                    inner: vec![],
                })
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(15)).await;
        }
        assert!(
            !obstacles
                .get_latest()
                .unwrap()
                .iter()
                .any(|o| o.kind == ObstacleKind::Robot)
        );
        task.abort();
        let _ = task.await;
        context.shutdown().unwrap();
    }

    #[test]
    fn camera_motion_composes_and_does_not_cross_reset_epochs() {
        let mut motion = CameraMotion::default();
        let first = Isometry3::new(
            nalgebra::vector![2.0, 3.0, 1.0],
            nalgebra::vector![0.1, 0.2, 0.3],
        );
        let second = first
            * Isometry3::new(
                nalgebra::vector![0.2, 0.0, 0.01],
                nalgebra::vector![0.0, 0.1, 0.2],
            );
        let third = second
            * Isometry3::new(
                nalgebra::vector![0.3, 0.0, 0.0],
                nalgebra::vector![0.0, -0.1, 0.0],
            );
        let (initial, delta) = motion.observe(0, Time::from_nanos(1), first);
        assert!(delta.is_none());
        assert!(initial.delta.is_none());
        assert!(
            initial
                .current_left_camera_to_visual_odometer
                .translation
                .vector
                .norm()
                < 1e-6
        );
        let (_, a) = motion.observe(0, Time::from_nanos(2), second);
        let (total, b) = motion.observe(0, Time::from_nanos(3), third);
        let embedded = total
            .delta
            .as_ref()
            .expect("production odometer carries its delta");
        assert_eq!(embedded.previous_time, Time::from_nanos(2));
        assert_eq!(
            embedded.current_left_camera_to_previous_left_camera,
            b.as_ref()
                .unwrap()
                .current_left_camera_to_previous_left_camera,
        );
        let composed = a.unwrap().current_left_camera_to_previous_left_camera
            * b.unwrap().current_left_camera_to_previous_left_camera;
        assert!(
            (composed.translation.vector
                - total
                    .current_left_camera_to_visual_odometer
                    .translation
                    .vector)
                .norm()
                < 1e-5
        );
        assert!(
            composed
                .rotation
                .angle_to(&total.current_left_camera_to_visual_odometer.rotation)
                < 1e-5
        );
        let (_, reset_delta) = motion.observe(1, Time::from_nanos(4), first);
        assert!(reset_delta.is_none());
    }
}

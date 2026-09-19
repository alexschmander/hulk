use std::{
    f32::consts::{FRAC_PI_2, PI, TAU},
    sync::Arc,
};

use color_eyre::Report;
use coordinate_systems::{Camera, Field, Pixel, Robot};
use eframe::egui::{Color32, Stroke};
use linear_algebra::{Isometry3, Point2, Point3, point};
use projection::{camera_matrix::CameraMatrix, intrinsic::Intrinsic};
use ros_z::{
    qos::{QosDurability, QosProfile},
    time::Time,
};
use ros_z_debug::ObservationPolicy;
use types::{
    field_dimensions::{FieldDimensions, Half, Side},
    time_wrapper::TimeWrapper,
    visual_localization::{
        ASSOCIATION_GEOMETRY_TOPIC, AssociationGeometry, LOCALIZATION_POSE_3D_TOPIC,
        VISUAL_LOCALIZATION_TOPIC, VisualLocalizationFrame,
    },
};

use super::super::image_overlay::{
    ImageOverlay, ImageOverlayPainter, OverlayObservation, interpolate_transform, valid_intrinsics,
};
use crate::repaint::ObservationContext;

const NEAR_Z: f32 = 1.0e-4;
const FIELD_STROKE: Stroke = Stroke {
    width: 2.0,
    color: Color32::from_rgb(80, 220, 255),
};
const RESIDUAL_STROKE: Stroke = Stroke {
    width: 2.0,
    color: Color32::from_rgb(255, 80, 200),
};

pub(in crate::panels::image) struct ProjectedFieldLinesOverlay {
    camera_matrix: OverlayObservation<TimeWrapper<CameraMatrix>>,
    localization: OverlayObservation<TimeWrapper<Option<Isometry3<Field, Robot>>>>,
    dimensions: OverlayObservation<FieldDimensions>,
    intrinsics: OverlayObservation<Intrinsic>,
    associations: OverlayObservation<TimeWrapper<VisualLocalizationFrame>>,
    association_geometry: OverlayObservation<TimeWrapper<AssociationGeometry>>,
}

impl ImageOverlay for ProjectedFieldLinesOverlay {
    type Sample = ProjectedFieldSample;
    const NAME: &'static str = "Projected Field Lines";
    const STORAGE_KEY: &'static str = "projected_field_lines";

    fn new<C: ObservationContext>(context: &C) -> Result<Self, Report> {
        Ok(Self {
            camera_matrix: OverlayObservation::new(context, "camera_matrix")?,
            localization: OverlayObservation::new(context, LOCALIZATION_POSE_3D_TOPIC)?,
            dimensions: OverlayObservation::with_policy(
                context,
                "field_dimensions",
                ObservationPolicy::default().with_subscriber_qos(QosProfile {
                    durability: QosDurability::TransientLocal,
                    ..Default::default()
                }),
            )?,
            associations: OverlayObservation::new(context, VISUAL_LOCALIZATION_TOPIC)?,
            intrinsics: OverlayObservation::new(context, "debug/calibrated_intrinsics")?,
            association_geometry: OverlayObservation::new(context, ASSOCIATION_GEOMETRY_TOPIC)?,
        })
    }

    fn prepare(&self, image_time: Time) -> Option<Self::Sample> {
        let epoch = self.association_geometry.latest()?.value.inner.epoch;
        let geometries = self.association_geometry.payload_history();
        let poses = self.localization.payload_history();
        let (before, after, fraction) = super::super::image_overlay::bracket(&poses, image_time)?;
        for pose in [before, after] {
            // Geometry is epoch state, published after pose_3d; do not wait for its exact stamp.
            let geometry = geometries
                .iter()
                .rev()
                .find(|g| g.value.time <= pose.time)?;
            if geometry.value.inner.epoch != epoch || geometry.value.inner.local_to_field.is_none()
            {
                return None;
            }
        }
        let field_to_robot = interpolate_transform(before.inner?, after.inner?, fraction)?;
        let matrix = self.camera_matrix.camera_at(image_time)?;
        let projection = FieldProjection {
            field_to_camera: matrix.robot_to_camera * field_to_robot,
            // Untimed current calibration/config is captured at commit, not followed while painting.
            intrinsics: self
                .intrinsics
                .latest()
                .filter(|s| valid_intrinsics(&s.value))
                .map_or(matrix.intrinsics, |s| s.value),
        };
        // Latched configuration, captured once per snapshot rather than read during painting.
        let dimensions = self
            .dimensions
            .latest()
            .map_or(FieldDimensions::SPL_2025, |record| record.value);
        let sample = ProjectedFieldSample {
            projection,
            dimensions,
            epoch,
            pose_time: before.time,
            associations: self
                .associations
                .at_time(image_time)
                .filter(|s| s.value.inner.epoch == epoch),
        };
        projection_valid(&sample, epoch, &poses, &geometries).then_some(sample)
    }

    fn paint(painter: &ImageOverlayPainter, sample: &Self::Sample) {
        let projection = &sample.projection;
        projection.draw_field(painter, &sample.dimensions);
        if let Some(associations) = &sample.associations {
            for association in &associations.value.inner.associations {
                if !association.detection.inner.iter().all(|v| v.is_finite()) {
                    continue;
                }
                if let Some(projected) = projection.project(association.field_point) {
                    painter.line_segment(association.detection, projected, RESIDUAL_STROKE);
                    painter.circle_filled(projected, 3.5, RESIDUAL_STROKE.color);
                }
            }
        }
    }
}

impl ProjectedFieldLinesOverlay {
    #[cfg(test)]
    pub(in crate::panels::image) fn for_test(
        context: &crate::panel::PanelCreationContext<'_>,
        node: Arc<ros_z::node::Node>,
    ) -> Self {
        use super::super::image_overlay::tests::observation;
        Self {
            camera_matrix: observation(context, Arc::clone(&node), "camera_matrix"),
            localization: observation(context, Arc::clone(&node), LOCALIZATION_POSE_3D_TOPIC),
            dimensions: observation(context, Arc::clone(&node), "field_dimensions"),
            intrinsics: observation(context, Arc::clone(&node), "debug/calibrated_intrinsics"),
            associations: observation(context, Arc::clone(&node), VISUAL_LOCALIZATION_TOPIC),
            association_geometry: observation(context, node, ASSOCIATION_GEOMETRY_TOPIC),
        }
    }

    pub(in crate::panels::image) fn unavailable(&self, time: Time) -> bool {
        // Explicit invalid state is unavailable, not an outstanding geometry result.
        self.localization
            .payload_history()
            .iter()
            .rev()
            .find(|s| s.value.time <= time)
            .is_some_and(|s| s.value.inner.is_none())
            || self
                .association_geometry
                .payload_history()
                .iter()
                .rev()
                .find(|s| s.value.time <= time)
                .is_some_and(|s| s.value.inner.local_to_field.is_none())
    }

    pub(in crate::panels::image) fn enrich_residual(
        &self,
        sample: &mut ProjectedFieldSample,
        time: Time,
    ) {
        if sample.associations.is_none() {
            sample.associations = self
                .associations
                .at_time(time)
                .filter(|s| s.value.inner.epoch == sample.epoch);
        }
    }

    pub(in crate::panels::image) fn valid(&self, sample: &ProjectedFieldSample) -> bool {
        let Some(geometry) = self.association_geometry.latest() else {
            return false;
        };
        projection_valid(
            sample,
            geometry.value.inner.epoch,
            &self.localization.payload_history(),
            &self.association_geometry.payload_history(),
        )
    }
}

type LocalizationSample =
    Arc<ros_z_debug::SampleRecord<TimeWrapper<Option<Isometry3<Field, Robot>>>>>;

fn projection_valid(
    sample: &ProjectedFieldSample,
    epoch: u64,
    poses: &[LocalizationSample],
    geometries: &[Arc<ros_z_debug::SampleRecord<TimeWrapper<AssociationGeometry>>>],
) -> bool {
    epoch == sample.epoch
        && !poses
            .iter()
            .any(|s| s.value.time >= sample.pose_time && s.value.inner.is_none())
        && !geometries.iter().any(|s| {
            s.value.inner.epoch == epoch
                && s.value.time >= sample.pose_time
                && s.value.inner.local_to_field.is_none()
        })
}

pub(in crate::panels::image) struct ProjectedFieldSample {
    projection: FieldProjection,
    dimensions: FieldDimensions,
    epoch: u64,
    pose_time: Time,
    associations: Option<Arc<ros_z_debug::SampleRecord<TimeWrapper<VisualLocalizationFrame>>>>,
}

struct FieldProjection {
    field_to_camera: Isometry3<Field, Camera>,
    intrinsics: Intrinsic,
}

impl FieldProjection {
    fn project(&self, point: Point3<Field>) -> Option<Point2<Pixel>> {
        self.project_camera(self.field_to_camera * point)
    }

    fn project_camera(&self, point: Point3<Camera>) -> Option<Point2<Pixel>> {
        if !point.inner.iter().all(|v| v.is_finite()) || point.z() < NEAR_Z {
            return None;
        }
        let pixel = self.intrinsics.project(point.coords());
        pixel.inner.iter().all(|v| v.is_finite()).then_some(pixel)
    }

    fn segment(&self, start: Point2<Field>, end: Point2<Field>) -> Option<[Point2<Pixel>; 2]> {
        let mut start = self.field_to_camera * start.extend(0.0);
        let mut end = self.field_to_camera * end.extend(0.0);
        if !start
            .inner
            .iter()
            .chain(end.inner.iter())
            .all(|v| v.is_finite())
            || (start.z() < NEAR_Z && end.z() < NEAR_Z)
        {
            return None;
        }
        // A pinhole maps straight lines to straight lines; clip instead of sampling.
        if start.z() < NEAR_Z || end.z() < NEAR_Z {
            let t = (NEAR_Z - start.z()) / (end.z() - start.z());
            let mut clipped = start + (end - start) * t;
            clipped.inner.z = NEAR_Z;
            if start.z() < NEAR_Z {
                start = clipped
            } else {
                end = clipped
            }
        }
        Some([self.project_camera(start)?, self.project_camera(end)?])
    }

    fn draw_field(&self, painter: &ImageOverlayPainter, d: &FieldDimensions) {
        let line = |a, b| {
            if let Some([a, b]) = self.segment(a, b) {
                painter.line_segment(a, b, FIELD_STROKE);
            }
        };
        line(d.t_crossing(Side::Left), d.t_crossing(Side::Right));
        for side in [Side::Left, Side::Right] {
            line(d.corner(Half::Own, side), d.corner(Half::Opponent, side));
        }
        self.arc(
            painter,
            d.center(),
            d.center_circle_diameter / 2.0,
            0.0,
            TAU,
        );
        for half in [Half::Own, Half::Opponent] {
            line(d.corner(half, Side::Left), d.corner(half, Side::Right));
            line(
                d.goal_box_corner(half, Side::Left),
                d.goal_box_corner(half, Side::Right),
            );
            line(
                d.penalty_box_corner(half, Side::Left),
                d.penalty_box_corner(half, Side::Right),
            );
            let spot = d.penalty_spot(half);
            let r = d.penalty_marker_size / 2.0;
            line(point![spot.x() - r, 0.0], point![spot.x() + r, 0.0]);
            line(point![spot.x(), -r], point![spot.x(), r]);
            for side in [Side::Left, Side::Right] {
                line(
                    d.goal_box_corner(half, side),
                    d.goal_box_goal_line_intersection(half, side),
                );
                line(
                    d.penalty_box_corner(half, side),
                    d.penalty_box_goal_line_intersection(half, side),
                );
                let start = match (half, side) {
                    (Half::Own, Side::Left) => -FRAC_PI_2,
                    (Half::Own, Side::Right) => 0.0,
                    (Half::Opponent, Side::Left) => PI,
                    (Half::Opponent, Side::Right) => FRAC_PI_2,
                };
                self.arc(
                    painter,
                    d.corner(half, side),
                    d.corner_arc_radius,
                    start,
                    start + FRAC_PI_2,
                );
            }
        }
    }

    fn arc(
        &self,
        painter: &ImageOverlayPainter,
        center: Point2<Field>,
        radius: f32,
        start: f32,
        end: f32,
    ) {
        if !radius.is_finite() || radius <= 0.0 {
            return;
        }
        let at = |angle: f32| {
            point![
                center.x() + radius * angle.cos(),
                center.y() + radius * angle.sin(),
                0.0
            ]
        };
        self.arc_segments(&at, start, end, &mut |a, b| {
            painter.line_segment(a, b, FIELD_STROKE);
        });
    }

    fn arc_segments(
        &self,
        at: &impl Fn(f32) -> Point3<Field>,
        start: f32,
        end: f32,
        draw: &mut impl FnMut(Point2<Pixel>, Point2<Pixel>),
    ) {
        let count = ((end - start).abs() / TAU * 256.0).ceil().clamp(1.0, 256.0) as usize;
        let mut previous = self.project(at(start));
        for index in 1..=count {
            let next = self.project(at(start + (end - start) * index as f32 / count as f32));
            if let (Some(a), Some(b)) = (previous, next) {
                draw(a, b);
            }
            previous = next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn geometry_interpolation_is_bounded_reset_aware_and_independent_of_associations() {
        use crate::panels::image::image_overlay::tests::{observation, publish_until};
        use crate::{backend::RobotBackend, panel::PanelCreationContext};
        let backend = Arc::new(
            RobotBackend::new(
                tokio::runtime::Handle::current(),
                None,
                "/image_geometry_test".into(),
            )
            .await
            .unwrap(),
        );
        let context = PanelCreationContext {
            backend,
            value: None,
            egui_context: eframe::egui::Context::default(),
            render_state: None,
        };
        let ros = ros_z::context::ContextBuilder::default()
            .build()
            .await
            .unwrap();
        let node = Arc::new(
            ros.create_node("image_geometry_publisher")
                .with_namespace("/image_geometry_test")
                .build()
                .await
                .unwrap(),
        );
        let overlay = ProjectedFieldLinesOverlay::for_test(&context, Arc::clone(&node));
        let camera_pub = node
            .publisher::<TimeWrapper<CameraMatrix>>("camera_matrix")
            .build()
            .await
            .unwrap();
        let pose_pub = node
            .publisher::<TimeWrapper<Option<Isometry3<Field, Robot>>>>(LOCALIZATION_POSE_3D_TOPIC)
            .build()
            .await
            .unwrap();
        let geometry_pub = node
            .publisher::<TimeWrapper<AssociationGeometry>>(ASSOCIATION_GEOMETRY_TOPIC)
            .build()
            .await
            .unwrap();
        let associations_pub = node
            .publisher::<TimeWrapper<VisualLocalizationFrame>>(VISUAL_LOCALIZATION_TOPIC)
            .build()
            .await
            .unwrap();
        let time = |millis: i64| Time::from_nanos(millis * 1_000_000);
        for (millis, x) in [(1000, 0.0), (1100, 2.0), (1301, 4.0)] {
            let matrix = CameraMatrix {
                robot_to_head: Isometry3::from_translation(x, 0.0, 0.0),
                ..Default::default()
            };
            publish_until(
                &camera_pub,
                &TimeWrapper {
                    time: time(millis),
                    inner: matrix,
                },
                || overlay.camera_matrix.at_time(time(millis)).is_some(),
            )
            .await;
            publish_until(
                &pose_pub,
                &TimeWrapper {
                    time: time(millis),
                    inner: Some(Isometry3::from_translation(0.0, x, 2.0)),
                },
                || overlay.localization.at_time(time(millis)).is_some(),
            )
            .await;
            publish_until(
                &geometry_pub,
                &TimeWrapper {
                    time: time(millis),
                    inner: AssociationGeometry {
                        epoch: 0,
                        state: types::localization::LocalizationState3D::Startup,
                        robot_to_local: Isometry3::identity(),
                        local_to_field: Some(linear_algebra::Isometry2::identity()),
                    },
                },
                || overlay.association_geometry.at_time(time(millis)).is_some(),
            )
            .await;
        }
        let sample = overlay
            .prepare(time(1050))
            .expect("bracketed pose and camera need no successful associations");
        assert!(sample.associations.is_none());
        assert_eq!(
            sample.projection.field_to_camera.translation(),
            point![1.0, 1.0, 2.0]
        );
        assert!(overlay.prepare(time(999)).is_none());
        assert!(overlay.prepare(time(1302)).is_none());
        assert!(
            overlay.prepare(time(1200)).is_some(),
            "same-epoch brackets have no arbitrary time cutoff"
        );
        assert!(
            overlay.prepare(time(1301)).is_some(),
            "exact samples need no interpolation"
        );
        // pose_3d arrives before association_geometry for this same publication cycle.
        publish_until(
            &pose_pub,
            &TimeWrapper {
                time: time(1302),
                inner: Some(Isometry3::identity()),
            },
            || overlay.localization.at_time(time(1302)).is_some(),
        )
        .await;
        publish_until(
            &camera_pub,
            &TimeWrapper {
                time: time(1302),
                inner: CameraMatrix::default(),
            },
            || overlay.camera_matrix.at_time(time(1302)).is_some(),
        )
        .await;
        assert!(
            overlay.prepare(time(1302)).is_some(),
            "same-epoch geometry state must not wait for the next exact geometry stamp"
        );
        let calibration_pub = node
            .publisher::<Intrinsic>("debug/calibrated_intrinsics")
            .build()
            .await
            .unwrap();
        let calibration = Intrinsic::new(nalgebra::vector![100.0, 200.0], point![320.0, 240.0]);
        publish_until(&calibration_pub, &calibration, || {
            overlay
                .intrinsics
                .latest()
                .is_some_and(|s| s.value == calibration)
        })
        .await;
        let calibrated = overlay.prepare(time(1050)).unwrap();
        assert_eq!(calibrated.projection.intrinsics, calibration);
        assert_eq!(
            sample.projection.intrinsics,
            Intrinsic::default(),
            "held snapshot must not follow live calibration"
        );
        for focal in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let invalid = Intrinsic::new(nalgebra::vector![focal, 1.0], point![0.0, 0.0]);
            publish_until(&calibration_pub, &invalid, || {
                overlay
                    .intrinsics
                    .latest()
                    .is_some_and(|s| s.value.focals.x.to_bits() == focal.to_bits())
            })
            .await;
            assert_eq!(
                overlay.prepare(time(1050)).unwrap().projection.intrinsics,
                Intrinsic::default()
            );
        }
        assert_eq!(calibrated.projection.intrinsics, calibration);
        let mut exact_held = overlay.prepare(time(1000)).unwrap();
        assert!(exact_held.associations.is_none());
        let held_transform = exact_held.projection.field_to_camera;
        let held_intrinsics = exact_held.projection.intrinsics;
        let associations = TimeWrapper {
            time: time(1000),
            inner: VisualLocalizationFrame {
                epoch: 0,
                robot_to_camera: Isometry3::identity(),
                robot_to_local: Isometry3::identity(),
                camera_intrinsic: Intrinsic::default(),
                associations: vec![],
            },
        };
        publish_until(&associations_pub, &associations, || {
            overlay.associations.at_time(time(1000)).is_some()
        })
        .await;
        overlay.enrich_residual(&mut exact_held, time(1050));
        assert!(
            exact_held.associations.is_none(),
            "late residual must still match the exact image"
        );
        overlay.enrich_residual(&mut exact_held, time(1000));
        let pinned_residual = Arc::clone(exact_held.associations.as_ref().unwrap());
        overlay.enrich_residual(&mut exact_held, time(1000));
        assert!(Arc::ptr_eq(
            exact_held.associations.as_ref().unwrap(),
            &pinned_residual
        ));
        assert_eq!(exact_held.projection.field_to_camera, held_transform);
        assert_eq!(exact_held.projection.intrinsics, held_intrinsics);
        assert!(
            overlay.prepare(time(1050)).unwrap().associations.is_none(),
            "residuals are exact only"
        );
        assert!(overlay.prepare(time(1000)).unwrap().associations.is_some());
        assert!(
            overlay.prepare(time(1301)).is_some(),
            "sparse associations cannot pace readiness"
        );
        publish_until(
            &pose_pub,
            &TimeWrapper {
                time: time(1350),
                inner: None,
            },
            || overlay.localization.at_time(time(1350)).is_some(),
        )
        .await;
        assert!(
            !overlay.valid(&sample),
            "explicit None invalidates held projection immediately"
        );
        assert!(overlay.prepare(time(1050)).is_none());
        // An epoch reset also invalidates already pinned inputs without a grace period.
        publish_until(
            &geometry_pub,
            &TimeWrapper {
                time: time(1400),
                inner: AssociationGeometry {
                    epoch: 1,
                    state: types::localization::LocalizationState3D::Startup,
                    robot_to_local: Isometry3::identity(),
                    local_to_field: None,
                },
            },
            || overlay.association_geometry.at_time(time(1400)).is_some(),
        )
        .await;
        assert!(!overlay.valid(&sample));
        assert!(overlay.prepare(time(1301)).is_none());
        publish_until(
            &pose_pub,
            &TimeWrapper {
                time: time(1500),
                inner: Some(Isometry3::identity()),
            },
            || overlay.localization.at_time(time(1500)).is_some(),
        )
        .await;
        publish_until(
            &camera_pub,
            &TimeWrapper {
                time: time(1500),
                inner: CameraMatrix::default(),
            },
            || overlay.camera_matrix.at_time(time(1500)).is_some(),
        )
        .await;
        publish_until(
            &geometry_pub,
            &TimeWrapper {
                time: time(1500),
                inner: AssociationGeometry {
                    epoch: 1,
                    state: types::localization::LocalizationState3D::Startup,
                    robot_to_local: Isometry3::identity(),
                    local_to_field: Some(linear_algebra::Isometry2::identity()),
                },
            },
            || overlay.association_geometry.at_time(time(1500)).is_some(),
        )
        .await;
        assert!(
            overlay.prepare(time(1500)).is_some(),
            "new epoch recovers with new aligned inputs"
        );
        assert!(!overlay.valid(&sample));
        let recovered = overlay.prepare(time(1500)).unwrap();
        publish_until(
            &geometry_pub,
            &TimeWrapper {
                time: time(1550),
                inner: AssociationGeometry {
                    epoch: 2,
                    state: types::localization::LocalizationState3D::Startup,
                    robot_to_local: Isometry3::identity(),
                    local_to_field: Some(linear_algebra::Isometry2::identity()),
                },
            },
            || overlay.association_geometry.at_time(time(1550)).is_some(),
        )
        .await;
        assert!(
            !overlay.valid(&recovered),
            "epoch alone invalidates projection, even without a None pose"
        );
        let epoch_two = ProjectedFieldSample {
            epoch: 2,
            ..recovered
        };
        publish_until(
            &geometry_pub,
            &TimeWrapper {
                time: time(1500),
                inner: AssociationGeometry {
                    epoch: 0,
                    state: types::localization::LocalizationState3D::Startup,
                    robot_to_local: Isometry3::identity(),
                    local_to_field: Some(linear_algebra::Isometry2::identity()),
                },
            },
            || {
                overlay
                    .association_geometry
                    .latest()
                    .is_some_and(|s| s.value.inner.epoch == 0)
            },
        )
        .await;
        assert!(
            !overlay.valid(&epoch_two),
            "restart to epoch zero invalidates held geometry"
        );
        assert_eq!(
            overlay.prepare(time(1500)).unwrap().epoch,
            0,
            "latest arrival wins even with lower payload time and epoch"
        );

        use crate::panels::image::{RenderedImageCache, image_overlay::tests::projection_overlays};
        use ros2::sensor_msgs::image::Image;
        let images = observation::<Image>(&context, Arc::clone(&node), "inputs/left_image");
        let image_pub = node
            .publisher::<Image>("inputs/left_image")
            .build()
            .await
            .unwrap();
        let mut image = Image {
            width: 1,
            height: 1,
            encoding: "rgb8".into(),
            step: 3,
            data: vec![0, 0, 0].into(),
            ..Default::default()
        };
        image.header.stamp = time(1500).to_wallclock().into();
        publish_until(&image_pub, &image, || images.latest().is_some()).await;
        let overlays = projection_overlays(overlay);
        let mut cache = RenderedImageCache::new("same-frame-projection-recovery");
        cache.refresh_candidates(&context.egui_context, images.get_all(), &overlays, true);
        assert!(overlays.ready(&cache.overlays));
        let pinned = Arc::clone(cache.sample.as_ref().unwrap());
        publish_until(
            &pose_pub,
            &TimeWrapper {
                time: time(1500),
                inner: None,
            },
            || overlays.prepare(time(1500)).field_unavailable,
        )
        .await;
        cache.refresh_candidates(&context.egui_context, vec![], &overlays, true);
        assert!(cache.projection_invalidated);
        publish_until(
            &pose_pub,
            &TimeWrapper {
                time: time(1500),
                inner: Some(Isometry3::identity()),
            },
            || !overlays.prepare(time(1500)).field_unavailable,
        )
        .await;
        cache.refresh_candidates(&context.egui_context, vec![], &overlays, true);
        assert!(
            overlays.ready(&cache.overlays),
            "same-frame localization recovery works after image history eviction"
        );
        assert!(!cache.projection_invalidated);
        assert!(Arc::ptr_eq(cache.sample.as_ref().unwrap(), &pinned));
    }

    #[test]
    fn projection_uses_extrinsics_and_calibrated_intrinsics_and_rejects_invalid_points() {
        let projection = FieldProjection {
            field_to_camera: Isometry3::from_translation(0.0, 0.0, 2.0),
            intrinsics: Intrinsic::new(nalgebra::vector![100.0, 200.0], point![320.0, 240.0]),
        };
        assert_eq!(
            projection.project(point![1.0, 1.0, 0.0]),
            Some(point![370.0, 340.0])
        );
        for point in [
            point![0.0, 0.0, -3.0],
            point![0.0, 0.0, -2.0],
            point![f32::NAN, 0.0, 0.0],
            point![f32::INFINITY, 0.0, 0.0],
        ] {
            assert!(projection.project(point).is_none());
        }
        let invalid = FieldProjection {
            intrinsics: Intrinsic::new(nalgebra::vector![f32::NAN, 1.0], point![0.0, 0.0]),
            ..projection
        };
        assert!(invalid.project(point![0.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn segments_clip_at_camera_plane_without_connecting_behind_camera() {
        let projection = FieldProjection {
            field_to_camera: Isometry3::from_rotation(linear_algebra::vector![
                0.0, -FRAC_PI_2, 0.0
            ]),
            intrinsics: Intrinsic::default(),
        };
        assert!(
            projection
                .segment(point![-2.0, 1.0], point![-1.0, 1.0])
                .is_none()
        );
        assert!(
            projection
                .segment(point![f32::NAN, 1.0], point![1.0, 1.0])
                .is_none()
        );
        let [a, b] = projection
            .segment(point![-1.0, 1.0], point![1.0, 1.0])
            .unwrap();
        assert!(a.inner.iter().chain(b.inner.iter()).all(|v| v.is_finite()));
        assert!((a.y() - 1.0 / NEAR_Z).abs() < 1.0);
        assert!((b.y() - 1.0).abs() < 1.0e-5);
        let reverse = projection
            .segment(point![1.0, 1.0], point![-1.0, 1.0])
            .unwrap();
        assert_eq!(reverse, [b, a]);
    }

    #[test]
    fn curves_use_at_most_256_segments_proportional_to_arc_length() {
        let projection = FieldProjection {
            field_to_camera: Isometry3::from_translation(0.0, 0.0, 1.0),
            intrinsics: Intrinsic::default(),
        };
        for (end, expected) in [(FRAC_PI_2, 64), (TAU, 256), (TAU * 2.0, 256)] {
            let mut count = 0;
            projection.arc_segments(&|t| point![t.cos(), t.sin(), 0.0], 0.0, end, &mut |_, _| {
                count += 1
            });
            assert_eq!(count, expected);
        }
    }

    #[test]
    fn curves_skip_hidden_pieces_without_bridging_gaps() {
        let projection = FieldProjection {
            field_to_camera: Isometry3::from_rotation(linear_algebra::vector![
                0.0, -FRAC_PI_2, 0.0
            ]),
            intrinsics: Intrinsic::default(),
        };
        let at = |t: f32| point![t.cos(), t.sin(), 0.0];
        let mut segments = Vec::new();
        projection.arc_segments(&at, 0.0, TAU, &mut |a, b| {
            segments.push((a, b));
        });
        let expected: Vec<_> = (0..256)
            .filter_map(|index| {
                Some((
                    projection.project(at(TAU * index as f32 / 256.0))?,
                    projection.project(at(TAU * (index + 1) as f32 / 256.0))?,
                ))
            })
            .collect();
        assert_eq!(segments, expected);
        assert_eq!(segments.len(), 126);
        assert!(segments.windows(2).any(|pair| pair[0].1 != pair[1].0));
        let hidden = FieldProjection {
            field_to_camera: Isometry3::from_translation(0.0, 0.0, -1.0),
            ..projection
        };
        hidden.arc_segments(&at, 0.0, TAU, &mut |_, _| {
            panic!("hidden arc must not be drawn");
        });
    }
}

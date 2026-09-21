use super::{
    Estimator, HUBER_THRESHOLD, MIN_REPROJECTION_DEPTH, WINDOW_NS, control_keys, seconds_per_knot,
};
use crate::{
    alignment::{seed_alignment, valid_visual_frame},
    camera::robot_to_camera,
};
use color_eyre::Result;
use linear_algebra::{IntoTransform, Point2, Point3};
use localization_fagra::{
    factors::{
        AdjacentVisualOdometry, FrameReprojections, ReprojectionObservation, VisualOdometry,
        VisualOdometryObservation,
    },
    variables::FieldAlignment,
};
use nalgebra::{Matrix2, SMatrix};
use projection::camera_matrix::CameraMatrix;
use types::{
    time_wrapper::TimeWrapper, visual_localization::VisualLocalizationFrame,
    visual_odometry::VisualOdometer,
};

impl Estimator {
    pub(crate) fn visual_rms(&self) -> Option<f64> {
        let frame = self.latest_visual_frame.as_ref()?;
        let (segment, tau) = self.segment_and_tau(frame.time).ok()?;
        let pose = self
            .spline(control_keys(&self.controls, segment).ok()?)
            .ok()?
            .pose(tau)
            .ok()?;
        let alignment = self.graph.get(self.alignment?).ok()?;
        let camera = self.graph.get(self.intrinsics).ok()?;
        let transform = frame.inner.robot_to_camera.inner.cast::<f64>()
            * pose.inner.inverse()
            * alignment.local_to_field.to_3d().inner.inverse();
        let mut squared = 0.0;
        for observation in &frame.inner.associations {
            let p = transform * observation.field_point.inner.cast::<f64>();
            if p.z <= MIN_REPROJECTION_DEPTH {
                return None;
            }
            let pixel = nalgebra::Vector2::new(
                camera.focal_lengths.x() * p.x / p.z + camera.optical_center.x(),
                camera.focal_lengths.y() * p.y / p.z + camera.optical_center.y(),
            );
            squared += (pixel - observation.detection.inner.coords.cast::<f64>()).norm_squared();
        }
        let rms = (squared / frame.inner.associations.len() as f64).sqrt();
        rms.is_finite().then_some(rms)
    }

    pub(crate) fn ingest_visual(
        &mut self,
        frame: TimeWrapper<VisualLocalizationFrame>,
    ) -> Result<bool> {
        let time = frame.time;
        let Some((segment, tau)) = self.check_time(time, "visual localization")? else {
            return Ok(false);
        };
        let mut frame = frame.inner;
        if frame.epoch != self.epoch || !valid_visual_frame(&frame) {
            return Ok(false);
        }
        let controls = self.ensure_segment(segment)?;
        let pose = self.spline(controls)?.pose(tau)?;
        let candidate = if self.alignment.is_none() {
            let Some(seed) = seed_alignment(
                frame.robot_to_local,
                frame.robot_to_camera,
                frame.camera_intrinsic,
                &mut frame.associations,
            ) else {
                return Ok(false);
            };
            Some(FieldAlignment {
                local_to_field: seed.inner.cast().framed_transform(),
            })
        } else {
            None
        };
        let alignment = match &candidate {
            Some(value) => value,
            None => self
                .graph
                .get(self.alignment.expect("existing alignment"))?,
        };
        let field_to_camera = frame.robot_to_camera.inner.cast::<f64>()
            * pose.inner.inverse()
            * alignment.local_to_field.to_3d().inner.inverse();
        if frame.associations.iter().any(|a| {
            let p = field_to_camera * a.field_point.inner.cast::<f64>();
            !p.iter().all(|v| v.is_finite()) || p.z <= MIN_REPROJECTION_DEPTH
        }) {
            tracing::warn!(?time, "discarding visual frame outside projection domain");
            return Ok(false);
        }
        if let Some(candidate) = candidate {
            self.alignment = Some(self.graph.add(candidate));
            for segment in self.segments() {
                self.add_containment(segment)?;
            }
        }
        let batch = self.graph.add_batch(FrameReprojections {
            controls,
            alignment: self.alignment.expect("initialized alignment"),
            intrinsics: self.intrinsics,
            duration: seconds_per_knot(),
            tau,
            robot_to_camera: frame.robot_to_camera.inner.cast().framed_transform(),
            pixel_information_root: Matrix2::identity()
                / self.parameters.visual_feature_noise_variance.sqrt(),
            huber_threshold: HUBER_THRESHOLD,
            min_depth: MIN_REPROJECTION_DEPTH,
        });
        for association in &frame.associations {
            self.graph.add_factor_to(
                batch,
                ReprojectionObservation {
                    field_point: Point3::wrap(association.field_point.inner.cast()),
                    detection: Point2::wrap(association.detection.inner.cast()),
                },
            )?;
            *self.measurements.entry(segment).or_default() += 1;
        }
        self.reprojection_batches.push((segment, batch));
        if self
            .latest_visual_frame
            .as_ref()
            .is_none_or(|old| time >= old.time)
        {
            self.latest_visual_frame = Some(TimeWrapper { time, inner: frame });
        }
        self.commit_time(time);
        Ok(true)
    }

    pub(crate) fn ingest_visual_odometry(
        &mut self,
        sample: VisualOdometer,
        previous_camera: Option<&CameraMatrix>,
        current_camera: Option<&CameraMatrix>,
    ) -> Result<bool> {
        if self.check_time(sample.time, "visual odometry")?.is_none() {
            return Ok(false);
        }
        if self
            .latest_vo_epoch
            .is_some_and(|epoch| sample.epoch < epoch)
        {
            tracing::warn!(epoch = sample.epoch, "discarding old VO epoch");
            return Ok(false);
        }
        self.latest_vo_epoch = Some(sample.epoch);
        let Some(delta) = sample.delta else {
            return Ok(false);
        };
        if !delta
            .current_left_camera_to_previous_left_camera
            .to_homogeneous()
            .iter()
            .all(|v| v.is_finite())
        {
            tracing::warn!("discarding invalid VO timestamp or transform");
            return Ok(false);
        }
        let (Some(previous_camera), Some(current_camera)) = (previous_camera, current_camera)
        else {
            tracing::warn!("discarding visual odometry without endpoint camera geometry");
            return Ok(false);
        };
        let Some((previous_segment, previous_tau)) =
            self.check_time(delta.previous_time, "visual odometry")?
        else {
            return Ok(false);
        };
        let Some((current_segment, current_tau)) =
            self.check_time(sample.time, "visual odometry")?
        else {
            return Ok(false);
        };
        if sample.time <= delta.previous_time || current_segment - previous_segment > 1 {
            tracing::warn!("discarding unsupported visual odometry interval");
            return Ok(false);
        }
        if delta.previous_time.as_nanos()
            < self
                .latest_time
                .max(sample.time)
                .as_nanos()
                .saturating_sub(WINDOW_NS)
        {
            tracing::warn!("discarding visual odometry outside optimization window");
            return Ok(false);
        }
        let transform = robot_to_camera(previous_camera)
            .inner
            .cast::<f64>()
            .inverse()
            * delta
                .current_left_camera_to_previous_left_camera
                .cast::<f64>()
            * robot_to_camera(current_camera).inner.cast::<f64>();
        let observation = VisualOdometryObservation {
            previous_tau,
            current_tau,
            current_to_previous: transform.framed_transform(),
        };
        let information_root = SMatrix::identity() * 10.0;
        if previous_segment == current_segment {
            let controls = self.ensure_segment(current_segment)?;
            let batch = *self
                .odometry_batches
                .entry(current_segment)
                .or_insert_with(|| {
                    self.graph.add_batch(VisualOdometry {
                        controls,
                        duration: seconds_per_knot(),
                        information_root,
                        huber_threshold: HUBER_THRESHOLD,
                    })
                });
            self.graph.add_factor_to(batch, observation)?;
        } else {
            let a = self.ensure_segment(previous_segment)?;
            let b = self.ensure_segment(current_segment)?;
            self.graph.add_factor(AdjacentVisualOdometry {
                controls: [a[0], a[1], a[2], a[3], b[3]],
                duration: seconds_per_knot(),
                observation,
                information_root,
                huber_threshold: HUBER_THRESHOLD,
            })?;
        }
        self.commit_time(sample.time);
        *self.measurements.entry(previous_segment).or_default() += 1;
        Ok(true)
    }
}

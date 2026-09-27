use super::{
    Estimator, HUBER_THRESHOLD, MIN_LANDMARK_RANGE, MIN_REPROJECTION_DEPTH, WINDOW_NS,
    control_keys, recovery::MotionRecord, seconds_per_knot,
};
use crate::alignment::{valid_visual_frame, valid_visual_rms};
use crate::heading::HeadingReference;
use color_eyre::{Result, eyre::eyre};
use coordinate_systems::{Field, Robot};
use linear_algebra::{IntoTransform, Isometry3, Orientation2, Point2, Point3};
use localization_fagra::factors::{
    AdjacentVisualOdometry, FrameReprojections, ReprojectionObservation, VisualOdometry,
    VisualOdometryObservation,
};
use nalgebra::SMatrix;
use ros_z::time::Time;
use types::camera_geometry::CameraGeometry;
use types::{
    time_wrapper::TimeWrapper, visual_localization::VisualLocalizationFrame,
    visual_odometry::VisualOdometer,
};

pub(super) struct PendingVisual {
    segment: i64,
    batch: fagra::BatchKey<FrameReprojections>,
    factors: Vec<fagra::FactorKey<ReprojectionObservation>>,
    frame: TimeWrapper<VisualLocalizationFrame>,
}

impl Estimator {
    pub(crate) fn field_heading_at(&self, time: Time) -> Option<Orientation2<Field, f64>> {
        let (segment, tau) = self.segment_and_tau(time).ok()?;
        let pose = self
            .spline(control_keys(&self.controls, segment).ok()?)
            .ok()?
            .pose(tau)
            .ok()?;
        let alignment = self.graph.get(self.alignment?).ok()?;
        Some(Orientation2::new(
            (alignment.local_to_field.to_3d().inner.rotation * pose.inner.rotation)
                .euler_angles()
                .2,
        ))
    }

    pub(crate) fn visual_rms(&self) -> Option<f64> {
        let frame = self
            .pending_visuals
            .iter()
            .map(|p| &p.frame)
            .chain(self.latest_visual_frame.iter())
            .max_by_key(|f| f.time)?;
        self.frame_rms(frame)
    }

    fn frame_rms(&self, frame: &TimeWrapper<VisualLocalizationFrame>) -> Option<f64> {
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

    /// Validate before either publication or marginalization can retain a bad field update.
    pub(super) fn validate_heading(&self, reference: &HeadingReference) -> Result<()> {
        let (&time, _) = self
            .attitudes
            .range(..=self.latest_time)
            .next_back()
            .ok_or_else(|| eyre!("missing IMU heading"))?;
        if self.latest_time.duration_since(time) > types::localization::MAX_IMU_ATTITUDE_GAP {
            return Err(eyre!("stale IMU heading"));
        }
        for time in std::iter::once(time).chain(self.pending_visuals.iter().map(|p| p.frame.time)) {
            let expected = reference.expected(
                self.attitude_at(time)
                    .ok_or_else(|| eyre!("missing exposure IMU heading"))?,
            );
            let field = self
                .field_heading_at(time)
                .ok_or_else(|| eyre!("missing field heading"))?;
            if expected.rotation_to(field).inner.angle().abs() > self.parameters.max_heading_error {
                return Err(eyre!("field heading disagrees with propagated IMU heading"));
            }
        }
        Ok(())
    }

    pub(super) fn validate_tilt(&self) -> Result<()> {
        let Some((&time, &attitude)) = self.attitudes.range(..=self.latest_time).next_back() else {
            return Ok(());
        };
        let (segment, tau) = self.segment_and_tau(time)?;
        let pose = self
            .spline(control_keys(&self.controls, segment)?)?
            .pose(tau)?;
        let predicted = pose.inner.rotation.inverse() * nalgebra::Vector3::z();
        let measured = attitude.inner.inverse() * nalgebra::Vector3::z();
        let error = predicted
            .cross(&measured)
            .norm()
            .atan2(predicted.dot(&measured));
        if error > self.parameters.max_tilt_error {
            return Err(eyre!("estimated tilt disagrees with IMU attitude"));
        }
        Ok(())
    }

    pub(super) fn validate_pending_visuals(&self) -> Result<()> {
        if self
            .pending_visuals
            .iter()
            .any(|pending| !valid_visual_rms(self.frame_rms(&pending.frame)))
        {
            return Err(eyre!("visual update failed pixel validation"));
        }
        Ok(())
    }

    pub(super) fn accept_visuals(&mut self) {
        for pending in self.pending_visuals.drain(..) {
            if self
                .latest_visual_frame
                .as_ref()
                .is_none_or(|old| pending.frame.time >= old.time)
            {
                self.latest_visual_frame = Some(pending.frame);
            }
        }
    }

    pub(super) fn discard_visuals(&mut self) -> Result<()> {
        for pending in self.pending_visuals.drain(..) {
            for key in &pending.factors {
                self.graph.remove_factor(*key)?;
            }
            self.graph.remove_batch(pending.batch)?;
            self.reprojection_batches
                .retain(|(_, batch)| *batch != pending.batch);
            *self.measurements.entry(pending.segment).or_default() -= pending.factors.len();
        }
        Ok(())
    }

    pub(crate) fn ingest_visual(
        &mut self,
        frame: TimeWrapper<VisualLocalizationFrame>,
    ) -> Result<bool> {
        let time = frame.time;
        let Some((segment, tau)) = self.check_time(time, "visual localization")? else {
            return Ok(false);
        };
        let frame = frame.inner;
        if frame.epoch != self.epoch
            || frame.generation != self.generation
            || !valid_visual_frame(&frame)
        {
            return Ok(false);
        }
        let Some(alignment_key) = self.alignment else {
            return Ok(false);
        };
        let controls = self.ensure_segment(segment)?;
        let pose = self.spline(controls)?.pose(tau)?;
        let alignment = self.graph.get(alignment_key)?;
        let field_to_camera = frame.robot_to_camera.inner.cast::<f64>()
            * pose.inner.inverse()
            * alignment.local_to_field.to_3d().inner.inverse();
        if frame.associations.iter().any(|a| {
            let p = field_to_camera * a.field_point.inner.cast::<f64>();
            !p.iter().all(|v| v.is_finite()) || p.coords.norm() <= MIN_LANDMARK_RANGE
        }) {
            tracing::warn!(?time, "discarding visual frame with invalid landmark range");
            return Ok(false);
        }
        let batch = self.graph.add_batch(FrameReprojections {
            controls,
            alignment: alignment_key,
            intrinsics: self.intrinsics,
            duration: seconds_per_knot(),
            tau,
            robot_to_camera: frame.robot_to_camera.inner.cast().framed_transform(),
            // Fixed calibration-based conversion; do not let optimized focal lengths
            // change measurement strength during a solve.
            angular_information_root: (f64::from(frame.camera_intrinsic.focals.x)
                * f64::from(frame.camera_intrinsic.focals.y))
            .sqrt()
                / self.parameters.visual_feature_noise_variance.sqrt(),
            huber_threshold: HUBER_THRESHOLD,
            min_range: MIN_LANDMARK_RANGE,
        });
        let mut factors = Vec::with_capacity(frame.associations.len());
        for association in &frame.associations {
            factors.push(self.graph.add_factor_to(
                batch,
                ReprojectionObservation {
                    field_point: Point3::wrap(association.field_point.inner.cast()),
                    detection: Point2::wrap(association.detection.inner.cast()),
                },
            )?);
            *self.measurements.entry(segment).or_default() += 1;
        }
        self.reprojection_batches.push((segment, batch));
        self.pending_visuals.push(PendingVisual {
            segment,
            batch,
            factors,
            frame: TimeWrapper { time, inner: frame },
        });
        self.commit_time(time);
        Ok(true)
    }

    pub(crate) fn ingest_visual_odometry(
        &mut self,
        sample: VisualOdometer,
        previous_camera: Option<&CameraGeometry>,
        current_camera: Option<&CameraGeometry>,
    ) -> Result<bool> {
        let Some((current_segment, _)) = self.check_time(sample.time, "visual odometry")? else {
            return Ok(false);
        };
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
        let Some((previous_segment, _)) =
            self.check_time(delta.previous_time, "visual odometry")?
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
        let transform = previous_camera
            .robot_to_camera
            .inner
            .cast::<f64>()
            .inverse()
            * delta
                .current_left_camera_to_previous_left_camera
                .cast::<f64>()
            * current_camera.robot_to_camera.inner.cast::<f64>();
        self.accept_motion(MotionRecord::Visual {
            previous_time: delta.previous_time,
            time: sample.time,
            transform: transform.framed_transform(),
        })
    }

    pub(super) fn insert_visual_odometry(
        &mut self,
        previous_time: Time,
        time: Time,
        transform: Isometry3<Robot, Robot, f64>,
    ) -> Result<()> {
        let (previous_segment, previous_tau) = self.segment_and_tau(previous_time)?;
        let (current_segment, current_tau) = self.segment_and_tau(time)?;
        let observation = VisualOdometryObservation {
            previous_tau,
            current_tau,
            current_to_previous: transform,
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
        *self.measurements.entry(previous_segment).or_default() += 1;
        Ok(())
    }
}

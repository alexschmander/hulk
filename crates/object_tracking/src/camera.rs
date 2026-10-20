//! Camera observation front-end: bounding boxes to measurements, and camera visibility.
//!
//! Everything is computed with the camera matrix of the image's acquisition time and expressed in
//! the odometry frame using the odometry pose of that same time.

use coordinate_systems::{Ground, Odometry, Pixel};
use linear_algebra::{IntoFramed, Isometry2, Point2, point, vector};
use nalgebra::{Matrix2, Vector2};
use projection::{Projection, camera_matrix::CameraMatrix};
use serde::{Deserialize, Serialize};
use types::bounding_box::BoundingBox;

use crate::measurement::MeasurementModel;

/// Pixel of a bounding box that corresponds to the object's ground anchor.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, ros_z::Message)]
pub enum GroundAnchor {
    /// Upright objects touching the ground, e.g. robots, people and goal posts.
    BottomCenter,
    /// Flat markings lying on the ground, e.g. line intersections.
    Center,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, ros_z::Message)]
pub struct ObservationParameters {
    /// Standard deviation of the anchor pixel, in pixels.
    pub pixel_standard_deviation: f32,
    /// Standard deviation of the camera pitch, in radians.
    ///
    /// Treated as independent per detection although a pitch error is shared by every detection
    /// of an image; this is a documented approximation.
    pub pitch_standard_deviation: f32,
    /// Additional isotropic ground position standard deviation, in meters.
    pub model_standard_deviation: f32,
    /// Additional bearing standard deviation, in radians.
    pub bearing_model_standard_deviation: f32,
    /// Bottom anchors closer than this many pixels to the lower image border are treated as
    /// cut off, so their ground contact is not trusted.
    pub image_border_margin: f32,
    /// Anchors closer than this many pixels below the horizon do not initialize tracks.
    pub horizon_margin: f32,
    /// Ground contacts projected farther than this do not initialize tracks.
    pub maximum_contact_distance: f32,
}

/// Camera geometry of one image.
pub struct CameraView<'a> {
    pub camera_matrix: &'a CameraMatrix,
    pub ground_to_odometry: Isometry2<Ground, Odometry>,
}

impl CameraView<'_> {
    pub fn camera_position(&self) -> Point2<Ground> {
        self.camera_matrix
            .ground_to_camera
            .inverse()
            .as_pose()
            .position()
            .xy()
    }

    pub fn camera_height(&self) -> f32 {
        self.camera_matrix
            .ground_to_camera
            .inverse()
            .as_pose()
            .position()
            .z()
    }

    /// Converts a detection into a ground contact or bearing measurement in the odometry frame.
    ///
    /// Returns a bearing through the box center when the anchor is cut off at the image border.
    pub fn measure(
        &self,
        bounding_box: &BoundingBox,
        anchor: GroundAnchor,
        parameters: &ObservationParameters,
    ) -> Option<MeasurementModel> {
        let area = bounding_box.area;
        let center_x = (area.min.x() + area.max.x()) / 2.0;
        let anchor_pixel = match anchor {
            GroundAnchor::BottomCenter => point![center_x, area.max.y()],
            GroundAnchor::Center => point![center_x, (area.min.y() + area.max.y()) / 2.0],
        };
        let cut_off = anchor == GroundAnchor::BottomCenter
            && area.max.y() >= self.camera_matrix.image_size.y() - parameters.image_border_margin;
        let near_horizon = self.camera_matrix.horizon.is_some_and(|horizon| {
            anchor_pixel.y() < horizon.y_at_x(anchor_pixel.x()) + parameters.horizon_margin
        });

        if !cut_off
            && let Some(contact) = self.ground_contact(anchor_pixel, near_horizon, parameters)
        {
            return Some(contact);
        }
        // The box center lies on the projection of the object's vertical axis, whose rays share
        // the object's azimuth even when the camera is pitched.
        let center = point![center_x, (area.min.y() + area.max.y()) / 2.0];
        self.bearing_measurement(center, parameters)
    }

    /// Line of sight to a ground contact, with its projected position if the anchor is clearly
    /// below the horizon and within the detector's trusted range.
    ///
    /// The angles stay valid when the projection is not trusted: selecting only near projections
    /// would bias distant objects towards the camera.
    fn ground_contact(
        &self,
        pixel: Point2<Pixel>,
        near_horizon: bool,
        parameters: &ObservationParameters,
    ) -> Option<MeasurementModel> {
        let ray = self.camera_matrix.ground_to_camera.inverse() * self.camera_matrix.bearing(pixel);
        let horizontal = Vector2::new(ray.x(), ray.y()).norm();
        if horizontal < 1e-3 {
            return None;
        }
        let height = self.camera_height().max(0.1);
        let depression = (-ray.z()).atan2(horizontal);
        let initialization = (!near_horizon)
            .then(|| self.projected_position(pixel, parameters))
            .flatten();
        let range = initialization
            .map(|(position, _)| (position - self.camera_position_in_odometry()).norm())
            .unwrap_or_else(|| {
                if depression > 0.0 {
                    (height / depression.tan()).min(parameters.maximum_contact_distance)
                } else {
                    parameters.maximum_contact_distance
                }
            });
        let (radians_per_pixel_x, radians_per_pixel_y) = self.radians_per_pixel();
        Some(MeasurementModel::GroundContact {
            origin: self.camera_position_in_odometry(),
            height,
            azimuth: crate::gaussian::wrap_angle(
                ray.y().atan2(ray.x()) + self.ground_to_odometry.inner.rotation.angle(),
            ),
            depression,
            angle_variances: Vector2::new(
                (parameters.pixel_standard_deviation * radians_per_pixel_x).powi(2),
                (parameters.pixel_standard_deviation * radians_per_pixel_y).powi(2)
                    + parameters.pitch_standard_deviation.powi(2),
            ),
            model_variance: parameters.model_standard_deviation.powi(2),
            range,
            initialization,
        })
    }

    fn camera_position_in_odometry(&self) -> Vector2<f32> {
        (self.ground_to_odometry * self.camera_position())
            .inner
            .coords
    }

    /// Ground projection of a pixel with its linearized covariance in the odometry frame.
    fn projected_position(
        &self,
        pixel: Point2<Pixel>,
        parameters: &ObservationParameters,
    ) -> Option<(Vector2<f32>, Matrix2<f32>)> {
        let ground = self.camera_matrix.pixel_to_ground(pixel).ok()?;
        let offset = (ground - self.camera_position()).inner;
        let distance = offset.norm();
        if distance > parameters.maximum_contact_distance || distance < 1e-3 {
            return None;
        }
        let pixel_variance = parameters.pixel_standard_deviation.powi(2);
        let pixel_covariance = self
            .camera_matrix
            .project_noise_to_ground(ground, vector![pixel_variance, pixel_variance])
            .ok()?;
        let height = self.camera_height().max(0.1);
        let range_per_radian = (height.powi(2) + distance.powi(2)) / height;
        let direction = offset / distance;
        let pitch_covariance = direction
            * direction.transpose()
            * (parameters.pitch_standard_deviation * range_per_radian).powi(2);
        let covariance_in_ground = pixel_covariance
            + pitch_covariance
            + Matrix2::identity() * parameters.model_standard_deviation.powi(2);

        let rotation = self.ground_to_odometry.inner.rotation.to_rotation_matrix();
        let position = (self.ground_to_odometry * ground).inner.coords;
        let covariance = rotation.matrix() * covariance_in_ground * rotation.matrix().transpose();
        covariance
            .iter()
            .all(|value| value.is_finite())
            .then(|| (position, (covariance + covariance.transpose()) * 0.5))
    }

    fn radians_per_pixel(&self) -> (f32, f32) {
        let field_of_view = self.camera_matrix.field_of_view;
        let image_size = self.camera_matrix.image_size;
        (
            field_of_view.x / image_size.x(),
            field_of_view.y / image_size.y(),
        )
    }

    fn bearing_measurement(
        &self,
        pixel: Point2<Pixel>,
        parameters: &ObservationParameters,
    ) -> Option<MeasurementModel> {
        let ray_in_ground =
            self.camera_matrix.ground_to_camera.inverse() * self.camera_matrix.bearing(pixel);
        let horizontal = Vector2::new(ray_in_ground.x(), ray_in_ground.y());
        if horizontal.norm() < 1e-3 {
            return None;
        }
        let origin = self.ground_to_odometry * self.camera_position();
        let angle_in_ground = horizontal.y.atan2(horizontal.x);
        let angle = angle_in_ground + self.ground_to_odometry.inner.rotation.angle();
        let (radians_per_pixel, _) = self.radians_per_pixel();
        let variance = (parameters.pixel_standard_deviation * radians_per_pixel).powi(2)
            + parameters.bearing_model_standard_deviation.powi(2);
        Some(MeasurementModel::Bearing {
            origin: origin.inner.coords,
            angle: crate::gaussian::wrap_angle(angle),
            variance,
        })
    }

    /// Fraction of an upright object of `height` at `position` that falls inside the image.
    ///
    /// The object's center column must lie within the image, shrunk by `margin` pixels; the
    /// returned value is the visible fraction of its vertical extent.
    pub fn visible_fraction(&self, position: Point2<Odometry>, height: f32, margin: f32) -> f32 {
        let ground = self.ground_to_odometry.inverse() * position;
        let Ok(bottom) = self.camera_matrix.ground_to_pixel(ground) else {
            return 0.0;
        };
        let top = if height > 0.0 {
            match self.camera_matrix.ground_with_z_to_pixel(ground, height) {
                Ok(top) => top,
                Err(_) => return 0.0,
            }
        } else {
            bottom
        };
        let width = self.camera_matrix.image_size.x();
        let image_height = self.camera_matrix.image_size.y();
        let inside_horizontally =
            |pixel: Point2<Pixel>| pixel.x() >= margin && pixel.x() <= width - margin;
        if !inside_horizontally(bottom) && !inside_horizontally(top) {
            return 0.0;
        }
        if height <= 0.0 {
            let inside = bottom.y() >= margin && bottom.y() <= image_height - margin;
            return if inside { 1.0 } else { 0.0 };
        }
        let upper = top.y().min(bottom.y());
        let lower = top.y().max(bottom.y());
        let extent = (lower - upper).max(1e-3);
        let visible = (lower.min(image_height) - upper.max(0.0)).max(0.0);
        (visible / extent).clamp(0.0, 1.0)
    }

    /// Distance from the camera to a point, on the ground plane.
    pub fn ground_distance(&self, position: Point2<Odometry>) -> f32 {
        let ground = self.ground_to_odometry.inverse() * position;
        (ground - self.camera_position()).norm()
    }

    pub fn odometry_to_ground(&self) -> Isometry2<Odometry, Ground> {
        self.ground_to_odometry.inverse()
    }
}

/// Converts a ground-frame vector into the odometry frame.
pub fn ground_point_to_odometry(
    ground_to_odometry: Isometry2<Ground, Odometry>,
    point: nalgebra::Point2<f32>,
) -> Point2<Odometry> {
    ground_to_odometry * point.framed()
}

/// An object that can hide others from the camera.
#[derive(Clone, Copy, Debug)]
pub struct Occluder {
    pub position: Vector2<f32>,
    pub position_covariance: Matrix2<f32>,
    pub radius: f32,
    pub existence: f32,
}

impl Occluder {
    /// Distance, bearing, angular half width and bearing variance as seen from `camera`.
    fn interval(&self, camera: Vector2<f32>) -> Option<(f32, f32, f32, f32)> {
        let offset = self.position - camera;
        let distance = offset.norm();
        if distance <= self.radius {
            return None;
        }
        let across = Vector2::new(-offset.y, offset.x) / distance;
        let bearing_variance =
            (across.transpose() * self.position_covariance * across)[0].max(0.0) / distance.powi(2);
        Some((
            distance,
            offset.y.atan2(offset.x),
            (self.radius / distance).asin(),
            bearing_variance,
        ))
    }
}

/// Expected probability of each object being visible past closer objects, as seen from `camera`
/// on the ground plane.
///
/// An object is hidden when a closer object blocks the line of sight to its center; detectors
/// still report partially hidden objects. The probability of a block accounts for the bearing
/// uncertainty of the closer object, so that a vaguely known object does not hide everything
/// behind its estimate. The hidden object's own uncertainty is left out: its misses in
/// consecutive images are not independent, and spreading it out of the shadow would delete
/// objects standing behind others within a few images. Blocks by different objects are combined as independent events, weighted by
/// their existence. This expected-visibility surrogate is an approximation of the coupled
/// occlusion model. Objects whose footprints overlap cannot both exist, so they do not occlude
/// each other.
pub fn expected_unoccluded_fractions(camera: Vector2<f32>, objects: &[Occluder]) -> Vec<f32> {
    let intervals: Vec<_> = objects
        .iter()
        .map(|object| object.interval(camera))
        .collect();

    intervals
        .iter()
        .enumerate()
        .map(|(index, interval)| {
            let Some((distance, center, _, _)) = *interval else {
                return 1.0;
            };
            intervals
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != index)
                .filter_map(|(other, interval)| Some((other, (*interval)?)))
                .filter(|(other, (other_distance, _, _, _))| {
                    let separation = (objects[*other].position - objects[index].position).norm();
                    *other_distance < distance
                        && separation >= objects[*other].radius + objects[index].radius
                })
                .map(|(other, (_, other_center, half_width, variance))| {
                    let offset = crate::gaussian::wrap_angle(other_center - center);
                    let blocked = probability_within(offset, half_width, variance);
                    1.0 - objects[other].existence.clamp(0.0, 1.0) * blocked
                })
                .product()
        })
        .collect()
}

/// Probability that a normal variable with `mean` and `variance` lies within `±half_width`.
fn probability_within(mean: f32, half_width: f32, variance: f32) -> f32 {
    if variance <= f32::EPSILON {
        return if mean.abs() < half_width { 1.0 } else { 0.0 };
    }
    let scale = (2.0 * variance).sqrt();
    (0.5 * (erf((half_width - mean) / scale) + erf((half_width + mean) / scale))).clamp(0.0, 1.0)
}

/// Error function with an absolute error below 1.5e-7 (Abramowitz and Stegun 7.1.26).
fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let polynomial = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_7 + t * (-1.453_152_1 + t * 1.061_405_4))));
    (1.0 - polynomial * (-x * x).exp()).copysign(x)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn fully_hidden_object_behind_certain_occluder() {
        let objects = [
            Occluder {
                position: Vector2::new(2.0, 0.0),
                position_covariance: Matrix2::zeros(),
                radius: 0.3,
                existence: 1.0,
            },
            Occluder {
                position: Vector2::new(4.0, 0.0),
                position_covariance: Matrix2::zeros(),
                radius: 0.3,
                existence: 1.0,
            },
        ];
        let fractions = expected_unoccluded_fractions(Vector2::zeros(), &objects);

        assert_relative_eq!(fractions[0], 1.0);
        assert_relative_eq!(fractions[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn overlapping_objects_do_not_occlude_each_other() {
        let objects = [
            Occluder {
                position: Vector2::new(2.0, 0.0),
                position_covariance: Matrix2::zeros(),
                radius: 0.2,
                existence: 1.0,
            },
            Occluder {
                position: Vector2::new(2.3, 0.0),
                position_covariance: Matrix2::zeros(),
                radius: 0.2,
                existence: 1.0,
            },
        ];
        let fractions = expected_unoccluded_fractions(Vector2::zeros(), &objects);
        assert_relative_eq!(fractions[1], 1.0);
    }

    #[test]
    fn uncertain_occluder_hides_in_expectation_only() {
        let objects = [
            Occluder {
                position: Vector2::new(2.0, 0.0),
                position_covariance: Matrix2::zeros(),
                radius: 0.3,
                existence: 0.25,
            },
            Occluder {
                position: Vector2::new(4.0, 0.0),
                position_covariance: Matrix2::zeros(),
                radius: 0.3,
                existence: 1.0,
            },
        ];
        let fractions = expected_unoccluded_fractions(Vector2::zeros(), &objects);

        assert_relative_eq!(fractions[1], 0.75, epsilon = 1e-6);
    }

    #[test]
    fn objects_in_other_directions_do_not_occlude() {
        let objects = [
            Occluder {
                position: Vector2::new(2.0, 2.0),
                position_covariance: Matrix2::zeros(),
                radius: 0.3,
                existence: 1.0,
            },
            Occluder {
                position: Vector2::new(4.0, 0.0),
                position_covariance: Matrix2::zeros(),
                radius: 0.3,
                existence: 1.0,
            },
        ];
        let fractions = expected_unoccluded_fractions(Vector2::zeros(), &objects);

        assert_relative_eq!(fractions[1], 1.0);
    }

    #[test]
    fn partially_covered_object_with_visible_center_is_not_hidden() {
        let objects = [
            Occluder {
                position: Vector2::new(2.0, 0.25),
                position_covariance: Matrix2::zeros(),
                radius: 0.2,
                existence: 1.0,
            },
            Occluder {
                position: Vector2::new(4.0, 0.0),
                position_covariance: Matrix2::zeros(),
                radius: 0.2,
                existence: 1.0,
            },
        ];
        let fractions = expected_unoccluded_fractions(Vector2::zeros(), &objects);

        assert_relative_eq!(fractions[1], 1.0);
    }

    #[test]
    fn vaguely_known_occluder_hides_only_where_it_probably_is() {
        let occluder = |variance: f32| Occluder {
            position: Vector2::new(2.0, 0.0),
            position_covariance: Matrix2::identity() * variance,
            radius: 0.2,
            existence: 1.0,
        };
        let target = Occluder {
            position: Vector2::new(4.0, 0.0),
            position_covariance: Matrix2::zeros(),
            radius: 0.2,
            existence: 1.0,
        };
        let visible = |variance| {
            expected_unoccluded_fractions(Vector2::zeros(), &[occluder(variance), target])[1]
        };

        assert_relative_eq!(visible(0.0), 0.0);
        // The occluder covers ±0.2 m of a lateral standard deviation of 1 m: about 16 %.
        assert_relative_eq!(visible(1.0), 1.0 - 0.1585, epsilon = 2e-3);
    }

    #[test]
    fn error_function_matches_reference_values() {
        assert_relative_eq!(erf(0.0), 0.0, epsilon = 1e-6);
        assert_relative_eq!(erf(0.5), 0.520_499_9, epsilon = 1e-6);
        assert_relative_eq!(erf(-1.0), -0.842_700_8, epsilon = 1e-6);
        assert_relative_eq!(erf(3.0), 0.999_977_9, epsilon = 1e-6);
    }
}

/// Synthetic pinhole camera for tests and simulation tooling.
pub mod synthetic {
    use std::f32::consts::FRAC_PI_2;

    use coordinate_systems::{Camera, Ground, Head, Robot};
    use linear_algebra::{IntoTransform, Isometry3, vector};
    use projection::camera_matrix::CameraMatrix;

    /// Camera at `height` above the ground origin looking along ground X, pitched down by `pitch`
    /// radians, with a 640×480 image and a horizontal field of view of about 90 degrees.
    pub fn forward_looking_camera(height: f32, pitch: f32) -> CameraMatrix {
        let ground_to_robot: Isometry3<Ground, Robot> =
            nalgebra::Isometry3::translation(0.0, 0.0, -height).framed_transform();
        let head_to_camera: Isometry3<Head, Camera> =
            (nalgebra::Isometry3::rotation(nalgebra::Vector3::x() * pitch)
                * nalgebra::Isometry3::rotation(nalgebra::Vector3::y() * -FRAC_PI_2)
                * nalgebra::Isometry3::rotation(nalgebra::Vector3::x() * FRAC_PI_2))
            .framed_transform();
        CameraMatrix::from_normalized_focal_and_center(
            nalgebra::vector![0.5, 0.5 * 640.0 / 480.0],
            nalgebra::point![0.5, 0.5],
            vector![640.0, 480.0],
            ground_to_robot,
            Isometry3::identity(),
            head_to_camera,
        )
    }
}

#[cfg(test)]
mod view_tests {
    use approx::assert_relative_eq;
    use geometry::rectangle::Rectangle;

    use super::*;
    use crate::camera::synthetic::forward_looking_camera;

    fn parameters() -> ObservationParameters {
        ObservationParameters {
            pixel_standard_deviation: 2.0,
            pitch_standard_deviation: 0.01,
            model_standard_deviation: 0.05,
            bearing_model_standard_deviation: 0.01,
            image_border_margin: 2.0,
            horizon_margin: 5.0,
            maximum_contact_distance: 10.0,
        }
    }

    fn box_around(camera: &CameraMatrix, ground: Point2<Ground>, height: f32) -> BoundingBox {
        let bottom = camera.ground_to_pixel(ground).unwrap();
        let top = camera.ground_with_z_to_pixel(ground, height).unwrap();
        BoundingBox {
            area: Rectangle {
                min: point![bottom.x() - 10.0, top.y()],
                max: point![bottom.x() + 10.0, bottom.y()],
            },
            confidence: 0.9,
        }
    }

    /// Box of an upright object clipped to the image, as a detector would report it.
    fn clipped_box_around(
        camera: &CameraMatrix,
        ground: Point2<Ground>,
        height: f32,
    ) -> BoundingBox {
        let bottom = camera.ground_to_pixel(ground).unwrap();
        let top = camera.ground_with_z_to_pixel(ground, height).unwrap();
        let center_x = (bottom.x() + top.x()) / 2.0;
        BoundingBox {
            area: Rectangle {
                min: point![center_x - 10.0, top.y().max(0.0)],
                max: point![center_x + 10.0, bottom.y().min(camera.image_size.y())],
            },
            confidence: 0.9,
        }
    }

    #[test]
    fn bottom_anchor_projects_to_ground_position_in_odometry() {
        let camera = forward_looking_camera(0.8, 0.4);
        let ground_to_odometry = Isometry2::<Ground, Odometry>::from_parts(
            vector![10.0, 0.0],
            std::f32::consts::FRAC_PI_2,
        );
        let view = CameraView {
            camera_matrix: &camera,
            ground_to_odometry,
        };
        let target = point![3.0, 0.5];
        let bounding_box = box_around(&camera, target, 0.6);

        let Some(MeasurementModel::GroundContact {
            origin,
            height,
            azimuth,
            depression,
            initialization: Some((position, covariance)),
            ..
        }) = view.measure(&bounding_box, GroundAnchor::BottomCenter, &parameters())
        else {
            panic!("expected a ground contact measurement");
        };

        assert_relative_eq!(position, Vector2::new(9.5, 3.0), epsilon = 1e-3);
        // The angles describe the same point as the projected position.
        let offset = position - origin;
        assert_relative_eq!(azimuth, offset.y.atan2(offset.x), epsilon = 1e-4);
        assert_relative_eq!(depression, height.atan2(offset.norm()), epsilon = 1e-4);
        // Range uncertainty (odometry Y here) dominates the lateral uncertainty.
        assert!(covariance[(1, 1)] > covariance[(0, 0)]);
    }

    #[test]
    fn cut_off_feet_become_a_bearing() {
        let camera = forward_looking_camera(0.8, 0.4);
        let view = CameraView {
            camera_matrix: &camera,
            ground_to_odometry: Isometry2::identity(),
        };
        let target = point![0.35, 0.1];
        assert!(camera.ground_to_pixel(target).unwrap().y() > 480.0);
        let bounding_box = clipped_box_around(&camera, target, 0.6);

        let Some(MeasurementModel::Bearing { angle, .. }) =
            view.measure(&bounding_box, GroundAnchor::BottomCenter, &parameters())
        else {
            panic!("expected a bearing measurement");
        };

        let camera_position = view.camera_position();
        let expected = (target.y() - camera_position.y()).atan2(target.x() - camera_position.x());
        assert_relative_eq!(angle, expected, epsilon = 0.03);
    }

    #[test]
    fn visibility_follows_the_image() {
        let camera = forward_looking_camera(0.8, 0.4);
        let view = CameraView {
            camera_matrix: &camera,
            ground_to_odometry: Isometry2::identity(),
        };

        assert_relative_eq!(view.visible_fraction(point![3.0, 0.0], 0.6, 5.0), 1.0);
        assert_relative_eq!(view.visible_fraction(point![-3.0, 0.0], 0.6, 5.0), 0.0);
        assert_relative_eq!(view.visible_fraction(point![0.5, 5.0], 0.6, 5.0), 0.0);
    }
}

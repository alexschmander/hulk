//! Measurement models expressed in the tracking frame.

use nalgebra::{Matrix1, Matrix2, SMatrix, Vector1, Vector2};
use serde::{Deserialize, Serialize};

use crate::gaussian::{Gaussian, Innovation, innovation, update, wrap_angle};

/// Observation of an object's position, with the leading two state entries as position.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub enum MeasurementModel {
    /// Ground-contact position with its covariance.
    Position {
        position: Vector2<f32>,
        covariance: Matrix2<f32>,
    },
    /// Ground contact seen from a camera at `origin`, `height` above the ground, as the azimuth
    /// and depression angle of the line of sight.
    ///
    /// Pixel and pitch errors are Gaussian in these angles, whereas the range of the projected
    /// ground position has a skewed, heavy-tailed error far from the camera. Comparing tracks in
    /// angles keeps distant detections from looking certain about wrong ranges.
    GroundContact {
        origin: Vector2<f32>,
        height: f32,
        azimuth: f32,
        depression: f32,
        /// Variances of azimuth and depression from pixel and pitch errors.
        angle_variances: Vector2<f32>,
        /// Variance of the object's anchor around its tracked position, in square meters.
        model_variance: f32,
        /// Ground range at which densities per square meter are converted, i.e. the projected
        /// range, limited to the detector's range.
        range: f32,
        /// Projected ground position with its linearized covariance, if trustworthy enough to
        /// initialize a track.
        initialization: Option<(Vector2<f32>, Matrix2<f32>)>,
    },
    /// Direction from `origin` to the object, for detections whose ground contact is not valid,
    /// e.g. with feet cut off by the image border.
    Bearing {
        origin: Vector2<f32>,
        angle: f32,
        variance: f32,
    },
}

impl MeasurementModel {
    /// Whether the measurement fixes a position, i.e. can initialize a new track.
    pub fn is_position(&self) -> bool {
        self.ground_position().is_some()
    }

    /// Ground position and its covariance, for measurements that fix a position.
    pub fn ground_position(&self) -> Option<(Vector2<f32>, Matrix2<f32>)> {
        match *self {
            Self::Position {
                position,
                covariance,
            } => Some((position, covariance)),
            Self::GroundContact { initialization, .. } => initialization,
            Self::Bearing { .. } => None,
        }
    }

    /// Density of false measurements in the units of the measurement likelihood, from densities
    /// per square meter of ground for ground positions and per radian for bearings.
    pub fn clutter_density(&self, per_square_meter: f32, per_radian: f32) -> f32 {
        match self {
            Self::Position { .. } | Self::GroundContact { .. } => {
                per_square_meter * self.square_meters_per_unit()
            }
            Self::Bearing { .. } => per_radian,
        }
    }

    /// Ground area in square meters per unit of this measurement's space, to express densities
    /// per square meter in the units of the measurement likelihood.
    pub fn square_meters_per_unit(&self) -> f32 {
        match *self {
            Self::GroundContact { height, range, .. } => {
                // One radian of azimuth spans `range` meters, one of depression
                // `(range² + height²) / height` meters.
                range * (range.powi(2) + height.powi(2)) / height.max(1e-3)
            }
            Self::Position { .. } | Self::Bearing { .. } => 1.0,
        }
    }
}

/// Innovation of a [`MeasurementModel`] against a track component.
#[derive(Clone, Copy, Debug)]
pub enum MeasurementInnovation<const N: usize> {
    Position {
        innovation: Innovation<N, 2>,
        noise: Matrix2<f32>,
    },
    GroundContact {
        innovation: Innovation<N, 2>,
        noise: Matrix2<f32>,
    },
    Bearing {
        innovation: Innovation<N, 1>,
        noise: Matrix1<f32>,
    },
}

impl<const N: usize> MeasurementInnovation<N> {
    pub fn compute(gaussian: &Gaussian<N>, model: &MeasurementModel) -> Option<Self> {
        match *model {
            MeasurementModel::Position {
                position,
                covariance,
            } => {
                let residual = position - gaussian.mean.fixed_rows::<2>(0);
                let innovation =
                    innovation(gaussian, residual, position_jacobian::<N>(), &covariance)?;
                Some(Self::Position {
                    innovation,
                    noise: covariance,
                })
            }
            MeasurementModel::GroundContact {
                origin,
                height,
                azimuth,
                depression,
                angle_variances,
                model_variance,
                ..
            } => {
                let offset = gaussian.mean.fixed_rows::<2>(0) - origin;
                let squared_range = offset.norm_squared();
                if squared_range < 1e-6 {
                    return None;
                }
                let range = squared_range.sqrt();
                let squared_slant = squared_range + height.powi(2);
                let mut jacobian = SMatrix::<f32, 2, N>::zeros();
                jacobian[(0, 0)] = -offset.y / squared_range;
                jacobian[(0, 1)] = offset.x / squared_range;
                let depression_per_range = -height / squared_slant;
                jacobian[(1, 0)] = depression_per_range * offset.x / range;
                jacobian[(1, 1)] = depression_per_range * offset.y / range;
                // The anchor's spread around the tracked position, seen at the predicted range.
                let noise = Matrix2::from_diagonal(&Vector2::new(
                    angle_variances.x + model_variance / squared_range,
                    angle_variances.y + model_variance * depression_per_range.powi(2),
                ));
                let residual = Vector2::new(
                    wrap_angle(azimuth - offset.y.atan2(offset.x)),
                    depression - height.atan2(range),
                );
                let innovation = innovation(gaussian, residual, jacobian, &noise)?;
                Some(Self::GroundContact { innovation, noise })
            }
            MeasurementModel::Bearing {
                origin,
                angle,
                variance,
            } => {
                let offset = gaussian.mean.fixed_rows::<2>(0) - origin;
                let squared_range = offset.norm_squared();
                if squared_range < 1e-6 {
                    return None;
                }
                let predicted = offset.y.atan2(offset.x);
                let mut jacobian = SMatrix::<f32, 1, N>::zeros();
                jacobian[(0, 0)] = -offset.y / squared_range;
                jacobian[(0, 1)] = offset.x / squared_range;
                let noise = Matrix1::new(variance);
                let innovation = innovation(
                    gaussian,
                    Vector1::new(wrap_angle(angle - predicted)),
                    jacobian,
                    &noise,
                )?;
                Some(Self::Bearing { innovation, noise })
            }
        }
    }

    pub fn log_likelihood(&self) -> f32 {
        match self {
            Self::Position { innovation, .. } => innovation.log_likelihood,
            Self::GroundContact { innovation, .. } => innovation.log_likelihood,
            Self::Bearing { innovation, .. } => innovation.log_likelihood,
        }
    }

    pub fn squared_mahalanobis_distance(&self) -> f32 {
        match self {
            Self::Position { innovation, .. } => innovation.squared_mahalanobis_distance,
            Self::GroundContact { innovation, .. } => innovation.squared_mahalanobis_distance,
            Self::Bearing { innovation, .. } => innovation.squared_mahalanobis_distance,
        }
    }

    pub fn is_position(&self) -> bool {
        matches!(self, Self::Position { .. } | Self::GroundContact { .. })
    }

    pub fn update(&self, gaussian: &Gaussian<N>) -> Option<Gaussian<N>> {
        match self {
            Self::Position { innovation, noise } => update(gaussian, innovation, noise),
            Self::GroundContact { innovation, noise } => update(gaussian, innovation, noise),
            Self::Bearing { innovation, noise } => update(gaussian, innovation, noise),
        }
    }
}

pub fn position_jacobian<const N: usize>() -> SMatrix<f32, 2, N> {
    let mut jacobian = SMatrix::<f32, 2, N>::zeros();
    jacobian[(0, 0)] = 1.0;
    jacobian[(1, 1)] = 1.0;
    jacobian
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{Matrix4, Vector4};

    use super::*;

    #[test]
    fn bearing_update_moves_track_onto_the_ray() {
        let prior = Gaussian::<4> {
            mean: Vector4::new(3.0, 1.0, 0.0, 0.0),
            covariance: Matrix4::from_diagonal(&Vector4::new(0.25, 0.25, 0.1, 0.1)),
        };
        let model = MeasurementModel::Bearing {
            origin: Vector2::zeros(),
            angle: 0.0,
            variance: 1e-4,
        };
        let innovation = MeasurementInnovation::compute(&prior, &model).unwrap();
        let posterior = innovation.update(&prior).unwrap();

        // The linearized update moves along the tangent, approximately preserving the range.
        assert!(posterior.mean[1].abs() < 0.1);
        assert_relative_eq!(
            posterior.mean.fixed_rows::<2>(0).norm(),
            10.0_f32.sqrt(),
            epsilon = 0.2
        );
    }

    fn ground_contact(position: Vector2<f32>, pixel_angle: f32) -> MeasurementModel {
        let height = 1.2_f32;
        MeasurementModel::GroundContact {
            origin: Vector2::zeros(),
            height,
            azimuth: position.y.atan2(position.x),
            depression: height.atan2(position.norm()),
            angle_variances: Vector2::repeat(pixel_angle.powi(2)),
            model_variance: 0.0,
            range: position.norm(),
            initialization: Some((position, Matrix2::identity())),
        }
    }

    #[test]
    fn ground_contact_matches_a_track_at_its_position() {
        let prior = Gaussian::<4> {
            mean: Vector4::new(7.0, 1.0, 0.0, 0.0),
            covariance: Matrix4::from_diagonal(&Vector4::new(0.01, 0.01, 0.1, 0.1)),
        };
        let model = ground_contact(Vector2::new(7.0, 1.0), 0.005);
        let innovation = MeasurementInnovation::compute(&prior, &model).unwrap();

        assert!(innovation.squared_mahalanobis_distance() < 1e-6);
        let posterior = innovation.update(&prior).unwrap();
        assert_relative_eq!(posterior.mean[0], 7.0, epsilon = 1e-4);
    }

    #[test]
    fn distant_ground_contact_constrains_the_bearing_more_than_the_range() {
        let prior = Gaussian::<4> {
            mean: Vector4::new(8.0, 0.0, 0.0, 0.0),
            covariance: Matrix4::from_diagonal(&Vector4::new(1.0, 1.0, 0.1, 0.1)),
        };
        let model = ground_contact(Vector2::new(7.0, 0.0), 0.005);
        let posterior = MeasurementInnovation::compute(&prior, &model)
            .unwrap()
            .update(&prior)
            .unwrap();

        assert!(posterior.covariance[(0, 0)] > 4.0 * posterior.covariance[(1, 1)]);
    }

    #[test]
    fn ground_contact_density_scale_grows_with_range() {
        let near = ground_contact(Vector2::new(2.0, 0.0), 0.005).square_meters_per_unit();
        let far = ground_contact(Vector2::new(8.0, 0.0), 0.005).square_meters_per_unit();
        assert!(far > 30.0 * near);
    }

    #[test]
    fn bearing_residual_is_wrapped() {
        let prior = Gaussian::<2> {
            mean: Vector2::new(-1.0, -0.01),
            covariance: Matrix2::identity() * 0.01,
        };
        let model = MeasurementModel::Bearing {
            origin: Vector2::zeros(),
            angle: std::f32::consts::PI - 0.01,
            variance: 1e-3,
        };
        let innovation = MeasurementInnovation::compute(&prior, &model).unwrap();

        assert!(innovation.squared_mahalanobis_distance() < 1.0);
    }
}

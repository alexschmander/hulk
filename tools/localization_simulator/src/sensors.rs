use field_mark_association::{
    DetectedVisualFeature, DetectedVisualFeatures, FieldFeatureLandmark, VisualFeatureClass,
    field_feature_landmarks,
};
use linear_algebra::Framed;
use nalgebra::{Isometry3, Point2, Point3, Translation3, UnitQuaternion, Vector2, Vector3};
use projection::{Projection as _, camera_matrix::CameraMatrix};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use ros_z::time::Time;
use types::{
    field_dimensions::FieldDimensions,
    visual_localization::FieldMarkAssociation,
    visual_odometry::{VisualOdometer, VisualOdometryDelta},
};

use crate::{config::SimulationConfig, production_vo::ProductionVoDiagnostics};

#[derive(Clone, Debug)]
pub(crate) struct VisualOdometryMeasurement {
    pub delta: Option<VisualOdometryDelta>,
    pub odometer: VisualOdometer,
    pub production_diagnostics: Option<ProductionVoDiagnostics>,
}

#[derive(Debug, Default)]
pub(crate) struct LandmarkObservations {
    pub detections: DetectedVisualFeatures,
    pub true_associations: Vec<FieldMarkAssociation>,
    pub ideal_visible_count: usize,
}

pub(crate) struct SyntheticSensors {
    config: SimulationConfig,
    vo_rng: ChaCha8Rng,
    landmark_rng: ChaCha8Rng,
    previous_camera_to_field: Option<Isometry3<f32>>,
    previous_time: Option<Time>,
    current_camera_to_visual_odometer: Isometry3<f32>,
    transition_index: usize,
    landmarks: Vec<FieldFeatureLandmark>,
}

impl SyntheticSensors {
    pub(crate) fn new(config: &SimulationConfig, field_dimensions: &FieldDimensions) -> Self {
        Self {
            config: config.clone(),
            vo_rng: ChaCha8Rng::seed_from_u64(config.seed ^ 0x564f_5f52_4e47),
            landmark_rng: ChaCha8Rng::seed_from_u64(config.seed ^ 0x4c4d_5f52_4e47),
            previous_camera_to_field: None,
            previous_time: None,
            current_camera_to_visual_odometer: Isometry3::identity(),
            transition_index: 0,
            landmarks: field_feature_landmarks(field_dimensions),
        }
    }

    pub(crate) fn measure_visual_odometry(
        &mut self,
        time: Time,
        camera_to_field: &Isometry3<f32>,
    ) -> VisualOdometryMeasurement {
        let previous_camera_to_field = self.previous_camera_to_field;
        let delta = previous_camera_to_field.zip(self.previous_time).map(
            |(previous_camera_to_field, previous_time)| {
                let exact_current_camera_to_previous_camera =
                    previous_camera_to_field.inverse() * camera_to_field;
                let measured = self.noisy_delta(exact_current_camera_to_previous_camera);
                self.current_camera_to_visual_odometer *= measured;
                self.transition_index += 1;
                VisualOdometryDelta {
                    previous_time,
                    current_time: time,
                    current_left_camera_to_previous_left_camera: measured,
                }
            },
        );
        self.previous_camera_to_field = Some(*camera_to_field);
        self.previous_time = Some(time);
        VisualOdometryMeasurement {
            delta,
            odometer: VisualOdometer {
                time,
                epoch: 0,
                current_left_camera_to_visual_odometer: self.current_camera_to_visual_odometer,
            },
            production_diagnostics: None,
        }
    }

    fn noisy_delta(&mut self, exact: Isometry3<f32>) -> Isometry3<f32> {
        let mut translation = Vector3::from(self.config.vo_translation_bias_per_step);
        let mut rotation = Vector3::from(self.config.vo_rotation_bias_per_step);
        for axis in 0..3 {
            translation[axis] +=
                self.vo_rng.sample::<f32, _>(StandardNormal) * self.config.vo_translation_sigma_m;
            rotation[axis] +=
                self.vo_rng.sample::<f32, _>(StandardNormal) * self.config.vo_rotation_sigma_rad;
        }
        if let Some(outlier) = &self.config.vo_outlier
            && outlier.transition_index == self.transition_index
        {
            translation += Vector3::from(outlier.translation);
            rotation += Vector3::from(outlier.rotation_scaled_axis);
        }
        Isometry3::from_parts(
            Translation3::from(translation),
            UnitQuaternion::from_scaled_axis(rotation),
        ) * exact
    }

    pub(crate) fn observe_landmarks(
        &mut self,
        camera_to_field: &Isometry3<f32>,
        camera_matrix: &CameraMatrix,
    ) -> LandmarkObservations {
        let field_to_camera = camera_to_field.inverse();
        let mut observations = LandmarkObservations::default();
        for landmark in &self.landmarks {
            let field_point = Point3::new(landmark.position.x(), landmark.position.y(), 0.0);
            let camera_point = field_to_camera * field_point;
            let Some(ideal_pixel) = project_in_bounds(camera_point, camera_matrix) else {
                continue;
            };
            observations.ideal_visible_count += 1;
            if self.landmark_rng.random::<f32>() < self.config.landmark_dropout_probability {
                continue;
            }
            let noise = Vector2::new(
                self.landmark_rng.sample::<f32, _>(StandardNormal),
                self.landmark_rng.sample::<f32, _>(StandardNormal),
            ) * self.config.landmark_pixel_sigma;
            let pixel = ideal_pixel + noise;
            if !pixel.x.is_finite()
                || !pixel.y.is_finite()
                || pixel.x < 0.0
                || pixel.y < 0.0
                || pixel.x >= camera_matrix.image_size.inner.x
                || pixel.y >= camera_matrix.image_size.inner.y
            {
                continue;
            }
            let framed_pixel = Framed::wrap(pixel);
            push_detection(
                &mut observations.detections,
                landmark.class,
                DetectedVisualFeature {
                    pixel: framed_pixel,
                    confidence: 1.0,
                },
            );
            observations.true_associations.push(FieldMarkAssociation {
                detection: framed_pixel,
                field_point: Framed::wrap(field_point),
            });
        }
        observations
    }
}

fn project_in_bounds(point: Point3<f32>, camera_matrix: &CameraMatrix) -> Option<Point2<f32>> {
    if point.coords.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let pixel = camera_matrix
        .camera_to_pixel(Framed::wrap(point.coords))
        .ok()?
        .inner;
    (pixel.x.is_finite()
        && pixel.y.is_finite()
        && pixel.x >= 0.0
        && pixel.y >= 0.0
        && pixel.x < camera_matrix.image_size.inner.x
        && pixel.y < camera_matrix.image_size.inner.y)
        .then_some(pixel)
}

fn push_detection(
    detections: &mut DetectedVisualFeatures,
    class: VisualFeatureClass,
    detection: DetectedVisualFeature,
) {
    match class {
        VisualFeatureClass::GoalPost => detections.goalposts.push(detection),
        VisualFeatureClass::LSpot => detections.l_spots.push(detection),
        VisualFeatureClass::TSpot => detections.t_spots.push(detection),
        VisualFeatureClass::XSpot => detections.x_spots.push(detection),
        VisualFeatureClass::PenaltySpot => detections.penalty_spots.push(detection),
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::{
        simulation::camera_matrix,
        trajectory::{Scenario, robot_to_field_from_camera_to_field},
    };

    fn exact_config(seed: u64) -> SimulationConfig {
        SimulationConfig {
            seed,
            landmark_pixel_sigma: 0.0,
            vo_translation_sigma_m: 0.0,
            vo_rotation_sigma_rad: 0.0,
            ..Default::default()
        }
    }

    fn assert_isometry_close(actual: &Isometry3<f32>, expected: &Isometry3<f32>) {
        assert_relative_eq!(
            actual.translation.vector,
            expected.translation.vector,
            epsilon = 1.0e-6
        );
        assert!(actual.rotation.angle_to(&expected.rotation) < 1.0e-6);
    }

    #[test]
    fn vo_delta_direction_and_cumulative_transform_are_consistent() {
        let mut sensors = SyntheticSensors::new(&exact_config(1), &FieldDimensions::SPL_2025);
        let first = Isometry3::translation(1.0, 0.0, 0.0);
        let second = Isometry3::from_parts(
            Translation3::new(1.0, 1.0, 0.0),
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.2),
        );
        let first_measurement = sensors.measure_visual_odometry(Time::from_nanos(0), &first);
        assert!(first_measurement.delta.is_none());
        assert_isometry_close(
            &first_measurement
                .odometer
                .current_left_camera_to_visual_odometer,
            &Isometry3::identity(),
        );

        let second_measurement =
            sensors.measure_visual_odometry(Time::from_nanos(20_000_000), &second);
        let expected_current_camera_to_previous_camera = first.inverse() * second;
        let measured = second_measurement.delta.unwrap();
        assert_isometry_close(
            &measured.current_left_camera_to_previous_left_camera,
            &expected_current_camera_to_previous_camera,
        );
        assert_isometry_close(
            &second_measurement
                .odometer
                .current_left_camera_to_visual_odometer,
            &measured.current_left_camera_to_previous_left_camera,
        );
    }

    #[test]
    fn exact_projection_returns_only_finite_in_bounds_pixels() {
        let scenario = Scenario::stationary();
        let camera_to_field = scenario.sample_camera_to_field(0.0);
        let robot_to_field = robot_to_field_from_camera_to_field(&camera_to_field);
        let matrix = camera_matrix(&robot_to_field);
        let mut sensors = SyntheticSensors::new(&exact_config(2), &FieldDimensions::SPL_2025);
        let observations = sensors.observe_landmarks(&camera_to_field, &matrix);

        assert!(observations.ideal_visible_count > 0);
        assert_eq!(
            observations.detections.supported_feature_count(),
            observations.true_associations.len()
        );
        for association in observations.true_associations {
            let pixel = association.detection.inner;
            assert!(pixel.coords.iter().all(|value| value.is_finite()));
            assert!((0.0..640.0).contains(&pixel.x));
            assert!((0.0..480.0).contains(&pixel.y));
        }
    }

    #[test]
    fn same_seed_repeats_sensor_sequence_with_tolerance() {
        let config = SimulationConfig {
            seed: 42,
            landmark_pixel_sigma: 1.5,
            landmark_dropout_probability: 0.2,
            vo_translation_sigma_m: 0.02,
            vo_rotation_sigma_rad: 0.01,
            ..Default::default()
        };
        let mut first = SyntheticSensors::new(&config, &FieldDimensions::SPL_2025);
        let mut second = SyntheticSensors::new(&config, &FieldDimensions::SPL_2025);
        let scenario = Scenario::six_dof_loop();
        for index in 0..8 {
            let time = Time::from_nanos(index * 20_000_000);
            let pose = scenario.sample_camera_to_field(index as f32 * 0.02);
            let first_vo = first.measure_visual_odometry(time, &pose);
            let second_vo = second.measure_visual_odometry(time, &pose);
            assert_isometry_close(
                &first_vo.odometer.current_left_camera_to_visual_odometer,
                &second_vo.odometer.current_left_camera_to_visual_odometer,
            );
            let robot = robot_to_field_from_camera_to_field(&pose);
            let matrix = camera_matrix(&robot);
            let first_landmarks = first.observe_landmarks(&pose, &matrix);
            let second_landmarks = second.observe_landmarks(&pose, &matrix);
            assert_eq!(
                first_landmarks.true_associations.len(),
                second_landmarks.true_associations.len()
            );
            for (left, right) in first_landmarks
                .true_associations
                .iter()
                .zip(&second_landmarks.true_associations)
            {
                assert_relative_eq!(
                    left.detection.inner,
                    right.detection.inner,
                    epsilon = 1.0e-6
                );
                assert_relative_eq!(
                    left.field_point.inner,
                    right.field_point.inner,
                    epsilon = 1.0e-6
                );
            }
        }
    }

    #[test]
    fn one_shot_outlier_is_applied_to_the_requested_transition() {
        let config = SimulationConfig {
            vo_translation_sigma_m: 0.0,
            vo_rotation_sigma_rad: 0.0,
            vo_outlier: Some(crate::VisualOdometryOutlier {
                transition_index: 0,
                translation: [5.0, 0.0, 0.0],
                rotation_scaled_axis: [0.0, 0.0, std::f32::consts::FRAC_PI_2],
            }),
            ..Default::default()
        };
        let pose = Isometry3::identity();
        let mut sensors = SyntheticSensors::new(&config, &FieldDimensions::SPL_2025);
        sensors.measure_visual_odometry(Time::from_nanos(0), &pose);
        let outlier = sensors
            .measure_visual_odometry(Time::from_nanos(20_000_000), &pose)
            .delta
            .expect("the second frame emits a delta")
            .current_left_camera_to_previous_left_camera;

        assert_relative_eq!(outlier.translation.vector.x, 5.0, epsilon = 1.0e-6);
        assert!((outlier.rotation.angle() - std::f32::consts::FRAC_PI_2).abs() < 1.0e-6);
    }
}

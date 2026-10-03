//! Source-time fixtures from the real flip; see recovered_frames.json for provenance.
use super::*;
use coordinate_systems::{Field, Ground, ImuReference, Local, Robot};
use field_mark_association::{
    DetectedVisualFeature, DetectedVisualFeatures, GlobalAssociationInput,
    GlobalLocalizerParameters, associate_global_visual_features,
};
use linear_algebra::{IntoTransform, Orientation2, Orientation3, Rotation3, point};
use nalgebra::{Quaternion, Translation3, UnitQuaternion};
use projection::intrinsic::Intrinsic;
use serde::Deserialize;
use types::visual_localization::FieldMarkAssociation;

#[derive(Deserialize)]
struct Recording {
    field: FieldDimensions,
    reference: Reference,
    frames: Vec<RecordedFrame>,
}

#[derive(Deserialize)]
struct Reference {
    time: i64,
    field_yaw: f64,
    imu_rpy: [f64; 3],
}

#[derive(Deserialize)]
struct RecordedFrame {
    sequence: u64,
    time: i64,
    camera_translation: [f32; 3],
    camera_quaternion: [f32; 4],
    intrinsics: [f32; 4],
    imu: [ImuSample; 2],
    detections: Vec<Detection>,
    associations: Vec<Association>,
}

#[derive(Deserialize)]
struct ImuSample {
    time: i64,
    rpy: [f64; 3],
    #[serde(default)]
    gyro: [f64; 3],
}
#[derive(Deserialize)]
struct Detection {
    class: String,
    confidence: f32,
    pixel: [f32; 2],
}
#[derive(Deserialize)]
struct Association {
    pixel: [f32; 2],
    field: [f32; 2],
}

fn attitude(rpy: [f64; 3]) -> Orientation3<ImuReference, f64> {
    Orientation3::from_euler_angles(rpy[0], rpy[1], rpy[2])
}

fn pose(translation: [f32; 3], q: [f32; 4]) -> nalgebra::Isometry3<f32> {
    nalgebra::Isometry3::from_parts(
        Translation3::from(translation),
        UnitQuaternion::new_normalize(Quaternion::new(q[3], q[0], q[1], q[2])),
    )
}

impl RecordedFrame {
    fn exposure_attitude(&self) -> Orientation3<ImuReference, f64> {
        let [a, b] = &self.imu;
        assert!(a.time <= self.time && self.time < b.time);
        attitude(a.rpy).slerp(
            attitude(b.rpy),
            (self.time - a.time) as f64 / (b.time - a.time) as f64,
        )
    }

    fn leveling(&self) -> Rotation3<Robot, Ground> {
        let (roll, pitch, _) = self.exposure_attitude().euler_angles();
        Rotation3::from_euler_angles(roll as f32, pitch as f32, 0.0)
    }

    fn visual(&self) -> VisualLocalizationFrame {
        VisualLocalizationFrame {
            epoch: 4,
            source: VisualAssociationSource::Global,
            generation: 0,
            robot_to_camera: pose(self.camera_translation, self.camera_quaternion)
                .framed_transform(),
            camera_intrinsic: Intrinsic::new(
                nalgebra::vector![self.intrinsics[0], self.intrinsics[1]],
                point![self.intrinsics[2], self.intrinsics[3]],
            ),
            associations: self
                .associations
                .iter()
                .map(|a| FieldMarkAssociation {
                    detection: point![a.pixel[0], a.pixel[1]],
                    field_point: point![a.field[0], a.field[1], 0.0],
                })
                .collect(),
        }
    }

    fn features(&self) -> DetectedVisualFeatures {
        let mut features = DetectedVisualFeatures::default();
        for d in &self.detections {
            let destination = match d.class.as_str() {
                "LSpot" => &mut features.l_spots,
                "TSpot" => &mut features.t_spots,
                other => panic!("unexpected fixture class {other}"),
            };
            destination.push(DetectedVisualFeature {
                pixel: point![d.pixel[0], d.pixel[1]],
                confidence: d.confidence,
            });
        }
        features
    }
}

#[test]
fn recorded_frames_recover_with_heading_guided_association() {
    let recording: Recording = serde_json::from_str(include_str!("recovered_frames.json")).unwrap();
    verify_recording(recording);
}

fn verify_recording(recording: Recording) {
    let reference = HeadingReference::new(
        Time::from_nanos(recording.reference.time),
        Orientation2::new(recording.reference.field_yaw),
        attitude(recording.reference.imu_rpy),
    );
    let parameters: Localization3dParameters = json5::from_str(include_str!(
        "../../../../../etc/parameters/base/localization3d.json5"
    ))
    .unwrap();
    for recorded in &recording.frames {
        assert!(recorded.time > recording.reference.time);
        let features = recorded.features();
        let associate = |frame: &VisualLocalizationFrame, heading| {
            associate_global_visual_features(GlobalAssociationInput {
                visual_features: &features,
                robot_to_ground: recorded.leveling(),
                robot_to_camera: frame.robot_to_camera,
                camera_intrinsic: frame.camera_intrinsic,
                field_dimensions: &recording.field,
                parameters: &GlobalLocalizerParameters::default(),
                heading,
            })
        };
        let mut corrected = recorded.visual();
        let unconstrained = associate(&corrected, None);
        let expected = reference.expected(recorded.exposure_attitude());
        let heading = types::localization::HeadingConstraint {
            expected,
            max_error: parameters.max_heading_error,
        };
        // This isolated seed test defines Local's yaw gauge at exposure.
        let robot_to_local: Rotation3<Robot, Local> = Rotation3::wrap(recorded.leveling().inner);
        if recorded.sequence < 3737 {
            assert!(
                unconstrained.associations.is_empty(),
                "{} must reject ambiguous geometry with exposure tilt",
                recorded.sequence
            );
            assert!(
                crate::alignment::seed_recovery_alignment(
                    &mut corrected.clone(),
                    robot_to_local,
                    heading,
                    &parameters.visual,
                )
                .is_none(),
                "{} must also reject the old ~90-degree hypothesis downstream",
                recorded.sequence
            );
        }
        let associated = associate(
            &corrected,
            Some(types::localization::HeadingConstraint {
                expected,
                max_error: parameters.max_heading_error,
            }),
        );
        assert_eq!(
            associated.associations.len(),
            recorded.associations.len(),
            "{}: heading-aware matcher must recover",
            recorded.sequence
        );
        corrected.associations = associated.associations;
        let selected = corrected.associations.clone();
        {
            let seed = crate::alignment::seed_recovery_alignment(
                &mut corrected,
                robot_to_local,
                heading,
                &parameters.visual,
            );
            let (pose, alignment) =
                seed.expect("later recorded geometry has the correct half-turn");
            let field = Orientation2::<Field, f64>::new(
                (alignment.to_3d().inner.rotation * pose.inner.rotation)
                    .euler_angles()
                    .2 as f64,
            );
            let error = expected.rotation_to(field).inner.angle().abs();
            assert!(
                error < 3.0_f64.to_radians(),
                "recorded recovery heading error: {} degrees",
                error.to_degrees()
            );
            // Association must preserve the oriented solution, not canonicalize
            // it back to the opposite half and rely on downstream repair.
            for (a, original) in corrected.associations.iter().zip(&selected) {
                assert_eq!(a.field_point, original.field_point);
            }
            assert!(alignment.inner.translation.vector.x > 2.0);
            eprintln!(
                "MCAP frame {}: {} associations, heading error {:.3} degrees",
                recorded.sequence,
                corrected.associations.len(),
                error.to_degrees()
            );
            // Exercise candidate construction and post-solve heading acceptance too.
            // The direct-MCAP test supplies the actual bracketing gyroscope samples.
            let camera = CameraGeometry {
                robot_to_camera: corrected.robot_to_camera,
                intrinsics: corrected.camera_intrinsic,
            };
            let mut localization = Localization::new(
                Time::from_nanos(recorded.imu[0].time),
                4,
                &parameters,
                &recording.field,
                &camera,
                pose,
            )
            .unwrap();
            localization.status.state = LocalizationState::LostTrack;
            localization.status.time = Time::from_nanos(recording.reference.time);
            localization.heading_reference = Some(reference);
            for sample in &recorded.imu {
                localization
                    .ingest_imu(
                        Time::from_nanos(sample.time),
                        ImuState {
                            roll_pitch_yaw: linear_algebra::Vector3::wrap(
                                nalgebra::Vector3::from(sample.rpy).cast(),
                            ),
                            angular_velocity: linear_algebra::Vector3::wrap(
                                nalgebra::Vector3::from(sample.gyro).cast(),
                            ),
                            ..Default::default()
                        },
                    )
                    .unwrap();
            }
            localization
                .ingest_visual_localization_frame(TimeWrapper {
                    time: Time::from_nanos(recorded.time),
                    inner: corrected,
                })
                .unwrap();
            let solved = localization.solve(Time::from_nanos(recorded.imu[1].time));
            assert_eq!(
                localization.status.state,
                LocalizationState::Tracking,
                "frame {}: {:?}",
                recorded.sequence,
                solved.diagnostics
            );
            let estimate = solved.estimate.unwrap();
            let heading = Orientation2::new(
                estimate
                    .robot_to_field
                    .unwrap()
                    .pose
                    .inner
                    .rotation
                    .euler_angles()
                    .2,
            );
            let expected = reference.expected(attitude(recorded.imu[1].rpy));
            let optimized_error = expected.rotation_to(heading).inner.angle().abs();
            assert!(optimized_error < parameters.max_heading_error);
            assert_eq!(estimate.time.as_nanos(), recorded.imu[1].time);
            eprintln!(
                "  accepted optimized heading error: {:.3} degrees",
                optimized_error.to_degrees()
            );
        }
    }
}

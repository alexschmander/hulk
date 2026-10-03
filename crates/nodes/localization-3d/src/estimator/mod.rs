use std::{collections::BTreeMap, ops::Range};

use color_eyre::{Result, eyre::eyre};
use coordinate_systems::{Field, Local, Robot};
use fagra::{BatchKey, Problem, StateKey};
use linear_algebra::{Framed, Isometry3, Vector2, Vector3, vector};
use localization_fagra::{
    factors::{
        CameraIntrinsicsPrior, FieldContainment, FootGround, FootObservation, FrameReprojections,
        ImuBiasPrior, ImuBiasWalk, ImuKinematics, MotionPrior, PreintegratedImu,
        ReprojectionObservation, TrajectoryPrior,
    },
    variables::{CameraIntrinsics, FieldAlignment, ImuBias, PoseControl, TrajectoryState},
};
use nalgebra::SMatrix;
use projection::intrinsic::Intrinsic;
use ros_z::time::Time;
use types::field_dimensions::FieldDimensions;
use types::{time_wrapper::TimeWrapper, visual_localization_next::VisualLocalizationFrame};

use crate::parameters::Localization3dParameters;

mod attitude;
mod bias;
mod inertial;
mod preintegration;
mod recovery;
mod vision;

mod window;

fagra::states! {
    States {
        controls: PoseControl,
        alignments: FieldAlignment,
        intrinsics: CameraIntrinsics,
        biases: ImuBias,
    }
}

fagra::factors! {
    Factors {
        trajectory_priors: TrajectoryPrior,
        bias_priors: ImuBiasPrior,
        bias_walks: ImuBiasWalk,
        intrinsics_priors: CameraIntrinsicsPrior,
        motion: MotionPrior,
        yaw: localization_fagra::factors::RelativeYaw,
        containment: FieldContainment,
        imu: ImuKinematics,
        preintegrated_imu: PreintegratedImu,
        feet: Batch<FootGround, FootObservation>,
        reprojections: Batch<FrameReprojections, ReprojectionObservation>,
    }
}

type Graph = Problem<States, Factors>;

pub struct Estimator {
    graph: Graph,
    origin: Time,
    pub epoch: u64,
    generation: u64,
    controls: BTreeMap<i64, StateKey<PoseControl<f64>>>,
    biases: BTreeMap<i64, StateKey<ImuBias>>,
    preintegration: preintegration::ImuIntervals,
    foot_batches: BTreeMap<i64, BatchKey<FootGround<f64>>>,
    /// Accepted batches awaiting retirement; pending batches belong to PendingVisual.
    reprojection_batches: Vec<(i64, BatchKey<FrameReprojections<f64>>)>,
    intrinsics: StateKey<CameraIntrinsics<f64>>,
    latest_visual_frame: Option<TimeWrapper<VisualLocalizationFrame>>,
    pending_visuals: Vec<vision::PendingVisual>,
    measurements: BTreeMap<i64, usize>,
    attitudes: BTreeMap<Time, linear_algebra::Orientation3<coordinate_systems::ImuReference, f64>>,
    motion_history: Vec<recovery::MotionRecord>,
    history_start: Time,
    alignment: Option<StateKey<FieldAlignment<f64>>>,
    latest_time: Time,
    parameters: Localization3dParameters,
    field_half_extents: Vector2<Field, f64>,
}

impl Estimator {
    pub fn new(
        origin: Time,
        epoch: u64,
        initial_pose: Isometry3<Robot, Local>,
        intrinsics: &Intrinsic,
        parameters: Localization3dParameters,
        field: &FieldDimensions,
    ) -> Result<Self> {
        let mut estimator = Self::empty(
            origin,
            epoch,
            intrinsics,
            parameters,
            vector![
                field.length as f64 * 0.5 + field.border_strip_width as f64,
                field.width as f64 * 0.5 + field.border_strip_width as f64,
            ],
        )?;
        estimator.initialize_biases(origin, <ImuBias as fagra::Variable>::identity())?;
        let pose = PoseControl {
            pose: Framed::wrap(initial_pose.inner.cast()),
        };
        for index in -1..=2 {
            estimator
                .controls
                .insert(index, estimator.graph.add(pose.clone()));
        }
        estimator.add_anchor(
            origin,
            TrajectoryState {
                pose: pose.pose,
                velocity: Vector3::wrap(nalgebra::Vector3::zeros()),
            },
            Some(estimator.parameters.initial_height_sigma),
        )?;
        estimator.add_motion_prior(0, false)?;
        Ok(estimator)
    }

    fn empty(
        origin: Time,
        epoch: u64,
        camera_intrinsic: &Intrinsic,
        parameters: Localization3dParameters,
        field_half_extents: Vector2<Field, f64>,
    ) -> Result<Self> {
        parameters.validate().map_err(|message| eyre!(message))?;
        let mut graph = Graph::new();
        let intrinsics_value = CameraIntrinsics {
            focal_lengths: Framed::wrap(camera_intrinsic.focals.cast()),
            optical_center: Framed::wrap(camera_intrinsic.optical_center.inner.cast()),
        };
        let intrinsics = graph.add(intrinsics_value.clone());
        graph.add_factor(CameraIntrinsicsPrior {
            intrinsics,
            reference: intrinsics_value,
            information_root: SMatrix::identity() / parameters.model.intrinsic_prior_sigma,
        })?;

        Ok(Self {
            graph,
            origin,
            epoch,
            generation: 0,
            controls: BTreeMap::new(),
            biases: BTreeMap::new(),
            preintegration: preintegration::ImuIntervals::new(&parameters.timing),
            foot_batches: BTreeMap::new(),
            reprojection_batches: Vec::new(),
            intrinsics,
            latest_visual_frame: None,
            pending_visuals: Vec::new(),
            measurements: BTreeMap::new(),
            attitudes: BTreeMap::new(),
            motion_history: Vec::new(),
            history_start: origin,
            alignment: None,
            latest_time: origin,
            parameters,
            field_half_extents,
        })
    }

    fn add_anchor(
        &mut self,
        time: Time,
        reference: TrajectoryState,
        height_sigma: Option<f64>,
    ) -> Result<()> {
        let (segment, tau) = self.segment_and_tau(time)?;
        let mut root = SMatrix::<f64, 9, 9>::zeros();
        let rotation = reference.pose.inner.rotation.to_rotation_matrix();
        // The residual is in reference body axes. Only Local XY/yaw fix gauge;
        // height and tilt are physical quantities constrained by sensor evidence.
        root.fixed_view_mut::<1, 3>(2, 0)
            .copy_from(&(rotation.matrix().row(2) / self.parameters.model.anchor_yaw_sigma));
        root.fixed_view_mut::<2, 3>(6, 6).copy_from(
            &(rotation.matrix().fixed_rows::<2>(0) / self.parameters.model.anchor_xy_sigma),
        );
        root.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(rotation.matrix() / self.parameters.initial_velocity_sigma));
        // Broad provisional startup height only; never count a visual seed as
        // another height observation alongside those same reprojections.
        if let Some(sigma) = height_sigma {
            root.fixed_view_mut::<1, 3>(8, 6)
                .copy_from(&(rotation.matrix().row(2) / sigma));
        }
        self.graph.add_factor(TrajectoryPrior {
            controls: control_keys(&self.controls, segment)?,
            duration: self.parameters.timing.trajectory_spacing.as_secs_f64(),
            tau,
            reference,
            information_root: root,
        })?;
        Ok(())
    }

    pub fn latest_time(&self) -> Time {
        self.latest_time
    }

    pub fn update_parameters(&mut self, parameters: Localization3dParameters) -> Result<()> {
        self.parameters
            .validate_update(&parameters)
            .map_err(|message| eyre!(message))?;
        self.parameters = parameters;
        Ok(())
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    fn segments(&self) -> Range<i64> {
        // Four consecutive controls support each segment. Growth and retirement
        // only change the ends of this contiguous range.
        let start = self
            .controls
            .first_key_value()
            .map_or(0, |(&index, _)| index + 1);
        let end = self
            .controls
            .last_key_value()
            .map_or(0, |(&index, _)| index - 1);
        start..end
    }
}

fn control_keys(
    controls: &BTreeMap<i64, StateKey<PoseControl<f64>>>,
    segment: i64,
) -> Result<[StateKey<PoseControl<f64>>; 4]> {
    let get = |index| {
        controls
            .get(&index)
            .copied()
            .ok_or_else(|| eyre!("missing control {index}"))
    };
    Ok([
        get(segment - 1)?,
        get(segment)?,
        get(segment + 1)?,
        get(segment + 2)?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use linear_algebra::{IntoTransform, point};
    use std::time::Duration;

    pub(super) fn estimator() -> Estimator {
        Estimator::new(
            Time::from_nanos(1_000_000_000),
            7,
            Isometry3::identity(),
            &Intrinsic::default(),
            json5::from_str(include_str!("../parameters_fixture.json5")).unwrap(),
            &FieldDimensions::SPL_2025,
        )
        .unwrap()
    }

    #[test]
    fn visual_ingestion_accepts_behind_camera_predictions_but_not_zero_range() {
        use types::visual_localization_next::FieldMarkAssociation;
        let mut estimator = estimator();
        estimator.alignment = Some(estimator.graph.add(FieldAlignment {
            local_to_field: nalgebra::Isometry2::identity().framed_transform(),
        }));
        let time = Time::from_nanos(1_000_000_000);
        let mut frame = TimeWrapper {
            time,
            inner: VisualLocalizationFrame {
                epoch: 7,
                source: types::visual_localization_next::VisualAssociationSource::Tracking,
                generation: 0,
                robot_to_camera: Isometry3::from_translation(0.0, 0.0, -1.0),
                camera_intrinsic: Intrinsic::new(
                    nalgebra::vector![300.0, 300.0],
                    point![160.0, 120.0],
                ),
                associations: [
                    (point![-1.0, -1.0, 0.0], point![150.0, 110.0]),
                    (point![1.0, -1.0, 0.0], point![170.0, 110.0]),
                    (point![0.0, 0.0, 0.0], point![160.0, 130.0]),
                ]
                .map(|(field_point, detection)| FieldMarkAssociation {
                    field_point,
                    detection,
                })
                .to_vec(),
            },
        };
        assert!(estimator.ingest_visual(frame.clone()).unwrap());
        // A finite bearing objective does not authorize a physically invalid tracking result.
        assert!(estimator.validate_pending_visuals().is_err());
        assert!(estimator.reprojection_batches.is_empty());
        frame.inner.robot_to_camera = Isometry3::identity();
        assert!(!estimator.ingest_visual(frame).unwrap());
        estimator.discard_visuals().unwrap();
        assert!(estimator.pending_visuals.is_empty());
        assert!(estimator.reprojection_batches.is_empty());
        assert_eq!(estimator.measurements.values().sum::<usize>(), 0);
    }

    #[test]
    fn configured_grids_preserve_stationary_seed_and_window_boundary() {
        let mut parameters: Localization3dParameters =
            json5::from_str(include_str!("../parameters_fixture.json5")).unwrap();
        parameters.timing.trajectory_spacing = Duration::from_millis(400);
        parameters.timing.bias_spacing = Duration::from_millis(1200);
        parameters.timing.optimization_window = Duration::from_millis(2500);
        let origin = Time::from_nanos(1_000_000_000);
        let mut estimator = Estimator::new(
            origin,
            7,
            Isometry3::from_translation(0.0, 0.0, 0.5),
            &Intrinsic::default(),
            parameters,
            &FieldDimensions::SPL_2025,
        )
        .unwrap();
        let latest = origin + Duration::from_millis(3600);
        let (segment, tau) = estimator.segment_and_tau(latest).unwrap();
        assert_eq!((segment, tau), (9, 0.0));
        let controls = estimator.ensure_segment(segment).unwrap();
        let state = estimator.spline(controls).unwrap().state(0.375).unwrap();
        assert!((state.pose.inner.translation.z - 0.5).abs() < 1e-12);
        assert!(state.velocity.norm() < 1e-12);
        let (_, bias_tau) = estimator.ensure_biases(latest).unwrap();
        assert_eq!(bias_tau, 0.0);
        assert_eq!(estimator.biases.len(), 5);
        estimator.commit_time(latest);
        estimator.commit_time(origin);
        assert_eq!(estimator.latest_time(), latest);
        assert_eq!(estimator.oldest_window_segment(), 2);
        let cutoff = latest - Duration::from_millis(2500);
        assert!(estimator.check_time(cutoff, "test").unwrap().is_some());
        assert!(
            estimator
                .check_time(cutoff - Duration::from_nanos(1), "test")
                .unwrap()
                .is_none()
        );
        assert!(
            estimator
                .segment_and_tau(origin - Duration::from_nanos(1))
                .is_err()
        );
    }
}

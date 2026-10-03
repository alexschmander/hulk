use std::{
    collections::BTreeMap,
    ops::Range,
    time::{Duration, Instant},
};

use color_eyre::{Result, eyre::eyre};
use coordinate_systems::{Field, Local, Robot};
use fagra::{
    BatchKey, DenseNormalCholesky, LevenbergMarquardt, OptimizeOptions, Problem, SolverError,
    StateKey,
};
use linear_algebra::{Framed, Isometry3, Vector2, Vector3, vector};
use localization_fagra::{
    factors::{
        AdjacentKinematicOdometry, AdjacentVisualOdometry, CameraIntrinsicsPrior, FieldContainment,
        FootGround, FootObservation, FrameReprojections, ImuBiasPrior, ImuBiasWalk, ImuKinematics,
        KinematicOdometry, MotionPrior, PreintegratedImu, ReprojectionObservation, TrajectoryPrior,
        VisualOdometry, VisualOdometryObservation,
    },
    variables::{CameraIntrinsics, FieldAlignment, ImuBias, PoseControl, TrajectoryState},
};
use nalgebra::SMatrix;
use projection::intrinsic::Intrinsic;
use ros_z::time::Time;
use types::{
    field_dimensions::FieldDimensions, localization::LocalizationEstimate,
    time_wrapper::TimeWrapper, visual_localization_next::VisualLocalizationFrame,
};

use crate::heading::HeadingReference;
use crate::{diagnostics::SolveDiagnostics, parameters::Localization3dParameters};

mod attitude;
mod bias;
mod covariance;
mod inertial;
mod kinematic_odometry;
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
        adjacent_odometry: AdjacentVisualOdometry,
        kinematic_odometry: KinematicOdometry,
        adjacent_kinematic_odometry: AdjacentKinematicOdometry,
        imu: ImuKinematics,
        preintegrated_imu: PreintegratedImu,
        feet: Batch<FootGround, FootObservation>,
        reprojections: Batch<FrameReprojections, ReprojectionObservation>,
        odometry: Batch<VisualOdometry, VisualOdometryObservation>,
    }
}

type Graph = Problem<States, Factors>;

pub struct SolveResult {
    pub estimate: Option<LocalizationEstimate>,
    pub diagnostics: SolveDiagnostics,
    pub converged: bool,
    pub motion_invalid: bool,
    pub visual_rejected: bool,
}

pub struct Estimator {
    graph: Graph,
    optimizer: LevenbergMarquardt<DenseNormalCholesky>,
    options: OptimizeOptions<f64>,
    origin: Time,
    epoch: u64,
    generation: u64,
    controls: BTreeMap<i64, StateKey<PoseControl<f64>>>,
    biases: BTreeMap<i64, StateKey<ImuBias>>,
    preintegration: preintegration::ImuIntervals,
    foot_batches: BTreeMap<i64, BatchKey<FootGround<f64>>>,
    odometry_batches: BTreeMap<i64, BatchKey<VisualOdometry<f64>>>,
    /// Accepted batches awaiting retirement; pending batches belong to PendingVisual.
    reprojection_batches: Vec<(i64, BatchKey<FrameReprojections<f64>>)>,
    alignment: Option<StateKey<FieldAlignment<f64>>>,
    intrinsics: StateKey<CameraIntrinsics<f64>>,
    latest_time: Time,
    last_converged_time: Option<Time>,
    latest_vo_epoch: Option<u64>,
    latest_kinematic_time: Option<Time>,
    measurements: BTreeMap<i64, usize>,
    latest_visual_frame: Option<TimeWrapper<VisualLocalizationFrame>>,
    pending_visuals: Vec<vision::PendingVisual>,
    attitudes: BTreeMap<Time, linear_algebra::Orientation3<coordinate_systems::ImuReference, f64>>,
    yaw_factors: BTreeMap<i64, fagra::FactorKey<localization_fagra::factors::RelativeYaw>>,
    current_yaw: Option<fagra::FactorKey<localization_fagra::factors::RelativeYaw>>,
    parameters: Localization3dParameters,
    field_half_extents: Vector2<Field, f64>,
    motion_history: Vec<recovery::MotionRecord>,
    history_start: Time,
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
            optimizer: {
                let mut optimizer = LevenbergMarquardt::new(DenseNormalCholesky::default());
                optimizer.options.max_trials = parameters.solver.max_trials;
                optimizer
            },
            options: OptimizeOptions {
                max_iterations: parameters.solver.max_iterations,
                gradient_tolerance: parameters.solver.gradient_tolerance,
                step_tolerance: parameters.solver.step_tolerance,
                cost_tolerance: parameters.solver.cost_tolerance,
            },
            origin,
            epoch,
            generation: 0,
            controls: BTreeMap::new(),
            biases: BTreeMap::new(),
            preintegration: preintegration::ImuIntervals::new(&parameters.timing),
            foot_batches: BTreeMap::new(),
            odometry_batches: BTreeMap::new(),
            reprojection_batches: Vec::new(),
            alignment: None,
            intrinsics,
            latest_time: origin,
            last_converged_time: None,
            latest_vo_epoch: None,
            latest_kinematic_time: None,
            measurements: BTreeMap::new(),
            latest_visual_frame: None,
            pending_visuals: Vec::new(),
            attitudes: BTreeMap::new(),
            yaw_factors: BTreeMap::new(),
            current_yaw: None,
            parameters,
            motion_history: Vec::new(),
            history_start: origin,
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

    pub fn update_parameters(&mut self, parameters: Localization3dParameters) -> Result<()> {
        self.parameters
            .validate_update(&parameters)
            .map_err(|message| eyre!(message))?;
        self.optimizer.options.max_trials = parameters.solver.max_trials;
        self.options = OptimizeOptions {
            max_iterations: parameters.solver.max_iterations,
            gradient_tolerance: parameters.solver.gradient_tolerance,
            step_tolerance: parameters.solver.step_tolerance,
            cost_tolerance: parameters.solver.cost_tolerance,
        };
        self.parameters = parameters;
        Ok(())
    }

    #[cfg(test)]
    pub fn solve(&mut self) -> SolveResult {
        self.solve_with_heading(None)
    }

    pub fn solve_with_heading(&mut self, heading: Option<&HeadingReference>) -> SolveResult {
        let had_visual_update = !self.pending_visuals.is_empty();
        let mut visual_rejected = false;
        let start = Instant::now();
        let mut diagnostics = SolveDiagnostics {
            time: self.latest_time,
            epoch: self.epoch,
            duration: Duration::ZERO,
            estimation_duration: Duration::ZERO,
            ingestion_duration: Duration::ZERO,
            iterations: None,
            lm_attempts: 0,
            lm_rejected_steps: 0,
            gradient_norm: None,
            motion_rebuilt: false,
            initial_cost: None,
            final_cost: None,
            termination: "failed".into(),
            state_count: 0,
            measurement_count: 0,
            failure: None,
            imu_bias: None,
        };
        let result = (|| -> Result<_> {
            let result = self
                .prepare_preintegration()
                .and_then(|()| self.prepare_attitude())
                .and_then(|()| {
                    let mut result = self.solve_graph(&mut diagnostics, heading);
                    if result.is_err() && had_visual_update {
                        self.discard_visuals()?;
                        visual_rejected = true;
                        result = self.solve_graph(&mut diagnostics, heading);
                    }
                    result
                });
            if result.is_ok() {
                self.accept_visuals();
            } else {
                self.discard_visuals()?;
            }
            if let Some(key) = self.current_yaw.take() {
                self.graph.remove_factor(key)?;
            }
            // Retire the restored graph after rejection too; continuous motion
            // must not grow an unbounded graph while field updates are withheld.
            self.retire_old_segments()?;
            result
        })();
        let (estimate, converged) = match result {
            Ok((estimate, converged)) => (Some(estimate), converged),
            Err(error) => {
                diagnostics.failure = Some(format!("{error:#}"));
                (None, false)
            }
        };
        if converged {
            self.last_converged_time = Some(self.latest_time);
        }
        if estimate.is_none() {
            diagnostics.imu_bias = None;
        }
        let motion_invalid = estimate.is_none() && self.alignment.is_some();
        diagnostics.duration = start.elapsed();
        diagnostics.state_count =
            self.controls.len() + self.biases.len() + 1 + usize::from(self.alignment.is_some());
        diagnostics.measurement_count = self.measurements.values().sum();
        SolveResult {
            estimate,
            diagnostics,
            converged,
            motion_invalid,
            visual_rejected,
        }
    }

    fn solve_graph(
        &mut self,
        diagnostics: &mut SolveDiagnostics,
        heading: Option<&HeadingReference>,
    ) -> Result<(LocalizationEstimate, bool)> {
        let (segment, tau) = self.segment_and_tau(self.latest_time)?;
        let controls = self.ensure_segment(segment)?;
        self.ensure_biases(self.latest_time)?;
        // LM rejects bad trials. Whole-update rollback is still needed when
        // application validation fails after otherwise accepted LM steps.
        let controls_before = self
            .controls
            .values()
            .map(|key| Ok((*key, self.graph.get(*key)?.clone())))
            .collect::<Result<Vec<_>>>()?;
        let alignment_before = self
            .alignment
            .map(|key| self.graph.get(key).cloned())
            .transpose()?;
        let intrinsics_before = self.graph.get(self.intrinsics)?.clone();
        let biases_before = self
            .biases
            .values()
            .map(|key| Ok((*key, self.graph.get(*key)?.clone())))
            .collect::<Result<Vec<_>>>()?;
        let blocks = self.estimate_covariance_blocks(controls)?;
        // LM's final gradient check already built the undamped model. Extract the
        // joint marginal from that model instead of visiting every factor again.
        let result = self
            .optimizer
            .solve_batch_with_covariance(
                &mut self.graph,
                &self.options,
                &blocks,
                &Default::default(),
            )
            .map(|(report, covariance)| {
                let joint = covariance::EstimateCovariance::from_fn(|r, c| {
                    if r < covariance.nrows() && c < covariance.ncols() {
                        covariance[(r, c)]
                    } else {
                        0.0
                    }
                });
                (report, joint)
            });
        let statistics = self.optimizer.statistics();
        diagnostics.lm_attempts = statistics.attempts;
        diagnostics.lm_rejected_steps = statistics.rejected_steps;
        diagnostics.gradient_norm = statistics.gradient_norm;
        let converged = result.is_ok();
        diagnostics.iterations = statistics.cost.map(|_| statistics.accepted_steps);
        diagnostics.final_cost = statistics.cost;
        // This fagra version returns the initial cost only in a successful report.
        // On failure after accepted steps it is unknown, not a reason to rebuild H.
        diagnostics.initial_cost = result
            .as_ref()
            .ok()
            .map(|(report, _)| report.initial_cost)
            .or_else(|| statistics.cost.filter(|_| statistics.accepted_steps == 0));
        diagnostics.termination = match &result {
            Ok((report, _)) => format!("{:?}", report.termination),
            Err(SolverError::NoProgress) => "NoProgress".into(),
            Err(SolverError::NoConvergence) => "MaxIterations".into(),
            // A converged numerical solve can still fail covariance rank checks.
            Err(_)
                if statistics
                    .gradient_norm
                    .is_some_and(|g| g <= self.options.gradient_tolerance) =>
            {
                "GradientTolerance".into()
            }
            Err(_) => "failed".into(),
        };
        let checked = (|| -> Result<_> {
            let covariance = match result {
                Ok((_, covariance)) => Some(covariance),
                // LM only commits cost-decreasing steps. Its recorded accepted
                // cost replaces the old evaluation-only solve on these paths.
                Err(SolverError::NoConvergence | SolverError::NoProgress)
                    if statistics.cost.is_some() =>
                {
                    None
                }
                Err(error) => return Err(error.into()),
            };
            if let Some(heading) = heading
                && self.alignment.is_some()
            {
                self.validate_heading(heading)?;
            }
            if !converged && !self.pending_visuals.is_empty() {
                return Err(eyre!("visual update did not converge"));
            }
            self.validate_pending_visuals()?;
            self.validate_tilt()?;
            let covariance = match covariance {
                Some(covariance) => covariance,
                None => {
                    // Accepted partial motion still needs a fresh, undamped
                    // covariance. Failed optimization has no cached-query API.
                    let covariance = self.graph.joint_covariance(&blocks)?;
                    covariance::EstimateCovariance::from_fn(|r, c| {
                        if r < covariance.nrows() && c < covariance.ncols() {
                            covariance[(r, c)]
                        } else {
                            0.0
                        }
                    })
                }
            };
            self.estimate_from_covariance(controls, tau, &covariance, diagnostics)
        })();
        let estimate = match checked {
            Ok(result) => result,
            Err(error) => {
                for (key, value) in controls_before {
                    self.graph.set(key, value)?;
                }
                if let Some((key, value)) = self.alignment.zip(alignment_before) {
                    self.graph.set(key, value)?;
                }
                self.graph.set(self.intrinsics, intrinsics_before)?;
                for (key, value) in biases_before {
                    self.graph.set(key, value)?;
                }
                return Err(error);
            }
        };
        Ok((estimate, converged))
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
    use booster::ImuState;
    use std::time::Duration;

    use linear_algebra::{IntoTransform, point};
    use projection::intrinsic::Intrinsic;
    use types::camera_geometry::CameraGeometry;

    use super::*;

    pub(super) fn estimator() -> Estimator {
        let camera = CameraGeometry {
            intrinsics: Intrinsic::new(nalgebra::vector![300.0, 300.0], point![160.0, 120.0]),
            ..Default::default()
        };
        Estimator::new(
            Time::from_nanos(1_000_000_000),
            7,
            nalgebra::Isometry3::identity().framed_transform(),
            &camera.intrinsics,
            Localization3dParameters {
                timing: Default::default(),
                model: Default::default(),
                solver: Default::default(),
                visual: Default::default(),
                imu_preintegration: Default::default(),
                imu_bias: Default::default(),
                kinematic_odometry_noise: Some(Default::default()),
                accelerometer: None,
                initial_height_sigma: 1.0,
                initial_velocity_sigma: 5.0,
                max_tilt_error: 20.0_f64.to_radians(),
                accelerometer_process_noise_variance: 10.0,
                visual_feature_noise_variance: 100.0,
                field_containment_sigma: 1.0,
                max_heading_error: 20.0_f64.to_radians(),
            },
            &FieldDimensions::SPL_2025,
        )
        .unwrap()
    }

    #[test]
    fn configured_foot_noise_changes_measurement_weight() {
        let template = estimator();
        let costs = [0.01, 0.04].map(|sigma| {
            let mut parameters = template.parameters.clone();
            parameters.model.foot_sigma = sigma;
            let mut estimator = Estimator::new(
                template.origin,
                template.epoch,
                Isometry3::identity(),
                &Intrinsic::default(),
                parameters,
                &FieldDimensions::SPL_2025,
            )
            .unwrap();
            estimator
                .insert_feet(
                    estimator.origin,
                    point![0.0, 0.0, -0.02],
                    point![0.0, 0.0, -0.02],
                )
                .unwrap();
            let evaluate = OptimizeOptions {
                gradient_tolerance: f64::MAX,
                ..estimator.options
            };
            estimator
                .optimizer
                .solve_batch(&mut estimator.graph, &evaluate)
                .unwrap()
                .initial_cost
        });
        // Two 2-cm penetrations: quadrupling sigma divides their likelihood cost by 16.
        assert!((costs[0] - 4.0).abs() < 1.0e-10);
        assert!((costs[0] / costs[1] - 16.0).abs() < 1.0e-10);
    }

    #[test]
    fn stationary_imu_produces_timestamped_local_estimate() {
        let mut estimator = estimator();
        // The initial graph is underconstrained: a completed numerical report
        // must survive covariance failure and must not leak into the next attempt.
        let failed = estimator.solve();
        assert!(failed.estimate.is_none());
        assert!(failed.diagnostics.failure.is_some());
        assert_eq!(failed.diagnostics.initial_cost, Some(0.0));
        assert_eq!(failed.diagnostics.final_cost, Some(0.0));
        assert_eq!(failed.diagnostics.iterations, Some(0));
        for index in 0..100 {
            let time = Time::from_nanos(1_000_000_000 + index * 2_000_000);
            assert!(estimator.ingest_imu(time, ImuState::default(),).unwrap());
        }
        let result = estimator.solve();
        assert!(
            result.diagnostics.failure.is_none(),
            "{:?}",
            result.diagnostics
        );
        let estimate = result.estimate.unwrap();
        assert_eq!(estimate.time, Time::from_nanos(1_198_000_000));
        assert_eq!(estimate.epoch, 7);
        assert!(estimate.robot_to_field.is_none());
        assert!(
            estimate
                .robot_to_local
                .covariance
                .iter()
                .all(|value| value.is_finite())
        );
        // A second solve must not retain the previous temporary yaw constraint.
        let repeated = estimator.solve().estimate.unwrap();
        assert!(
            (estimate.robot_to_local.covariance - repeated.robot_to_local.covariance).amax()
                < 1.0e-10
        );
        // Crossing the knot boundary must retain observability.
        let boundary = Time::from_nanos(1_200_000_000);
        estimator.ingest_imu(boundary, ImuState::default()).unwrap();
        assert_eq!(estimator.solve().estimate.unwrap().time, boundary);
    }

    #[test]
    fn cached_covariance_matches_fresh_model_and_partial_solves_remain_usable() {
        let mut estimator = estimator();
        for index in 0..100 {
            estimator
                .ingest_imu(
                    estimator.origin + Duration::from_millis(index * 2),
                    ImuState {
                        roll_pitch_yaw: vector![0.02, -0.01, 0.0],
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        estimator.prepare_preintegration().unwrap();
        estimator.prepare_attitude().unwrap();
        let (segment, _) = estimator.segment_and_tau(estimator.latest_time).unwrap();
        let controls = control_keys(&estimator.controls, segment).unwrap();
        let blocks = estimator.estimate_covariance_blocks(controls).unwrap();
        let (_, cached) = estimator
            .optimizer
            .solve_batch_with_covariance(
                &mut estimator.graph,
                &estimator.options,
                &blocks,
                &Default::default(),
            )
            .unwrap();
        let cached = covariance::EstimateCovariance::from_fn(|r, c| {
            if r < cached.nrows() && c < cached.ncols() {
                cached[(r, c)]
            } else {
                0.0
            }
        });
        let fresh = estimator.graph.joint_covariance(&blocks).unwrap();
        for r in 0..fresh.nrows() {
            for c in 0..fresh.ncols() {
                assert!(
                    (fresh[(r, c)] - cached[(r, c)]).abs() < 1e-9 * (1.0 + fresh[(r, c)].abs())
                );
            }
        }
        // Force a cost-decreasing partial result after changing the observation.
        // That path must compute a fresh covariance and preserve its termination,
        // rather than treating a gradient-tolerance=MAX probe as convergence.
        if let Some(key) = estimator.current_yaw.take() {
            estimator.graph.remove_factor(key).unwrap();
        }
        estimator
            .ingest_imu(
                estimator.latest_time + Duration::from_millis(2),
                ImuState {
                    roll_pitch_yaw: vector![0.03, -0.02, 0.0],
                    ..Default::default()
                },
            )
            .unwrap();
        estimator.options.max_iterations = 1;
        estimator.options.gradient_tolerance = 1e-30;
        let partial = estimator.solve();
        assert!(partial.estimate.is_some(), "{:?}", partial.diagnostics);
        assert!(!partial.converged);
        assert_eq!(partial.diagnostics.iterations, Some(1));
        assert!(partial.diagnostics.final_cost.is_some());
        assert!(partial.diagnostics.initial_cost.is_none());
    }

    #[test]
    fn late_imu_rebuild_preserves_cost_and_information() {
        let build = |late: bool| {
            let mut estimator = estimator();
            estimator.parameters.accelerometer = Some(Default::default());
            let mut indices: Vec<_> = (0..100).collect();
            if late {
                indices.swap(24, 70);
            }
            for index in indices {
                estimator
                    .ingest_imu(
                        estimator.origin + Duration::from_millis(index * 2),
                        ImuState {
                            angular_velocity: vector![0.1, -0.2, index as f32 * 0.001],
                            linear_acceleration: vector![0.2, -0.1, 9.81],
                            ..Default::default()
                        },
                    )
                    .unwrap();
                if late && index == 70 {
                    estimator.prepare_preintegration().unwrap();
                }
            }
            estimator.prepare_preintegration().unwrap();
            estimator.prepare_attitude().unwrap();
            estimator
        };
        let mut grouped = build(false);
        let mut singletons = build(true);
        let evaluate = OptimizeOptions {
            gradient_tolerance: f64::MAX,
            ..grouped.options
        };
        let grouped_cost = grouped
            .optimizer
            .solve_batch(&mut grouped.graph, &evaluate)
            .unwrap()
            .initial_cost;
        let singleton_cost = singletons
            .optimizer
            .solve_batch(&mut singletons.graph, &evaluate)
            .unwrap()
            .initial_cost;
        assert!((grouped_cost - singleton_cost).abs() < 1e-12);
        let grouped_controls = control_keys(&grouped.controls, 0).unwrap();
        let singleton_controls = control_keys(&singletons.controls, 0).unwrap();
        let grouped_blocks = grouped
            .estimate_covariance_blocks(grouped_controls)
            .unwrap();
        let singleton_blocks = singletons
            .estimate_covariance_blocks(singleton_controls)
            .unwrap();
        let a = grouped.graph.joint_covariance(&grouped_blocks).unwrap();
        let b = singletons
            .graph
            .joint_covariance(&singleton_blocks)
            .unwrap();
        for r in 0..a.nrows() {
            for c in 0..a.ncols() {
                assert!((a[(r, c)] - b[(r, c)]).abs() < 1e-9 * (1.0 + a[(r, c)].abs()));
            }
        }
    }

    #[test]
    fn measurements_older_than_window_are_rejected() {
        let mut estimator = estimator();
        estimator
            .ingest_imu(Time::from_nanos(4_000_000_000), ImuState::default())
            .unwrap();
        assert!(
            !estimator
                .ingest_imu(Time::from_nanos(1_500_000_000), ImuState::default())
                .unwrap()
        );
    }

    #[test]
    fn kinematic_odometry_stops_blind_motion_and_survives_marginalization() {
        use types::odometry::KinematicOdometryDelta;
        let run = |observe_stop: bool| {
            let mut estimator = estimator();
            let origin = estimator.origin;
            let mut last_delta = None;
            for index in 0..=80 {
                let time = origin + Duration::from_millis(index * 50);
                estimator.ingest_imu(time, ImuState::default()).unwrap();
                if index > 0 && (index <= 20 || observe_stop) {
                    let delta = KinematicOdometryDelta {
                        previous_time: time - Duration::from_millis(50),
                        time,
                        current_to_previous: linear_algebra::Isometry2::from_parts(
                            vector![if index <= 20 { 0.02 } else { 0.0 }, 0.0],
                            0.0,
                        ),
                    };
                    assert!(estimator.ingest_kinematic_odometry(delta).unwrap());
                    last_delta = Some(delta);
                }
                if index == 0 {
                    continue;
                }
                let solved = estimator.solve();
                assert!(
                    solved.diagnostics.failure.is_none(),
                    "{:?}",
                    solved.diagnostics
                );
            }
            assert!(
                !estimator
                    .ingest_kinematic_odometry(last_delta.unwrap())
                    .unwrap()
            );
            let (segment, tau) = estimator.segment_and_tau(estimator.latest_time).unwrap();
            let controls = control_keys(&estimator.controls, segment).unwrap();
            estimator.spline(controls).unwrap().state(tau).unwrap()
        };
        let stopped = run(true);
        assert!(stopped.velocity.norm() < 0.02, "{stopped:?}");
        assert!(
            (stopped.pose.inner.translation.vector.x - 0.4).abs() < 0.05,
            "{stopped:?}"
        );
        let unobserved = run(false);
        assert!(unobserved.velocity.x() > 0.2, "{unobserved:?}");
        assert!(
            unobserved.pose.inner.translation.vector.x > 1.0,
            "{unobserved:?}"
        );
    }

    #[test]
    fn head_motion_odometry_does_not_move_stationary_body_without_ground_geometry() {
        use types::visual_odometry::{VisualOdometer, VisualOdometryDelta};

        let mut estimator = estimator();
        let camera = |angle| CameraGeometry {
            robot_to_camera: Isometry3::wrap(
                nalgebra::Isometry3::from_parts(
                    nalgebra::Translation3::new(0.05, 0.0, 0.25),
                    nalgebra::UnitQuaternion::from_euler_angles(0.0, angle, 0.0),
                )
                .inverse(),
            ),
            ..Default::default()
        };
        let mut previous = camera(0.0);
        for index in 0..100 {
            let time = Time::from_nanos(1_000_000_000 + index * 2_000_000);
            estimator.ingest_imu(time, ImuState::default()).unwrap();
            if index > 0 {
                let current = camera(index as f32 * 0.005);
                let delta =
                    previous.robot_to_camera.inner * current.robot_to_camera.inner.inverse();
                assert!(
                    estimator
                        .ingest_visual_odometry(
                            VisualOdometer {
                                time,
                                epoch: 0,
                                delta: Some(VisualOdometryDelta {
                                    previous_time: time - Duration::from_millis(2),
                                    current_left_camera_to_previous_left_camera: delta,
                                }),
                                current_left_camera_to_visual_odometer: current
                                    .robot_to_camera
                                    .inner
                                    .inverse(),
                            },
                            Some(&previous),
                            Some(&current),
                        )
                        .unwrap()
                );
                previous = current;
            }
        }
        let result = estimator.solve();
        assert!(
            result.diagnostics.failure.is_none(),
            "{:?}",
            result.diagnostics
        );
        let pose = result.estimate.unwrap().robot_to_local.pose.inner;
        assert!(pose.translation.vector.norm() < 1e-5, "{pose:?}");
        assert!(pose.rotation.angle() < 1e-5, "{pose:?}");
    }

    #[test]
    fn solve_retires_controls_outside_two_second_window() {
        let mut estimator = estimator();
        for index in 0..=110 {
            estimator
                .ingest_imu(
                    Time::from_nanos(1_000_000_000 + index * 20_000_000),
                    ImuState::default(),
                )
                .unwrap();
        }
        estimator.solve().estimate.unwrap();
        assert_eq!(estimator.controls.first_key_value().unwrap().0, &0);
        assert_eq!(estimator.segments(), 1..12);
        // Late data inside the retained window changes the graph, not the output time.
        assert!(
            estimator
                .ingest_imu(Time::from_nanos(2_810_000_000), ImuState::default())
                .unwrap()
        );
        assert_eq!(
            estimator.solve().estimate.unwrap().time,
            Time::from_nanos(3_200_000_000)
        );
    }

    #[test]
    fn imu_bias_learns_from_motion_evidence_and_stays_bounded_across_retirement() {
        use types::visual_odometry::{VisualOdometer, VisualOdometryDelta};
        let mut estimator = estimator();
        estimator.parameters.accelerometer = Some(Default::default());
        estimator.parameters.kinematic_odometry_noise = None;
        let camera = CameraGeometry::default();
        let origin = estimator.origin;
        let bias_at = |t: f64| ImuBias {
            gyroscope: vector![0.003 + 0.00005 * t, -0.004, 0.006],
            accelerometer: vector![0.025 + 0.0001 * t, -0.03, 0.015],
        };
        let mut last_bias = None;
        let mut last_pose = None;
        for index in 0..=2000 {
            let t = index as f64 * 0.01;
            let time = origin + Duration::from_millis(index * 10);
            let truth_bias = bias_at(t);
            estimator
                .ingest_imu(
                    time,
                    ImuState {
                        angular_velocity: Vector3::wrap(truth_bias.gyroscope.inner.cast()),
                        linear_acceleration: Vector3::wrap(
                            (truth_bias.accelerometer.inner + nalgebra::vector![0.1, 0.0, 9.81])
                                .cast(),
                        ),
                        ..Default::default()
                    },
                )
                .unwrap();
            if index > 0 && index % 2 == 0 {
                estimator
                    .ingest_visual_odometry(
                        VisualOdometer {
                            time,
                            epoch: 0,
                            delta: Some(VisualOdometryDelta {
                                previous_time: time - Duration::from_millis(20),
                                current_left_camera_to_previous_left_camera:
                                    nalgebra::Isometry3::translation(
                                        (0.05 * (t * t - (t - 0.02).powi(2))) as f32,
                                        0.0,
                                        0.0,
                                    ),
                            }),
                            current_left_camera_to_visual_odometer: nalgebra::Isometry3::identity(),
                        },
                        Some(&camera),
                        Some(&camera),
                    )
                    .unwrap();
            }
            if index > 0 && index % 5 == 0 {
                let solved = estimator.solve();
                let estimate = solved.estimate.expect("biased but observable motion");
                last_pose = Some(estimate.robot_to_local.pose);
                last_bias = solved.diagnostics.imu_bias;
                assert!(
                    estimator.biases.len() <= 3,
                    "coarse calibration must retire with the window"
                );
                assert!(estimator.preintegration.interval_count() <= 22);
            }
        }
        let bias = last_bias.unwrap();
        assert!(
            (bias.accelerometer - bias_at(20.0).accelerometer).norm() < 0.012,
            "{bias:?}"
        );
        assert!(
            (bias.gyroscope - bias_at(20.0).gyroscope).norm() < 0.001,
            "{bias:?}"
        );
        // Real acceleration stays in the trajectory, not in the learned bias.
        assert!((last_pose.unwrap().translation().x() - 20.0).abs() < 0.2);
        assert!(estimator.biases.first_key_value().unwrap().0 >= &3);
        assert!(bias.covariance.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn rejected_visual_frame_does_not_advance_output_time() {
        let mut estimator = estimator();
        assert!(
            !estimator
                .ingest_visual(TimeWrapper {
                    time: Time::from_nanos(2_000_000_000),
                    inner: VisualLocalizationFrame {
                        epoch: 7,
                        source: types::visual_localization_next::VisualAssociationSource::Tracking,
                        robot_to_camera: nalgebra::Isometry3::identity().framed_transform(),
                        generation: 0,
                        camera_intrinsic: Intrinsic::new(
                            nalgebra::vector![300.0, 300.0],
                            point![160.0, 120.0],
                        ),
                        associations: Vec::new(),
                    },
                })
                .unwrap()
        );
        assert_eq!(estimator.latest_time, Time::from_nanos(1_000_000_000));
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
}

use std::{
    collections::BTreeMap,
    ops::Range,
    time::{Duration, Instant},
};

use color_eyre::{Result, eyre::eyre};
use coordinate_systems::{Field, Local, Robot};
use fagra::{BatchKey, GaussNewton, OptimizeOptions, Problem, SolverError, StateKey};
use linear_algebra::{Framed, Isometry3, Vector2, Vector3, vector};
use localization_fagra::{
    factors::{
        AdjacentKinematicOdometry, AdjacentVisualOdometry, CameraIntrinsicsPrior, FieldContainment,
        FootGround, FootObservation, FrameReprojections, ImuKinematics, ImuObservation,
        KinematicOdometry, MotionPrior, ReprojectionObservation, RollPitchPrior, TrajectoryPrior,
        VisualOdometry, VisualOdometryObservation,
    },
    variables::{CameraIntrinsics, FieldAlignment, PoseControl, TrajectoryState},
};
use nalgebra::SMatrix;
use ros_z::time::Time;
use types::camera_geometry::CameraGeometry;
use types::{
    field_dimensions::FieldDimensions, localization::LocalizationEstimate,
    time_wrapper::TimeWrapper, visual_localization::VisualLocalizationFrame,
};

use crate::{diagnostics::SolveDiagnostics, parameters::Localization3dParameters};

const KNOT_SPACING_NS: i64 = 200_000_000;
pub(crate) const OPTIMIZATION_WINDOW: Duration = Duration::from_secs(2);
const WINDOW_NS: i64 = OPTIMIZATION_WINDOW.as_nanos() as i64;
const HUBER_THRESHOLD: f64 = 2.0;
const MIN_REPROJECTION_DEPTH: f64 = 0.01;
const MIN_LANDMARK_RANGE: f64 = 0.01;

mod attitude;
mod covariance;
mod inertial;
mod kinematic_odometry;
mod vision;
mod window;

fagra::states! {
    States {
        controls: PoseControl,
        alignments: FieldAlignment,
        intrinsics: CameraIntrinsics,
    }
}

fagra::factors! {
    Factors {
        trajectory_priors: TrajectoryPrior,
        intrinsics_priors: CameraIntrinsicsPrior,
        motion: MotionPrior,
        tilt: RollPitchPrior,
        yaw: localization_fagra::factors::RelativeYaw,
        containment: FieldContainment,
        adjacent_odometry: AdjacentVisualOdometry,
        kinematic_odometry: KinematicOdometry,
        adjacent_kinematic_odometry: AdjacentKinematicOdometry,
        imu: Batch<ImuKinematics, ImuObservation>,
        feet: Batch<FootGround, FootObservation>,
        reprojections: Batch<FrameReprojections, ReprojectionObservation>,
        odometry: Batch<VisualOdometry, VisualOdometryObservation>,
    }
}

type Graph = Problem<States, Factors>;

pub(crate) struct SolveResult {
    pub estimate: Option<LocalizationEstimate>,
    pub diagnostics: SolveDiagnostics,
    pub converged: bool,
}

pub(crate) struct Estimator {
    graph: Graph,
    optimizer: GaussNewton,
    options: OptimizeOptions<f64>,
    origin: Time,
    epoch: u64,
    controls: BTreeMap<i64, StateKey<PoseControl<f64>>>,
    imu_batches: BTreeMap<i64, BatchKey<ImuKinematics<f64>>>,
    foot_batches: BTreeMap<i64, BatchKey<FootGround<f64>>>,
    odometry_batches: BTreeMap<i64, BatchKey<VisualOdometry<f64>>>,
    reprojection_batches: Vec<(i64, BatchKey<FrameReprojections<f64>>)>,
    alignment: Option<StateKey<FieldAlignment<f64>>>,
    intrinsics: StateKey<CameraIntrinsics<f64>>,
    latest_time: Time,
    latest_vo_epoch: Option<u64>,
    latest_kinematic_time: Option<Time>,
    measurements: BTreeMap<i64, usize>,
    latest_visual_frame: Option<TimeWrapper<VisualLocalizationFrame>>,
    attitudes: BTreeMap<Time, nalgebra::UnitQuaternion<f64>>,
    yaw_factors: BTreeMap<i64, fagra::FactorKey<localization_fagra::factors::RelativeYaw>>,
    current_yaw: Option<fagra::FactorKey<localization_fagra::factors::RelativeYaw>>,
    parameters: Localization3dParameters,
    field_half_extents: Vector2<Field, f64>,
    field: FieldDimensions,
}

impl Estimator {
    pub(crate) fn new(
        origin: Time,
        epoch: u64,
        initial_pose: Isometry3<Robot, Local>,
        camera: &CameraGeometry,
        parameters: Localization3dParameters,
        field: &FieldDimensions,
    ) -> Result<Self> {
        let mut graph = Graph::new();
        let pose = PoseControl {
            pose: Framed::wrap(initial_pose.inner.cast()),
        };
        let mut controls = BTreeMap::new();
        for index in -1..=2 {
            controls.insert(index, graph.add(pose.clone()));
        }
        let intrinsics_value = CameraIntrinsics {
            focal_lengths: Framed::wrap(camera.intrinsics.focals.cast()),
            optical_center: Framed::wrap(camera.intrinsics.optical_center.inner.cast()),
        };
        let intrinsics = graph.add(intrinsics_value.clone());
        let initial_controls = control_keys(&controls, 0)?;
        let mut anchor_root = SMatrix::<f64, 9, 9>::identity() * 1.0e3;
        // The local pose defines the gauge. Initial velocity is only a guess,
        // not an observed zero-velocity constraint (one m/s standard deviation).
        anchor_root
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&nalgebra::Matrix3::identity());
        graph.add_factor(TrajectoryPrior {
            controls: initial_controls,
            duration: seconds_per_knot(),
            tau: 0.0,
            reference: TrajectoryState {
                pose: pose.pose,
                velocity: Vector3::wrap(nalgebra::Vector3::zeros()),
            },
            information_root: anchor_root,
        })?;
        graph.add_factor(CameraIntrinsicsPrior {
            intrinsics,
            reference: intrinsics_value,
            information_root: SMatrix::identity() * 1.0e3,
        })?;

        let mut estimator = Self {
            graph,
            optimizer: GaussNewton::default(),
            options: OptimizeOptions {
                max_iterations: 5,
                // Ten micrometres / microradians is below sensor precision.
                step_tolerance: 1.0e-5,
                cost_tolerance: 1.0e-8,
                ..Default::default()
            },
            origin,
            epoch,
            controls,
            imu_batches: BTreeMap::new(),
            foot_batches: BTreeMap::new(),
            odometry_batches: BTreeMap::new(),
            reprojection_batches: Vec::new(),
            alignment: None,
            intrinsics,
            latest_time: origin,
            latest_vo_epoch: None,
            latest_kinematic_time: None,
            measurements: BTreeMap::new(),
            latest_visual_frame: None,
            attitudes: BTreeMap::new(),
            yaw_factors: BTreeMap::new(),
            current_yaw: None,
            parameters,
            field: *field,
            field_half_extents: vector![
                field.length as f64 * 0.5 + field.border_strip_width as f64,
                field.width as f64 * 0.5 + field.border_strip_width as f64,
            ],
        };
        let information_root = MotionPrior::information_root(
            seconds_per_knot(),
            0.01,
            estimator.parameters.accelerometer_process_noise_variance,
        )?;
        estimator.graph.add_factor(MotionPrior {
            controls: initial_controls,
            duration: seconds_per_knot(),
            information_root,
            use_start_velocity: true,
        })?;
        Ok(estimator)
    }

    pub(crate) fn latest_time(&self) -> Time {
        self.latest_time
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

    pub(crate) fn update_parameters(&mut self, parameters: Localization3dParameters) {
        self.parameters = parameters;
    }

    pub(crate) fn solve(&mut self) -> SolveResult {
        let start = Instant::now();
        let mut diagnostics = SolveDiagnostics {
            time: self.latest_time,
            epoch: self.epoch,
            duration: Duration::ZERO,
            iterations: None,
            initial_cost: None,
            final_cost: None,
            termination: "failed".into(),
            state_count: 0,
            measurement_count: 0,
            failure: None,
        };
        let result = (|| -> Result<_> {
            self.prepare_attitude()?;
            let result = self.solve_graph(&mut diagnostics);
            if let Some(key) = self.current_yaw.take() {
                self.graph.remove_factor(key)?;
            }
            self.retire_old_segments()?;
            result
        })();
        let (estimate, converged) = match result {
            Ok((estimate, converged)) => (Some(estimate), converged),
            Err(error) => {
                diagnostics.failure = Some(error.to_string());
                (None, false)
            }
        };
        diagnostics.duration = start.elapsed();
        diagnostics.state_count = self.controls.len() + 1 + usize::from(self.alignment.is_some());
        diagnostics.measurement_count = self.measurements.values().sum();
        SolveResult {
            estimate,
            diagnostics,
            converged,
        }
    }

    fn solve_graph(
        &mut self,
        diagnostics: &mut SolveDiagnostics,
    ) -> Result<(LocalizationEstimate, bool)> {
        let (segment, tau) = self.segment_and_tau(self.latest_time)?;
        let controls = self.ensure_segment(segment)?;
        // Full-step GN can accept uphill intermediate steps. Restore the whole
        // attempt on failure so later measurements never inherit a failed trial.
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
        // ponytail: an evaluation-only pass also assembles Jacobians; replace with
        // a cost-only Problem API when fagra exposes one.
        let evaluation = OptimizeOptions {
            gradient_tolerance: f64::MAX,
            ..self.options
        };
        let initial_cost = self
            .optimizer
            .solve_batch(&mut self.graph, &evaluation)?
            .initial_cost;
        let result = self.optimizer.solve_batch(&mut self.graph, &self.options);
        let converged = result.is_ok();
        let result = match result {
            Err(SolverError::NoConvergence) => {
                self.optimizer.solve_batch(&mut self.graph, &evaluation)
            }
            result => result,
        };
        let report = match result {
            Ok(report) if report.final_cost <= initial_cost => report,
            result => {
                for (key, value) in controls_before {
                    self.graph.set(key, value)?;
                }
                if let Some((key, value)) = self.alignment.zip(alignment_before) {
                    self.graph.set(key, value)?;
                }
                self.graph.set(self.intrinsics, intrinsics_before)?;
                return Err(match result {
                    Err(error) => eyre!(error),
                    Ok(_) => eyre!("Gauss-Newton increased the objective"),
                });
            }
        };
        diagnostics.iterations = Some(if converged {
            report.iterations
        } else {
            self.options.max_iterations
        });
        diagnostics.initial_cost = Some(initial_cost);
        diagnostics.final_cost = Some(report.final_cost);
        diagnostics.termination = if converged {
            format!("{:?}", report.termination)
        } else {
            "MaxIterations".into()
        };
        Ok((self.estimate_with_covariance(controls, tau)?, converged))
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

fn seconds_per_knot() -> f64 {
    KNOT_SPACING_NS as f64 * 1.0e-9
}

#[cfg(test)]
mod tests {
    use booster::ImuState;
    use std::time::Duration;

    use linear_algebra::{IntoTransform, point};
    use projection::intrinsic::Intrinsic;

    use super::*;

    fn estimator() -> Estimator {
        let camera = CameraGeometry {
            intrinsics: Intrinsic::new(nalgebra::vector![300.0, 300.0], point![160.0, 120.0]),
            ..Default::default()
        };
        Estimator::new(
            Time::from_nanos(1_000_000_000),
            7,
            nalgebra::Isometry3::identity().framed_transform(),
            &camera,
            Localization3dParameters {
                kinematic_odometry_noise: Default::default(),
                accelerometer_process_noise_variance: 10.0,
                visual_feature_noise_variance: 100.0,
                field_containment_sigma: 1.0,
                tracking_timeout: Duration::from_secs(2),
                visual_tracking_timeout: Duration::from_secs(2),
            },
            &FieldDimensions::SPL_2025,
        )
        .unwrap()
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
            assert!(
                estimator
                    .ingest_imu(
                        time,
                        ImuState {
                            roll_pitch_yaw: Vector3::wrap(nalgebra::Vector3::zeros()),
                            angular_velocity: Vector3::wrap(nalgebra::Vector3::zeros()),
                            linear_acceleration: Vector3::wrap(nalgebra::Vector3::zeros()),
                        },
                    )
                    .unwrap()
            );
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
    fn covariance_remains_observable_across_knot_boundary() {
        let mut estimator = estimator();
        for index in 0..=100 {
            estimator
                .ingest_imu(
                    Time::from_nanos(1_000_000_000 + index * 2_000_000),
                    ImuState::default(),
                )
                .unwrap();
        }
        estimator.solve().estimate.unwrap();
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
    fn rejected_visual_frame_does_not_advance_output_time() {
        let mut estimator = estimator();
        assert!(
            !estimator
                .ingest_visual(TimeWrapper {
                    time: Time::from_nanos(2_000_000_000),
                    inner: VisualLocalizationFrame {
                        epoch: 7,
                        robot_to_camera: nalgebra::Isometry3::identity().framed_transform(),
                        robot_to_local: nalgebra::Isometry3::identity().framed_transform(),
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
    fn startup_replaces_drifted_height_prior_with_landmark_fit() {
        use types::visual_localization::FieldMarkAssociation;
        let camera = CameraGeometry {
            robot_to_camera: Isometry3::wrap(
                nalgebra::Isometry3::from_parts(
                    nalgebra::Translation3::new(0.12, -0.04, 0.2),
                    nalgebra::UnitQuaternion::from_euler_angles(std::f32::consts::PI, 0.0, 0.0),
                )
                .inverse(),
            ),
            intrinsics: Intrinsic::new(nalgebra::vector![300.0, 300.0], point![160.0, 120.0]),
        };
        let truth = nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(-2.0, 1.0, 0.55),
            nalgebra::UnitQuaternion::from_euler_angles(0.1, -0.05, 0.4),
        );
        let associations: Vec<_> = [
            point![-3.0, 0.0, 0.0],
            point![-1.0, 0.0, 0.0],
            point![-2.0, 2.0, 0.0],
        ]
        .into_iter()
        .map(|field_point| {
            let p = camera.robot_to_camera.inner * truth.inverse() * field_point.inner;
            FieldMarkAssociation {
                field_point,
                detection: camera.intrinsics.project(Vector3::wrap(p.coords)),
            }
        })
        .collect();
        for wrong_height in [0.55, 4.0] {
            let mut estimator = estimator();
            let wrong_pose = Isometry3::wrap(nalgebra::Isometry3::from_parts(
                nalgebra::Translation3::new(100.0, -50.0, wrong_height),
                nalgebra::UnitQuaternion::from_euler_angles(0.1, -0.05, 0.0),
            ));
            estimator = Estimator::new(
                estimator.origin,
                estimator.epoch,
                wrong_pose,
                &camera,
                estimator.parameters.clone(),
                &estimator.field,
            )
            .unwrap();
            estimator
                .ingest_imu(Time::from_nanos(1_800_000_000), ImuState::default())
                .unwrap();
            let time = Time::from_nanos(1_100_000_000);
            assert!(
                estimator
                    .ingest_visual(TimeWrapper {
                        time,
                        inner: VisualLocalizationFrame {
                            epoch: 7,
                            robot_to_local: wrong_pose,
                            robot_to_camera: camera.robot_to_camera,
                            camera_intrinsic: camera.intrinsics,
                            associations: associations.clone(),
                        }
                    })
                    .unwrap()
            );
            assert_eq!(estimator.origin, time);
            let controls = control_keys(&estimator.controls, 0).unwrap();
            let seeded = estimator.spline(controls).unwrap().pose(0.0).unwrap();
            assert!((seeded.inner.translation.vector.z - 0.55).abs() < 1e-5);
            for index in 0..100 {
                estimator
                    .ingest_imu(
                        time + Duration::from_millis(index * 2),
                        ImuState {
                            roll_pitch_yaw: Vector3::wrap(nalgebra::Vector3::new(0.1, -0.05, 0.0)),
                            ..Default::default()
                        },
                    )
                    .unwrap();
            }
            let solved = estimator.solve();
            assert!(
                solved.diagnostics.failure.is_none(),
                "{:?}",
                solved.diagnostics
            );
            let field = solved.estimate.unwrap().robot_to_field.unwrap().pose.inner;
            assert!(
                (field.translation.vector - truth.translation.vector.cast::<f64>()).norm() < 1e-4
            );
            assert!(field.rotation.angle_to(&truth.rotation.cast::<f64>()) < 1e-4);
            assert!(estimator.visual_rms().unwrap() < 0.01);
        }
    }

    #[test]
    fn visual_ingestion_accepts_behind_camera_predictions_but_not_zero_range() {
        use types::visual_localization::FieldMarkAssociation;
        let mut estimator = estimator();
        estimator.alignment = Some(estimator.graph.add(FieldAlignment {
            local_to_field: nalgebra::Isometry2::identity().framed_transform(),
        }));
        let time = Time::from_nanos(1_000_000_000);
        let mut frame = TimeWrapper {
            time,
            inner: VisualLocalizationFrame {
                epoch: 7,
                robot_to_local: Isometry3::identity(),
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
        assert!(estimator.visual_rms().is_none());
        frame.inner.robot_to_camera = Isometry3::identity();
        assert!(!estimator.ingest_visual(frame).unwrap());
    }
}

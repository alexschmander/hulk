use std::time::SystemTime;

use factrs::{
    containers::{Key, ValuesOrder},
    core::{SE3, SO3},
    optimizers::{OptError, OptStatus},
    residuals::ErasedResidual,
    traits::{Optimizer, Variable},
    variables::{SE2, SE23},
};
use nalgebra::{DMatrix, DVector, SMatrix, Vector3};

use crate::{
    factors::{
        gaussian_process_prior::GaussianProcessPriorFactor,
        visual_odometry::{AdjacentVisualOdometryFactor, VisualOdometryFactor},
        visual_reprojection::VisualReprojectionFactor,
    },
    schur_marginalization::marginalize,
    splines::SE23Spline,
    symbols::{CameraIntrinsics, LocalToField, State},
    tau,
};

use super::{
    BackendOptimizerStatus, BackendSolveDiagnostics, OptimizationResult,
    ResidualDiagnosticsAccumulator, VinsBackend, VinsBackendError,
};

impl VinsBackend {
    pub(super) fn optimize_and_publish(
        &mut self,
    ) -> Result<Option<OptimizationResult>, VinsBackendError> {
        let result = self.optimize();
        if self.result_sender.send(result.clone()).is_err() {
            return Err(VinsBackendError::FrontendDisconnected);
        }

        Ok(result)
    }

    pub(super) fn optimize(&mut self) -> Option<OptimizationResult> {
        self.last_optimizer_status = None;
        let time = self.last_knot_time?;
        log::info!("solving graph with {} values", self.values.len());

        self.remove_current_spline_orientation_factor();
        self.add_current_spline_orientation_factor();

        let optimizer_status = match self.optimizer.optimize(&mut self.values) {
            Ok(OptStatus::Converged) => BackendOptimizerStatus::Converged,
            Ok(OptStatus::MaxIterations) => {
                log::warn!("optimizer failed to converge: max iterations reached");
                BackendOptimizerStatus::MaxIterations
            }
            Err(OptError::FailedToStep) => {
                log::warn!("optimizer failed: failed to step");
                BackendOptimizerStatus::FailedToStep
            }
            Err(OptError::InvalidSystem) => {
                log::warn!("optimizer failed: invalid system");
                BackendOptimizerStatus::InvalidSystem
            }
        };
        let robot_to_field_covariance = (optimizer_status == BackendOptimizerStatus::Converged)
            .then(|| self.robot_to_field_covariance_at(time))
            .flatten();
        self.remove_current_spline_orientation_factor();

        if matches!(
            optimizer_status,
            BackendOptimizerStatus::Converged | BackendOptimizerStatus::MaxIterations
        ) {
            let cutoff_time = time - self.config.max_optimization_window;
            if let Some(smallest_interval_index_in_window) = self
                .interval_assigner
                .assign_or_initialize_interval(cutoff_time)
            {
                marginalize(
                    &mut self.optimizer,
                    &mut self.values,
                    State(smallest_interval_index_in_window),
                );
            }
        }

        self.last_optimizer_status = Some(optimizer_status);

        let interval_start_time = self
            .interval_assigner
            .current_or_initialize_interval_start_time(time)?;
        let interval_start_index = self
            .interval_assigner
            .assign_or_initialize_interval(interval_start_time)?;
        let start = self.values.get(State(interval_start_index))?.clone();
        let end = self.values.get(State(interval_start_index + 1))?.clone();
        let interval_end_time = interval_start_time + self.config.knot_spacing;
        let latest_robot_to_local = SE23Spline::new(
            start,
            end,
            self.config.knot_spacing.as_secs_f64(),
        )
        .evaluate(tau(interval_start_time, interval_end_time, time));
        let camera_intrinsics = self.values.get(CameraIntrinsics(0))?.clone();
        let local_to_field = self.values.get(LocalToField(0)).cloned();
        let latest_visual_robot_to_local = self
            .latest_visual_measurement_time
            .and_then(|time| self.pose_at(time));
        Some(OptimizationResult {
            time,
            generation: self.generation,
            latest_robot_to_local,
            local_to_field,
            robot_to_field_covariance,
            camera_intrinsics,
            latest_visual_measurement_time: self.latest_visual_measurement_time,
            latest_visual_robot_to_local,
            optimizer_status,
        })
    }

    fn robot_to_field_covariance_at(&self, time: SystemTime) -> Option<SMatrix<f64, 6, 6>> {
        let interval_start_time = self
            .interval_assigner
            .current_or_initialize_interval_start_time(time)?;
        let interval_start_index = self
            .interval_assigner
            .assign_or_initialize_interval(interval_start_time)?;
        let alignment = self.values.get(LocalToField(0))?;
        self.robot_to_field_covariance(
            State(interval_start_index),
            State(interval_start_index + 1),
            alignment,
            tau(
                interval_start_time,
                interval_start_time + self.config.knot_spacing,
                time,
            ),
        )
    }

    fn robot_to_field_covariance(
        &self,
        start_key: State,
        end_key: State,
        alignment: &SE2,
        tau: f64,
    ) -> Option<SMatrix<f64, 6, 6>> {
        let start = self.values.get(start_key)?;
        let end = self.values.get(end_key)?;
        let order = ValuesOrder::from_values(&self.values);
        let (hessian, _) = self
            .optimizer
            .graph()
            .linearize(&self.values)
            .dense_normal_equations(&order);
        if !hessian.iter().all(|value| value.is_finite()) {
            return None;
        }

        let indices: Vec<_> = [
            Key::from(start_key),
            Key::from(end_key),
            Key::from(LocalToField(0)),
        ]
        .into_iter()
        .flat_map(|key| {
            let index = order.get(key)?;
            Some(index.idx..index.idx + index.dim)
        })
        .flatten()
        .collect();
        if indices.len() != 21 {
            return None;
        }

        let mut selector = DMatrix::zeros(order.dim(), indices.len());
        for (column, &row) in indices.iter().enumerate() {
            selector[(row, column)] = 1.0;
        }
        let solved = hessian.cholesky()?.solve(&selector);
        if !solved.iter().all(|value| value.is_finite()) {
            return None;
        }
        let joint_covariance =
            DMatrix::from_fn(21, 21, |row, column| solved[(indices[row], column)]);

        let jacobian = composed_pose_jacobian(
            start,
            end,
            alignment,
            tau,
            self.config.knot_spacing.as_secs_f64(),
        )?;
        let covariance = jacobian * joint_covariance * jacobian.transpose();
        let covariance = SMatrix::<f64, 6, 6>::from_fn(|row, column| {
            (covariance[(row, column)] + covariance[(column, row)]) * 0.5
        });
        covariance
            .iter()
            .all(|value| value.is_finite())
            .then_some(covariance)
    }

    fn pose_at(&self, time: SystemTime) -> Option<SE23<f64>> {
        let interval_start_time = self
            .interval_assigner
            .current_or_initialize_interval_start_time(time)?;
        let interval_index = self
            .interval_assigner
            .assign_or_initialize_interval(interval_start_time)?;
        let start = self.values.get(State(interval_index))?.clone();
        let end = self.values.get(State(interval_index + 1))?.clone();
        let interval_end_time = interval_start_time + self.config.knot_spacing;
        Some(
            SE23Spline::new(start, end, self.config.knot_spacing.as_secs_f64()).evaluate(tau(
                interval_start_time,
                interval_end_time,
                time,
            )),
        )
    }

    pub(super) fn solve_diagnostics(
        &self,
        optimizer_status: BackendOptimizerStatus,
    ) -> BackendSolveDiagnostics {
        let graph = self.optimizer.graph();
        let mut visual_odometry = self.residual_diagnostics::<VisualOdometryFactor>();
        visual_odometry.extend(self.residual_diagnostics::<AdjacentVisualOdometryFactor>());
        let visual_reprojection = self.residual_diagnostics::<VisualReprojectionFactor>();

        BackendSolveDiagnostics {
            optimizer_status,
            value_count: self.values.len(),
            factor_count: graph.len(),
            total_error: graph.error(&self.values),
            visual_odometry: visual_odometry.finish(),
            visual_reprojection: visual_reprojection.finish(),
            gaussian_process_prior: self
                .residual_diagnostics::<GaussianProcessPriorFactor>()
                .finish(),
        }
    }

    fn residual_diagnostics<R>(&self) -> ResidualDiagnosticsAccumulator
    where
        R: ErasedResidual + 'static,
    {
        let graph = self.optimizer.graph();
        let mut diagnostics = ResidualDiagnosticsAccumulator::default();
        for index in 0..graph.len() {
            let factor = graph.at(index);
            if !factor.is_residual::<R>() {
                continue;
            }
            let Ok(error) = factor.try_error(&self.values) else {
                continue;
            };
            let Ok(dim) = factor.try_dim_out(&self.values) else {
                continue;
            };
            diagnostics.add(error, dim);
        }
        diagnostics
    }
}

fn composed_pose_jacobian(
    start: &SE23,
    end: &SE23,
    alignment: &SE2,
    tau: f64,
    dt: f64,
) -> Option<SMatrix<f64, 6, 21>> {
    const EPSILON: f64 = 1.0e-6;
    let base = compose_pose(start, end, alignment, tau, dt);
    let mut jacobian = SMatrix::<f64, 6, 21>::zeros();
    for column in 0..21 {
        let (variable, component) = if column < 9 {
            (0, column)
        } else if column < 18 {
            (1, column - 9)
        } else {
            (2, column - 18)
        };
        let dimension = if variable == 2 { 3 } else { 9 };
        let mut delta = DVector::zeros(dimension);
        delta[component] = EPSILON;
        let (plus_start, plus_end, plus_alignment) = match variable {
            0 => (start.oplus(delta.as_view()), end.clone(), alignment.clone()),
            1 => (start.clone(), end.oplus(delta.as_view()), alignment.clone()),
            _ => (start.clone(), end.clone(), alignment.oplus(delta.as_view())),
        };
        delta[component] = -EPSILON;
        let (minus_start, minus_end, minus_alignment) = match variable {
            0 => (start.oplus(delta.as_view()), end.clone(), alignment.clone()),
            1 => (start.clone(), end.oplus(delta.as_view()), alignment.clone()),
            _ => (start.clone(), end.clone(), alignment.oplus(delta.as_view())),
        };
        let plus = compose_pose(&plus_start, &plus_end, &plus_alignment, tau, dt).ominus(&base);
        let minus = compose_pose(&minus_start, &minus_end, &minus_alignment, tau, dt).ominus(&base);
        jacobian.set_column(column, &((plus - minus) / (2.0 * EPSILON)));
    }
    jacobian
        .iter()
        .all(|value| value.is_finite())
        .then_some(jacobian)
}

fn compose_pose(start: &SE23, end: &SE23, alignment: &SE2, tau: f64, dt: f64) -> SE3 {
    let pose = SE23Spline::new(start.clone(), end.clone(), dt).evaluate(tau);
    let half_yaw = alignment.theta() * 0.5;
    SE3::from_rot_trans(
        SO3::from_xyzw(0.0, 0.0, half_yaw.sin(), half_yaw.cos()),
        Vector3::new(alignment.x(), alignment.y(), 0.0),
    )
    .compose(&SE3::from_rot_trans(
        pose.rot().clone(),
        pose.xyz().into_owned(),
    ))
}

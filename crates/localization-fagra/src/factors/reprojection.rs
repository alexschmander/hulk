use coordinate_systems::{Field, NormalizedDeviceCoordinates, Pixel, Robot};
use fagra::{
    BlockId, EvaluationError, FactorBatch, FactorSelection, JacobianBlock, LinearizationSink,
    StateKey, StateStore,
};
use linear_algebra::{Isometry3, Point2, Point3};
use nalgebra::{Matrix2, RealField, SMatrix};

use super::common;
use crate::variables::{CameraIntrinsics, FieldAlignment, PoseControl};

/// A detected image point associated with a known, fixed field landmark.
#[derive(Clone, Debug)]
pub struct ReprojectionObservation<R: RealField + Copy = f64> {
    pub field_point: Point3<Field, R>,
    pub detection: Point2<Pixel, R>,
}

/// A frame's reprojections share trajectory evaluation and camera geometry.
///
/// Transform the fixed field point into local coordinates using inverse field
/// alignment (leaving z unchanged), then into the robot and optical camera frames.
/// Raw residual: pinhole projection minus detected pixel coordinates. Whiten in
/// pixel x/y order and apply Huber independently to each 2D observation.
///
/// Reject invalid associations before insertion. An active observation outside
/// the valid projection domain returns `EvaluationError::InvalidEvaluation` in
/// both cost and linearization; it must not become a zero-cost observation.
#[derive(Clone, Debug)]
pub struct FrameReprojections<R: RealField + Copy = f64> {
    pub controls: [StateKey<PoseControl<R>>; 4],
    pub alignment: StateKey<FieldAlignment<R>>,
    pub intrinsics: StateKey<CameraIntrinsics<R>>,
    pub duration: R,
    pub tau: R,
    /// Optical convention: x right, y down, z forward, before perspective division.
    pub robot_to_camera: Isometry3<Robot, NormalizedDeviceCoordinates, R>,
    pub pixel_information_root: Matrix2<R>,
    pub huber_threshold: R,
    /// Positive minimum optical z in metres. Depth must be finite and strictly
    /// greater than this value; otherwise evaluation fails.
    pub min_depth: R,
}

impl<R, S> FactorBatch<S> for FrameReprojections<R>
where
    R: RealField + Copy,
    S: StateStore<PoseControl<R>> + StateStore<FieldAlignment<R>> + StateStore<CameraIntrinsics<R>>,
{
    type Scalar = R;
    type Factor = ReprojectionObservation<R>;

    fn visit_variables(&self, _factor: &Self::Factor, mut visitor: impl FnMut(BlockId)) {
        for key in self.controls {
            visitor(key.block_id());
        }
        visitor(self.alignment.block_id());
        visitor(self.intrinsics.block_id());
    }

    fn cost(
        &self,
        states: &S,
        factors: FactorSelection<'_, Self::Factor>,
    ) -> Result<R, EvaluationError> {
        if factors.is_empty() {
            return Ok(R::zero());
        }
        self.validate()?;
        let pose = common::spline(states, &self.controls, self.duration)?.pose(self.tau)?;
        let geometry = Geometry::new(
            &pose.inner,
            states.get(self.alignment)?,
            states.get(self.intrinsics)?,
            &self.robot_to_camera.inner,
        )?;
        let mut cost = R::zero();
        for (_, observation) in factors {
            let projection = geometry.project(observation, self.min_depth)?;
            cost += common::huber(
                &(self.pixel_information_root * projection.error),
                self.huber_threshold,
            )?
            .0;
        }
        common::checked_cost(cost)
    }

    fn linearize<L: LinearizationSink<Scalar = R>>(
        &self,
        states: &S,
        factors: FactorSelection<'_, Self::Factor>,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        if factors.is_empty() {
            return Ok(());
        }
        self.validate()?;
        let spline = common::spline(states, &self.controls, self.duration)?;
        let pose = spline.linearize()?.pose(self.tau)?;
        let geometry = Geometry::new(
            &pose.pose.inner,
            states.get(self.alignment)?,
            states.get(self.intrinsics)?,
            &self.robot_to_camera.inner,
        )?;
        let robot_to_camera = geometry
            .robot_to_camera
            .rotation
            .to_rotation_matrix()
            .into_inner();
        let local_to_robot = geometry
            .local_to_robot
            .rotation
            .to_rotation_matrix()
            .into_inner();
        for (id, observation) in factors {
            let projection = geometry.project(observation, self.min_depth)?;
            let residual = self.pixel_information_root * projection.error;
            let scale = common::huber(&residual, self.huber_threshold)?.1;
            let z_inv = projection.camera.z.recip();
            let x = projection.camera.x * z_inv;
            let y = projection.camera.y * z_inv;
            let j_projection = SMatrix::<R, 2, 3>::new(
                geometry.fx * z_inv,
                R::zero(),
                -geometry.fx * x * z_inv,
                R::zero(),
                geometry.fy * z_inv,
                -geometry.fy * y * z_inv,
            );
            let root = self.pixel_information_root * scale;
            let camera = root * j_projection * robot_to_camera;
            let mut inverse_pose = SMatrix::<R, 3, 6>::zeros();
            inverse_pose
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&projection.robot.cross_matrix());
            inverse_pose
                .fixed_view_mut::<3, 3>(0, 3)
                .set_diagonal(&nalgebra::Vector3::repeat(-R::one()));
            let h_pose = camera * inverse_pose;
            let jacobians = pose.jacobians.map(|j| h_pose * j);
            // Right perturbation of alignment: q_local' = Exp(-delta) q_local.
            let alignment = SMatrix::<R, 3, 3>::new(
                projection.local.y,
                -R::one(),
                R::zero(),
                -projection.local.x,
                R::zero(),
                -R::one(),
                R::zero(),
                R::zero(),
                R::zero(),
            );
            let alignment = camera * local_to_robot * alignment;
            let intrinsics = root
                * SMatrix::<R, 2, 4>::new(
                    x,
                    R::zero(),
                    R::one(),
                    R::zero(),
                    R::zero(),
                    y,
                    R::zero(),
                    R::one(),
                );
            let blocks: [_; 6] = std::array::from_fn(|i| match i {
                0..=3 => JacobianBlock::new(self.controls[i], &jacobians[i]),
                4 => JacobianBlock::new(self.alignment, &alignment),
                _ => JacobianBlock::new(self.intrinsics, &intrinsics),
            });
            sink.factor(id, |sink| sink.residual(&(residual * scale), &blocks))?;
        }
        Ok(())
    }
}

impl<R: RealField + Copy> FrameReprojections<R> {
    fn validate(&self) -> Result<(), EvaluationError> {
        common::positive(self.min_depth)?;
        common::positive(self.huber_threshold)?;
        common::finite(self.pixel_information_root.iter())
    }
}

struct Geometry<R: RealField + Copy> {
    field_to_local: nalgebra::Isometry2<R>,
    local_to_robot: nalgebra::Isometry3<R>,
    robot_to_camera: nalgebra::Isometry3<R>,
    fx: R,
    fy: R,
    cx: R,
    cy: R,
}

struct Projection<R: RealField + Copy> {
    local: nalgebra::Vector3<R>,
    robot: nalgebra::Vector3<R>,
    camera: nalgebra::Vector3<R>,
    error: nalgebra::Vector2<R>,
}

impl<R: RealField + Copy> Geometry<R> {
    fn new(
        pose: &nalgebra::Isometry3<R>,
        alignment: &FieldAlignment<R>,
        intrinsics: &CameraIntrinsics<R>,
        extrinsic: &nalgebra::Isometry3<R>,
    ) -> Result<Self, EvaluationError> {
        common::finite(
            alignment
                .local_to_field
                .inner
                .translation
                .vector
                .iter()
                .chain(
                    [
                        alignment.local_to_field.inner.rotation.re,
                        alignment.local_to_field.inner.rotation.im,
                    ]
                    .iter(),
                )
                .chain(extrinsic.translation.vector.iter())
                .chain(extrinsic.rotation.coords.iter())
                .chain(intrinsics.optical_center.inner.coords.iter()),
        )?;
        let fx = intrinsics.focal_lengths.inner.x;
        let fy = intrinsics.focal_lengths.inner.y;
        common::positive(fx)?;
        common::positive(fy)?;
        Ok(Self {
            field_to_local: alignment.local_to_field.inner.inverse(),
            local_to_robot: pose.inverse(),
            robot_to_camera: *extrinsic,
            fx,
            fy,
            cx: intrinsics.optical_center.inner.x,
            cy: intrinsics.optical_center.inner.y,
        })
    }

    fn project(
        &self,
        observation: &ReprojectionObservation<R>,
        min_depth: R,
    ) -> Result<Projection<R>, EvaluationError> {
        common::finite(
            observation
                .field_point
                .inner
                .coords
                .iter()
                .chain(observation.detection.inner.coords.iter()),
        )?;
        let xy =
            self.field_to_local * nalgebra::Point2::from(observation.field_point.inner.coords.xy());
        let local = nalgebra::Vector3::new(xy.x, xy.y, observation.field_point.inner.z);
        let robot = (self.local_to_robot * nalgebra::Point3::from(local)).coords;
        let camera = (self.robot_to_camera * nalgebra::Point3::from(robot)).coords;
        common::finite(camera.iter())?;
        if camera.z <= min_depth {
            return Err(EvaluationError::InvalidEvaluation);
        }
        let inverse_z = camera.z.recip();
        let error = nalgebra::Vector2::new(
            self.fx * camera.x * inverse_z + self.cx - observation.detection.inner.x,
            self.fy * camera.y * inverse_z + self.cy - observation.detection.inner.y,
        );
        common::finite(error.iter())?;
        Ok(Projection {
            local,
            robot,
            camera,
            error,
        })
    }
}

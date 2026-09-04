use factrs::{
    linalg::{ForwardProp, Numeric, VectorX},
    traits::Residual,
    variables::{MatrixLieGroup, SE2, SE23},
};

#[derive(Debug, Clone)]
pub struct FieldContainmentFactor {
    x_limit: f64,
    y_limit: f64,
    sigma: f64,
}

#[factrs::mark]
impl Residual for FieldContainmentFactor {
    type Input = (SE23, SE2);
    type Differ = ForwardProp;

    fn dim_out(&self) -> usize {
        2
    }

    fn residual<T: Numeric>(
        &self,
        (robot_to_local, local_to_field): (SE23<T>, SE2<T>),
    ) -> VectorX<T> {
        let mut residuals = VectorX::<T>::zeros(self.dim_out());
        let local_position = robot_to_local.xyz().fixed_rows::<2>(0).into_owned();
        let position = local_to_field.apply(local_position.as_view());
        residuals[0] = soft_limit_residual(position.x, self.x_limit, self.sigma);
        residuals[1] = soft_limit_residual(position.y, self.y_limit, self.sigma);
        residuals
    }
}

impl FieldContainmentFactor {
    pub fn new(x_limit: f64, y_limit: f64, sigma: f64) -> Self {
        assert!(
            x_limit.is_finite() && x_limit > 0.0,
            "x limit must be finite and positive"
        );
        assert!(
            y_limit.is_finite() && y_limit > 0.0,
            "y limit must be finite and positive"
        );
        assert!(
            sigma.is_finite() && sigma > 0.0,
            "field containment sigma must be finite and positive"
        );

        Self {
            x_limit,
            y_limit,
            sigma,
        }
    }
}

fn soft_limit_residual<T: Numeric>(value: T, limit: f64, sigma: f64) -> T {
    let limit = T::from(limit);
    let sigma = T::from(sigma);
    if value > limit {
        (value - limit) / sigma
    } else if value < -limit {
        (value + limit) / sigma
    } else {
        T::zero()
    }
}

#[cfg(test)]
mod tests {
    use factrs::{
        core::{SO3, Vector3},
        traits::Residual,
        traits::Variable,
        variables::{SE2, SE23},
    };
    use nalgebra::vector;

    use super::*;

    fn state(position: Vector3) -> SE23 {
        SE23::from_rot_vel_trans(SO3::identity(), Vector3::zeros(), position)
    }

    #[test]
    fn residual_is_zero_inside_and_on_boundary() {
        let factor = FieldContainmentFactor::new(6.5, 5.0, 1.0);

        let inside = factor.residual((state(vector![1.0, -2.0, 0.0]), SE2::identity()));
        let boundary = factor.residual((state(vector![6.5, -5.0, 0.0]), SE2::identity()));

        assert!(inside.iter().all(|value| value.abs() < 1.0e-9));
        assert!(boundary.iter().all(|value| value.abs() < 1.0e-9));
    }

    #[test]
    fn residual_penalizes_outside_position_with_sign() {
        let factor = FieldContainmentFactor::new(6.5, 5.0, 0.5);

        let positive = factor.residual((state(vector![7.0, 5.25, 0.0]), SE2::identity()));
        let negative = factor.residual((state(vector![-7.0, -5.25, 0.0]), SE2::identity()));

        assert!((positive[0] - 1.0).abs() < 1.0e-9);
        assert!((positive[1] - 0.5).abs() < 1.0e-9);
        assert!((negative[0] + 1.0).abs() < 1.0e-9);
        assert!((negative[1] + 0.5).abs() < 1.0e-9);
    }

    #[test]
    fn residual_uses_local_to_field_alignment() {
        let factor = FieldContainmentFactor::new(6.5, 5.0, 0.5);

        let residual = factor.residual((state(vector![0.0, 0.0, 0.0]), SE2::new(0.0, 7.0, 0.0)));

        assert!((residual[0] - 1.0).abs() < 1.0e-9);
    }
}

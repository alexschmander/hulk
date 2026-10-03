use super::Estimator;

use coordinate_systems::ImuReference;
use linear_algebra::Orientation3;

use ros_z::time::Time;

impl Estimator {
    pub fn attitude_at(&self, time: Time) -> Option<Orientation3<ImuReference, f64>> {
        let (&before, a) = self.attitudes.range(..=time).next_back()?;
        if before == time {
            return Some(*a);
        }
        let (&after, b) = self.attitudes.range(time..).next()?;
        if after.duration_since(before) > self.parameters.timing.max_imu_gap {
            return None;
        }
        let fraction = (time.as_nanos() - before.as_nanos()) as f64
            / (after.as_nanos() - before.as_nanos()) as f64;
        Some(a.slerp(*b, fraction))
    }
}

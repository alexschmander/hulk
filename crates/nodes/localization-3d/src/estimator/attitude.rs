use super::{Estimator, KNOT_SPACING_NS, control_keys, seconds_per_knot};
use color_eyre::Result;
use localization_fagra::factors::RelativeYaw;
use nalgebra::UnitQuaternion;
use ros_z::time::Time;

impl Estimator {
    /// Boundary attitudes summarize the source-time IMU stream. Relative headings
    /// carry no absolute-yaw anchor. Rebuilding the small active set handles late samples.
    pub(super) fn prepare_attitude(&mut self) -> Result<()> {
        for segment in self.segments() {
            let start = Time::from_nanos(self.origin.as_nanos() + segment * KNOT_SPACING_NS);
            let end = Time::from_nanos(start.as_nanos() + KNOT_SPACING_NS);
            let Some(a) = self.attitude_at(start) else {
                continue;
            };
            let Some(b) = self.attitude_at(end) else {
                continue;
            };
            if let Some(key) = self.yaw_factors.remove(&segment) {
                self.graph.remove_factor(key)?;
            }
            let key = self.graph.add_factor(RelativeYaw {
                controls: control_keys(&self.controls, segment)?,
                duration: seconds_per_knot(),
                end_tau: 1.0,
                measured_yaw_change: yaw_change(a, b),
                information_root: (2.0e-5_f64).sqrt().recip(),
            })?;
            self.yaw_factors.insert(segment, key);
        }
        if let Some((&time, &orientation)) = self.attitudes.last_key_value() {
            let (segment, tau) = self.segment_and_tau(time)?;
            let start = Time::from_nanos(self.origin.as_nanos() + segment * KNOT_SPACING_NS);
            if tau > 0.0
                && let Some(anchor) = self.attitude_at(start)
            {
                self.current_yaw = Some(self.graph.add_factor(RelativeYaw {
                    controls: control_keys(&self.controls, segment)?,
                    duration: seconds_per_knot(),
                    end_tau: tau,
                    measured_yaw_change: yaw_change(anchor, orientation),
                    information_root: (2.0e-5_f64).sqrt().recip(),
                })?);
            }
        }
        Ok(())
    }

    fn attitude_at(&self, time: Time) -> Option<UnitQuaternion<f64>> {
        let (&before, a) = self.attitudes.range(..=time).next_back()?;
        if before == time {
            return Some(*a);
        }
        let (&after, b) = self.attitudes.range(time..).next()?;
        let fraction = (time.as_nanos() - before.as_nanos()) as f64
            / (after.as_nanos() - before.as_nanos()) as f64;
        Some(a.slerp(b, fraction))
    }
}

fn yaw_change(a: UnitQuaternion<f64>, b: UnitQuaternion<f64>) -> f64 {
    let a = a * nalgebra::Vector3::x();
    let b = b * nalgebra::Vector3::x();
    let difference = b.y.atan2(b.x) - a.y.atan2(a.x);
    difference.sin().atan2(difference.cos())
}

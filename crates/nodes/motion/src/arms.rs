use super::*;
use kinematics::joints::leg::LegJoints;

impl MotionState {
    pub(super) fn generate_walking_arm_joints(
        &self,
        legs: &LowerBodyJoints<MotorCommand>,
        clock: &Clock,
        parameters: &ArmParameters,
        joint_limits: &JointLimits,
    ) -> Result<UpperBodyJoints<MotorCommand>> {
        let elapsed = clock.now().duration_since(self.last_arms.time);

        let legs = LowerBodyJoints {
            left_leg: clamped_positions(&legs.left_leg, joint_limits.position.left_leg),
            right_leg: clamped_positions(&legs.right_leg, joint_limits.position.right_leg),
        };

        let ratio =
            (elapsed.as_secs_f32() / parameters.arm_blend_duration.as_secs_f32()).clamp(0.0, 1.0);
        let mut target = Joints::fill(0.0);
        for (left, arm, leg_angles, initial, sign) in [
            (
                true,
                &mut target.left_arm,
                &legs.left_leg,
                self.last_arms.inner.left_arm,
                1.0,
            ),
            (
                false,
                &mut target.right_arm,
                &legs.right_leg,
                self.last_arms.inner.right_arm,
                -1.0,
            ),
        ] {
            let (sole, knee) = leg(leg_angles, left);
            arm.shoulder_pitch = sole.x() * parameters.shoulder_pitch_scale;
            arm.shoulder_roll = sign
                * (parameters.shoulder_roll_degrees.to_radians()
                    + (sign * knee.y() - parameters.knee_lateral_offset).max(0.0)
                        * parameters.shoulder_roll_scale);
            arm.shoulder_yaw = 0.0;
            arm.elbow = sign
                * (parameters.elbow_degrees.to_radians()
                    + sole.x() * parameters.shoulder_pitch_scale * parameters.elbow_scale);
            *arm = initial * (1.0 - ratio) + *arm * ratio;
        }
        let joints = target.map(|position| MotorCommand {
            position,
            kp: parameters.kp,
            kd: parameters.kd,
            ..MotorCommand::zeros()
        });
        ensure!(
            joints_are_finite(&joints),
            "non-finite generated arm joints"
        );
        Ok(UpperBodyJoints {
            left_arm: joints.left_arm,
            right_arm: joints.right_arm,
        })
    }
}

fn clamped_positions(leg: &LegJoints<MotorCommand>, limits: LegJoints<[f32; 2]>) -> LegJoints<f32> {
    LegJoints {
        hip_pitch: leg.hip_pitch.position,
        hip_roll: leg.hip_roll.position,
        hip_yaw: leg.hip_yaw.position,
        knee: leg.knee.position,
        ankle_up: leg.ankle_up.position,
        ankle_down: leg.ankle_down.position,
    }
    .clamp(limits.map(|[min, _]| min), limits.map(|[_, max]| max))
}

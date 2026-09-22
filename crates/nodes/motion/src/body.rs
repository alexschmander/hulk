//! Scheduling around the existing typed inference services. At most one request
//! is in flight; faster head ticks hold the last complete body command unchanged.
use super::*;
use ros_z::time::Time;
use tokio::task::JoinSet;

const POLICY_PERIOD: Duration = Duration::from_millis(20);

pub(super) struct CachedBody {
    pub joints: Joints<MotorCommand>,
    pub requested_at: Time,
    valid_until: Time,
    sensor_time: Time,
}

enum BodyOutput {
    Lower(InferenceResponse<LowerBodyJoints<MotorCommand>>),
    Whole(InferenceResponse<Joints<MotorCommand>>),
}

struct Completed {
    requested_at: Time,
    valid_until: Time,
    output: BodyOutput,
}

#[derive(Default)]
pub(super) struct BodySchedule {
    pub cached: Option<CachedBody>,
    policy: Option<motion_inference::config::Policy>,
    next_request: Option<Time>,
    pub pending_since: Option<Time>,
    pending_valid_until: Option<Time>,
    workers: JoinSet<Result<Completed>>,
}

impl BodySchedule {
    pub fn clear(&mut self) {
        // Dropping the set also drops completed replies from the previous mode.
        // An already executing remote request cannot be cancelled, but its result
        // can no longer reach this coordinator. The next request has a new generation.
        *self = Self::default();
    }

    pub fn validate(&self, now: Time, p: &Parameters) -> Result<()> {
        for (requested_at, valid_until) in [
            self.cached
                .as_ref()
                .map(|c| (c.requested_at, c.valid_until)),
            self.pending_since.zip(self.pending_valid_until),
        ]
        .into_iter()
        .flatten()
        {
            ensure!(
                now >= requested_at && now < valid_until,
                "body inference result/request expired"
            );
        }
        if let Some(cached) = &self.cached {
            ensure!(
                now >= cached.sensor_time && now < cached.sensor_time + p.maximum_sensor_age,
                "held body inference observation expired"
            );
        }
        Ok(())
    }
}

impl MotionState {
    pub(super) fn update_body(
        &mut self,
        command: InferenceCommand,
        clock: &Clock,
        p: &Parameters,
        limits: &JointLimits,
    ) -> Result<()> {
        let now = clock.now();
        let policy = command.policy();
        if self.body.policy.is_some_and(|previous| previous != policy) {
            self.body.clear();
            self.generation = self.generation.saturating_add(1);
            self.last_policy = None;
        }
        self.body.policy = Some(policy);

        if let Some(completed) = self.body.workers.try_join_next() {
            self.body.pending_since = None;
            self.body.pending_valid_until = None;
            let completed = completed.wrap_err("inference request task stopped")??;
            ensure!(
                now >= completed.requested_at && now < completed.valid_until,
                "body inference deadline expired before dispatch"
            );
            let (joints, execution) = match completed.output {
                BodyOutput::Lower(output) => {
                    let arms =
                        self.generate_walking_arm_joints(&output.joints, clock, &p.arms, limits)?;
                    (
                        Joints::from_head_and_body(
                            kinematics::joints::head::HeadJoints::fill(MotorCommand::zeros()),
                            BodyJoints::from_lower_and_upper(*output.joints, arms),
                        ),
                        output.execution,
                    )
                }
                BodyOutput::Whole(output) => (*output.joints, output.execution),
            };
            ensure!(
                execution.policy == policy,
                "inference returned the wrong policy"
            );
            let RobotCommand::Custom { joints_command } = (RobotCommand::Custom {
                joints_command: joints,
            })
            .clamp(limits)?
            else {
                unreachable!();
            };
            self.last_arms = TimeWrapper {
                time: now,
                inner: joints_command.upper_body_as_ref().map(|j| j.position),
            };
            self.last_policy = Some(execution);
            self.body.cached = Some(CachedBody {
                joints: joints_command,
                requested_at: completed.requested_at,
                valid_until: completed.valid_until,
                sensor_time: execution.sensor_time,
            });
        }
        self.body.validate(now, p)?;
        if self.body.workers.is_empty() && self.body.next_request.is_none_or(|due| now >= due) {
            // Preserve the 50 Hz schedule, but skip missed slots; never catch up by
            // advancing a policy several times on the same observation.
            let mut next = self.body.next_request.unwrap_or(now) + POLICY_PERIOD;
            if next <= now {
                next = now + POLICY_PERIOD;
            }
            self.body.next_request = Some(next);
            self.body.pending_since = Some(now);
            self.body.pending_valid_until = Some(now + p.inference_timeout);
            let generation = self.generation;
            let timeout = p.inference_timeout;
            let request = InferenceRequest {
                generation,
                requested_at: now,
                valid_until: now + timeout,
                command,
            };
            match command {
                InferenceCommand::Walk(command) => {
                    let client = self.walk_inference_client.clone();
                    self.body.workers.spawn(async move {
                        let request = InferenceRequest {
                            command,
                            generation: request.generation,
                            requested_at: now,
                            valid_until: request.valid_until,
                        };
                        let output = client.call_with_timeout_async(&request, timeout).await??;
                        Ok(Completed {
                            requested_at: now,
                            valid_until: request.valid_until,
                            output: BodyOutput::Lower(output),
                        })
                    });
                }
                InferenceCommand::Kick(command) => {
                    let client = self.kick_inference_client.clone();
                    self.body.workers.spawn(async move {
                        let request = InferenceRequest {
                            command,
                            generation: request.generation,
                            requested_at: now,
                            valid_until: request.valid_until,
                        };
                        let output = client.call_with_timeout_async(&request, timeout).await??;
                        Ok(Completed {
                            requested_at: now,
                            valid_until: request.valid_until,
                            output: BodyOutput::Lower(output),
                        })
                    });
                }
                InferenceCommand::GetUp(command) => {
                    let client = self.get_up_inference_client.clone();
                    self.body.workers.spawn(async move {
                        let request = InferenceRequest {
                            command,
                            generation: request.generation,
                            requested_at: now,
                            valid_until: request.valid_until,
                        };
                        let output = client.call_with_timeout_async(&request, timeout).await??;
                        Ok(Completed {
                            requested_at: now,
                            valid_until: request.valid_until,
                            output: BodyOutput::Whole(output),
                        })
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;

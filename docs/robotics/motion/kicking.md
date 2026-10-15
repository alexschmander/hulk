# Kick and get-up requests

Behavior publishes `types::motion_command::MotionCommand` on
`behavior/motion_command`. Its `Kick` and `StandUp` variants carry policy choices
through the motion node to inference. Motion publishes `RobotCommand` on
`commands/robot_command`, containing the desired control mode and joint commands.

## Behavior inputs

`BodyMotion::Kick` and `MotionCommand::Kick` carry:

- `ball_position`: observed ball position in Ground coordinates, in metres.
- `ball_velocity`: current ball velocity in Ground coordinates, in m/s.
- `kick_direction`: desired kick orientation in Ground coordinates.
- `target_speed`: desired outgoing ball speed in m/s.
- `soft`: select the soft-kick policy.
- `quick`: enable the normal kick policy's quick flag.
- `strong`: enable the normal kick policy's strong flag, requesting maximum speed.

Behavior stores its selected target in `Blackboard::kick_target` for strength
selection. The kick command carries the resulting direction and inference
inputs. `last_kick_target` holds the Field target across ticks for remote and
penalty kicks.

The behavior `kick` action prefers the visual percept for position. It transforms
that observation through timestamped odometry and advances it by its age using
`world_state.ball.ball_in_ground_velocity`. The tracked ball must be our own
estimate and agree with the visual ball within `ball_velocity_matching_distance`.
Otherwise the visual position uses zero velocity. Observations older than
`ball_observation_timeout` fall back to tracking. Motion compensates the remaining
command age and robot movement before submitting the inference request.
Interception computes a future point locally to choose direction; the request
still describes the ball at the current time.

Behavior actions can call `use_kick_strength` with `KickStrength::Soft`,
`Medium`, or `Strong`. Each selection sets the policy, flags and speed together.
The base configuration selects these profiles:

| Strength | Policy | Requested speed | `strong` flag | `quick` flag |
| --- | --- | --- | --- | --- |
| Soft | SoftKick | 0.1 m/s | false | false |
| Medium | Kick | 1.0 m/s | false | false |
| Strong | Kick | 3.4 m/s | false | false |

The strong behavior profile uses the normal policy at maximum target speed.
`strong_uses_maximum_flag` enables the network's separate maximum-strength flag
if needed. Trials found this flag weaker on one approach side and less precise
on the other than the scaled maximum-speed request. `quick` enables the training
flag that allows deviation for a faster kick; leave it false for precise passes.
The separate inaccurate-kick input in slot 49 stays zero to preserve aim.

Setting `kicking.soft` makes ordinary kicks soft and prevents automatic promotion
to a goal shot. Otherwise ordinary kicks use Medium and distant targets can
select Strong. A continuing kick retains the complete profile while refreshing
position, velocity and aim. Goalkeeper actions select the ordinary profile;
penalty actions can select Strong when permitted. The walk fallback preserves
settings while a transition is blocked.

`StandUp { fast }` selects the fast or slow get-up policy. The behavior action
reads `behavior_node.stand_up.fast` (default `false`) on entry and retains that
choice until a different command is produced.

## Behavior selection parameters

These parameters belong to `behavior_node.kicking`. All distances are in metres.

| Parameter | Base default | Purpose |
| --- | --- | --- |
| `target_speed` | `1.0` | Medium profile's requested outgoing speed in m/s. |
| `soft_target_speed` | `0.1` | Soft profile's requested outgoing speed in m/s. |
| `strong_target_speed` | `3.4` | Strong profile's requested outgoing speed in m/s. |
| `strong_uses_maximum_flag` | `false` | Also set the network's strong flag for the Strong profile. |
| `allow_strong_kicks` | `true` | Permit strong striker and penalty kicks. |
| `strong_kick_min_target_distance` | `6.0` | Select strong striker kicks at or beyond this ball-to-target distance. Penalty selection only checks permission. |
| `kick_activation_distance` | `1.5` | Centre of the normal striker's walk-to-kick handoff band. |
| `kick_activation_hysteresis` | `0.4` | Total width of that band: enter below 1.3, leave above 1.7. |
| `ball_observation_timeout` | `100 ms` | Maximum visual observation age and odometry association gap. |
| `ball_velocity_matching_distance` | `0.3` | Maximum disagreement between projected visual position and own tracked ball before using tracked velocity. |
| `approach_ball_standoff` | `0.3` | Walking standoff behind the ball and set-play alignment position. |
| `minimum_interception_forward_distance` | `0.3` | Minimum Ground x coordinate of the predicted interception point, for both striker and goalkeeper. |

`disable_strong_kick` selects Soft or Medium according to `kicking.soft`, including
that profile's speed and flags. The `kick_strength_subtree` retains settings while
a kick continues.

Strength selection and kick direction use the same target position. The separate
`substates.distance_for_kick` and `substates.distance_for_kick_hysteresis`
parameters control proximity to the set-play alignment position.

## Inference and simulation

Motion compensates ball position, velocity and aim without applying policy limits.
Inference caps ball velocity and clamps target speed to the selected policy's
limits. Strong normal
kicks use the maximum speed regardless of `target_speed`. Soft kicks ignore the
strong and quick flags. Base speed limits are `[0.5, 3.4]` m/s for normal kicks and
`[0.1, 1.7]` m/s for soft kicks.

Inference multiplies target speed by **0.1 exactly once**, in tensor slot 56.
Behavior and injected commands supply m/s, so a 3.4 m/s request becomes 0.34.
Multiplying at the caller as well weakens the kick. Kick joint velocity uses
raw finite differences, scaled by 0.1, rather than the walking slew limiter.
The previous frequency action uses the policy action limit of 2.0 for kick
observations; only physical phase integration clips it to 0.5. Walking retains
its reference controller's clipped frequency feedback.

Inference caps the ball-position observation at 2.0 m. The handoff band stays
inside this cap. Behavior approaches balls at up to 1.8 m/s using
`walking.speed.kicking`; `walking.speed.velocity_fade_distance` is 1.0 m.
Motion's direction-dependent velocity limits still apply.

The behavior simulator approximates the speed rules with configurable limits.
It does not model the learned quick-kick movement or fast/slow recovery timing;
get-up clears the simulated fallen state immediately.

## Calibration reference

The installed `K1_kick_inner.onnx` matches B-Human's published model byte for
byte, SHA-256 `7fb9ea33c36e14332861025f22e92f2f4a7ae7681f4532bdd6e1e78975fe3452`.
The input contract comes from
[`t1_ball.py`](https://github.com/bhuman/MachineLearning/blob/412978cdff3a0e2758c5bffc0bced4f236062c7c/IsaacGymRL/envs/t1_ball.py),
inherited by `k1_ball.py`, and
[`K1_Ball.yaml`](https://github.com/bhuman/MachineLearning/blob/412978cdff3a0e2758c5bffc0bced4f236062c7c/IsaacGymRL/envs/K1_Ball.yaml).
In particular slot 8 carries direction divided by pi, not ball height, and the
quick input is `flag_allow_deviation`. Training used a 290 g, 95 mm radius ball.
The simulator uses a 370 g ball, the midpoint of the robot's 350 to 390 g range,
and retains its 105 mm radius.

Headless trials use production behavior, motion, inference and MuJoCo contacts.
With the filtering profile, medium requests contacted both an approaching ball
at 0.5 m/s and a crossing ball at 0.8 m/s at each of 350, 370 and 390 g.
The six passes travelled 3.10 to 3.73 m in six seconds, with direction errors
of 4.1 to 5.1 degrees relative to the commanded aim at first contact.
Trials switch to Stand one second after contact to measure a single kick.
Policy inputs arrived every 20 ms. The 95th-percentile sensor age was 2 ms,
with a maximum of 4 ms in those trials.
The soft policy at its minimum request still travels about 1.5 to 1.7 m in ten
seconds for stationary balls. It does not reliably absorb incoming speed or
redirect crossing balls: the moving soft trials travelled 3.04 and 2.12 m,
and the crossing direction error was 26 degrees. Use the normal policy for
moving-ball passes. Medium passes can hop up to 4 cm; a strictly grounded
3 to 5 m pass remains a calibration limitation. Strong stationary shots reached
3.5 to 3.94 m/s and travelled 5.9 to 7.7 m in six seconds. Strong approaching
and crossing trials reached 3.48 and 2.93 m/s, with direction errors of 0.85
and 6.6 degrees and lift of 8 and 13 cm.

Walking commands of 1.8 m/s forward, 0.8 m/s backward and 1.0 m/s sideways kept
the robot upright. Measured cruise velocities over the final three seconds
were about 1.54, 0.76 and 0.65 m/s along the requested axis, respectively.
Backward and lateral trials also drifted across the requested direction, so
these command limits do not imply equal measured speed or precise tracking.

At 370 g, a normal 3.4 m/s request produced a 3.52 m/s peak and 5.90 m travel
in six seconds. Changing only the observation scale from 0.1 to 0.01 produced
1.87 m/s and 3.33 m. These are fixed-window travel distances. The 3.52 m/s shot
was still moving at 0.42 m/s when that six-second trial ended.

Twenty-second repeats let the ball settle almost to rest:

| Request | Peak speed | Travel from first contact | Final ball speed |
| --- | --- | --- | --- |
| Normal 3.4 m/s, existing scale | 3.47 m/s | 7.43 m | 0.0031 m/s |
| Normal 3.4 m/s, extra 0.1 scale | 1.87 m/s | 4.03 m | 0.0018 m/s |
| Medium 1.0 m/s | 2.06 m/s | 4.53 m | 0.0020 m/s |

The simulator's range depends on carpet and contact settings;
these measurements do not establish a speed-to-distance formula for the robot.
See the simulator's [motion trial instructions](../../../tools/simulate/README.md#motion-policy-trials)
to repeat these measurements.

## Serialized commands

Injected kick commands use the `Kick` variant with all fields supplied explicitly.
Stand-up commands use `{"StandUp":{"fast":true}}` for fast recovery or
`{"StandUp":{"fast":false}}` for slow recovery.

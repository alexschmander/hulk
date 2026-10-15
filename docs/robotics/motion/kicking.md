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

Ground has +x forward and +y left, with robot yaw and an origin halfway between
the soles. Position, absolute ball velocity and aim all use these axes. Velocity
does not subtract robot motion and does not rotate into the requested kick
direction. Aim is a counterclockwise angle from forward.

Behavior stores its selected target in `Blackboard::kick_target` for strength
selection. The kick command carries the resulting direction and inference
inputs. `last_kick_target` holds the Field target across ticks for remote and
penalty kicks.

The behavior `kick` action prefers the visual percept for position. It transforms
that observation through timestamped odometry and advances it by its age using
velocity from the local `ball_filter/ball_position` topic. The tracked snapshot
has its own source timestamp; behavior rotates its velocity and advances its
position into the same Ground frame before matching it to the visual ball.
The tracked position must agree within `ball_velocity_matching_distance`.
Otherwise the visual position uses zero velocity. Observations older than
`ball_observation_timeout` fall back to tracking. Motion compensates the remaining
command age and robot movement before submitting the inference request.
Behavior also propagates the field pose through odometry before computing aim,
and publishes the command with the timestamp of that common frame. Own tracked
ball positions in the behavior state use this updated field pose as well.
A field pose without matching odometry is unavailable to behavior.
Interception computes a future point locally to choose direction; the request
still describes the ball at the current time.

Behavior actions can call `use_kick_strength` with `KickStrength::Soft`,
`Medium`, or `Strong`. Each selection sets the policy, flags and speed together.
The base configuration selects these profiles:

| Strength | Policy | Requested speed | `strong` flag | `quick` flag |
| --- | --- | --- | --- | --- |
| Soft | SoftKick | 0.1 m/s | false | false |
| Medium | SoftKick | 0.3 m/s | false | false |
| Strong | Kick | 3.4 m/s | false | false |

Soft and Medium use the same soft policy at different requested speeds. Both
clear the strong and quick flags. The strong behavior profile uses the normal
policy at maximum target speed.
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
| `target_speed` | `0.3` | Medium soft-policy request, calibrated with strike extension for 3 to 5 m passes. |
| `soft_target_speed` | `0.1` | Soft profile's requested outgoing speed in m/s. |
| `strong_target_speed` | `3.4` | Strong profile's requested outgoing speed in m/s. |
| `strong_uses_maximum_flag` | `false` | Also set the network's strong flag for the Strong profile. |
| `allow_strong_kicks` | `true` | Permit strong striker, kickoff follow-up and penalty kicks. |
| `strong_kick_min_target_distance` | `6.0` | Select strong striker kicks at or beyond this ball-to-target distance. Penalty selection only checks permission. |
| `kick_activation_distance` | `1.5` | Centre of the normal striker's walk-to-kick handoff band. |
| `kick_activation_hysteresis` | `0.4` | Total width of that band: enter below 1.3, leave above 1.7. |
| `receding_ball_kick_activation_distance` | `0.6` | Handoff centre when the ball's field velocity projects away from the robot faster than 0.05 m/s. With default hysteresis, enter below 0.4 m. Release an early kick above 0.65 m. |
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

## Kickoff

Our striker arms a single kickoff sequence in Set. It also accepts the filtered
whistle-in-Set kickoff flag when joining Playing. The sequence survives the GC
clearing that flag and resets on leaving Playing. Opponent restarts, set plays
and penalty shootouts do not arm it. A ball already more than 0.4 m from the
centre does not start the routine, and an unfinished routine expires after 20 s.

The opening touch uses Soft at 45 degrees to the left of the attacking field
axis. Its field target stays fixed while the robot turns. Walking aligns with
the kick direction and stops 0.6 m behind the ball before handing over. The soft
policy remains active until the ball has moved 0.25 m, followed by 350 ms of
follow-through.

The striker then uses walking to recover behind the ball, facing the opponent
goal centre. It aims for a standoff of 0.85 m with the ball 0.08 m to its left,
waits for ball speed below 0.2 m/s, and stands for 600 ms before selecting Strong.
That stand lets the walking gait settle and its phase return to zero. A direct
handoff from walking produced substantially weaker shots in the calibration.
Disabling strong kicks selects the configured ordinary profile for the follow-up.

The behavior blackboard exposes `kickoff` stages for simulator inspection. Once
the strong shot has moved the ball 0.5 m and completed 500 ms of follow-through,
ordinary striker behavior resumes.

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
The foot-avoidance bias uses foot points relative to the leveled sole midpoint,
matching the ball frame; torso-relative foot points would introduce a false
bias as the robot leans sideways.

### Contact calibration

`motion_inference.kick.contact` extends the learned strike along the requested
kick direction. It changes the policy's current and previous ball-position
features together, after foot avoidance. It leaves the ball estimate, command,
velocity, direction and target-speed scaling unchanged. This is a strike
calibration, not additional observation-age compensation.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `forward_offset` | `0.12 m` | Normal-policy extension along the kick direction. |
| `soft_forward_offset` | `0.12 m` | Maximum soft-policy extension. |
| `soft_speed_range` | `[0.1, 0.3] m/s` | Ramp soft extension from zero to its maximum as the requested speed increases. |
| `incoming_speed_limit` | `0.05 m/s` | Ramp extension to zero when ball velocity opposes the kick direction at this speed. |

Both offsets can be set to zero to reproduce the uncalibrated policy. They are
limited to 0.15 m and require a node restart, like the other inference parameters.
The final position observation retains its 2 m magnitude cap.
The minimum-speed soft touch keeps its short stroke. Medium uses a lower speed
request with extension to preserve pass range while improving contact timing
and follow-through. Requested speed is a network input, not a prediction of the
calibrated shot's measured speed.

Incoming balls need the original strike location. Applying the full extension
to a filtered 0.5 m/s approaching ball increased direction error from about
2 degrees to 14 degrees. The incoming-speed ramp disables that extension.

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

### Initial calibration at 2060822ca

These baseline measurements precede the coordinate audit corrections below.
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

### Coordinate audit

The Motion & behavior profile supplies exact ball truth at the published scene
timestamp, approximately every 33 ms. It still needs age and frame compensation
at the 500 Hz behavior and 50 Hz policy loops. Filtering supplies synthetic camera
detections to the real visual selector and ball filter, so its velocity is an
estimate rather than perfect truth.

An independent physical-frame audit found three mismatches:

- Simulator Ground used foot-body origins instead of the sole midpoint. A tilted
  ankle regression measured an 18 mm horizontal origin error before correction.
- Behavior computed aim with the older field pose while the ball was already
  advanced to a newer Ground frame. A +45 degree ideal-profile trial had
  3.75 degrees of 95th-percentile request aim error before correction.
- The foot-avoidance bias compared torso-relative foot points with a Ground
  ball position. Translating both soles relative to the torso changed the bias
  by 20 mm even with their separation unchanged.

The corrected implementation uses the physical sole midpoint, aligns field pose,
visual ball and own tracked velocity to a common timestamp, and expresses the
foot points in that same Ground frame. The coordinate audit compares actual
inference requests and tensor inputs with physical sole positions, trunk yaw and
ball state without calling the simulator's Ground transform helper.

Eight trials cover +45 and -45 degree starts in Motion & behavior, Filtering and
Body state & odometry, a translated +45 degree start, and a stationary ball during
a turn from 90 degrees. Each audits inputs after a 250 ms observation warmup and
before contact. The stationary-ball case retains zero world ball velocity while
the robot translates and rotates. The tensor's deliberate foot bias is reported
separately from raw request position error.

Largest scenario 95th-percentile errors in those eight trials:

| Profile | Request position | Absolute velocity | Request aim |
| --- | --- | --- | --- |
| Motion & behavior | 3.1 mm | 0.006 m/s | 0.29 degrees |
| Filtering | 13 mm | 0.184 m/s | 0.37 degrees |
| Body state & odometry | 61 mm | 0.192 m/s | 0.57 degrees |

After these corrections, representative twenty-second stationary-ball runs
travelled 1.69 m for Soft, 4.48 m for Medium and 7.25 m for Strong. Their final
speeds were 0.0009, 0.0020 and 0.0030 m/s, respectively. These runs used 370 g,
with the ball starting at world coordinates `[0.8, 0.08]` m.

These checks establish frame consistency for the tested cases. Moving balls,
especially at oblique angles, still produced weak or inaccurate medium kicks
despite accurate ideal inputs. Filtered velocity also has measurable estimation error; coordinate
corrections alone do not establish reliable moving-ball control.

## Soft-policy passes and autonomous kickoff calibration

The coordinate-audit results above used the earlier normal-policy Medium profile.
At this earlier calibration, Medium selected SoftKick at a 1.0 m/s request. Direct stationary-ball trials
at 350, 370 and 390 g, including centre and both lateral approach offsets,
travelled 3.52 to 4.01 m in 20 s. Final speeds were 0.0013 to 0.0025 m/s and direction
errors about 1.5 to 3 degrees. A default autonomous Filtering trial travelled
3.78 m and ended at 0.0018 m/s. The requested speed is a policy input; measured
peak speeds in those trials were about 1.5 to 1.8 m/s.

Kickoff trials used one robot at `[-1.5, 0]` m, a stationary centre ball and
scripted Ready, Set, whistle and Playing inputs through the production GC filters.
They used default behavior parameters without a strength override. Motion &
behavior passed at 350, 370 and 390 g; Filtering passed at 370 g. The soft opening
touch travelled at 48.5 to 48.7 degrees against a 45-degree target and moved the ball
about 1.8 m before the follow-up. The strong shot reached about 3.0 to 3.14 m/s with
1.3 to 1.7 degrees of error against the actual field goal centre. The first contact
occurred around 2.3 s and the strong contact around 13.8 s. Measured lift remained
about 3 cm.

The kickoff trial compares the strong trajectory directly with the field goal,
rather than transforming a potentially older behavior command using the robot's
current yaw. Those two aim measurements can differ during the kicking motion.
Thirty-five-second Filtering runs measured 6.57 to 7.05 m from strong contact,
ending at 0.0014 to 0.0016 m/s. The 25-second kickoff runs still had 0.036 to 0.071 m/s
of ball speed, so their distances are not stopping distances. These are isolated
two-kick trials with a scripted referee, not full matches with other players.

For rolling-away balls, walking now retains the approach until a closer handoff.
Velocity estimation initially reported only about 0.14 to 0.17 m/s after a 0.6 m/s
velocity step, so the receding detector uses a 0.05 m/s threshold. It also releases
an early kick above 0.65 m rather than retaining the wide stationary hysteresis
band. The sequential regression covers a stationary estimate becoming receding.
Coordinate tests rotate and translate the approach to check the velocity projection.

With a 0.6 m/s rolling-away ball, default autonomous Filtering improved from
0.60 m/s peak and 1.47 m travel to 1.31 m/s and 3.12 m. The latter ended at
0.00018 m/s after 30 s. Motion & behavior reached 1.60 m/s and 3.61 m at 20 s,
with 0.012 m/s remaining. These trials use the normal behavior selection and
isolate the kick by standing one second after contact.

Moving-ball quality still varies. At 1.0 m/s initial rolling-away speed,
Filtering reached 1.25 m/s and 2.98 m travel, just short of the 3 m quality check.
The 0.8 m/s crossing fixture travelled 3.29 m in 10 s but missed the target
direction by about 15 degrees. An approaching-ball fixture reached 3.68 m in
10 s with about 1.7 degrees of direction error. Those ten-second measurements
still had 0.057 to 0.070 m/s of ball speed. Further moving-ball accuracy tuning is
needed, especially for crossing balls.

### Contact timing and strike extension

The physics-step contact audit reproduced late strikes with the preceding calibration.
These comparisons use a 370 g, 105 mm radius ball, unchanged motor gains and the same physical initial conditions.
Baseline trials disable both contact offsets; their Medium request is 1.0 m/s.
Calibrated trials use the settings above, including the 0.3 m/s Medium request.

| Fixture | Contact / peak foot speed | Peak lead before contact | Velocity-line miss from centre | Follow-through past original centre |
| --- | --- | --- | --- | --- |
| Strong kickoff follow-up, Filtering | 80% → 97% | 46 → 14 ms | 7.6 → 4.1 mm | 89 → 149 mm |
| Stationary Medium, direct policy | 59% → 85% | 96 → 48 ms | 27.3 → 24.2 mm | -14 → 41 mm |
| Medium chasing 0.6 m/s initial ball velocity, Filtering | 55% → 81% | 110 → 64 ms | 35.7 → 20.7 mm | -41 → 37 mm |

Speed refers to the actual contacting material point on the foot, projected along the intended shot axis, sampled one physics step before loaded contact.
Peak speed is the preceding 300 ms maximum of that point, rather than a hypothetical no-ball replay.
The audit also reports trunk-relative swing speed, relative foot/ball velocity, contact height, lateral offset and incidence angle.
Follow-through is the maximum in the next 150 ms relative to the ball's original centre.
The independent velocity-line metric measures whether the relative contact trajectory passes through the ball centre.
A below-centre contact can still have a nearly central rising trajectory.

The strong kickoff follow-up increased from 3.12 to 3.79 m/s, with about 3.7 degrees of field-goal direction error after calibration.
Direct normal-policy shots at the 350 and 390 g mass limits reached 4.55 to 4.59 m/s and contacted at about 94% of their pre-impact peak.
The stationary Medium comparison increased from 1.76 to 1.93 m/s; its physical contact was farther below centre, about 33 mm versus 20 mm.
For the chased ball the contact height changed from 14 to 27 mm below centre, while trajectory alignment and follow-through improved.
These settings improve timing and trajectory; they do not produce an equatorial contact in every approach.

Trials continued long enough to measure near-rest travel:

| Fixture | Duration | Travel from relevant contact | Final ball speed |
| --- | --- | --- | --- |
| Direct Medium, 350 g | 20 s | 4.14 m | 0.0016 m/s |
| Direct Medium, 390 g | 20 s | 4.44 m | 0.0032 m/s |
| Default Filtering chase, initial 0.6 m/s | 30 s | 4.17 m | 0.0003 m/s |
| Default Filtering chase, initial 1.0 m/s | 30 s | 4.17 m | 0.0005 m/s |
| Default Filtering kickoff, strong follow-up | 35 s | 9.06 m | 0.0018 m/s |

The chase fixtures contact after about 5.8 and 7.2 s, when rolling friction has reduced ball speed to approximately 0.05 m/s.
They establish improved chase completion, not accurate interception of a ball still rolling at its initial speed.
Starting the 0.6 m/s rolling ball closer, at x = 0.35 m, also improved contact ratio from 54% to 79%, but the policy still chased until the ball slowed to about 0.053 m/s.
The 0.8 m/s crossing fixture contacted at about 0.26 m/s, travelled 3.76 m in 20 s and ended at 0.0023 m/s, with 13.1 degrees of direction error.

The approaching-ball fixture at `[0.8, 0.08]` m and `[-0.5, 0]` m/s still contacts the right shin before the foot, including with the previous settings.
Increasing the soft-policy request up to 1.7 m/s did not remove that failure.
Starting at y = 0 or -0.08 m produced clean foot contacts, implicating approach placement rather than insufficient requested power alone.
With the incoming extension disabled, the original +0.08 m fixture travelled 3.34 m and stayed within 1.3 degrees of aim, but fails the foot-first contact check.
Off-angle moving approaches also retain glancing-contact and direction errors.
Mirrored 45-degree input-frame audits passed their position, velocity and aim checks; this does not establish strike quality.

Soft at 0.1 m/s retains its decelerating stroke and no contact extension.
It travelled 1.57 m in 12 s with 0.015 m/s remaining; extending this stroke during probes made it too powerful for the short-touch profile.
The calibrated Medium mass-boundary trials lifted by less than 1 cm.
All 127 unit checks and the three opt-in live contact regression fixtures passed during this calibration.
Disabling extension reproduced the strong, Medium, chase and kickoff late-contact quality failures.

## Serialized commands

Injected kick commands use the `Kick` variant with all fields supplied explicitly.
Stand-up commands use `{"StandUp":{"fast":true}}` for fast recovery or
`{"StandUp":{"fast":false}}` for slow recovery.

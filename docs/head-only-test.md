# Seated head motion test

Branch: `head-only-motion`, based on `motion-inference`. Worktree: `../head-only-motion` relative to `new-motion`.

This runs the production head controller and hardware command path with damping on
every body joint (`kp=0`, `kd=1`, velocity and torque targets zero). It does not
start behavior, walking inference, or get-up. **Seat and support the robot before
running it: damping cannot maintain a sitting or standing pose.**

The SDK uses one mode for the robot, so the test enters **Custom** mode with active
head commands and damped body commands. It returns to SDK **Damping** afterward.
It never requests Prepare or Walking.

## Run

The prepared ARM64 kit is in `target/head-only-kit`. Use the robot's actual IP:

```bash
cd /home/ThagonDuarte/hulk/.worktrees/head-only-motion
./scripts/head-only-test run 10.1.24.42 zero 10
```

The helper uploads the kit to a separate directory under
`/home/booster/hulk/head-only-tests`, stops the regular `hulk` service, and runs the
test. It requires SSH access as `booster`, sudo, and the existing `hulk-runtime`
container and SDK/Zenoh bridge. The regular application binary and parameters are
not replaced. The normal service remains stopped after the test.

The DDS bridge allowlist must permit `rt/joint_ctrl` in the Zenoh-to-DDS
direction (the bridge configuration may use the DDS name `joint_ctrl`). A
successful Custom-mode RPC does not prove joint commands can cross the bridge.
The first four recordings on 2026-09-20 were made before that allowlist entry was
added and cannot establish active tracking performance.

Once the first test works, collect these runs with the same supported posture:

```bash
# Center hold at the scan's pitch (angles in radians).
./scripts/head-only-test run 10.1.24.42 hold 15 0 0.7
# The right endpoint that failed to settle in the earlier recording.
./scripts/head-only-test run 10.1.24.42 hold 15 -0.95 0.7
# Three-waypoint scan, including pitch changes.
./scripts/head-only-test run 10.1.24.42 scan 45
# Original side-to-side search for comparison with earlier recordings.
./scripts/head-only-test run 10.1.24.42 search 30
./scripts/head-only-test fetch 10.1.24.42
```

Test duration starts when head control becomes active. Startup is limited to 30
seconds; test duration is limited to 600 seconds. Recording subscriptions must be
ready before head control is enabled. Ctrl-C, SIGTERM, completion, or a reported
fault stop the test and request firmware damping. From another terminal:

```bash
./scripts/head-only-test stop 10.1.24.42
```

Loss of fresh sensor data, commands, head-service replies, or hardware state
disables head output and latches a fault. Restart the test after resolving the
cause; Prepare does not rearm this runtime. Keep the robot's physical stop
available for failures outside the process, including loss of transport.

## Scan path

`scan` now uses production **LookAround**: left `(yaw=0.95, pitch=0.5)` -> center
`(0, 0.7)` -> right `(-0.95, 0.5)` -> center, then repeats. Angles are radians.
It exercises yaw reversals, a center waypoint, and pitch changes. Arrival, dwell,
and timeout use the existing LookAround parameters (currently 1 s dwell and 2 s
maximum per waypoint). `look-around` remains an equivalent name.

`search` preserves the old scan: center once, then alternate left/right at pitch
0.7. Use it for comparisons with recordings from before this change. The production
lost-ball search behavior itself is unchanged. Each experiment now also records
its actual `head_motion`, so the two scan types can be distinguished explicitly.

## Head update rate

The default remains 50 Hz. Select 100 or 200 Hz per run without changing the
packaged parameters or gains:

```bash
./scripts/head-only-test run 10.1.24.43 scan 30 --rate-hz 100
./scripts/head-only-test run 10.1.24.43 scan 30 --rate-hz 200
```

For a comparison with the tuned yaw gains, run the same three holds and scan at
50, 100, and 200 Hz (12 runs, 3.75 minutes active with the default durations):

```bash
./scripts/head-only-test campaign 10.1.24.43 --kp 12 --kd 1 --rate-hz 50 100 200 --dry-run
./scripts/head-only-test campaign 10.1.24.43 --kp 12 --kd 1 --rate-hz 50 100 200
```

Pitch stays at kp=10/kd=1.2. The campaign manifest records each rate alongside its
gains; each experiment records its rate override and the effective parameter
layers. Leave the head untouched and keep the support identical across runs.

`motion.control_period` selects 20, 10, or 5 ms and requires a process restart.
The existing head service evaluates a new trajectory sample on each coordinator
tick. The normal motion stack requests body inference every 20 ms and holds the
complete body result between replies, including velocities, torque, and gains.
Inference runs independently of head-service calls. A held body result retains
its original request and sensor timestamps; fresh head output cannot extend its
validity. Policy changes discard old results, and get-up still owns all joints.

Hardware sends accepted RobotCommands immediately. Its existing
`joint_control_message_interval` controls watchdog checks and protective output,
not a second timer sampling active head targets. The run's rate override sets both
periods together. Overruns skip missed ticks rather than publishing catch-up bursts.
The motion services, RobotCommand topic, SDK command format, and trajectory limits
are unchanged. Measure actual publication intervals in the recording: a selected
rate is not a guarantee of delivery through the bridge or firmware.

For the combined simulator branch:

```bash
./simulator --head-only --head-rate-hz 200
```

The rate is applied again on simulator reset. Physics advances on the simulator's
logical clock; inspect recorded timing before treating a busy graphical run as a
precise rate comparison with the robot.

## Gain campaign

The campaign runs a fixed matrix for one selected joint. Defaults are:

| Stage | kp | kd |
|---|---:|---:|
| Baseline | 10 | 1.2 |
| Higher proportional gain | 12 | 1.2 |
| Higher derivative gain | 10 | 1.4 |
| Combined | 12 | 1.4 |

For each pair, it holds center `(0, 0.7)`, left `(0.95, 0.5)`, and right
`(-0.95, 0.5)` for 15 seconds each, then runs the three-waypoint scan for 30 seconds.
That is 16 recordings and 5 minutes of active testing, plus startup/transfers and
2 seconds in damping between runs. Every run starts a separate timed process and
returns to firmware damping. The other head joint is explicitly fixed at kp=10,
kd=1.2. Body damping and trajectory limits remain unchanged; the campaign uses 50 Hz unless
`--rate-hz` is specified.

Preview without contacting the robot or requiring a prepared kit:

```bash
./scripts/head-only-test campaign 10.1.24.43 --dry-run
```

Run the yaw campaign with the robot seated/supported and the bridge configured:

```bash
./scripts/head-only-test campaign 10.1.24.43
```

Remain with the robot, leave the head untouched during measurements, and use
Ctrl-C if oscillation or roughness increases. Ctrl-C stops the campaign and
requests damping; no subsequent gain setting starts. `stop ROBOT_IP` during an
active run also prevents progression once that run's result is downloaded. The
campaign does not automatically judge physical smoothness or choose a winning
setting. Completion means the timed run finished and finalized its outputs; it
does not prove commands crossed the bridge or the head tracked correctly.

It downloads and verifies `result.json` after every run. A failed run, interruption,
incomplete result, or failed download stops progression. Completed recordings and
available partial recordings are retained. Each run has a unique campaign/index/
waypoint name. The local manifest lives at
`logs/head-only/ROBOT_IP/BUILD_ID/campaigns/CAMPAIGN_ID.json`, with exact commands,
gains, run IDs, statuses, and results. Each run's recording directory is a sibling
of `campaigns/`, named by its `run_id`. Do not rebuild the kit during a campaign;
the script stops if it detects a different build ID. Restarting a campaign creates
new IDs and does not overwrite earlier runs. The normal HULK service stays stopped.

After examining yaw results, the same campaign can isolate pitch, or use a custom
matrix and durations:

```bash
./scripts/head-only-test campaign 10.1.24.43 --joint pitch
./scripts/head-only-test campaign 10.1.24.43 --joint yaw --kp 10 11 12 --kd 1.2 1.3 --hold-seconds 15 --scan-seconds 30 --dry-run
```

Values are traversed with kd outermost, kp innermost. A custom matrix replaces the
defaults for the specified gain; include the baseline explicitly when comparing.
The campaign uses the current default LookAround waypoint coordinates for its
holds; keep those scan waypoints unchanged when comparing campaign results.
Python 3 on the laptop is required; no extra Python packages are needed.

Compare final five-second hold error, overshoot, settling, measured torque,
constant-speed ripple, and scan arrival/timeouts. Raising kp may reduce endpoint
error but amplify the response to position steps. Compare each single-gain change
against baseline before interpreting the combined change. The earlier holds made
before the bridge correction cannot replace this baseline.

## Manual gain testing

Manual runs remain available, independently of campaigns:

```bash
./scripts/head-only-test run 10.1.24.43 hold 15 -0.95 0.5 --yaw-kp 12 --yaw-kd 1.2
./scripts/head-only-test run 10.1.24.43 scan 30 --yaw-kp 12 --yaw-kd 1.4
./scripts/head-only-test run 10.1.24.43 scan 30 --pitch-kp 12 --pitch-kd 1.4
./scripts/head-only-test run 10.1.24.43 search 30 --yaw-kp 10 --yaw-kd 1.2
./scripts/head-only-test fetch 10.1.24.43
```

Optional flags `--yaw-kp`, `--yaw-kd`, `--pitch-kp`, `--pitch-kd` override only
specified active head gains for that run. Others retain their parameter-layer
values. They do not modify deployment parameters or head damping-mode gains.
No flags means ordinary baseline parameters, even after a tuned run.
The current active defaults are yaw kp=12/kd=1.0 and pitch kp=10/kd=1.2.
The campaign matrix above explicitly pins its gains for comparison with the earlier
experiments; it does not inherit these changed yaw defaults.

Manual run names include explicit gain flags. `--run-id NAME` optionally supplies
an exact directory name (letters, digits, underscores, hyphens; maximum 121
characters); an existing run name is rejected. Campaigns use this option to link
recordings to their manifest. `experiment.json` records `gain_overrides`,
`parameters/test/head_motion.json5` contains the override layer, and
`head_motion/diagnostics` records the complete evaluated parameters.

## Recordings

`fetch` downloads to `logs/head-only/ROBOT_IP/BUILD_ID/`. Each run contains:

- `recording.mcap`, finalized on normal shutdown or a handled fault;
- `parameters/`, the layers actually used for that run;
- `experiment.json` and `result.json`, arguments, timing, completion reason, and errors;
- `BUILD.txt` and `SHA256SUMS`, source revision and binary/parameter hashes.

The console logs sit beside each run directory as `.out` and `.err` files.

The test records `inputs/low_state`, `joint_limits`, `behavior/motion_command`,
`commands/robot_command`, `motion/execution`, `motion/timing`, `hardware_interface/status`,
`hardware_interface/joint_command`, `hardware_interface/command_timing`, and
`head_motion/diagnostics`. Images and unrelated perception topics are omitted.
These diagnostic topics are also included in the normal base recorder config.

Head diagnostics contain each request's observation, observation source and receipt
times, evaluation times, reference position/velocity/acceleration, output commands,
arrival progress, scan phase/dwell/deadline state, constraints, reseeds, errors, and
the exact head parameters used. Hardware timing connects incoming RobotCommand
source/receipt times to the raw SDK publish start/end times. These are host publish
times; motor-controller receipt is not acknowledged by that transport.

`motion/timing` records coordinator start/end times and the original request times
of the held body result and any pending inference. This distinguishes expected
body repetition from head repeats and from expired inference.

Compare reference and final commanded trajectories against measured motion. Smooth
references with uneven hardware publication point to dispatch timing. Regular final
commands with rough measured movement or persistent endpoint error point farther
downstream, toward actuator tracking, gains, friction, or load. A seated run removes
walking disturbances from that comparison; it does not by itself prove the original
root cause. Use the same targets, parameters, and diagnostic topics for a subsequent
simulator comparison and walking recording.

## Rebuild after editing

```bash
./scripts/head-only-test prepare
```

This builds locally with `podman:localhost/k1sdk:1.3.0` and replaces the local kit;
it does not contact a robot. `HEAD_TEST_BUILD_ENV` overrides the SDK environment.
Run `fetch` before rebuilding to collect the previous kit's recordings.

The implementation is covered by motion fault tests and a complete timed test
against an isolated fake SDK transport. The latter checks raw body commands, mode
requests, final damping, and MCAP integrity. Physical smoothness still needs the
live tests above.

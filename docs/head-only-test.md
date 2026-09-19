# Seated head motion test

Branch: `head-only-motion`, based on `motion-inference`. Worktree: `../head-only-motion` relative to `new-motion`.

This runs the production head controller and hardware command path with damping on
every body joint (`kp=0`, `kd=1`, velocity and torque targets zero). It does not
start behavior, walking inference, or get-up. **Seat and support the robot before
running it: damping cannot maintain a sitting or standing pose.**

The SDK uses one mode for the robot, so the test enters **Custom** mode with active
head commands and damped body commands. It returns to SDK **Damping** afterward.
It never requests Prepare or Walking.

## Tomorrow

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

Once the first test works, collect these runs with the same supported posture:

```bash
# Center hold at the scan's pitch (angles in radians).
./scripts/head-only-test run 10.1.24.42 hold 15 0 0.7
# The right endpoint that failed to settle in the earlier recording.
./scripts/head-only-test run 10.1.24.42 hold 15 -0.95 0.7
# Search scan, with the current production trajectory and gains.
./scripts/head-only-test run 10.1.24.42 scan 45
# Optional localization scan.
./scripts/head-only-test run 10.1.24.42 look-around 30
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

## Recordings

`fetch` downloads to `logs/head-only/ROBOT_IP/BUILD_ID/`. Each run contains:

- `recording.mcap`, finalized on normal shutdown or a handled fault;
- `parameters/`, the layers actually used for that run;
- `experiment.json` and `result.json`, arguments, timing, completion reason, and errors;
- `BUILD.txt` and `SHA256SUMS`, source revision and binary/parameter hashes.

The console logs sit beside each run directory as `.out` and `.err` files.

The test records `inputs/low_state`, `joint_limits`, `behavior/motion_command`,
`commands/robot_command`, `motion/execution`, `hardware_interface/status`,
`hardware_interface/joint_command`, `hardware_interface/command_timing`, and
`head_motion/diagnostics`. Images and unrelated perception topics are omitted.
The two new diagnostic topics are also included in the normal base recorder config.

Head diagnostics contain each request's observation, observation source and receipt
times, evaluation times, reference position/velocity/acceleration, output commands,
arrival progress, scan phase/dwell/deadline state, constraints, reseeds, errors, and
the exact head parameters used. Hardware timing connects incoming RobotCommand
source/receipt times to the raw SDK publish start/end times. These are host publish
times; motor-controller receipt is not acknowledged by that transport.

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

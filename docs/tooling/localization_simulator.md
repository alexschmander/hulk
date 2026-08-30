# Localization Simulator

## Purpose

Provide a deterministic, interactive environment for diagnosing localization drift, jumps, and
field-symmetry flips without running camera neural networks or robot hardware.

The simulator renders the SPL field, moves a ground-truth camera rig along a six-degree-of-freedom
trajectory, synthesizes localization inputs, runs the production association and localization
implementations, and displays estimated poses against ground truth.

## Requirements

### Correctness

- Use the current `FieldDimensions::SPL_2025` geometry, production semantic landmark map,
  stateless field-mark associator,
  localization configuration, factor-graph frontend/backend, global lock, and live visual-odometry
  propagation where applicable.
- Keep transform directions explicit and use typed coordinate-system transforms at boundaries.
- Generate the frame-to-frame visual-odometry delta and cumulative visual odometer from the same
  noisy measurement.
- Run simulation on a fixed logical clock. Rendering frame rate and playback speed must not alter
  generated inputs or localization results.
- Use independent, seeded random-number streams for each simulated sensor.
- Recreate and replay localization state when restarting. History inspection must not mutate the
  estimator.
- Record raw backend, live-propagated, and ground-truth poses separately.

### Simulation

- Support six-degree-of-freedom camera trajectories represented by timestamped positions and unit
  quaternions.
- Include straightforward built-in trajectories for stationary and moving tests.
- Provide a free-fly controller for exploratory movement and allow recorded motion to be replayed.
- Generate configurable visual-odometry translation/rotation noise and bias.
- Generate configurable field-mark pixel noise and dropout.
- Support deterministic one-shot visual-odometry outliers.
- Offer two field-mark modes:
  - known ground-truth correspondences, to isolate the estimator;
  - production field-mark association from class-grouped synthetic pixel detections.

### User Interface

- Render the field, ground-truth camera pose, backend estimate, live estimate, and their trails.
- Provide play, pause, single-step, restart, playback-speed, trajectory, association-mode, noise,
  and random-seed controls.
- Show simulation time, position/orientation errors, global-lock state, optimizer status, and factor
  residual summaries.
- Keep controls and colors documented in the application itself.

### Testing

- The simulation core must run without a renderer.
- Repeated runs with the same scenario and seed must produce identical results within floating-point
  tolerance.
- Include stationary, six-degree-of-freedom, visual-odometry-outlier, and association-mode tests.

## Non-Goals

- Photorealistic image rendering, occlusion, or neural-network execution.
- Robot dynamics, joint-level motion, or contact simulation.
- Foot-height factors. They are intentionally omitted so visual localization and odometry can be
  evaluated independently; add them only if a concrete localization experiment requires contact
  constraints.
- A generic ROS or MCAP replayer.
- Reproducing nondeterministic operating-system scheduling of the live ROS node.

## Launch and controls

Launch the native viewer from the repository development environment:

```sh
nix develop --command cargo run -p localization_simulator
```

Use the left panel to select a scenario and sensor settings, then press **Apply configuration /
rebuild**. Playback always advances the estimator in fixed 20 ms steps; the speed control only
changes how quickly those steps are requested. While paused, the history slider inspects recorded
samples without changing estimator state.

The collapsible **VO bias and one-shot outlier** section configures per-step translation/rotation
bias and a deterministic transition-indexed SE(3) outlier. Input edits are staged: playback is
disabled until **Apply configuration / rebuild** is pressed, preventing old results from being
mistaken for the newly displayed settings.

The `pose_teleport` scenario changes the ground-truth robot pose discontinuously. The `vo_fault`
scenario remains physically smooth and enables a deterministic VO-only transform outlier. This
keeps robot relocation and sensor corruption independently testable.

The 3D view uses left-drag to orbit, right-drag to pan, and the mouse wheel to zoom. Cyan is truth,
yellow is the raw backend estimate, and magenta is the live estimate.

Flight recording is available while paused. **W/S** move along camera-local z, **A/D** along
camera-local x, **Q/E** vertically in the field, arrow keys control yaw and pitch, and **Z/C** control
roll. Stopping creates and selects a deterministic Custom scenario. The path field and Load/Save
buttons read and write that scenario as position/quaternion JSON5 keyframes.

## Headless analysis

Run the same deterministic simulator without opening a window and write a complete JSON report:

```sh
nix develop --command cargo run -p localization_simulator -- headless \
  --scenario six-dof-loop --output localization-report.json
```

Use `--scenario-file path.json5` for a custom trajectory and `--config path.json5` for complete
sensor settings. `--seed` and `--association known-correspondences|production-association` override
those individual settings. Without `--output`, the report is written to standard output. The
`vo-fault` preset installs its standard one-shot VO outlier unless a configuration file is supplied.
Run `localization_simulator headless --help` for all built-in scenario names and options.

Report schema version 2 includes the effective scenario, sensor configuration, SPL field dimensions,
and bundled production localization and association parameters. Every 20 ms sample contains:

- Timestamp in nanoseconds.
- Truth, raw backend, live, and cumulative noisy odometry SE(3) poses with explicit frame directions.
- Synthetic IMU values and every timestamped noisy VO transition passed to the frontend.
- Translation in meters and quaternion rotation in `[x, y, z, w]` order.
- Backend and live translation and rotation error against truth.
- Emitted semantic landmark pixels and accepted pixel-to-field correspondences.
- Global visual-lock state, visible/emitted/associated landmark counts, and solve diagnostics.

The summary includes lock acquisition time, RMS/max/final pose error, and maximum consecutive pose
step for detecting jumps. Reports contain the complete timeline rather than only the summary, so
analysis scripts can derive additional metrics without rerunning the simulation.

Version 2 removes `landmark_frame.backend_reset_robot_to_field` and
`landmark_frame.associations[].source`. Consumers of version 1 reports must branch on the top-level
schema version before decoding landmark frames.

## Scenario format

Positions are meters in the field frame. Each pose maps camera coordinates into the field frame,
and quaternions use `[x, y, z, w]` order. Timestamps are seconds, must be strictly increasing, and
must span exactly from zero to a duration aligned to the 20 ms simulation tick. Scenarios are
limited to 600 seconds.

```json5
{
  name: "short_forward_motion",
  duration_seconds: 1.0,
  camera_to_field_keyframes: [
    {
      time_seconds: 0.0,
      position: [-3.0, 0.0, 0.55],
      quaternion_xyzw: [-0.5, 0.5, -0.5, 0.5],
    },
    {
      time_seconds: 1.0,
      position: [-2.5, 0.0, 0.55],
      quaternion_xyzw: [-0.5, 0.5, -0.5, 0.5],
    },
  ],
}
```

Run deterministic headless coverage with:

```sh
nix develop --command cargo test -p localization_simulator
```

For a manual smoke test, run the viewer, step and restart a built-in trajectory, compare both
association modes, inspect earlier history while paused, and enable a one-shot VO outlier.

## Tasks

- [x] Expose the production semantic landmark list without duplicating field geometry.
- [x] Add a small synchronous localization runner around the existing production configuration and
      `VinsFrontend`/`VinsBackend` APIs.
- [x] Implement trajectory interpolation and built-in scenarios.
- [x] Implement deterministic synthetic camera, IMU, visual-odometry, and field-mark measurements.
- [x] Implement known-correspondence and production-association modes.
- [x] Run localization at explicit fixed checkpoints and retain result history.
- [x] Add a Bevy field scene with truth/backend/live markers and trails.
- [x] Add free-fly recording, playback controls, and diagnostics UI.
- [x] Add headless deterministic regression tests.
- [x] Document launch and scenario-authoring commands.

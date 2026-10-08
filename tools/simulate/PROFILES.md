# Simulator profiles implementation plan

Status: implemented on `twix/motion-simulator`, with the retained profile-4 limitation documented below.
Profiles 1–3 pass their runtime gates; profile 4 is wired but its startup acquisition is blocked by the production field-association search budget.
The camera-mount correction and prior-aligned spawn/reset placement have been implemented separately as prerequisites.
The [README](README.md) describes the current simulator.

## Scope and constraints

Implement four cumulative profiles, selectable in the Simulator panel before startup.
Keep the fifth profile, Vision, for later.
Synthetic image-space observations are in scope because filtering and localization consume them; camera rendering, image transport, neural detection and actual stereo VO are deferred.

All wiring, substitute publishers, calibration overrides and tests belong in `tools/simulate`, with panel/settings changes in `tools/twix`.
Reuse production node entry points and message types unchanged.
Do not edit `crates/nodes`, supporting robotics crates, or shared robot parameter files.
If testing exposes a robotics defect that cannot be resolved at the simulator boundary, report it separately; fixing it requires explicit approval.
Do not conceal such defects with ground truth.

Continue in the existing isolated worktree and branch.
Use linear commits and rebase workflows.
Keep one controlled robot and the existing physics, fly camera, gizmo, ball controls and Booster mode emulation.
Multiple robotics processes, dynamic controlled robots and router-side namespace rewriting remain separate work.

Microphone capture and audio processing are out of scope.
Whistle detection is replaced by a UI-generated detection stream; the real whistle filter runs.

## Profile boundaries

| Profile | Additional real nodes | Remaining simulated observations |
| --- | --- | --- |
| Motion & behavior | Common stack below | Raw joint/IMU samples; perfect ground/camera geometry, field pose, selected ball and obstacles |
| Filtering | `ball_filter`, `visual_kick_ball_selector`, `obstacle_filter`, `search_suggestor` | Raw sensors; synthetic detections; perfect ground/camera geometry, planar odometry and field pose |
| Body state & odometry | `kinematics_provider`, `support_foot_estimator`, `ground_provider`, `camera_matrix_calculator`, `odometry` | Raw sensors, camera calibration, synthetic detections and perfect field pose |
| Localization | `field_mark_association`, `localization_3d`, `localization_2d` | Raw sensors, camera calibration, synthetic detections and camera-motion measurements substituting for stereo VO |

The Rust crate names for the last two nodes are `localization-3d` and `localization-2d`; their runtime node names are `localization3d` and `localization2d`.
The profiles are cumulative in robotics coverage.
Higher profiles remove substitute publishers from production topics whose outputs are now computed by real nodes.
All physical reference streams remain published under `ground_truth/` in every profile, regardless of which estimators are running.

### Common stack

Run these in every profile:

| Responsibility | Nodes |
| --- | --- |
| Behavior | `behavior_node`, `ball_state_composer`, `rule_obstacle_composer` |
| Motion and protection | `motion`, `motion_inference`, `head_motion`, `hardware_interface`, `fall_detection`, `safe_pose_checker` |
| Raw Booster inputs and body buttons | `low_state_bridge`, `button_event_bridge`, `button_event_handler` |
| Gamepad | `controller_handler`, using the existing controller input and behavior path |
| Whistle | `whistle_filter` |
| HSL communications and state | `message_handler`, `message_filter`, `game_controller_filter`, `game_controller_state_filter`, `primary_state_filter`, `player_states_receiver`, `team_ball_filter` |
| Shared configuration and field side | `global_parameter_provider`, `world_to_field_provider` |

`low_state_bridge`, `controller_handler`, `whistle_filter` and `world_to_field_provider` extend the original motion-only simulator.
State filtering for messages, buttons and whistles is common infrastructure, not part of the selectable Filtering increment.

The simulator continues to host these entry points in process.
This work does not claim to simulate the entire unchanged `hulk_ros_z` executable.

## Topic ownership

Every functional output has one designated producer.
Inactive substitutes must not even declare the conflicting production publisher.
Truth for comparison always uses `ground_truth/...` topics, within the robot's existing ROS namespace and global Zenoh scope.
Use that prefix directly, without an additional `simulator/` component.
Production nodes do not subscribe to the reference namespace.

Publish the same available physical reference streams in every profile, even when their real node runs or their production counterpart is unnecessary in that profile.
Mirror production topic suffixes and types where meaningful, for example:

| Always-published reference | Meaning |
| --- | --- |
| `ground_truth/ground_to_field` | Exact planar field pose |
| `ground_truth/robot_to_ground`, `ground_truth/ground_to_robot` | Physical body/ground geometry |
| `ground_truth/robot_kinematics`, `ground_truth/support_foot_state` | MuJoCo link transforms and support derived from physical contacts |
| `ground_truth/camera_matrix` | Exact physical camera geometry and intrinsics |
| `ground_truth/inputs/odometry` | Exact planar movement relative to the trial's odometry origin |
| `ground_truth/ball_filter/ball_position`, `ground_truth/visual_kick/ball_position` | Ideal scalar ball reference with the documented selection policy |
| `ground_truth/obstacles` | Physical obstacle positions |
| `ground_truth/detected_objects` | Ideal visible observations before optional observation perturbations |
| `ground_truth/visual_odometry/current_left_camera_to_previous_left_camera` | Exact camera-motion delta |
| `ground_truth/visual_odometry/current_left_camera_to_visual_odometer` | Exact accumulated camera motion and epoch |

Also expose all scene balls with stable identities under `ground_truth/balls` so a filter choosing a different ball can be evaluated correctly.
Reuse existing types where their meanings match.
Hypothesis confidence, search policy and behavior decisions do not have unique physical ground truth; do not manufacture reference answers for those.
Include timestamps and define how contact truth maps to support states.
Truth remains independent of estimated geometry and configured sensor noise.

Where a profile needs an ideal production input, publish that same physical sample to its production topic too.
Removing this second publication in a higher profile never disables its `ground_truth/` publication.
The table below describes only production ownership, not whether the reference is available.

| Output | Motion & behavior | Filtering | Body state & odometry | Localization |
| --- | --- | --- | --- | --- |
| Raw `rt/low_state`, `rt/joint_states` | Simulator | Simulator | Simulator | Simulator |
| `inputs/low_state`, `inputs/imu_state`, `inputs/serial_motor_states`, `inputs/parallel_motor_states` | Low-state bridge | Low-state bridge | Low-state bridge | Low-state bridge |
| Raw `rt/button_event` | Simulator button controls | Same | Same | Same |
| `inputs/controller_input` | Controller handler | Same | Same | Same |
| `detected_whistle` / `filtered_whistle` | Simulator / whistle filter | Same | Same | Same |
| `robot_kinematics`, `support_foot_state` | Truth | Truth | Real estimators | Real estimators |
| `robot_to_ground`, `ground_to_robot` | Truth | Truth | Ground provider | Ground provider |
| `camera_matrix` | Truth | Truth | Camera-matrix calculator | Camera-matrix calculator |
| `inputs/camera_info` | Simulator calibration | Simulator calibration | Simulator calibration | Simulator calibration |
| `inputs/odometry` | Truth | Truth | Odometry node | Odometry node |
| `current_odometry_to_last_odometry` | Not needed | No producer, matching production | No producer, matching production | No producer, matching production |
| `detected_objects` and its announcements | Not needed | Synthetic observations | Synthetic observations | Synthetic observations, including field marks |
| `ball_filter/ball_position`, `visual_kick/ball_position` | Truth | Real filters/selector | Real filters/selector | Real filters/selector |
| `obstacles` | Truth | Obstacle filter | Obstacle filter | Obstacle filter |
| Ball hypotheses and search suggestions | Not needed | Real filter/suggestor | Real filter/suggestor | Real filter/suggestor |
| Both `visual_odometry/...` outputs | Not needed | Not needed | Not needed | Simulated camera motion |
| Field associations and localization pose/hint | Not needed | Not needed | Not needed | Real association/localization nodes |
| `ground_to_field` | Truth | Truth | Truth | Localization 2D projection |
| `position_of_interest` | Minimal tool fallback using the selected ball topic | Same, using filtered ball | Same | Same |

Do not retain the current attention fallback's direct access to the scene ball in profiles 2–4.
It should follow the currently published ball position, with a fixed forward target when no usable ball exists.
This preserves the necessary substitute for the stub `active_vision` without revealing hidden balls to higher profiles or implementing a second active-vision algorithm.
Search suggestions still reach behavior through its existing subscription.

## Common input and control contracts

### Booster sensors and modes

Replace the current direct low-state/IMU/joint publishers with CDR packets on `rt/low_state` and `rt/joint_states`, using the already scoped Zenoh session.
Generate both from one sampled observation and one timestamp.
The low-state bridge matches joint position, velocity and effort by their `f32` bit patterns; construct the JointState's `f64` values from those exact `f32` values, in the same joint order.
Its header supplies the source timestamp propagated to downstream sensor topics.

Keep serial ankle states as modeled today.
Parallel motor states remain unavailable; do not fabricate a linkage model.
Use the real raw button bridge and handler, plus existing SDK request/reply handling for Damping, Prepare and Custom modes.
Retain startup upright, through the existing safe-pose/button sequence, into Initial.
An already-running HSL GameController may subsequently change primary state.

### Gamepad remote control

Run the existing `controller_handler` on the Twix host.
It polls OS gamepads at 20 ms intervals and publishes `ControllerInput` on `inputs/controller_input`.
Behavior already implements Start-edge toggling, walking axes, head commands and kick controls; the primary-state filter handles the existing damping button.
Preserve that mapping and precedence, including explicit motion overrides in Twix.

No controller connected is a normal state.
Connect/disconnect and held-button behavior must work without restarting.
Behavior rejects controller samples older than 250 ms; test the resulting existing fallback rather than adding a competing simulator command path.
If the OS gamepad backend itself cannot initialize, report gamepad unavailability without treating its optional task's clean exit as failure of the whole robot.
Other required node exits remain fatal.

For a controller hosted in another process/machine, the supported boundary is the same `ControllerInput` topic, ROS namespace and global Zenoh scope.
An explicit advanced input-source choice, Local gamepad or External publisher, must prevent a local disconnected publisher from overwriting external input.
Local is the default; external mode keeps all consumers active and does not require a publisher at Start.
Do not add a custom remote-control panel or a second controller message protocol.
Test external delivery separately from the OS-device capture path.

### Whistle button

Add one Whistle button in the running toolbar.
It activates a short logical whistle pulse in a worker-owned publisher of `Whistle` on `detected_whistle`.
Keep publishing false samples while idle, so the filter clears and subsequent whistles can trigger.

The payload is a vector of per-channel booleans, not a timestamped event structure.
Use the configured detector channel count and logical frame period from sample count/sample rate, without starting audio capture or detection.
The current defaults are six channels and 64 ms per frame.
Use a 750 ms pulse, all channels true, then false.
This covers the upstream GameController's 500 ms update interval; a 200 ms pulse passed whistle filtering but missed the Set-to-Playing check in the actual roundtrip.
Repeated clicks during a pulse extend it; they do not enqueue unbounded whistles.
Source metadata uses the common wall clock.
The real filter supplies its own detection time and applies its own threshold, currently six detections in a 20-entry buffer.
Do not override its output or force acceptance when parameters are changed.
This corrects the earlier single-event suggestion.

Test repeated separated clicks, a below-threshold sequence, clearing the buffer, and a whistle in Set through the real GameController state pipeline.
That pipeline processes whistles when GameController updates arrive; the UI does not fabricate those updates or write primary state directly.

### HSL GameController and parameters

Retain the actual upstream HSL GameController UDP roundtrip, return messages and team messages in all profiles.
External GameController absence is not a startup failure.
Retain session-only parameter layers and the existing scoped transport.
Add only simulator calibration overrides for newly enabled nodes.
Physics field geometry, landmark locations and ball radius must use the same real field parameters.

## Profile 1: Motion & behavior

This is the default and the migration target for saved panels without a profile.
Supply ideal downstream inputs while exercising real behavior, safety and actuation.

Keep one stable selected scene ball for the scalar ball inputs, initially the first remaining ball in creation order, and document this choice.
No ball means `None`.
Publish its actual planar position/velocity and a fresh observation time.
Additional balls remain physical objects here; profile 2 observes all visible balls and lets the filter select its hypothesis.
Publish actual modeled obstacles, initially fixed goalposts; do not create fake controlled robots just to populate the obstacle list.
Use existing obstacle constructors and field dimensions for their footprint.

Publish both directions of the ideal ground transform where needed, ideal camera geometry and the field pose with the current game-controller field-side convention.
Keep truth samples available separately for later comparisons.
Preserve current parameter-based motion overrides, selection, dragging, deletion and mode controls.

Acceptance: start upright in Initial without GC, execute gamepad/head commands and motion overrides, enter Damping/Prepare through raw buttons and SDK requests, update behavior when the selected ball moves/disappears, and complete the actual HSL roundtrip.
Verify sensor timestamps through the raw bridge rather than only direct message injection into downstream consumers.

## Profile 2: Filtering

Replace scalar ball, visual-kick ball and obstacle truth publishers with real nodes.
Publish synthetic `TimeWrapper<Vec<Object<RobocupObjectLabel>>>` on `detected_objects` using the existing announcing-publisher API.
Ball filtering and obstacle filtering use FutureMap subscribers; a plain publisher is insufficient.
An empty visible scene still produces completed, empty frames so filters can age hypotheses and the visual-kick selector can expire its held ball.

### Synthetic observations

Use the MuJoCo camera's true optical pose, intrinsics and world geometry to project all visible balls and goalposts.
Later profiles use this identical physical source, even when their robotics `camera_matrix` is estimated.
Never generate observations by inverting the estimated matrix, which would cancel the very errors under test.

Use the production detection conventions: ball box center projects the ball center; goalpost and robot box bottom-center represents the ground contact point.
Supply finite, nondegenerate boxes and confidence.
Do not shift these anchors by blindly clamping a partly offscreen box.
Skip observations whose required anchor is not visible.
Apply positive-depth and image-bounds checks plus simple geometric occlusion against the scene.
This is idealized visibility, not pixel segmentation.

Preserve actual ball height when projecting.
The current ball filter assumes a ball center at ball-radius height when reconstructing it, so airborne-ball errors are an expected estimator limitation, not a reason to publish a corrected ground point.

Use fixed goalposts for interactive obstacle testing.
Add deterministic robot-shaped observation fixtures in tests without adding an uncontrolled-robot UI.
Dynamic controlled robots remain deferred.

### Odometry inputs

Publish ground-truth planar odometry on `inputs/odometry`, using its existing announcing-publisher contract.
Anchor it at startup, independently of field side.
Use the Ground frame convention expected by the real estimator/filter.

There is no producer for `current_odometry_to_last_odometry` in this checkout, including the main `hulk_ros_z` setup.
Leave it unproduced in the simulator as well, as requested.
The obstacle filter's missing-input fallback is identity, so its prediction does not compensate existing hypotheses for observer movement through that input.
Fresh detections still update them.
Do not add the previously proposed adapter or fail readiness because this intentionally dangling subscriber is empty.
Keep moving-observer tests and ground-truth comparisons to expose the actual production behavior; they must not assume compensation that production lacks.

Acceptance: ball acquisition, motion/velocity estimation, multiple hypotheses, visibility loss and reacquisition, deletion expiry, kick-ball selection, goalpost obstacles while the observer walks/turns, and useful search suggestions after ball loss.
Compare estimates to independent truth.
Start with exact detections, then use repeatable perturbation fixtures to test dropouts and noisy pixel coordinates.

## Profile 3: Body state & odometry

Run the real serial-joint-to-kinematics, support-foot, ground-transform, camera-matrix and odometry chain.
Remove ground/camera truth publishers and truth odometry from the production topic set, retaining all corresponding `ground_truth/` streams.
Keep field pose perfect so this profile isolates body geometry and relative movement errors from global localization errors.
The unused obstacle-delta input stays unproduced, just as in profile 2 and production.

Supply `inputs/camera_info` with the actual intrinsics of the simulated optical camera.
Publish it periodically as well as at startup because the existing consumer uses a normal cache, and late subscribers must receive it.
Camera calibration is static metadata, not an image pipeline.
A stereo baseline is not needed before profile 5.
Define the existing modeled optical camera as the left measurement camera for these profiles; do not silently add a half-baseline offset just because MuJoCo has `ipd`.

Before declaring this profile usable, compare MuJoCo link poses with real kinematics over representative head/leg poses.
Check joint order/signs, foot-to-sole offsets, Ground's origin, IMU axes and gravity, camera axes, mounting orientation and intrinsics.
Use physics/contact geometry for independent reference, not the estimator's answer.

No new calibration procedure is needed.
Populate CameraInfo consistently with the ideal simulated camera: width/height, focal lengths, optical center, matching K/P, identity rectification and zero distortion.
The current 640x544, 94-degree vertical FOV camera gives fx=fy approximately 253.644 pixels and center [320, 272].
The production projection currently reads intrinsics from P, so filling K alone is insufficient.

Camera mounting is a separate input.
`camera_matrix_calculator` gets head-to-camera geometry from fixed kinematic dimensions plus `camera_to_head_pitch`; it cannot learn that mounting from CameraInfo.
It also supports correction rotations, currently zero.
The simulator MJCF mount now matches the production nominal geometry: the existing head-to-camera translation and approximately 0.21258 degrees downward pitch.
The former approximately 12.18-degree upward mounting has been corrected in the simulator asset.
The camera regression compares its full physical transform with production kinematics.
Robotics parameters and correction rotations are unchanged.
Publish the matching CameraInfo when profile 3 is implemented; no additional calibration procedure or mounting-parameter override is needed.

Support-foot output is change-driven, so readiness must accept its retained state instead of demanding a new support message every sensor cycle.
When support is unavailable, preserve the real nodes' `None`/missing-output behavior and visible staleness; never restore perfect transforms to keep the display looking healthy.

Acceptance: stationary stability, head-only motion, both-support and single-support poses, walking/turning, fall and recovery, camera reprojection errors, and odometry drift measured against truth.
Explicitly test with simulated calibration error to show it reaches the filter outputs.
Existing fall/IMU/safe-pose gates remain active.

## Profile 4: Localization

Run actual field-mark association, 3D localization and the 2D projection.
Remove the production `ground_to_field` truth publisher.
Do not inject field associations, the association pose hint, backend resets or absolute field pose from truth.

### Field observations

Extend the existing synthetic detection frame with goalposts, L/T/X intersections and penalty spots using the production labels.
Reuse public `FieldDimensions` geometry helpers to define their physical positions.
Match the real association node's feature conventions: bottom-center for posts, box centers for field marks.
The detections contain class, pixel box and confidence, never the true landmark ID.
The unchanged association node must find correspondences and initialize/recover localization itself.
Keep field geometry in the world frame when sides change.

### Ground-truth stereo-VO substitute

Let `C(t)` be the left optical camera-to-world transform from a physics sample.
Publish the existing message types and topics:

| Topic | Payload |
| --- | --- |
| `visual_odometry/current_left_camera_to_previous_left_camera` | `VisualOdometryDelta` with previous/current timestamps and `inverse(C(previous)) * C(current)` |
| `visual_odometry/current_left_camera_to_visual_odometer` | `VisualOdometer` with current time, epoch and `inverse(C(epoch_start)) * C(current)` |

The first sample of each epoch is identity and produces no delta.
Both outputs derive from one trajectory and share timestamps.
Include head articulation, body roll/pitch, translation and vertical motion.
The production localization code uses camera extrinsics at both endpoints to recover robot motion.

Start with exact motion.
Test noise/drift by perturbing one coherent camera trajectory and deriving both outputs from it, rather than adding unrelated errors to each topic.
A tracking reset increments the epoch, starts from identity and never creates a cross-epoch delta.
Losing field observations must not silently stop VO; these are independent observation sources with independent loss tests.

Localization needs fresh camera matrices at both VO endpoints, currently within 100 ms.
Schedule observations so the real geometry chain can produce these values, without delaying the high-rate sensor worker.
Missing geometry must produce an observable dropped/deferred measurement, not substitution by perfect geometry.

### Initialization at the production prior

Spawn and Reset pose use the placement already expected by localization: `x=-field.length/2, y=-field.width/2, yaw=+90 degrees`, looking into the field.
MuJoCo grounds the prepared robot physically instead of imposing a nominal body height.
Use the configured field dimensions.
This placement is shared by all four profiles and is implemented in the current simulator.

During Damping, localization resets to its existing startup prior and discards normal measurement processing.
Leaving Damping allows field association/acquisition again.
The published `localization` value is None for a startup prior, but the 3D pose/hint topics carry the prior, and the 2D projector can publish it as `ground_to_field`.
Seeing a field-pose message alone therefore does not prove a visual fix.
Distinguish the prior from an acquired result in profile-4 readiness/status.

Per the user's decision, keep baseline acquisition at this expected placement and skip the additional center-spawn, displaced/rotated-robot recovery, symmetry-branch and GameController side-change test scenarios.
Do not add compensating logic for those cases.
Preserve the existing robotics initialization and branch selection.

Profile-4 validation covers acquisition from the expected placement, head-only camera motion, ordinary movement tracking, coherent VO epochs and timestamps, and topic ownership.
Behavior consumes the real resulting field pose.
Verify there is no truth publisher on production `ground_to_field` and no synthetic association hint.
The ground-truth references remain available for normal Twix inspection.

## Time, pause, editing and lifecycle

Retain wall-clock ROS-Z time and the existing live external-network behavior.
This is not deterministic stepped robotics time.
Sample one timestamp per physics observation and propagate it into raw JointState headers, wrapped geometry, announcements, detections, VO and truth diagnostics.
Do not call `now()` independently for different views of the same sample.

Keep raw sensors at the physics-worker cadence, currently typically 2 ms.
During scene rebuilds, skip camera frames that cannot obtain coherent scene geometry and continue stationary raw sensor publication.
Expose capture/publication times and cumulative skipped frames through `diagnostics/camera_frames`.
Start synthetic detections and VO at 30 Hz, independent of repaint rate; use the same frame schedule for their shared camera observations.
Publish truth behavior inputs on the same 33 ms camera-frame cadence.
These are simulator rates, not claims about hardware.
Bound queues, skip missed timer ticks and report overruns.
Do not replay bursts of stale observations after a slow scene rebuild.

The sensor worker must not wait for filtering, association, optimization or UI work.
Give observation projection/publication a bounded snapshot stream.
Before announcing a frame, obtain any required odometry/geometry samples within a bounded window; once announced, complete its payload.
Test the actual FutureMap timing contracts, including the ball filter's short odometry safety lag.
Keep sample time distinct from transport publication time.
If a dependency times out, count the omission; do not invent a successful measurement.

During Pause and scene recompilation, publish a consistent stationary sensor view with fresh timestamps and unchanged positions, zero joint/angular velocities and stationary gravity-specific acceleration.
Do not repeatedly restamp a moving IMU sample while publishing identity camera motion.
Preserve resumable physics state separately.
Nodes, GC, whistles and controller input continue in wall time; filter timeouts continue too.
This extends the existing frozen-snapshot behavior where necessary for inertial localization consistency.

Treat scene changes explicitly:

| Change | Required semantics |
| --- | --- |
| Add/delete/move ball | Natural observation acquisition/expiry; preserve robot odometry and VO epoch across model recompilation |
| Robot gizmo move/rotation | Pause during drag; new VO epoch for discontinuities; actual localization reacquisition without truth reseeding; additional recovery tests excluded |
| Reset pose | Paused physical reset with the same discontinuity semantics; existing filters retained |
| Stop/Start | Clean trial reset of tasks, caches, filter state, odometry anchors and session parameters |
| Change profile | Requires Stop/Start; immutable publisher ownership during a run |

Startup must proceed in phases: validate profile/configuration and parameter layers; obtain field dimensions and prepare physics; start transport/subscribers and sensor publication; run the real startup button sequence; wait for the profile's required outputs.
Avoid waiting for sensor-derived outputs before the physics worker exists.
Use observable readiness and bounded deadlines, not sleeps chosen to hide races.

Profile 1 readiness includes bridged sensors, safe pose, primary state and common outputs.
Profile 2 additionally proves detections/odometry are being consumed and filters emit, including empty output.
Profile 3 additionally requires kinematics, support state and fresh ground/camera/odometry output while upright.
Profile 4 proves its measurement and localization tasks are alive; a globally acquired pose is a separate status because the current view may be ambiguous or empty.

Tag tasks with node names for failures.
Roll back partial starts and close sensors, SDK receivers and sockets on cancellation or Stop.
Repeated start/stop must not leave controllers, UDP listeners or Zenoh publishers behind.
HSL disconnection, missing gamepad and temporarily unobservable landmarks are distinct from task failure.

## Code and UI shape

Add a serializable `Profile` enum with stable IDs `motion_behavior`, `filtering`, `body_state_odometry`, `localization`, defaulting to the first.
One internal profile description determines node startup and substitute ownership.
Keep this a closed, typed configuration; do not build a generic plugin system or arbitrary graph editor.
Tests should inspect the same description used to launch, then verify actual outputs.

Keep responsibilities small in number:

| Area | Owner |
| --- | --- |
| Profile selection, labels and node/substitute membership | New tool-local profile module |
| Context, node lifecycle, parameter layers and supervision | Existing `robotics.rs`, refactored around profile membership |
| Coherent physical observations and scene discontinuities | Existing `robot_io.rs` and `simulation.rs` |
| Raw Booster sensor encoding | Tool-local sensor publisher beside existing SDK/button substitutes |
| Ideal behavior inputs, always-published references and synthetic detections | Tool-local input modules replacing the monolithic `BehaviorInputs` ownership |
| Camera-motion trajectory and VO messages | Tool-local VO substitute |
| Whistle pulse scheduling | Tool-local control publisher, driven by toolbar intent |
| Profile selection/persistence and running status | `tools/twix/src/panels/simulator.rs` plus existing simulator toolbar |

Only add module splits where there is substantial behavior to own.
Reuse existing types and projection math; add dependencies in `tools/simulate/Cargo.toml` on the already available nodes, ROS2 sensor types and `ros-z-streams`.
Ordinary Twix builds remain independent of the optional simulator dependencies.
Lockfile changes, if any, must follow actual dependency resolution rather than unrelated upgrades.

Before Start, show the four choices and a short sentence describing the selected profile's remaining perfect inputs.
Disable configuration changes during startup and while running; capture an immutable start configuration.
Persist the profile with the existing panel settings and retain the legacy panel storage ID.
Do not offer Vision until it exists.
Keep detailed node coverage in an optional disclosure.

During a run, display the selected profile and concise startup/failure/acquisition status.
Add Whistle next to the existing body buttons.
Keep robot tuning in Twix's Parameter panel and output analysis in Text/Plot/Map; do not reintroduce a simulator parameter editor.
Optional observation noise/loss belongs in simulator parameters, with zero-noise defaults.
A full fault editor and scenario UI are deferred; use seeded test fixtures first.

Publish the full physical reference set under `ground_truth/` in all profiles, with reused message types where practical and minimal tool-owned types otherwise.
Include sample times and discontinuity/epoch information.
Changing profiles must not remove references needed to compare the real estimators.
Put publication/drop timing in ordinary simulator diagnostics, rather than calling it ground truth.
Compare estimates at their measurement times, not simply their latest arrival times.
Reuse Twix for plotting those errors.

## Implementation sequence and validation gates

Deliver linear, reviewable increments.
Each gate must pass before the next profile is presented as supported.

1. **Profile assembly and lifecycle.** Add enum, saved settings, immutable startup configuration and node/substitute ownership.
   Explicitly carry all 19 currently running nodes into profile 1: `behavior_node`, `ball_state_composer`, `rule_obstacle_composer`, `motion`, `motion_inference`, `head_motion`, `hardware_interface`, `fall_detection`, `safe_pose_checker`, `button_event_bridge`, `button_event_handler`, `global_parameter_provider`, `message_handler`, `message_filter`, `game_controller_filter`, `game_controller_state_filter`, `primary_state_filter`, `player_states_receiver` and `team_ball_filter`.
   Keep their SDK and input dependencies intact.
   Check saved-layout migration, startup cancellation, repeat start/stop, and default Twix compilation without the simulator feature.
   Keep unfinished profiles unavailable until their implementation gates pass.
2. **Complete the profile-1 node set and shared controls.** Add the four missing common nodes explicitly: `low_state_bridge`, `controller_handler`, `whistle_filter` and `world_to_field_provider`.
   Profile 1 is not complete until the 23-node common set is wired and validated; profiles 2-4 inherit it, with the documented external controller-source exception.
   Implement raw paired sensor packets, gamepad source selection and whistle pulses.
   Remove direct sensor publishers.
   Exercise real button/mode paths, actual HSL roundtrip, raw timestamps, no-gamepad startup and external controller disconnect.
   Verify the native OS gamepad path with a real or virtual input device.
3. **Coherent truth and geometry.** Capture unified samples, camera calibration, all ball poses and discontinuities.
   Establish independent transform fixtures, physical camera conventions and paused-sensor semantics.
   Publish all reference streams under `ground_truth/`, regardless of selected profile; test their presence across all four configurations.
   Move attention fallback to the active ball topic.
   Finish and verify Motion & behavior.
4. **Filtering.** Add synthetic detections and announcing truth odometry; enable the four filtering nodes.
   Leave `current_odometry_to_last_odometry` unproduced.
   Verify hypothesis lifecycle, moving-observer obstacles, empty frames, partial visibility, physical ball height, and FutureMap ordering/latency.
   Enable profile 2.
5. **Body state & odometry.** Add real estimators and periodic camera calibration; remove perfect transforms/odometry from their production topics.
   Verify calibration parity first, then stationary, walking, fall and recovery cases.
   Enable profile 3.
6. **Localization.** Add labeled field observations and coherent VO streams; run association/localization and remove the field-pose substitute.
   Test acquisition, head motion, ordinary tracking, coherent VO epochs and topic ownership.
   Skip the additional recovery, symmetry and GC side-change scenarios as requested.
   Enable profile 4 only after its actual observable behavior is documented.
7. **Cross-profile acceptance and documentation.** Run a scenario matrix through the same production profile launcher, complete native Twix checks, update the README/preset and replace stale audit statements about current controls/coverage.
   Verify the final diff contains no robotics/shared-parameter edits and history remains linear.

Use tests that cross the actual boundary, not replicas of node implementations:

| Layer | Evidence required |
| --- | --- |
| Deterministic input fixtures | Raw packet roundtrip/pairing, analytic camera/VO transforms, visibility/box anchors, coherent timestamps, epoch/reset behavior, seeded observation faults |
| Node integration | Production filters and localizer consume the generated streams; measured output errors, expiry and reacquisition match the named fixture expectations |
| Ownership/isolation | Exactly one producer for designated functional topics; all references remain under `ground_truth/` in every profile; no truth fallback in higher profiles; intentionally missing obstacle-delta input stays absent; scopes `42`/`43` stay isolated for raw and ROS traffic |
| Lifecycle/load | Hidden panel, pause/resume, slow scene rebuild, ball edits, missing observations, failed start, repeated Stop/Start, no sensor starvation |
| External controls | Actual HSL runtime roundtrip including whistle-in-Set; real/virtual OS gamepad and separate external publisher; held/released body buttons |
| Native UI | Persisted selector, disabled changes during startup/run, profile status, Whistle, existing camera/gizmo controls, useful startup errors |

For transform/serialization fixtures use tight numerical assertions.
For integrated estimators define tolerances per scenario and units before accepting results, based on the configured estimator noise and observation rate.
Record acquisition time, position/yaw error, drift and dropped samples; do not choose a broad tolerance after seeing a failure.
Control fixture inputs and random seeds, but do not promise bitwise deterministic asynchronous/physics runs.

Run focused simulator and Twix tests, formatting, and the appropriate clippy/build checks.
Full-stack tests need MuJoCo, motion models and ONNX Runtime as documented; the first four profiles require no detection model or image GPU pipeline.
Preserve the existing real-HSL test harness.
Runtime validation is required for these implementations.

## Explicit deferrals and implementation risks

- Defer rendered stereo images, `image_receiver`/X5 emulation, `detection`, `stereo_visual_odometry`, and the old segment/image pipeline.
  Do not add their dependencies or present their coverage in these four profiles.
- Do not run the stub nodes `active_vision`, `world_state_composer`, `time_to_reach_kick_position`, `motor_commands_collector` or `trigger` as evidence of additional coverage.
  Only the minimal attention substitute is needed here.
- Booster odometer/fall-state firmware topics, LED emulation and MCAP recording are separate optional features.
  They are not prerequisites for these profile boundaries.
- The camera mount has been aligned with the production parameters.
  Retain its transform check and verify joint/sole frame parity when enabling profile 3.
  Any actual robotics correction still requires explicit approval.
- The obstacle-delta subscriber intentionally remains dangling, matching main `hulk_ros_z`.
  Record its identity fallback and any resulting moving-observer tracking errors; do not add simulator compensation for missing production logic.
- Start at the production localization prior.
  Additional arbitrary-placement, symmetry and GC side-change tests are excluded by the user's decision; do not make them implementation gates or introduce simulator workarounds for them.
- Correct sensor publication during pauses and scene edits matters more once IMU integration runs.
  Measuring queue delays and coherent stationary observations is a prerequisite, not a later performance polish.

## Validation record and retained limitations

Profiles 1–3 have passed the same production launcher with real physics, motion models and ONNX Runtime.
The latest profile-3 fixture measured 0.047 m maximum walking odometry error over three seconds, less than 1 mm paused drift, and about 0.1 mm initial camera translation difference.
The filter fixtures cover rolling-ball velocity, seeded 2 px noise, total detection loss, reacquisition, multiple hypotheses, nearest kick-ball selection and deletion expiry.
The robot-obstacle fixture exercises the actual filter through announcing publishers and verifies acquisition and timeout.
Raw sensor pairing and source timestamps are tested through the unchanged low-state bridge.
A 300 ms model-lock fixture verifies continuing raw sensors, omitted camera frames and resumed coherent publication.
Both raw SDK request/reply and ROS topics are isolated between global scopes `42` and `43`.

The actual HSL runtime roundtrip passes Ready, pickup penalty, penalty removal, accepted return messages and whistle-in-Set with no fabricated GameController packets.
Twix tests pass with the simulator feature enabled and disabled.
Native Twix starts profiles 1–3 and preserves the profile choice across application restart.
The local controller node starts without a device and the external controller path passes walking/disconnect tests.
OS gamepad capture remains unverified because this environment exposes neither an input device nor `uinput`.

Moving-observer obstacle comparisons expose existing production limitations rather than compensating for them.
Goalpost outputs differed from arrival-time truth by up to 0.44 m in the walking fixture; those outputs contain no measurement timestamp, so this includes processing delay.
The missing odometry-delta input remains unproduced.
The obstacle filter's fixed map also offsets post centers by `(goal_post_diameter - line_width)/2` relative to the shared field helper, or 0.02 m with the default dimensions.
These findings do not authorize robotics changes.

Profile 4's 13-landmark startup frame reproduces a separate production limitation.
The field-association solver stops at 2,501 lookup hits and rejects the truncated search, despite having valid candidate matches.
A debugger replay allowing the complete search uses 6,654 lookups and 56 refined seeds, and returns an acquired pose without changing observations, geometry or acceptance gates.
The proposed correction raises `MAX_TRIPLET_LOOKUP_HITS` from 2,500 to 10,000 in `crates/nodes/field_mark_association/src/global_association/solver/types.rs`.
The user chose to retain the existing robotics code and report the limitation instead of applying that correction.
No robotics source file has been changed for this implementation.
The captured frame is retained in `tests/fixtures/localization_startup.json`; its acquisition regression test is explicitly ignored with the production limitation stated in its reason.
The separate `localization_acquisition_and_tracking` test also remains ignored and will fail at acquisition with current production code.
The shared startup test validates profile 4's live input chain, VO ingestion, topic ownership and honest acquisition status, without claiming an acquired pose or validated tracking.
Profile 4 is implemented with this accepted limitation; acquisition and tracking remain unvalidated.

## Source references

- Current assembly and inputs: [robotics.rs](src/robotics.rs), [behavior_inputs.rs](src/behavior_inputs.rs), [robot_io.rs](src/robot_io.rs), [simulation.rs](src/simulation.rs), [Twix panel](../twix/src/panels/simulator.rs).
- Raw sensor matching: [low_state_bridge](../../crates/nodes/low_state_bridge/src/lib.rs).
- Common controls: [controller_handler](../../crates/nodes/controller_handler/src/lib.rs), [behavior input freshness](../../crates/nodes/behavior_node/src/node.rs), [whistle_filter](../../crates/nodes/whistle_filter/src/lib.rs), [game state filter](../../crates/nodes/game_controller_state_filter/src/lib.rs).
- Filtering contracts: [ball_filter](../../crates/nodes/ball_filter/src/lib.rs), [obstacle_filter](../../crates/nodes/obstacle_filter/src/lib.rs), [announcing publisher](../../crates/ros-z-streams/src/announce.rs).
- Estimated geometry: [ground_provider](../../crates/nodes/ground_provider/src/lib.rs), [camera_matrix_calculator](../../crates/nodes/camera_matrix_calculator/src/lib.rs), [odometry](../../crates/nodes/odometry/src/lib.rs).
- Localization contracts: [feature extraction](../../crates/nodes/field_mark_association/src/features.rs), [VO messages](../../crates/types/src/visual_odometry.rs), [localization node](../../crates/nodes/localization-3d/src/node.rs), [startup prior](../../crates/nodes/localization-3d/src/pose.rs), [symmetry branch selection](../../crates/nodes/field_mark_association/src/global_association/solver/output.rs), [IMU yaw constraints](../../crates/localization-factrs/src/factors/imu/relative_yaw.rs), [field side provider](../../crates/nodes/world_to_field_provider/src/lib.rs), [2D projection](../../crates/nodes/localization-2d/src/lib.rs).

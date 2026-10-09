# Simulator in Twix

The Simulator panel runs one to five K1 robots per team with independent real motion, inference, fall detection, behavior and HSL message nodes.
MuJoCo supplies physics and simulated sensor inputs.
The panel reuses `egui_bevy::BevyWidget`, as the legacy simulator did.
All simulator wiring and hardware substitutes live under `tools/simulate`; robotics crates need no simulator integration code.

Select a cumulative profile before starting.
The simulator sits on three explicitly requested upstream squashes: `oleflb/feat/localization-improvements`, `schluis/dev/ball-filter-preferred-20261004` and `BenSampaolo/ball-search-behavior`.
The new localization solver acquires at the default spawn; the former lookup-budget limitation no longer applies.
The walking/head-motion fixture measured 0.0209 m maximum position error and 0.0072 rad maximum yaw error over three seconds of simulation time, using production parameters and no truth pose injection.
The [profile plan](PROFILES.md) records topic ownership and validation contracts.

| Profile | Additional real nodes | Remaining ideal inputs |
| --- | --- | --- |
| Motion & behavior | Common 24-node stack, including raw sensor/button bridges, gamepad, LED handler, whistle filter and HSL communications | Body geometry, odometry, field pose, first ball, other robots and fixed goalposts |
| Filtering | Ball filter, visual-kick selector, obstacle filter, search suggestor | Body geometry, odometry and field pose; synthetic camera detections |
| Body state & odometry | Kinematics, support foot, ground provider, camera matrix and odometry | Field pose; synthetic camera detections and camera calibration |
| Localization | Field-mark association and 3D/2D localization | Synthetic camera detections, calibration and coherent camera-motion measurements |

Changing a profile requires Stop/Start.
Profile selection and gamepad source persist with the panel; old layouts default to Motion & behavior and Local gamepad.
Vision and microphone processing remain out of scope.

## Run

Fetch Git LFS assets, including the K1 meshes and `etc/neural_networks/*.onnx`.
The simulator needs MuJoCo 3.9 and a compatible ONNX Runtime shared library.
`mujoco-rs` downloads MuJoCo during the first build.
Set `ORT_DYLIB_PATH` to your ONNX Runtime library if it is not available in the loader's search path.

```sh
./twix --simulator /hulks/1
```

The launcher enables Twix's optional `simulator` Cargo feature and configures its MuJoCo library path.
Robot N uses global Zenoh prefix `hulk_simulator/hulks/N` and ROS namespace `/hulks/N`, including raw Booster topics, discovery, parameters and RPCs.
The top-left namespace selector switches Twix and simulator body controls between `/hulks/1` through `/hulks/5` and `/opponents/1` through `/opponents/5`.
The namespace selector autocompletes live graph namespaces, including robots discovered across the simulator scopes.
Type to filter, use arrow keys and Enter or click a suggestion; Ctrl+Space opens completions without typing.
Team names identify namespaces; GameController uses team 24 for HULKs and team 5 for opponents.
Selecting an absent player dims its name and disables its body controls.
Without `--router`, simulator mode starts a private loopback router automatically.
Ordinary Twix builds keep their existing dependencies.
With Cargo directly, enable `--features simulator`, pass `--zenoh-namespace hulk_simulator`, and set the MuJoCo library path yourself.
The current Nix Twix package builds ordinary Twix; it does not bundle the simulator.

Open the **Simulator** preset or add a **Simulator** panel.
Select one to five HULKs, zero to five opponents, a field location, profile and gamepad source, then press **Start simulator**.
The field preview draws the selected location's parameter dimensions and the spawn slots for both teams.
Parameter and model directories are under **Parameter and model paths**.
Twix selects the robot's namespace automatically.
Only one simulator can run per Twix process.
Restoring a saved layout does not start robotics or bind UDP sockets.

Robots start upright and paused in **Initial**.
Initial players alternate between both long touchlines in their own half, starting 0.5 m from the goal line and spaced 0.9 m toward midfield, facing inward.
**Add robot** starts the next available player, at a vacant slot along the sideline, without moving existing robots.
Placement follows the known GameController field side, defaulting to Home for HULKs and Away for opponents before the first packet.
A running match continues during startup; a paused match keeps the existing clock and poses frozen.
Reset pose returns the selected robot to its original spawn slot, mirrored if GameController has changed its field side.
Floating player numbers use each robot's actual commanded LED color; the selected player's number is ringed.
Startup uses the real button bridge, handler and safe-pose check to pass through Prepare to Initial.
While running, the panel header is the toolbar.
Its first row controls the whole simulation: **Run**/**Pause**, simulation time, **Whistle**, **Add ball**, **Add robot** and **Stop simulator**.
Its second row controls the selected player: LED, **F1**, **Stand**, **Walk**, **Reset pose** and the fly camera.
Narrow panels show icons for the scene actions; hover them for their names.
Press **Run** to advance physics.
If GameController is already running, its game state takes over.

**F1**, **Stand** and **Walk** emit CDR `ButtonEventMsg` packets on `rt/button_event` within the simulator's global Zenoh scope.
The unchanged `button_event_bridge` and `button_event_handler` process them.
Tap F1 for Damping, tap Stand for Prepare; hold Stand for one second and release to enter Initial once the pose is safe.
Hold Walk and release to enter Playing from Initial.
The one-second hold threshold is a simulator convention.
A bar under the held button fills toward the threshold.
Short presses emit PressDown/PressUp/SingleClick; long presses emit PressDown/LongPressStart/LongPressHold/LongPressEnd/PressUp.
The simulator implements passive damping and a two-second Prepare joint trajectory behind the existing Booster mode RPC, including requests from hardware_interface.
These are approximations of the manufacturer's controller, not firmware emulation.
See the [K1 body controls](https://docs.booster.tech/docs/product-manual/k1/basic-operations/body-operations/).
HULK's long-press actions differ from Booster's default firmware WALK action.

The header's **LED** rectangle displays the actual color commanded by `led_handler` through `hardware_interface` and `rt/LightControlApiTopicReq`.
The simulator acknowledges the Booster light RPC and follows color changes and Stop-state blinking in every profile.
Before the first command or after releasing LED control, the rectangle is neutral; the tooltip reports that no command is active.
Firmware-owned LED colors after release are not simulated.
Inspect primary state, localization and Game Controller connection status in Twix or GameController; the simulator header does not repeat them.

The default **Local gamepad** source runs the real controller handler on the Twix host.
It routes input only to the robot selected in Twix.
Release and press Start after switching robots to enable routing; holding Start across a switch does not activate the new robot.
Subsequent Start presses toggle behavior's remote mode; walking axes, head controls and kicks use the existing robot mappings.
Inactive robots receive disconnected controller inputs.
Inspect `inputs/controller_input` and `behavior/blackboard.remote_control_enabled` in Twix to check capture and activation.
**External controller** disables only the local producer and accepts `ControllerInput` on `inputs/controller_input` in the same robot namespace and global Zenoh scope.
Behavior stops using stale controller input after its existing 250 ms freshness window.
No connected gamepad is required for startup.

**Whistle** publishes a 750 ms detection pulse to every robot through the real whistle filter, with false samples between pulses.
Repeated clicks extend the current pulse.
The pulse spans the GameController's 500 ms update interval, because the state filter checks whistles when those updates arrive.
Use the real GameController to enter Set before testing the whistle-to-Playing transition.

Motion overrides use the Parameter panel: select the relative node `behavior_node`, path `control.injected_motion_command`.
Set it to `null` to let behavior choose motion.
The preset opens this parameter; press **Refresh** once the simulator has started.

**M** or **Fly camera** captures/releases the mouse; **Esc** releases it.
While captured, a legend at the bottom of the scene lists the movement keys.
While captured, use W/A/S/D to move, Q/E for down/up, and Shift for faster movement.
**Add ball** places a ball one meter from the origin.
Select and drag it on the horizontal plane, or press **Delete** while pointing inside the scene to remove it.
Select the robot to show translation arrows and rotation rings.
Dragging pauses physics and restores its previous running state on release.
Reset pose pauses and returns the robot to its standing spawn pose.
Stop and Start rebuild the node stack and clear a latched emergency stop.
Blue arrows show commanded translation, green shows yaw rate, amber shows kicks.

The physics worker advances a shared ROS-Z logical clock independently of the visible panel and repaint rate.
Each running physics step advances robotics time by the MuJoCo timestep, and every sensor view of that step shares its timestamp.
Pause and scene recompilation freeze both clocks and sensor sampling.
Resume continues from that time without catching up elapsed wall time; pose resets and scene edits never rewind the clock.
A new robot initializes from stationary sensor samples at the shared clock time, then joins the common physics scene in Initial.
Camera frames use simulation time at approximately 30 Hz.
`diagnostics/camera_frames` reports capture/publication times and cumulative overwritten queued snapshots.
A whistle clicked while paused is retained for resume; its 750 ms pulse uses simulation time.

The external GameController keeps its own match clock, and incoming packets, Twix parameters and body buttons remain live while paused.
Behavior-generated return messages pause with behavior, so the GameController may mark the robot disconnected during a long pause.
Local gamepad capture and its source timestamps use wall time, matching external gamepads and the existing behavior freshness check.
LED blinking, transport deadlines and UI button holds also retain their existing wall-time behavior.
This freezes robotics clock-driven processing; it does not suspend every asynchronous callback or promise deterministic execution.
Neural policies and contacts still need validation against hardware.

## GameController

Use [RoboCup-HumanoidSoccerLeague/GameController](https://github.com/RoboCup-HumanoidSoccerLeague/GameController).
Start an actual game for team 24 with up to five players, and team 5 when opponents are enabled.
A simulator-owned UDP relay receives state on UDP 3838 and forwards the original bytes to each production endpoint on a separate private port.
Every robot parses the message, applies its own player number and emits a real return message, which the relay sends to the external sender on UDP 3939.
Simulated teammates broadcast on an ephemeral shared port over loopback.
The relay forwards a copy to GameController on UDP 10024 for accounting; ordinary robot broadcasts on 10024 do not enter the simulated team.

The simulator runs `message_handler`, `message_filter`, `game_controller_filter`, `game_controller_state_filter`, `primary_state_filter`, `player_states_receiver` and `team_ball_filter`.
Behavior sends the real return and team messages through that handler.
The simulator does not publish fabricated filtered game or primary states.
Use GameController for game phases, penalties, scores, sides and restarts.

GameController and Twix can run on the same machine, in either startup order.
Select a GameController network interface that can broadcast to the simulator host.
When using separate machines, ensure UDP can cross the network.
The automated roundtrip below runs both applications in one private network without changing the host's interfaces.

## Parameters and observations

The parameter layers are the selected root's `base`, this tool's `parameters`, the chosen `location`, private player/network/calibration overrides, then an empty writable session layer for each robot.
Twix writes to the last layer, which is deleted when the simulator stops.
Save an intentional tuning result separately if it should survive a restart.
Repository robot parameters stay intact.
Select the ONNX model directory separately in the panel before starting.
It defaults to the repository's `etc/neural_networks` and is resolved independently of the working directory.
The tool calibrates safe-pose checks and the stand-up pose to its SDK preparation pose.
These are simulated hardware calibration values, not changes to the checks.

Use Twix's Parameter panel for behavior, motion, inference, fall detection and ball parameters.
The location selector discovers all directories under the chosen parameter root's `location` directory.
Field dimensions load from layered `global.json5` files and are passed to every robot's real global parameter provider.
Goal height loads from the location's `simulator.json5` file.
The added `hsl_small`, `hsl_middle` and `hsl_large` locations contain the exemplary 2026 HSL v1.1.1 fields of 9×6 m, 14×9 m and 22×14 m.
Add another location directory to make it available without changing simulator code.
Field geometry is shared and selected before startup; edit the location files and restart to change it.
Runtime goal-height edits are rejected, and changing a robot's global field dimensions stops the worker with an error to prevent divergent worlds.
The simulator node owns ball contact and mass settings.
Use Text, Plot, Map and Behavior tree panels for outputs.

Every profile publishes independent references under `ground_truth/`, even when the corresponding real estimator runs.
These include body/link/camera transforms, contact support, field pose, planar odometry, all scene balls with stable identities, scalar ball references, other-robot and fixed goalpost obstacles, ideal visible detections and both camera-motion streams.
Production nodes never consume this reference namespace.
Profile 1 uses the first remaining ball in creation order for its scalar ball inputs; higher profiles observe all visible balls and let the real filter choose.

Synthetic detections use the physical optical camera, image bounds and scene occlusion.
Ball boxes preserve the projected 3D center; goalpost boxes preserve their ground-contact anchor.
CameraInfo describes the simulator's actual intrinsics and zero distortion.
The camera-motion substitute includes body and articulated head motion, with coherent delta/accumulated poses and a new epoch after a robot gizmo move or reset.
Ball edits preserve the epoch.

Set `simulator.observations.pixel_noise_std_dev`, `detection_dropout_probability` and `seed` through the Parameter panel for repeatable detection faults.
Defaults are exact observations with no loss.
Noise never changes ground truth.
No producer is added for the obsolete `current_odometry_to_last_odometry` topic; the preferred obstacle filter uses absolute odometry.
During a three-second simulated walking fixture, goalpost outputs differed from current truth by up to 6.11 cm.
These outputs lack a measurement timestamp, so that diagnostic includes processing delay as well as hypothesis lag.
The obstacle filter also places mapped posts 2 cm beyond the shared field helper's goal-line centers with the default field parameters.

This tool hosts selected unchanged node entry points in Twix; it does not launch the complete `hulk_ros_z` executable.
Image rendering/transport, neural detection and actual stereo VO are deferred to a later profile.
See the [UI and multiple-robot audit](AUDIT.md) for the process boundary.

## Self-play

Set the opponent count above zero before starting, or choose **opponents** from **Add robot** during a session.
Both teams run the same selected profile and share the ball, contacts, field and simulation clock.
Each side uses player numbers 1 through 5 and starts on its own half's sidelines.
Dynamic additions use the team's current GameController field side and skip occupied slots.
Badges show H1–H5 or O1–O5 with the robot's actual LED color; hovering shows its namespace.
Select `/opponents/N` in Twix to inspect or control an opponent, including the body buttons and local gamepad.
Switching robots requires releasing and pressing Start again before gamepad input reaches the new robot.

Run the external HSL GameController with teams 24 and 5, matching the simulator's field class and player count.
Use its normal controls for kickoff, penalties, goals and side assignment.
There is no automatic referee or scoring detection.
Before receiving GameController data, HULKs defaults to the home side and opponents to away.
Changing sides through GameController changes the robots' field interpretation; reposition robots with the gizmo or reset pose as needed.

The unmodified production protocol hardcodes HULKs' team number as 24.
The simulator adapts opponents' incoming protocol-20 packets by exchanging team identifiers 24 and 5, including the kicking team, while preserving team order and all other fields.
It rewrites opponents' protocol-4 return team identifier to 5; poses remain in the production team's coordinate frame.
Each team has a separate private UDP teammate channel, with copies forwarded to GameController ports 10024 and 10005 for message accounting.
Opponent adaptation rejects unsupported packet versions and other team pairings.
All raw Booster and ROS traffic is scoped under `hulk_simulator/hulks/N` or `hulk_simulator/opponents/N`.
No robotics source changes are required.

## Verify

```sh
export MUJOCO_DOWNLOAD_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/mujoco-rs"
export LD_LIBRARY_PATH="$MUJOCO_DOWNLOAD_DIR/mujoco-3.9.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo test -p simulate --lib
cargo test -p twix --features simulator --bin twix
# With ONNX Runtime/models available and no simulator using the GameController ports:
cargo test -p simulate startup_pause_resume_and_scene_edits_preserve_time -- --ignored
SIMULATOR_TEST_PROFILE=localization cargo test -p simulate five_players_dynamic -- --ignored
SIMULATOR_TEST_OPPONENTS=2 SIMULATOR_TEST_ADD_RUNNING=1 cargo test -p simulate five_players_dynamic -- --ignored

git clone https://github.com/RoboCup-HumanoidSoccerLeague/GameController /tmp/hsl-game-controller
HSL_PLAYERS=5 SIMULATOR_TEST_ROBOTS=5 SIMULATOR_TEST_OPPONENTS=5 HSL_TEST_FILTER=five_players_dynamic tools/simulate/tests/hsl/roundtrip.sh /tmp/hsl-game-controller
```

The last test requires Linux user/network namespaces, `ip`, Python 3, and the upstream Rust build dependencies including libclang.
It compiles the upstream `game_controller_runtime` used by the GUI, drives its normal action API, and connects it to HULK's production message nodes in the same private network.
It checks ten accepted return streams, team-specific pickup penalties and removal, kickoff ownership, opposing field sides, production walking commands, and a whistle-in-Set transition through the real state filters while upstream still reports Set.
It then verifies that the upstream match countdown advances while robotics time is frozen, return messages stop, and connection status recovers after resume.
No synthetic GameController packets are used in this test.
The multi-robot runtime test also sends distinct teammate markers through the real message handlers and filters, requiring same-team delivery and rejecting cross-team delivery.
The GUI itself is not driven.
Tested against upstream commit `af7d12962c671b631dee7b293df70e6f7b8bb491`, protocol 20, return protocol 4.

The ordinary unit suite also checks UDP team-message reception, SDK request/reply isolation for global Zenoh prefixes `42` and `43`, physics model recompilation, sensor transforms, command vectors, fall-detection poses, gizmo geometry, and raw body-button delivery through the real bridge, handler and primary-state filter.
The optional startup test launches profiles sequentially and checks producer ownership, independent truth, whistles, external gamepad/disconnect, filtering and body estimation.
Set `SIMULATOR_TEST_PROFILE` to `motion_behavior`, `filtering`, `body_state_odometry` or `localization` to select one.
It pauses longer than a whistle pulse and verifies frozen physics, robotics time, sensor timestamps and behavior timers, followed by resume without a catch-up jump.
It also holds the scene-edit lock for 300 ms and checks that the shared clock freezes, caches remain fresh in simulation time and camera frames resume coherently.
For profile 4 it checks the live input chain, VO ingestion and acquisition status.
The captured-frame regression `localization_startup_frame_acquires_with_production_parameters` now runs in the ordinary unit suite and passes with the new production association solver.
Run `localization_acquisition_and_tracking` explicitly with `--ignored` to exercise live acquisition and walking with ONNX Runtime and motion models.
OS gamepad capture could not be tested here because no input device or `uinput` was available; external gamepad messages and the no-device local startup path were tested.

On the tested Linux environment, dynamically loading ONNX Runtime 1.22.1 caused an exit-time crash in its global environment destructor.
Preloading the same library made startup, rendering and shutdown succeed:

```sh
LD_PRELOAD="$ORT_DYLIB_PATH${LD_PRELOAD:+:$LD_PRELOAD}" ./twix --simulator /hulks/1
```

This is an environment-specific workaround, not a change to motion inference.

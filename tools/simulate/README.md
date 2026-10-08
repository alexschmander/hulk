# Simulator in Twix

The Simulator panel runs one K1 with the real motion, inference, fall detection, behavior and HSL message nodes.
MuJoCo supplies physics and simulated sensor inputs.
The panel reuses `egui_bevy::BevyWidget`, as the legacy simulator did.
All simulator wiring and hardware substitutes live under `tools/simulate`; robotics crates need no simulator integration code.

Select a cumulative profile before starting.
Profiles 1–3 are validated; profile 4 currently remains in acquisition at the default spawn because the production field-association solver exhausts its lookup budget.
The robotics correction is intentionally deferred at the user's request; no landmarks or truth poses are substituted to bypass it.
The [profile plan](PROFILES.md) records topic ownership and validation contracts.

| Profile | Additional real nodes | Remaining ideal inputs |
| --- | --- | --- |
| Motion & behavior | Common 23-node stack, including raw sensor/button bridges, gamepad, whistle filter and HSL communications | Body geometry, odometry, field pose, first ball and fixed goalposts |
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
./twix --simulator /simulator/robot
```

The launcher enables Twix's optional `simulator` Cargo feature and configures its MuJoCo library path.
It also scopes Twix and the simulator to the global Zenoh prefix `hulk_simulator`, including raw Booster topics and discovery.
Without `--router`, simulator mode starts a private loopback router automatically.
Ordinary Twix builds keep their existing dependencies.
With Cargo directly, enable `--features simulator`, pass `--zenoh-namespace hulk_simulator`, and set the MuJoCo library path yourself.
The current Nix Twix package builds ordinary Twix; it does not bundle the simulator.

Open the **Simulator** preset or add a **Simulator** panel.
Select a profile, then press **Start simulator**.
Twix selects the robot's namespace automatically.
Only one simulator can run per Twix process.
Restoring a saved layout does not start robotics or bind UDP sockets.

The robot starts upright and paused in **Initial**, at localization's expected sideline placement: field coordinates `(-length/2, -width/2)`, facing `+90°` into the field.
Reset pose returns it to that placement.
Startup uses the real button bridge, handler and safe-pose check to pass through Prepare to Initial.
Press **Run** to advance physics.
If GameController is already running, its game state takes over.

**F1**, **Stand** and **Walk** emit CDR `ButtonEventMsg` packets on `rt/button_event` within the simulator's global Zenoh scope.
The unchanged `button_event_bridge` and `button_event_handler` process them.
Tap F1 for Damping, tap Stand for Prepare; hold Stand for one second and release to enter Initial once the pose is safe.
Hold Walk and release to enter Playing from Initial.
The one-second hold threshold is a simulator convention.
Short presses emit PressDown/PressUp/SingleClick; long presses emit PressDown/LongPressStart/LongPressHold/LongPressEnd/PressUp.
The simulator implements passive damping and a two-second Prepare joint trajectory behind the existing Booster mode RPC, including requests from hardware_interface.
These are approximations of the manufacturer's controller, not firmware emulation.
See the [K1 body controls](https://docs.booster.tech/docs/product-manual/k1/basic-operations/body-operations/).
HULK's long-press actions differ from Booster's default firmware WALK action.

The default **Local gamepad** source runs the real controller handler on the Twix host.
Press the controller's Start button to toggle behavior's remote mode; walking axes, head controls and kicks use the existing robot mappings.
**External controller** disables only the local producer and accepts `ControllerInput` on `inputs/controller_input` in the same robot namespace and global Zenoh scope.
Behavior stops using stale controller input after its existing 250 ms freshness window.
No connected gamepad is required for startup.

**Whistle** publishes a 750 ms detection pulse through the real whistle filter, with false samples between pulses.
Repeated clicks extend the current pulse.
The pulse spans the GameController's 500 ms update interval, because the state filter checks whistles when those updates arrive.
Use the real GameController to enter Set before testing the whistle-to-Playing transition.

Motion overrides use the Parameter panel: select `/simulator/robot/behavior_node`, path `control.injected_motion_command`.
Set it to `null` to let behavior choose motion.
The preset opens this parameter; press **Refresh** once the simulator has started.

**M** or **Fly camera (M)** captures/releases the mouse; **Esc** releases it.
While captured, use W/A/S/D to move, Q/E for down/up, and Shift for faster movement.
**Add ball** places a ball one meter from the origin.
Select and drag it on the horizontal plane, or press **Delete** while pointing inside the scene to remove it.
Select the robot to show translation arrows and rotation rings.
Dragging pauses physics and restores its previous running state on release.
Reset pose pauses and returns the robot to its standing spawn pose.
Stop and Start rebuild the node stack and clear a latched emergency stop.
Blue arrows show commanded translation, green shows yaw rate, amber shows kicks.

The physics worker publishes sensors independently of the visible panel and repaint rate.
Scene recompilation freezes physics while the worker keeps publishing the last stationary raw sensor sample, so adding or deleting a ball does not interrupt sensor delivery.
Camera frames are skipped during a scene rebuild rather than mixing old pixels with a newer pose.
`diagnostics/camera_frames` reports capture/publication times and cumulative skipped frames, including overwritten queued snapshots.
Pausing freezes physics while publishing stationary observations; the real nodes, network and wall clock continue running.
This is not deterministic stepped robotics time.
Neural policies and contacts still need validation against hardware.

## GameController

Use [RoboCup-HumanoidSoccerLeague/GameController](https://github.com/RoboCup-HumanoidSoccerLeague/GameController).
Start an actual game for team 24, default player 3.
The default HULK team broadcast address is `10.0.255.255`.
GameController state arrives on UDP 3838, HULK returns status to its sender on UDP 3939, and team communication uses UDP 10024.

The simulator runs `message_handler`, `message_filter`, `game_controller_filter`, `game_controller_state_filter`, `primary_state_filter`, `player_states_receiver` and `team_ball_filter`.
Behavior sends the real return and team messages through that handler.
The simulator does not publish fabricated filtered game or primary states.
Use GameController for game phases, penalties, scores, sides and restarts.

GameController and Twix can run on the same machine, in either startup order.
Both receivers share the team's UDP port and receive team broadcasts.
Select a GameController network interface whose broadcast address matches HULK's configured address.
When using separate machines, ensure UDP can cross the network.
The automated roundtrip below runs both applications in one private network without changing the host's interfaces.

## Parameters and observations

The parameter layers are the selected root's `base`, this tool's `parameters`, session calibration, then an empty writable session layer.
Twix writes to the last layer, which is deleted when the simulator stops.
Save an intentional tuning result separately if it should survive a restart.
Repository robot parameters stay intact.
Select the ONNX model directory separately in the panel before starting.
It defaults to the repository's `etc/neural_networks` and is resolved independently of the working directory.
The tool calibrates safe-pose checks and the stand-up pose to its SDK preparation pose.
These are simulated hardware calibration values, not changes to the checks.

Use Twix's Parameter panel for behavior, motion, inference, fall detection, field and ball parameters.
Field dimensions come from the real global parameter provider; there is no second simulator field configuration.
The simulator node owns ball contact and mass settings.
Use Text, Plot, Map and Behavior tree panels for outputs.

Every profile publishes independent references under `ground_truth/`, even when the corresponding real estimator runs.
These include body/link/camera transforms, contact support, field pose, planar odometry, all scene balls with stable identities, scalar ball references, fixed goalpost obstacles, ideal visible detections and both camera-motion streams.
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
No producer is added for `current_odometry_to_last_odometry`, matching production; the obstacle filter therefore retains its existing identity fallback for that input.
During a three-second walking fixture, goalpost outputs differed from current truth by up to 44 cm.
These outputs lack a measurement timestamp, so that diagnostic includes processing delay as well as hypothesis lag.
The obstacle filter also places mapped posts 2 cm beyond the shared field helper's goal-line centers with the default field parameters.

This tool hosts selected unchanged node entry points in Twix; it does not launch the complete `hulk_ros_z` executable.
Image rendering/transport, neural detection and actual stereo VO are deferred to a later profile.
See the [UI and multiple-robot audit](AUDIT.md) for the process boundary.

## Verify

```sh
export MUJOCO_DOWNLOAD_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/mujoco-rs"
export LD_LIBRARY_PATH="$MUJOCO_DOWNLOAD_DIR/mujoco-3.9.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo test -p simulate --lib
cargo test -p twix --features simulator --bin twix
# With ONNX Runtime/models available and no simulator using the GameController ports:
cargo test -p simulate startup_reaches_initial_and_scene_edits_keep_sensors_live -- --ignored

git clone https://github.com/RoboCup-HumanoidSoccerLeague/GameController /tmp/hsl-game-controller
tools/simulate/tests/hsl/roundtrip.sh /tmp/hsl-game-controller
```

The last test requires Linux user/network namespaces, `ip`, Python 3, and the upstream Rust build dependencies including libclang.
It compiles the upstream `game_controller_runtime` used by the GUI, drives its normal action API, and connects it to HULK's production message nodes in the same private network.
It checks Ready, a pickup penalty, penalty removal, accepted return messages, and a whistle-in-Set transition through the real state filters while upstream still reports Set.
No synthetic GameController packets are used in this test.
The GUI itself is not driven.
Tested against upstream commit `af7d12962c671b631dee7b293df70e6f7b8bb491`, protocol 20, return protocol 4.

The ordinary unit suite also checks UDP team-message reception, SDK request/reply isolation for global Zenoh prefixes `42` and `43`, physics model recompilation, sensor transforms, command vectors, fall-detection poses, gizmo geometry, and raw body-button delivery through the real bridge, handler and primary-state filter.
The optional startup test launches profiles sequentially and checks producer ownership, independent truth, whistles, external gamepad/disconnect, filtering and body estimation.
Set `SIMULATOR_TEST_PROFILE` to `motion_behavior`, `filtering`, `body_state_odometry` or `localization` to select one.
It also holds the scene-edit lock for 300 ms and checks that raw sensors remain fresh while camera frames are skipped and resume coherently.
For profile 4 it checks the live input chain, VO ingestion and acquisition status.
The separate `localization_startup_frame_acquires_with_production_parameters` and `localization_acquisition_and_tracking` tests remain explicitly ignored because of the known production search-budget limitation.
Running either explicitly currently fails at acquisition; tracking is not validated.
OS gamepad capture could not be tested here because no input device or `uinput` was available; external gamepad messages and the no-device local startup path were tested.

On the tested Linux environment, dynamically loading ONNX Runtime 1.22.1 caused an exit-time crash in its global environment destructor.
Preloading the same library made startup, rendering and shutdown succeed:

```sh
LD_PRELOAD="$ORT_DYLIB_PATH${LD_PRELOAD:+:$LD_PRELOAD}" ./twix --simulator /simulator/robot
```

This is an environment-specific workaround, not a change to motion inference.

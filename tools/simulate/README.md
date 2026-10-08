# Motion simulator in Twix

The Motion simulator panel runs one K1 with the real motion, inference, fall detection,
behavior and HSL message nodes. MuJoCo supplies physics and simulated sensor inputs.
The panel reuses `egui_bevy::BevyWidget`, as the legacy simulator did. All simulator
wiring and hardware substitutes live under `tools/simulate`; robotics crates need no
simulator integration code.

## Run

Fetch Git LFS assets, including the K1 meshes and `etc/neural_networks/*.onnx`.
The simulator needs MuJoCo 3.9 and a compatible ONNX Runtime shared library.
`mujoco-rs` downloads MuJoCo during the first build. Set `ORT_DYLIB_PATH` to your
ONNX Runtime library if it is not available in the loader's search path.

```sh
./twix --simulator /simulator/robot
```

The launcher enables Twix's optional `simulator` Cargo feature and configures its
MuJoCo library path. It also scopes Twix and the simulator to the global Zenoh
prefix `hulk_simulator`, including raw Booster topics and discovery. Without
`--router`, simulator mode starts a private loopback router automatically. Ordinary
Twix builds keep their existing dependencies. With Cargo directly, enable
`--features simulator`, pass `--zenoh-namespace hulk_simulator`, and set the MuJoCo
library path yourself.
The current Nix Twix package builds ordinary Twix; it does not bundle the simulator.

Open the **Motion simulator** preset or add a **Motion simulator** panel. Press
**Start simulator**, then **Inspect this robot in Twix** to select its namespace in
other panels. Only one simulator can run per Twix process. Restoring a saved layout
does not start robotics or bind UDP sockets until you press Start.

The simulation starts paused. Press **Stand: prepare**, then **Run**. Once the robot
settles in the preparation pose, **Stand: enable** follows the real safe-pose and
motion state machines. Motion overrides are edited in the existing Parameter panel:
select `/simulator/robot/behavior_node`, path `control.injected_motion_command`.
Set it to `null` to let behavior choose motion. The preset opens this parameter;
press **Refresh** once the simulator has started. Overrides still pass through the
real motion safety state machine, so use the stand buttons to prepare and enable.

**Add ball** places a ball at the configured x/y position. **Placement** contains
ball position, removal, and paused robot placement. Robot x/y use field coordinates;
z is height above its grounded spawn pose. Roll, pitch and yaw are degrees.
Reset pose pauses and repositions the robot. Stop and Start rebuild the node stack
and clear a latched emergency stop. Drag to orbit, right-drag to pan, scroll to zoom.
Blue arrows show commanded translation, green shows yaw rate, amber shows kicks.

The physics worker publishes sensors independently of the visible panel and repaint
rate. Pausing freezes physics while publishing stationary observations; the real
nodes, network and wall clock continue running. This is not deterministic stepped
robotics time. Neural policies and contacts still need validation against hardware.

## GameController

Use [RoboCup-HumanoidSoccerLeague/GameController](https://github.com/RoboCup-HumanoidSoccerLeague/GameController).
Start an actual game for team 24, default player 3. The default HULK team broadcast
address is `10.0.255.255`. GameController state arrives on UDP 3838, HULK returns
status to its sender on UDP 3939, and team communication uses UDP 10024.

The simulator runs `message_handler`, `message_filter`, `game_controller_filter`,
`game_controller_state_filter`, `primary_state_filter`, `player_states_receiver`
and `team_ball_filter`. Behavior sends the real return and team messages through
that handler. The simulator does not publish fabricated filtered game or primary
states. Use GameController for game phases, penalties, scores, sides and restarts.

Run GameController on a separate machine or in a separate network namespace.
Both GameController and HULK bind the team's UDP port, so starting both in the same
host network namespace conflicts. Zenoh namespaces do not isolate UDP sockets.
The automated roundtrip below creates private networks without changing the host's
interfaces. Ensure UDP can cross the network when using separate machines.

## Parameters and observations

The parameter layers are the selected root's `base`, this tool's `parameters`,
session calibration, then an empty writable session layer. Twix writes to the last
layer, which is deleted when the simulator stops. Save an intentional tuning result
separately if it should survive a restart. Repository robot parameters stay intact.
Select the ONNX model directory separately in the panel before starting. It defaults
to the repository's `etc/neural_networks` and is resolved independently of the
working directory.
The tool calibrates safe-pose checks and the stand-up pose to its SDK preparation
pose. These are simulated hardware calibration values, not changes to the checks.

Use Twix's Parameter panel for behavior, motion, inference, fall detection, field
and ball parameters. Field dimensions come from the real global parameter provider;
there is no second simulator field configuration. The simulator node owns ball
contact and mass settings. Use Text, Plot, Map and Behavior tree panels for outputs.

Ground-truth robot pose, camera geometry and the first ball supply behavior inputs.
Additional balls remain physical props. Images, detection, microphones and hardware
receivers are not simulated. This tool hosts selected unchanged node entry points
in Twix; it does not yet launch the complete `hulk_ros_z` executable. See the
[UI and multiple-robot audit](AUDIT.md) for the remaining boundary.

## Verify

```sh
export MUJOCO_DOWNLOAD_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/mujoco-rs"
export LD_LIBRARY_PATH="$MUJOCO_DOWNLOAD_DIR/mujoco-3.9.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo test -p simulate --lib
cargo test -p twix --features simulator --bin twix

git clone https://github.com/RoboCup-HumanoidSoccerLeague/GameController /tmp/hsl-game-controller
tools/simulate/tests/hsl/roundtrip.sh /tmp/hsl-game-controller
```

The last test requires Linux user/network namespaces, `ip`, `nsenter`, Python 3,
and the upstream Rust build dependencies including libclang. It compiles the
upstream `game_controller_runtime` used by the GUI, drives its normal action API,
and connects it to HULK's production message nodes in two private networks. It
checks Ready, a pickup penalty, penalty removal, and GameController's acceptance of
return messages. No synthetic GameController packets are used in this test. The GUI
itself is not driven. Tested against upstream commit
`af7d12962c671b631dee7b293df70e6f7b8bb491`, protocol 20, return protocol 4.

The ordinary unit suite also checks UDP team-message reception, SDK request/reply
isolation for global Zenoh prefixes `42` and `43`, physics model recompilation,
sensor transforms, command vectors and fall-detection poses.

On the tested Linux environment, dynamically loading ONNX Runtime 1.22.1 caused
an exit-time crash in its global environment destructor. Preloading the same library
made startup, rendering and shutdown succeed:

```sh
LD_PRELOAD="$ORT_DYLIB_PATH${LD_PRELOAD:+:$LD_PRELOAD}" ./twix --simulator /simulator/robot
```

This is an environment-specific workaround, not a change to motion inference.

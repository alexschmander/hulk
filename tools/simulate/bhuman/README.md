# B-Human simulator players

Each B-Human opponent runs in its own process, with native K1 cognition and motion
module graphs from Code Release 2026. StrategyBehaviorControl and SkillBehaviorControl
choose motions, and the native MotionEngine and generators produce the joint commands.
The simulator supplies perfect pose, ball and player models. No image processing,
localization, camera capture, microphone capture or whistle detector runs.

## Build and select opponents

The native worker supports Linux on x86-64. Install Git, Python 3.12 or newer,
CMake 3.24 or newer, Ninja and Clang. The build uses mold when available and
otherwise uses the compiler's default linker. Fetch the simulator Git LFS assets
as described in the [simulator README](../README.md), then launch:

```sh
./twix --simulator --bhuman /hulks/1
```

The launcher builds the pinned worker, enables Twix's `bhuman` Cargo feature,
and supplies the worker and source paths. It configures MuJoCo and uses the
worker's ONNX Runtime for HULK motion inference, including the preload workaround.
An existing `ORT_DYLIB_PATH` takes precedence. Dependencies are cached under
`$XDG_CACHE_HOME/hulk-bhuman/<revision>`, using `~/.cache` as the default cache root.
Builds use all detected CPU threads by default. Subsequent launches rebuild
the worker incrementally.

In the simulator setup, set the opponent count above zero and select
**B-Human 2026**. The worker and source fields are filled from the launcher.
To reuse a separately built worker, set both `BHUMAN_WORKER` and `BHUMAN_SOURCE`
before launching; this skips the native build.
HULKs keep their selected estimator profile; B-Human uses perfect world models.
The MCP startup contract currently selects HULKs self-play.

For a separate worker build, run `python3 tools/simulate/bhuman/build.py`.
It prints the worker and source paths. Use `--cache PATH`, `--source PATH` to
reuse an existing release checkout, or `--jobs N` to set build concurrency.
With Cargo directly, enable `--features bhuman`, pass
`--zenoh-namespace hulk_simulator`, and configure the worker paths, MuJoCo and
ONNX Runtime yourself.

The build pins B-Human commit `2e3374ba896da8288607f9a1881b876642fe18b8` and its
CompiledNN and SimRobot submodules, FlatBuffers 25.2.10 and ONNX Runtime 1.22.0.
Downloaded archives have SHA-256 checks. A CMake overlay adds the adapter and
replaces the native team socket implementation, without patching B-Human source.
The sparse checkout includes the native strategy and tactic configuration.
Ordinary HULK builds do not require the worker or FlatBuffers.

## Whistles, buttons and LEDs

The existing simulator whistle action broadcasts a detection to every robot.
For B-Human it sets `Whistle` with a logical timestamp, confidence 2 and two
listening channels. Native GameStateProvider decides whether to accept it,
including SET transitions and teammate whistles. Audio recognition stays excluded.

Body controls queue pressed/released states for native `KeyStates`. One edge is
consumed per cognition frame so a short tap between frames is preserved.
KeyStateEnhancer and the native game-state and behavior handlers measure holds
on simulation time. F1 enters Unstiff/Damping. Holding Stand for over two
simulated seconds activates the robot. Pressing Stand while active enters Prepare;
holding Walk for over two simulated seconds can leave Prepare when native checks
allow it. The progress bar freezes with simulation time. UI long-press event
labels do not override native timing. Players start active through B-Human's
own one-shot simulator startup request.

Native LEDHandler generates the RGB channel states. The simulator preserves
Off, On, Half and the normal/fast blinking variants, and renders them using the
same 400/80 ms clock divisions as BoosterProvider. The LED indicator freezes
while paused.

## Runtime and interfaces

`protocol.fbs` defines version 2 of the size-prefixed FlatBuffers protocol over a
private Unix socket. Initialization checks the pinned release and the ordered
22-joint list. Every request and reply carries a process epoch and sequence.
The host sends SI units; the adapter converts translations to native millimetres.
The serial mapping omits the T1 waist joint.

Motion executes every 2 ms; cognition executes every 20 ms. Both run synchronously
on the worker's IPC thread, with separate native blackboards and module graphs.
Their native representation packets carry cognition requests to motion and
motion feedback to cognition. RobotHealth carries the actual cognition heartbeat.
Physics snapshots and worker exchanges happen outside the shared physics mutex.

The host supplies field geometry through a temporary configuration layer,
including ball radius, so native tactics and the MuJoCo field use the same dimensions.
The pinned source checkout remains unchanged. World coordinates rotate with the
robot's GameController field side. Other robots retain team and player identities.

GameController v20 packets pass through native GameStateProvider, with penalties,
restart state and packet freshness intact. The host undoes the existing HULK
opponent identity translation before delivering packets to native team 5, and
sends return packets through the shared simulator network.

Native TeamMessageHandler retains its codec, clock synchronization and send-budget
policy. Its channel uses the worker connection instead of UDP. The host delivers
outgoing packets to other B-Human players on the same team after all robots finish
the current frame, and forwards packets for GameController budget accounting.
Private worker connections isolate simultaneous matches.

Actuation follows native calibration offsets, stiffness, motor gains, damping,
head-pitch feedforward and get-up torque control. Host torque is clamped to both
the returned limit and the model limit. Reset starts a fresh process and epoch;
pose resets clear native button and teammate queues. Exchanges have wall-time
deadlines, and failures stop the physics worker. Stop requires acknowledgment and
successful exit; drop kills and reaps the child. `Robot::step` is blocking.

## Verify

Fetch simulator Git LFS assets and configure MuJoCo as described in the
[simulator README](../README.md), then run:

```sh
export BHUMAN_WORKER=/path/to/bhuman-motion-worker
export BHUMAN_SOURCE=/path/to/BHumanCodeRelease
cargo test -p simulate --features bhuman --lib bhuman:: -- --include-ignored
cargo test -p simulate --features bhuman --lib robot_io::tests
cargo run -p simulate --features bhuman --bin bhuman-motion-smoke -- --soccer "$BHUMAN_WORKER" "$BHUMAN_SOURCE"
```

The tests exercise autonomous motion selection, whistle confidence, penalties,
button timing, native LED states, teammate encoding/decoding, mixed HULK/B-Human
physics, pause, independent workers, reset, shutdown, crashes and stale requests.
The physical soccer fixture starts in Initial, then switches to Playing with a
ball in MuJoCo. It requires movement without falling and reports displacement.
Omit `--soccer` for the explicit Prepare/Stand motion fixture. An optional final
argument sets the number of 2 ms frames.

After changing the schema, run the build script with `--generate-rust` to
regenerate checked-in Rust bindings. C++ bindings regenerate during native builds.

## Attribution

This product includes software developed by B-Human (http://www.b-human.de).
The fetched release retains its license and third-party notices. See its
`License.md` before redistribution or use in competition.

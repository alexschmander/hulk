# Simulator UI and multiple-robot audit

## UI ownership

The old left palette and right Motion studio are removed. The palette could create
robots without an owning robotics instance, so those robots were misleading test
subjects. The new panel creates one controlled robot and adds balls with a button.

| Former control | Owner after integration |
| --- | --- |
| Body/head motion command form, inject/clear | Twix Parameter panel, `behavior_node.control.injected_motion_command` |
| Parameter groups, snapshots, apply/discard, validation | Twix Parameter panel and the nodes' real parameter services |
| Game state, penalties, field side | External HSL GameController over the real network pipeline |
| Motion/actuator/behavior readouts | Twix Text, Plot and Behavior tree panels |
| Robot and ball palette | One controlled robot; Add ball in the simulator |
| Pause, reset and object placement | Simulator toolbar and Placement section |
| Walk/kick arrows and camera | Simulator scene |
| Reset robot and stack | Reset pose; Stop/Start for a new stack |
| Physical stand-button presses | Simulator toolbar, through the real button topic |
| Track first ball and fill kick ground truth | Removed convenience commands; not provided by Twix's parameter editor |

The last row is the material capability Twix does not already duplicate. The old
UI continuously rewrote head targets and filled kick targets from scene state. That
hid inputs from the normal parameter workflow. The new tool supplies normal ball
observations to behavior; explicit overrides use the coordinates entered in Twix.
Adding a small, explicit scene-to-parameter action later would be possible without
restoring a custom motion editor. The audit does not claim Twix already has these
helpers. Graphical drag/rotate gizmos were also replaced by numeric placement.

## Current process boundary

Twix owns one in-process robotics context, the SDK substitute and a physics worker.
The launcher scopes both Twix and the simulator to `hulk_simulator/` at the Zenoh
session level. The panel requires that scope before it can start.
Closing the panel stops the worker and node tasks and releases their sockets. The
simulator binds the same production message nodes as `hulk_ros_z`. There is no
separate simulator game-state implementation and no robotics-crate delta beyond
the explicitly requested upstream branch merges.

This is sufficient to test the selected motion and message stack. It is not yet a
full executable-level `hulk_ros_z` simulation. That binary also starts device
receivers, camera/detection, microphones and other perception producers. Launching
it unchanged alongside today's ground-truth producers would introduce competing
publishers and device dependencies. Full executable support needs tool-side device
or transport substitutes and process isolation, then end-to-end validation.

## Global Zenoh prefix

The desired outside key for robot 42 is `42/rt/...`, while its Booster code keeps
using `rt/...`. Zenoh's session `namespace` configuration supplies this transformation
for publications, subscriptions and request/reply traffic. The included two-robot
transport test checks command and reply isolation with prefixes `42` and `43`.
ROS-Z exposes configuration overrides through the environment, so an unchanged
process can use `ZENOH_CONFIG_OVERRIDE='namespace="42"'`.

This prefix is different from a ROS namespace such as `/42`. It scopes *all* session
keys, including ROS-Z discovery and parameter services. A matching scoped Twix
connection, or a bridge that also translates discovery, is required for inspection.
Adding a namespace to a routing-only `zenohd` is not proof that forwarded traffic
from unscoped peers is rewritten. The tested mechanism is session namespacing.
A router-only solution still needs a gateway/plugin experiment with request/reply,
liveliness and discovery, not only topic prefixing.

References: [Zenoh namespace introduction](https://zenoh.io/blog/2025-04-14-zenoh-gozuryu/),
[Zenoh configuration](https://github.com/eclipse-zenoh/zenoh/blob/main/DEFAULT_CONFIG.json5).

## Work to support dynamic robots

| Area | Existing support | Remaining work and relative difficulty |
| --- | --- | --- |
| Physics objects | Add/remove/recompile preserves existing state; each object has a unique entity prefix | Moderate: own a binding and SDK controller per robot and route observations by identity |
| Booster transport | Namespaced sessions prove commands and replies do not cross robots | Small for scoped sessions; larger for transparent router-only rewriting |
| Robotics identity | Binary derives ROS namespace from hardware identity; player/team are parameters | Moderate: allocate consistent hardware ID, robot prefix and unique player number; independent writable parameter directories |
| Lifecycle | Single simulator owns its tasks and sockets | Moderate: launch/monitor each child, await readiness, remove robot only after its process exits, roll back failed starts |
| HSL networking | Full one-robot message path and real GC returns work | Substantial: one network namespace and IP per robot, broadcast reachability and return routing; Zenoh prefixes alone cannot solve UDP port collisions |
| Full `hulk_ros_z` | Node functions are reusable unchanged | Substantial: emulate raw device inputs and decide how real perception receives simulated camera/audio data without duplicate ground-truth publishers |
| Twix inspection | One ROS namespace can be selected | Moderate: select the matching global Zenoh session scope and preserve service/discovery behavior |

A practical first increment is two isolated robotics processes and two physics
robots with fixed identities, before adding dynamic UI. Validate addressed SDK
queries/replies, independent parameter edits, disconnect/restart, real GC returns,
team messages and cleanup after a failed launch. Dynamic Add/Remove should be the
last step once these cases pass. The complete executable-level version is a
separate development effort, not a palette button or a topic rename.

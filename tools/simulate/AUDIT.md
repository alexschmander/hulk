# Simulator UI and multiple-robot audit

## UI ownership

The old left palette and right Motion studio are removed.
The palette could create robots without an owning robotics instance, so those robots were misleading test subjects.
The panel starts a configured team and adds controlled robots and balls with toolbar buttons.

| Former control | Owner after integration |
| --- | --- |
| Body/head motion command form, inject/clear | Twix Parameter panel, `behavior_node.control.injected_motion_command` |
| Parameter groups, snapshots, apply/discard, validation | Twix Parameter panel and the nodes' real parameter services |
| Game state, penalties, field side | External HSL GameController over the real network pipeline |
| Motion/actuator/behavior readouts | Twix Text, Plot and Behavior tree panels |
| Robot and ball palette | One to five controlled robots; Add robot and Add ball in the simulator |
| Pause, reset and object placement | Simulator toolbar; ball dragging and robot translation/rotation gizmo |
| Walk/kick arrows and camera | Simulator scene |
| Reset robot and stack | Reset pose; Stop/Start for a new stack |
| Profile and gamepad source | Simulator configuration before Start |
| Whistle | Simulator toolbar publishes detections through the real whistle filter |
| Physical F1/Stand/Walk buttons | Simulator toolbar, emitting raw `rt/button_event` through the unchanged bridge and handler |
| Track first ball and fill kick ground truth | Removed convenience commands; not provided by Twix's parameter editor |

The last row is the material capability Twix does not already duplicate.
The old UI continuously rewrote head targets and filled kick targets from scene state.
That hid inputs from the normal parameter workflow.
The new tool supplies normal ball observations to behavior; explicit overrides use the coordinates entered in Twix.
Adding a small, explicit scene-to-parameter action later would be possible without restoring a custom motion editor.
The audit does not claim Twix already has these helpers.
The simulator retains ball dragging and robot translation/rotation gizmos.

## Current process boundary

Twix owns one robotics context and SDK substitute per player, plus a shared physics worker and logical clock.
Robot N uses the Zenoh session scope `hulk_simulator/hulks/N/` and ROS namespace `/hulks/N`.
The panel requires that scope before it can start.
Closing the panel cancels pending additions, drains the physics worker and node tasks, and releases their sockets.
The simulator lease stays owned by the team until teardown completes.
The simulator binds the same production message nodes as `hulk_ros_z`.
There is no separate simulator game-state implementation.
The simulator invokes the production message and button nodes unchanged.

The four cumulative profiles cover motion/behavior, filtering, body estimation/odometry and localization, with independent references under `ground_truth/`.
It is not yet a full executable-level `hulk_ros_z` simulation.
That binary also starts device receivers, camera/detection, microphones and other perception producers.
Launching it unchanged alongside today's ground-truth producers would introduce competing publishers and device dependencies.
Full executable support needs tool-side device or transport substitutes and process isolation, then end-to-end validation.

## Global Zenoh prefix

The desired outside key for robot 42 is `42/rt/...`, while its Booster code keeps using `rt/...`.
Zenoh's session `namespace` configuration supplies this transformation for publications, subscriptions and request/reply traffic.
The included two-robot transport test checks command and reply isolation with prefixes `42` and `43`.
ROS-Z exposes configuration overrides through the environment, so an unchanged process can use `ZENOH_CONFIG_OVERRIDE='namespace="42"'`.

This prefix is different from a ROS namespace such as `/42`.
It scopes *all* session keys, including ROS-Z discovery and parameter services.
A matching scoped Twix connection, or a bridge that also translates discovery, is required for inspection.
Adding a namespace to a routing-only `zenohd` is not proof that forwarded traffic from unscoped peers is rewritten.
The tested mechanism is session namespacing.
A router-only solution still needs a gateway/plugin experiment with request/reply, liveliness and discovery, not only topic prefixing.

References: [Zenoh namespace introduction](https://zenoh.io/blog/2025-04-14-zenoh-gozuryu/), [Zenoh configuration](https://github.com/eclipse-zenoh/zenoh/blob/main/DEFAULT_CONFIG.json5).

## Multiple robots

Players 1 through 5 now have independent SDK controllers, raw sensor/button streams, node stacks, writable parameters and observation publishers.
Adding a robot rebuilds MuJoCo while preserving existing state and SDK controller modes.
Private UDP state/return ports avoid production endpoint collisions without changing the robotics nodes.
An ephemeral loopback team channel keeps physical teammate broadcasts out of the simulation, while forwarding copies of simulated team messages to GameController for accounting.
The upstream GameController runtime has accepted all five return streams and delivered a player-3 penalty through the unchanged production filters.

Twix recreates inspection panels on a scoped backend switch while retaining the Simulator panel and scene.
Relative topic and node names follow the selected player; intentionally absolute inspection paths remain absolute.
One local gamepad is captured centrally and gated per selection with a fresh Start press.
The remote-enabled state is read from the real behavior blackboard rather than inferred from UI input.

This implements in-process multi-robot profiles, not separate full `hulk_ros_z` processes.
Full executable support still needs device substitutes, a perception strategy, child-process supervision and end-to-end validation.
Opposing teams and removing individual robots are outside the current one-team implementation.

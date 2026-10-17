# Simulator MCP design

Status: initial implementation. The contract remains version `draft-1`; see [running the server](../../tools/simulate/README.md#agent-access-through-mcp) for setup and the implementation limits below.

Give an agent the simulator controls and ROS-Z access a developer uses to investigate a robot. The agent can launch a headless session, discover interfaces, observe signals, change inputs and parameters, inspect visuals, and recover earlier recorded data. These operations apply across motion, behavior, perception, estimation, hardware substitutes, and communications.

The [tool contract](simulator_mcp_contract.schema.json) defines argument and result schemas. Its root validates a tool name and arguments. The `x-tools` catalog associates each tool with its input, output, and, for asynchronous work, completion schema. This is an application contract, separate from MCP wire messages. An implementation must resolve the local schema references before registering self-contained MCP tool schemas.

## Agreed requirements

- Agents initially launch and own headless simulators for unattended use.
- The interface covers general simulator operation and robotics framework access.
- Framework access uses ROS-Z libraries directly. Dynamic discovery should make newly added robot topics and services accessible without adding a specialized MCP tool.
- Live data comes through ROS-Z. Persistent subscriptions and retained history use `ros-z-debug`.
- Logs, on-demand simulator views, plots, and short videos are available.
- MCAP is an optional fallback for historical data that was recorded but is unavailable through the agent's observations. It must work during an active simulation, including for topics the agent never observed.
- Regression runners and behavior-specific workflows can compose this interface. They do not define its scope.

## Ownership and existing code

| Owner | Responsibility |
| --- | --- |
| `tools/simulate` | Headless lifecycle, physics and scene controls, simulated inputs, shared clock, robot stack, recording checkpoints, and renderable scene state. |
| `ros-z` | Graph and schema discovery, parameters, message publication, and service calls. Includes canonical JSON input conversion and dynamic native service invocation. |
| `ros-z-debug` | Observation handles, status, retention, sample metadata, field selection, and queryable history. Includes receipt-time indexing and payload-byte retention; coverage remains conservative. |
| Recording adapters in MCP and `mcap-replay` | Query immutable recording checkpoints, decode archived schemas, and preserve publication identity. |
| MCP adapter | Session and operation handles, tool validation, result budgets, cancellation, artifact references, and composition of these interfaces. |
| MuJoCo renderer and shared simulator scene geometry | Scene images, maps, topic images, plots, and video frames from identified data. |

The MCP adapter should call library APIs. Extract useful CLI diagnostics into shared library code when needed. Keep `ros-z-debug` focused on observations; writes go through `ros-z`. Framework access must not depend on the Twix window or on simulator-specific topic names.

The MCP [worker](../../tools/simulate/src/mcp/worker.rs) constructs a minimal Bevy app with the same physics worker, robot team, models, and goal/ball geometry used by Twix. It owns commands on one thread. Twix keeps its existing interactive controls.

The initial connection lifecycle creates targets for the owned simulator. The target abstraction also permits connection to an independently running framework instance later. General ROS-Z operations remain the same in either case.

## Process and handle lifetimes

Use a local stdio MCP executable that supervises one simulator worker process at a time. A fresh session creates a fresh worker, private router and transport scopes, logical clock, node stack, and recording destination. Process separation lets the MCP server report a native worker crash and retain access to completed recordings. It adds startup and supervision work; performance should be measured before introducing worker reuse.

The MCP process owns observation handles, decoded caches, and recording queries. The worker owns physics and the robot stack. Keep worker diagnostics separate from MCP standard output, which carries protocol messages.

A session starts paused. Startup succeeds when the scene and robot stack are initialized and the physics worker can accept commands. Estimator acquisition is reported separately and is not implied by startup success. Missing models, libraries, or recording destinations produce actionable startup errors.

| Handle | Meaning and lifetime |
| --- | --- |
| `session_id` | One simulator lifetime. Stop ends execution; restart uses a new ID and clock origin. |
| `target_id` | A robot namespace within one connection and transport scope, or the session root for graph inspection. A reused namespace in a later session gets a new ID. |
| `entity_id` | One scene object within a session. Bevy entity IDs and raw MuJoCo indices remain internal. |
| `observation_id` | Owned subscriptions and retention settings. It survives individual tool calls until released or the MCP server exits. |
| `operation_id` | Startup, advancement, service calls, rendering, or another outstanding action. Results remain readable for the server lifetime, subject to advertised bounded metadata retention. |
| `query_id` and cursor | A frozen query result and page position. Proposed cursor lifetime is five wall-clock minutes. Expiry is an explicit error. |
| Artifact URI | An image, plot, clip, or exported data file. The result includes its MIME type and size; saved files survive session stop. |

Stopping cancels inputs and operations that need a live worker, joins the worker, and finalizes its recording. Observation inputs close; retained samples remain available until the observation is released or its advertised retention expires. Historical target identities remain usable for recording queries while the server retains that session's metadata.

An MCP transport disconnect triggers bounded cleanup of its owned worker. A forced termination may lose the uncheckpointed recording tail; already published checkpoints remain readable. No session or operation recovery across an MCP server restart is promised in the first implementation.

## Proposed tool catalog

Fields and alternatives are defined in the JSON schema. Dynamic ROS-Z payloads are further validated against the endpoint schema. Optional fields with a default use that default when omitted.

| Tool | Main inputs | Result or completion |
| --- | --- | --- |
| `server_info` | None | Capabilities, supported configurations, limits, simulator input schemas, and rendering prerequisites. |
| `simulator_start` | Configuration, request ID | Operation completing with session state and target IDs. |
| `simulator_status` | Session | Lifecycle, clock, readiness, targets, recording health, and active operations. |
| `simulator_control` | Session and pause, resume, speed, advance, or stop command | Operation completing with reached time and stop reason. |
| `scene_read` | Session | Physics snapshot, stable entities, and scene revision. |
| `scene_edit` | Session, expected scene revision, one edit | Operation completing with the resulting scene. |
| `simulator_input` | Session and body button, controller, whistle, or referee input | Operation reporting whether the input was applied. |
| `rosz_graph` | Target, entity kind, name prefix, cursor | Observed graph snapshot with types, endpoint ownership, and QoS. |
| `rosz_schema` | Target and topic, service, or parameter node | Canonical JSON schemas and examples. |
| `rosz_observe` | Target, sources, retention, request ID | Observation handle and subscription status. |
| `rosz_read` | Target, sources, latest or historical query, optional recording fallback | Bounded samples with provenance and coverage, or an operation if checkpoint/query work takes longer. |
| `rosz_unobserve` | Observation, request ID | Releases the observation's retained resources. |
| `rosz_wait` | Conditions with target IDs, wall timeout | Operation completing when fresh observations satisfy the condition. |
| `rosz_parameter_read` | Target, node, optional path | Effective value, revision, layers, and source information. |
| `rosz_parameter_write` | Target, node, set/reset/reload, request ID | Committed revision and changed paths. |
| `rosz_publish` | Target, topic, schema hash, payload, bounded delivery | Publication identity and number of samples sent. |
| `rosz_call` | Target, service, schema hash, request, wall timeout | Decoded service response. |
| `logs_read` | Target, filters, optional interval, cursor | Structured entries and history availability. |
| `view_render` | Scene, map, published image, or plot view | Artifact plus snapshot/data provenance. |
| `video_capture` | Session, camera, simulated duration, frame rate, wall timeout | Bounded future scene clip and capture metadata. |
| `operation_status` | Operation, optional bounded wait | Current stage, outcome, result, or failure. |
| `operation_cancel` | Operation, request ID | Current cancellation/effect status. |

Stable tools discover topics and schemas on demand. Do not generate a new MCP tool for every node, topic, or field.

### Common values and bounds

- Time values are decimal nanosecond strings to avoid integer precision loss in clients. Simulation receipt time is elapsed from the session's clock origin. Wall receipt time is Unix epoch time. Source time preserves the publisher's clock.
- Historical intervals are half-open, `[start, end)`. Reject reversed or empty intervals.
- Scene coordinates use a fixed, right-handed world frame: x along the field length, y along its width, z upward. Positions use meters, velocities use SI units, and orientations use normalized quaternions in x/y/z/w order. This frame does not flip at halftime. Convert Bevy coordinates inside the simulator; label robotics field frames separately.
- Resolve robot targets explicitly. MCP operations do not inherit a selected robot from a UI.
- A source is a topic plus optional ROS-Z `ValuePath`. Keep topic and field path separate to avoid ambiguity with dots in topic names.
- Wide integers in dynamic JSON use decimal strings, byte buffers use an advertised encoded representation, and nonfinite floats use explicit tags. Preserve the existing dynamic enum representation. `rosz_schema` must describe the actual MCP encoding for each endpoint, including write payloads.
- Read defaults are 1,000 samples and 256 KiB, with maximum inline limits of 10,000 samples and 1 MiB. Large individual values return an artifact or an explicit omission status. Pagination limits and missing data are reported separately.
- Observation defaults are 60 seconds of simulation receipt time, 16,384 samples per topic, and 64 MiB per observation. Callers can select a different retention clock. Server-wide admission limits are also advertised. Reject an unsupported request instead of silently reducing its retention promise.
- A field selection reduces returned data. The underlying subscription may still decode and retain a full message, which matters for images and large arrays.

Byte retention charges serialized payload size. Decoded values, schemas, transport queues, and temporary query results also consume memory; these are not process RSS limits. Field paths use upstream `ValuePath` syntax: `.pose.x`, `.joints[3]`, or `.state::Walking.speed`; empty selects the root. Present optionals use `{"$type":"some","value":...}`, absent optionals use `null`.

## Commands and completion

Mutating calls include a caller-generated `request_id`. Within the server lifetime, repeating the same tool, target, request ID, and normalized arguments returns the same operation or handle. Reusing that ID with different arguments returns an error. Idempotency is not guaranteed after a server crash or for a remote service that has already acted without returning a response.

Operation results and request IDs are retained for the server lifetime. Admission stops at 2,048 operations or 100,000 request IDs. Frozen pagination snapshots expire after five minutes, with a combined 64 MiB budget. Unpaginated reads do not consume that budget.

Long work returns an operation promptly. The operation moves through queued, running, and a terminal succeeded, failed, or cancelled state. Stages identify work such as initializing, armed, advancing, checkpointing, or encoding. `operation_status.wait_ms` waits up to 30 wall-clock seconds for progress or completion.

An operation also reports its effect as none, applied, partial, or unknown. Validation rejection has no effect. A service timeout after dispatch can have an unknown effect and must not trigger an automatic retry. Cancellation stops future work at the next supported boundary; it does not undo already applied scene edits, publications, parameter commits, or remote service effects.

Errors include a code, message, retryability, and effect status. This version uses `request_failed` and `operation_failed`; messages distinguish specific causes. Read-only failures have no simulator effect. After a mutation may have dispatched, failure conservatively reports an unknown effect.

### Simulation time

- `pause` completes after the physics worker reaches a step boundary. Asynchronous transport and already-started node work can still finish.
- `resume` continues autonomous playback. Playback speed changes wall pacing without changing the physics timestep.
- `advance` requires a paused, ready session and takes either a step count or a simulated duration. Duration rounds upward to a whole physics step; return the actual step count and time reached. The worker remains paused after completion, timeout, cancellation, or condition-triggered stopping.
- Only one operation advances physics at a time. Reads, existing observations, and an armed video capture can run alongside it. Conflicting control commands return `session_busy`, except pause/stop can interrupt advancement.
- Conditions use a small schema-defined predicate language with field selection and comparisons. They are evaluated inside the server. A condition's target must belong to the session being advanced.
- Establish subscriptions before advancing. `rosz_observe` being ready means subscriptions are armed; first samples may still require physics to run.
- `rosz_wait` does not advance time. It evaluates fresh samples received after arming, with `changed` measured against the value at arming. A paused simulation may reach the wall timeout without producing a new value.
- Asynchronous conditions can stop later than the physical event. Return the triggering sample and reached physics time. Do not promise a condition breakpoint at the exact originating physics tick.

Physics advancement and downstream processing have distinct completion. If the caller needs a particular output, it follows advancement with a fresh-data read or wait. Do not label the entire distributed stack synchronized merely because physics has stopped.

### Scene and inputs

Scene edits require pause and an expected scene revision. The revision changes on structural/placement edits, not every physics step. One call performs one edit; it does not imply a transaction over arbitrary multi-robot changes. Creating a robot completes only after it joins the physics scene and its framework interfaces are available. Failures report any partial effect.

Reset pose changes physical pose and joint state. A fresh experiment uses a new session to reset estimator, controller, referee, clock, parameter-overlay, and noise-generator state.

Body buttons and referee actions use existing simulator input paths. `server_info` exposes valid referee names and argument schemas; the referee checks current-state legality. A reported applied input does not prove the robot has reached the desired state.

A body-button hold follows the simulator's existing one-second wall-time convention. Controller input uses a bounded wall-time lease, publishes at the rate needed by the existing freshness checks, and releases the input when it expires or is cancelled. The simulator owns routing to the specified robot and arbitrates with any host controller source.

## ROS-Z observation and interaction

Use `ros-z-debug` observations across tool calls. Share underlying subscriptions for identical targets, topics, schema hashes, and compatible QoS where practical, while keeping handle ownership and memory accounting explicit. Retargeting an existing observation to another robot is excluded from this draft; open a new handle to avoid mixing histories.

A latest-value read uses an existing observation or a transient subscription, with a one-second wait by default. A transient read releases its subscription after returning and does not retain continuing history. A latest-value query cannot opt into recording fallback; archived values belong in an explicitly historical query. Requested freshness is measured against the target's collection clock and reported with the sample times.

Graph discovery is an observed local view with a revision and observation time. It is not proof that every distributed endpoint has been found. A schema hash pins decoding and input validation. Ambiguous topic types require a selection, and schema changes create explicit history boundaries.

Parameter writes preserve ROS-Z validation and expected-revision checks. A set batch is atomic within one node. The target layer is explicit; the simulator's disposable overlay is discoverable rather than hardcoded into the MCP tool. A committed parameter revision is distinct from a later node output reflecting it. Reload also reports the resulting revision.

Publishing adds a publisher with a declared lifetime and QoS. It does not replace a production publisher or override a node's internal state. Multiple publishers remain visible in graph queries. Once-only delivery releases the publisher after the send operation; a bounded repeated delivery maintains it until completion/cancellation. Retained delivery after the operation's lifetime is not promised.

Generic service calls require discovered request/response schemas and the ordinary ROS-Z wire semantics. Implement a reusable dynamic client and JSON input conversion; avoid encoding transport details independently inside the MCP adapter.

Logs use the existing log topic and paged log-history service, including structured fields and severity filters. Report when only a bounded live tail is available.

## History and optional recording fallback

A historical read is addressed by target and source names, not an observation ID. It can therefore ask about a topic that has never been observed. Merely reading history does not start a continuing subscription.

The default query uses existing observation caches. With `allow_recording_fallback: true`, consult the matching recording only where retained observations are absent, evicted, or have known gaps. If coverage is uncertain, an archive read over the uncertain interval may supplement the cache. Do not infer complete coverage from the timestamp of the oldest and newest samples.

For each source, return available intervals, subscription state, known loss, unknown coverage, and whether data came from observations or recording. Distinguish an interval with no received publications from an interval the server cannot inspect. An archive cannot recover topics that were not recorded, dropped samples, absent schemas, or private node state.

Queries freeze their selected caches and recording checkpoint references. Cursors page through that frozen result even while physics and recording continue. Pinning and decoding obey query memory budgets. `truncated` and a next cursor mean there are more results; coverage describes missing data.

Fallback data is decoded and returned directly. It is not republished into the live graph. A recording from a different session is never selected merely because it uses the same robot namespace.

### Clock and identity rules

The [existing cache](../../crates/ros-z-debug/src/retention.rs) retains windows in publisher source time. The [simulator recorder](../../tools/simulate/src/telemetry/recording.rs) indexes MCAP messages by simulation time at receipt and preserves source time separately. The public query supports source, simulation receipt, or wall receipt time and always labels the chosen basis.

Extend observation metadata and retention/indexing to keep observer receipt times using the session clock and wall clock. Preserve the original source timestamp and its known or unknown clock domain. Archived data uses recorded receipt times. A source-time archive query needs a suitable index or bounded scan; it cannot assume a receipt-time chunk range is the same range.

Observer receipt and recorder receipt are different events. Keep both if the same publication is found in both sources. A sample found in the requested interval by either collector is returned once, with all available receipts; `selected_time_ns` uses the observation receipt if it falls inside the interval, otherwise the recording receipt. Source-time queries use the unchanged publisher timestamp. This is a union of recorded observations, not a promise of one global acquisition instant.

Deduplicate by original publisher identity and sequence, qualified by session and topic, rather than value or timestamp. Preserve schema hashes. The existing `_recording/receipt` metadata can support this identity, but the recording reader must decode and correlate it. If correlation is unavailable in an older file, report the limitation and avoid silently dropping potentially distinct samples.

### Checkpoints during an active session

Required behavior: a fallback read can request an immutable readable checkpoint while the simulator keeps running. The response identifies the cutoff and coverage for each recorder. Multi-robot checkpoints need not have identical cutoff times.

The implementation rotates a robot's recorder into independently finalized MCAP segments and publishes a session manifest:

1. Establish an ordered checkpoint barrier through recorder ingestion and serialization. Include all accepted work before that barrier, including pending schema resolution.
2. Finish the current segment and its index, flush it, and publish its manifest entry atomically. Continue accepting later samples into bounded queues and a subsequent segment.
3. Return a checkpoint reference containing the immutable segment set, per-recorder cutoff, and recording health.
4. Query those immutable segments while recording proceeds.

Checkpoint requests are serialized per worker. Each explicit fallback query currently rotates the recorder; frozen pagination reuses the same checkpoint. Checkpoints do not prove that every remote publisher delivered every sample. Drop counters describe known admission and schema-resolution losses.

The manifest names finalized segments and stores barrier metadata. A crash can leave an unreadable current tail; prior finalized segments remain available and the result reports a partial tail. `mcap-replay` accepts the manifest, its directory, or `recording.mcap` beside a manifest. It builds an anonymous combined indexed file before replay, which requires temporary disk space and startup time proportional to recorded data.

## Visual inspection

Headless means no interactive window. Offscreen scene rendering can still need a graphics adapter and driver. Report rendering support separately from physics readiness and create rendering resources on demand.

- Scene images use a frozen physics/scene snapshot and return its simulation time.
- Maps plot explicitly selected topics with `x/y` or `position.x/y` fields and retain contributing sample times. They do not infer coordinate transforms. Optional truth overlays use the world frame.
- Plots use the same `rosz_read` query and coverage rules, including opt-in recording fallback. Numeric values become curves; booleans and enums appear as labeled state marks. An oversized plot fails with a request to narrow its interval.
- Image inspection requires an actual published image with a supported schema. Synthetic detections do not imply a camera image exists.
- Video capture arms a future simulated interval and does not itself advance physics. Wait for its armed stage, then resume or advance. Clips are animated GIFs, up to 30 simulated seconds and 30 fps. Frame delays follow actual sampled simulation times. Missed frame times and partial clips are reported. A wall timeout bounds capture if the session stays paused.

A historical MCAP lookup does not reconstruct arbitrary past 3D scenes or guarantee an unseen past video. Recorded signals can be plotted. Image inspection currently reads an actual live published image: encoded image bytes or an `rgb8` record.

## Developer walkthrough

These requests are validated against the contract. Substitute IDs and revisions returned by earlier calls.

### Start and discover

```json
{"tool":"server_info","arguments":{}}
```

```json
{"tool":"simulator_start","arguments":{"request_id":"start-1","configuration":{"location":"hsl_small","robot_count":1,"opponent_count":0,"profile":"filtering","referee":"automatic","rendering":"on_demand"}}}
```

```json
{"tool":"operation_status","arguments":{"operation_id":"op-start-1","wait_ms":30000}}
```

Startup completion returns a paused session `session-1` and robot target `robot-1` for `/hulks/1`. Inspect the graph and the parameter schema before choosing a write.

```json
{"tool":"rosz_graph","arguments":{"target_id":"robot-1","kind":"all","limit":100}}
```

```json
{"tool":"rosz_schema","arguments":{"target_id":"robot-1","kind":"parameters","name":"simulator"}}
```

### Observe and change a parameter

```json
{"tool":"rosz_observe","arguments":{"target_id":"robot-1","request_id":"observe-1","sources":[{"topic":"primary_state"},{"topic":"behavior/motion_command"}],"retention":{"duration_ms":60000,"max_samples_per_topic":16384,"max_bytes":67108864}}}
```

Proceed once those subscriptions are armed. This need not imply a first sample has arrived while physics is paused.

```json
{"tool":"rosz_parameter_read","arguments":{"target_id":"robot-1","node":"simulator","path":"observations.pixel_noise_std_dev"}}
```

Use the returned revision and writable overlay layer. The example assumes revision 7 and layer `live` were returned. Field names come from the discovered schema.

```json
{"tool":"rosz_parameter_write","arguments":{"target_id":"robot-1","node":"simulator","request_id":"parameter-1","action":{"kind":"set","expected_revision":7,"writes":[{"path":"observations.pixel_noise_std_dev","value":2.0,"layer":"live"}]}}}
```

Read the write operation to completion before advancing. Its completion reports the committed revision; it does not claim all nodes have processed the new configuration.

### Advance and inspect

```json
{"tool":"simulator_control","arguments":{"session_id":"session-1","request_id":"advance-1","command":{"kind":"advance","duration_ns":"3000000000","wall_timeout_ms":30000}}}
```

```json
{"tool":"operation_status","arguments":{"operation_id":"op-advance-1","wait_ms":30000}}
```

```json
{"tool":"rosz_read","arguments":{"target_id":"robot-1","sources":[{"topic":"primary_state"},{"topic":"behavior/motion_command"}],"query":{"kind":"history","range":{"clock":"simulation_receipt","start_ns":"0","end_ns":"3000000000"}},"limit":1000}}
```

```json
{"tool":"view_render","arguments":{"request_id":"plot-1","view":{"kind":"plot","target_id":"robot-1","sources":[{"topic":"primary_state"}],"range":{"clock":"simulation_receipt","start_ns":"0","end_ns":"3000000000"}}}}
```

```json
{"tool":"view_render","arguments":{"request_id":"scene-1","view":{"kind":"scene","session_id":"session-1","camera":{"kind":"preset","name":"field"}},"size":{"width":1280,"height":720}}}
```

The state plot and scene capture answer different questions. Each artifact identifies its interval or snapshot; a plot with no transitions can be a valid observation.

### Ask about an unobserved topic

The agent now asks about camera frame loss during those first three seconds. It never observed `diagnostics/camera_frames`.

```json
{"tool":"rosz_read","arguments":{"target_id":"robot-1","sources":[{"topic":"diagnostics/camera_frames","path":".skipped_frames"}],"query":{"kind":"history","range":{"clock":"simulation_receipt","start_ns":"0","end_ns":"3000000000"}},"allow_recording_fallback":true,"limit":1000}}
```

The server uses the active session's recording, requesting a checkpoint if necessary. It returns a read result directly or an operation whose completion is a read result. Samples identify recording provenance, original source times, and receipt times. Coverage identifies the queryable interval and any loss. The session remains available for further controls.

Without fallback, the same request reports an unobserved interval. It does not invent zeros, create a historical subscription, or interpret missing data as no camera activity.

### Capture a future clip and close

```json
{"tool":"video_capture","arguments":{"session_id":"session-1","request_id":"clip-1","camera":{"kind":"preset","name":"follow_robot","target_id":"robot-1"},"duration_ns":"1000000000","fps":15,"wall_timeout_ms":30000}}
```

Inspect the returned operation until its stage is armed. Then advance another simulated second.

```json
{"tool":"simulator_control","arguments":{"session_id":"session-1","request_id":"advance-2","command":{"kind":"advance","duration_ns":"1000000000","wall_timeout_ms":30000}}}
```

After advancement and encoding complete, read the clip operation for its artifact. Stop explicitly when investigation is finished.

```json
{"tool":"simulator_control","arguments":{"session_id":"session-1","request_id":"stop-1","command":{"kind":"stop"}}}
```

Stop completion includes recording finalization. Stored observations and recording-backed historical queries remain readable under the handle rules above.

## Implementation limits and verification

The server owns one active Linux worker and private Zenoh router. It does not attach to an existing simulator or real robot. Startup means physics and nodes were constructed; graph discovery and estimator readiness can still be settling. Stop finalizes recordings, while a native crash only preserves completed checkpoints. Framework clocks and task scheduling remain asynchronous, so identical steps do not guarantee identical behavior.

Live data and history use ROS-Z and `ros-z-debug`. The server does not poll MCAP for latest state. History coverage is conservatively `unknown` or `not_observed`: timestamps alone do not establish complete delivery. Live source clock domains are currently reported as unknown. Source-schema changes fail reads that specify a hash; publication and service calls require one. Native dynamic service calls require one unambiguous discovered server. They cannot cancel a remote side effect after dispatch.

Rendering uses headless EGL and the MuJoCo scene, without Twix's UI overlays. Rendering holds the physics lock for a snapshot and can slow wall-clock playback. Synthetic perception profiles do not publish camera images. Map layers need coordinates already expressed in the desired frame; plots require bounded intervals. Artifacts and recordings remain on disk until the user removes their directories.

The [stdio integration test](../../tools/simulate/tests/mcp_smoke.py) validates outputs and operation completions against the contract while exercising startup, exact stepping, ROS-Z reads/writes/services, optional recording fallback, publication deduplication, visuals, and shutdown history. Rust tests cover JSON/CDR round trips, receipt-clock retention, checkpoint ordering, and replay across finalized segments.

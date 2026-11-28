# Detection Replay

`detection-replay` prerenders detections from ROS-Z MCAP recordings and compares models on a synchronized image feed. Inference uses the production detection implementation through `run_boxed_with_model_info`. The repository wrapper builds and runs the tool in release mode. `view`, `verify`, `index`, and `snapshot` build without ONNX Runtime and work with saved detections on machines without CUDA.

## Recording index

Build the random-access JPEG proxy and import any recorded `detected_objects` messages:

```bash
./detection-replay index /path/to/recording.mcap
```

The source MCAP is opened read-only. A recording that ends in an incomplete final chunk remains usable: complete chunks are indexed and the damaged tail is reported. The tool supports `inputs/left_image` and `inputs/stereo_image_pair` recordings. Prerender resumes seek through the MCAP chunk index when a summary is available; damaged or unindexed recordings keep a direct-seek CDR copy of original images in the cache.

By default the cache is written next to the recording as `recording.detection-replay-cache`. Use `--cache-dir` to put it elsewhere.

All commands accept an inclusive frame-ID range through `--start-frame` and `--end-frame`. The index still scans the MCAP once to establish stable global frame IDs, but prerendering and viewing are restricted to the selected range.

## Prerendering

Run models sequentially so their GPU timings and predictions do not interfere:

```bash
./detection-replay prerender /path/to/recording.mcap \
  --model baseline=/path/to/baseline.onnx \
  --model candidate=/path/to/candidate.onnx \
  --provider cuda --gpu 1 \
  --start-frame 5000 --end-frame 5500
```

Each input frame is published only after the preceding result has been received. Predictions are matched by the image header timestamp and written in atomic chunks. Repeating the command resumes at the first uncached frame.

Use `--limit 10` for a short provider/model compatibility check. CUDA is the default provider. Select `automatic`, `tensorrt`, `cuda`, or `cpu` with `--provider`; `--gpu` selects the physical device index for NVIDIA providers. Replay exposes that physical GPU as logical device zero through `CUDA_VISIBLE_DEVICES` before ONNX Runtime or worker threads start, avoiding cross-thread CUDA context mismatches on nonzero devices. `tensorrt` retains CUDA as an operator fallback, while `automatic` follows the available production order TensorRT, CUDA, then CPU in this NVIDIA build. The default confidence threshold is `0.05` and the default NMS IoU is `0.4`; use `--confidence` and `--iou` to change them. Thresholds, model hash, provider, and device are part of the cache identity, so switching provider cannot silently mix results or timing measurements in one run.

Models must accept `raw_bytes_input` as a rank-three `uint8` tensor with shape `[height / 2, width / 2, 6]`. Its contiguous bytes are the full-resolution NV12 Y plane followed by the half-height interleaved UV plane; image width and height must be multiples of 32.

Object detections require either `object_output`, or both `hslvision_output` and `nao_output`, as `float32` tensors. Legacy object rows use shape `[1, 300, 6]` and contain `[x_min, y_min, x_max, y_max, confidence, class_index]`. The native Hydra contract uses `object_output [1, 300, 9]` and appends `[anchor_x, anchor_y, anchor_presence]`. Coordinates are full input-image pixel coordinates. Class indices must be finite integral values from 0 through 6: Ball, GoalPost, LSpot, PenaltySpot, Robot, TSpot, and XSpot. Anchors are retained without a presence cutoff for every class except Robot, whose anchor is always marked unavailable. The object and its anchor remain paired through the confidence filter and class-agnostic NMS. The legacy two-head models use the selection behavior from `hslvision-plus-old-goalposts`: HSLVision supplies all classes except GoalPost, and NAO supplies GoalPost.

Native Hydra models must provide `line_output [1, 300, 21]` alongside the nine-column object output. A line row contains `[x_min, y_min, x_max, y_max, confidence, class_index]` followed by five ordered `(x, y, presence)` triples. The only valid line class is `0`. Lines with confidence at least `0.39` are retained independently of object filtering, and neither object NMS nor cross-head NMS is applied to them. All five points are retained without a presence cutoff, so each accepted row draws four ordered segments. Published line coordinates are clipped to the inclusive native image bounds and shifted by `-0.5` to the pixel-centre convention. Replay adds the half pixel back when mapping them into the image texture. Object coordinates and anchors are not shifted.

The native supplement also retains all 300 unfiltered rows from each head in model order. Raw object rows keep the network's anchor values even for Robot and explicitly mark that anchor as unsupervised. Raw line rows keep their original vertex coordinates and presences. The viewer's raw-head mode has independent per-head score filters so low-scoring or object-NMS-suppressed rows remain available without making the default view unreadable.

`person_pose_output` is optional and must be `float32 [1, 300, 57]`: six pose-object values followed by 17 `(x, y, confidence)` triples in COCO order: nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left hand, right hand, left hip, right hip, left knee, right knee, left foot, right foot. Pose class indices use the COCO `YOLOObjectLabel` mapping (`0` is Person), not the RoboCup object mapping. Coordinates use the same full-image pixel space. For older human-pose models, `pose_output [1, 300, 57]` remains supported when the model has no `branches` metadata.

`pose_output` may instead carry field poses as `float32 [1, 300, 9]`: six pose-object values followed by one `(x, y, confidence)` keypoint. Detection recognizes this form through the Hydra `branches` metadata and maps the metadata's class names to GoalPost, LSpot, TSpot, PenaltySpot, and XSpot rather than assuming a numeric class order. It publishes the keypoint through `detected_field_features`, so replay and robotics use the same field-feature message.

`robot_pose_output` is optional and must be `float32 [1, 300, 48]`: six pose-object values followed by 14 `(x, y, confidence)` triples in this DHRP order: nose, neck, right shoulder, right elbow, right wrist, left shoulder, left elbow, left wrist, right hip, right knee, right ankle, left hip, left knee, left ankle. The DHRP model has one class (`0`, Robot); the runtime maps it to the RoboCup Robot label. Replay caches these timestamp-matched robot skeletons separately and distinguishes unavailable output from a valid empty result.

`field_feature_output` is also optional and must be `float32 [1, 300, 4]`. Each row is `[x, y, confidence, class_index]` in full-image pixel coordinates. Class indices are GoalPost, LSpot, TSpot, PenaltySpot, and XSpot. Replay stores the timestamp-matched field-feature points separately from the object and pose chunks so caches created before field-feature capture remain readable.

The `prerender` subcommand uses the x86-64 ONNX Runtime 1.22 CUDA distribution. The wrapper enables the Cargo feature `prerender` automatically for this command. If you invoke Cargo directly, use `cargo run --release -p detection-replay --features prerender -- prerender …`. The wrapper prevents a CPU-only system runtime discovered through `pkg-config` from overriding the CUDA runtime and adds CUDA 12 user-space libraries from the multi-task YOLO virtual environment to the loader path when they are available. Operators unsupported by the selected NVIDIA provider may still use ORT's CPU fallback. Robot builds dynamically load their runtime image and retain the default provider order compiled into that deployment. Only the generic provider choice crosses into the detection node; replay-specific CLI parsing, physical-device remapping, CUDA library discovery, and defaults remain confined to this tool.

## Viewing

```bash
./detection-replay view /path/to/recording.mcap --start-frame 5000 --end-frame 5500
```

The viewer initially opens model controls and model viewports as tabs in the upper dock, with the timeline in a lower dock. Tabs can be moved between dock nodes, reordered, detached, closed, and reopened from the top-bar `View` menu. Model viewports remain synchronized and share pan and zoom.

The timestamp-proportional timeline shows source capture gaps and prediction availability for every run. Drag to scrub, scroll to zoom around the pointer, Shift+scroll to pan, and double-click to reset to the selected CLI frame range. Press `B` to toggle a bookmark and Page Up/Page Down to visit bookmarks; bookmarks are persisted in the recording-specific cache and persistence failures are shown in Models.

The Models tab manages cached runs. Model runs can be renamed, hidden, or permanently deleted after confirming the `Are you sure?` dialog. Renames and hidden state persist in recording-specific cache metadata. Hidden runs are removed from the timeline, viewport tabs, and `View` menu but remain in Models for unhiding. The source-derived Recorded baseline can be hidden but cannot be renamed or deleted.

The recording and its cache may be synced between machines. Preserve the recording's file size and modification time so the viewer can reuse the bundled index, JPEG proxy, and model runs at the new path. If no matching valid recording cache is available, the tool prints an indexing message and builds the proxy before opening the viewer.

`Show field features` overlays cached field-feature points and is enabled by default. `Show object boxes`, `Show object anchors`, and `Show field lines` control the native Hydra layers independently. Object boxes and anchors share the object display-confidence filter. Anchor presence uses the keypoint-confidence filter for display, and `Show object anchor details` labels its coordinates and presence.

FieldLines use their own score filter and display only the five ordered keypoints and four connecting segments. Under a run in the Models tab, expand `FieldLine details` to inspect class, score, and the ordered coordinates and predicted presence for each row. `Show raw Hydra head rows` exposes pre-threshold and pre-NMS rows with separate object and line score filters. It also adds the original tensor rows to the FieldLine details. Presence is a network score.

`Show robot poses` is enabled by default. `Show person poses` controls the human-pose overlay. Pose bounding boxes use the display-confidence threshold, while skeleton lines and keypoints use the separate keypoint-confidence threshold. The viewport distinguishes an unavailable Hydra supplement from a present but empty result. Older runs remain usable and report uncaptured outputs as unavailable. Prerender them again to populate the Hydra overlays.

Hydra anchors and lines are stored in a versioned sidecar beside each primary prediction chunk. The sidecar has one timestamp-matched supplement per frame and one optional anchor slot per surviving object. The primary object cache layout is unchanged, so existing replay caches remain readable.

Useful controls:

- `Space`: play or pause
- Left/right arrow: step one frame
- Drag: pan every viewport
- Scroll: zoom every viewport
- Double-click: reset pan and zoom

An unavailable prediction is shown separately from a valid empty detection result.

## Validation

Force a lazy decode of every primary and supplemental cache chunk without initializing ONNX Runtime:

```bash
./detection-replay verify /path/to/recording.mcap
```

`verify` checks each cached prediction's frame ID and timestamp against the recording index, validates object/anchor alignment, and requires every captured native Hydra frame to contain all 300 raw rows from both heads. Its per-run summary separates present Hydra messages, missing messages, empty object/line frames, retained detections, and raw-row totals. This is the preferred integrity check after relocating or downloading a prerender; `view` then provides normal interactive playback without inference.

Use `--run MODEL_KEY_OR_LABEL` to verify a delivered run independently of unrelated legacy caches in the same cache directory.

Export deterministic visual evidence from the same cached viewer data as SVG:

```bash
./detection-replay snapshot /path/to/recording.mcap \
  --run candidate --frame 1000 --output frame-1000.svg

./detection-replay snapshot /path/to/recording.mcap \
  --run candidate --frame 1000 --output frame-1000-raw.svg \
  --raw --details --object-confidence 0.05 --line-confidence 0.20
```

Snapshots follow the viewer's native-coordinate and half-pixel conventions and draw FieldLines as points and connected segments. `--details` labels object anchors. Raw mode is intended for evidence and diagnostics.

Use a short range before prerendering an entire recording:

```bash
./detection-replay index /path/to/recording.mcap --start-frame 100 --end-frame 109
./detection-replay prerender /path/to/recording.mcap \
  --model test=etc/neural_networks/model.onnx \
  --provider cuda --gpu 0 \
  --start-frame 100 --end-frame 109 --limit 10
```

The prerender must report 10 of 10 frames without a provider or shape error. Repeating it should immediately report the cached 10 frames. Open the same range with `view`; the timeline should show ten source frames, the run should be `complete`, empty detections should differ from unavailable predictions, and any damaged MCAP tail should appear as a warning.

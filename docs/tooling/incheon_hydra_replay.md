# Incheon Native Hydra Replay

This branch's Incheon replay target is the native 544 x 448 NV12 two-head Hydra contract described in [detection_replay.md](detection_replay.md). The indexed July 5 recording contains 21,492 consecutive complete frames, IDs 0 through 21,491. The MCAP warning after that range is a damaged tail after the last complete frame, not a missing indexed frame.

## Delivery acceptance

The delivered model run must satisfy all of the following:

- The pinned selected ONNX SHA-256 matches the delivered checksum manifest, and the replay manifest's BLAKE3 model digest resolves to that same file.
- Every indexed frame has a primary prediction and a present Hydra supplemental message, including valid empty results.
- Every supplemental message contains 300 raw object rows and 300 raw line rows in model order.
- Post-NMS anchors remain aligned one-for-one with surviving objects. Robot has no postprocessed anchor, while its raw network anchor remains inspectable and is labelled unsupervised.
- FieldLine rows retain all five ordered vertices and all predicted presence scores. The overlay draws only the five points and four connecting segments.
- Normal playback can show object boxes/labels/scores, supervised anchors, and FieldLines together or as separate layers. Raw diagnostic mode exposes pre-threshold and pre-NMS rows with independent per-head filters and is off by default.
- Object anchor details are available on the image. FieldLine details under each run in the Models tab report class, score, and each line vertex's index, coordinates, and predicted presence.
- Raw object labels and FieldLine details retain the original top-300 row index.
- `detection-replay verify` successfully decodes every primary and sidecar chunk after relocating the download to a different absolute path with the ONNX unavailable. `detection-replay view` opens the same cache without inference.
- Combined and detailed screenshots from representative actual Incheon frames demonstrate native-coordinate alignment. Fixture evidence alone is not sufficient.

The fixture export is suitable for deterministic routing and edge-case checks only. Full replay acceptance uses the final selected ONNX and the actual Incheon recording.

## Delivered run

The selected model is `hydra-native.onnx`, SHA-256
`226cafaad243a91c978dea9f927eec8f10e5b0b029f7940a34b7e80d09f0f2d0`.
The cached run ID is `model-1ae20a4d3605c424`, with model BLAKE3 digest
`696b9b7912c43308d32f1c35783ca7191bfcf2c18a90ffed8c75673471d86a3c`.

Strict verification of frame IDs 0 through 21,491 reports:

- 21,492 primary prediction frames and 21,492 Hydra supplemental messages;
  no supplemental message is missing.
- 6,447,600 raw object rows and 6,447,600 raw FieldLine rows, exactly 300
  rows per head per frame.
- 135,667 post-NMS object detections, including 2,874 valid empty object
  frames, and 128,612 supervised object anchors.
- 105,366 retained FieldLines, including 2,186 valid empty FieldLine frames.

The delivery bundle uses the recording fingerprint
`cfda24d6a890c817205ff558f7b50d9fbb40fb22f4dd4b762d3183ca29cd7f0b`.
It includes the source MCAP, proxy/index, recorded baseline, selected model run,
selected ONNX, checksums, and visual evidence. `original-frames.cdr` is omitted:
the viewer uses the proxy, while a future inference run falls back to the
included MCAP when the derived CDR is unavailable.

After extracting the private bundle, verify and open it from this branch with:

```sh
./detection-replay verify /path/to/incheon-hydra26s-replay-delivery-2026-11-23/recording.mcap \
  --cache-dir /path/to/incheon-hydra26s-replay-delivery-2026-11-23/recording.detection-replay-cache \
  --run model-1ae20a4d3605c424 --start-frame 0 --end-frame 21491

./detection-replay view /path/to/incheon-hydra26s-replay-delivery-2026-11-23/recording.mcap \
  --cache-dir /path/to/incheon-hydra26s-replay-delivery-2026-11-23/recording.detection-replay-cache
```

The viewer builds without ONNX Runtime or CUDA and plays the saved cache without
inference. Keep the extracted MCAP and cache together. Tar extraction preserves
the MCAP timestamp so the viewer can reuse the bundled index and JPEG proxy at
the new path.

The evidence directory contains first, quarter, middle, three-quarter,
and last-frame combined overlays plus separated object and FieldLine detail views for frame
10,746. Supplemental frame-10,746 evidence records a below-threshold object
row and a below-threshold FieldLine row by original tensor index. FieldLine
evidence intentionally contains no detection boxes.

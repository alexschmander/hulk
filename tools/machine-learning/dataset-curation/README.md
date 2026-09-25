# Dataset curation

A single uv application for extracting images, computing frozen embeddings,
selecting diverse shortlists, and reviewing quality and visual redundancy.
Implementation lives directly in `dataset_curation/`, with tests in `tests/`.
There is no nested distribution, build backend, or package installation.

Keep code and its environment here. Keep each dataset's configuration, raw
inputs, extracted images, manifests, embeddings, and review decisions beside
that dataset. The application has no HSLVision-specific paths or selection quotas.

## Setup and commands

From this directory:

```bash
uv sync --locked
uv run python -m dataset_curation extract --config /path/to/dataset/curation/config.toml
uv run python -m dataset_curation embed --config /path/to/dataset/curation/config.toml
uv run python -m dataset_curation report --config /path/to/dataset/curation/config.toml
```

The lockfile targets Python 3.12 and the CUDA 12.6 PyTorch distribution, which
supports the RTX 2070. Embedding also supports `device = "cpu"`. Extraction
and report generation do not load PyTorch or download model weights.

For gated models, obtain access on the model's Hugging Face page, then run
`uv run hf auth login`. Credentials and downloaded models use Hugging Face's
normal user cache, outside both the repository and the dataset.

## Job configuration

Save a TOML file beside the dataset. Paths under `[dataset]` are resolved
relative to the configuration file (`root`) or dataset root (`raw`, `output`).
The optional reference root is relative to the configuration file. Paths do
not depend on the shell's working directory.

```toml
schema_version = 1

[dataset]
title = "Camera dataset selection pilot"
root = ".."
raw = "raw-source"
output = "curation/pilot"

[sampling]
mode = "pilot"                   # "all" exports every temporal group
fps = 3
burst_size = 3
timestamp_gap_seconds = 10
workers = 4

[[sources]]
name = "camera-exports"
kind = "image_sequence"
pattern = "camera-exports/session-*" # directories relative to raw
filename_pattern = 'frame_(\d+)'
frame_rate = 6                     # known export rate, not inferred from names
pilot_count = 1000

[[sources]]
name = "robot-recordings"
kind = "ros_z_mcap"
pattern = "robot-recordings/*/recording.mcap"
topic = "inputs/stereo_image_pair"
stereo = true
pilot_count = 300
eligibility_question = "Is the target visible?" # optional review question

[embedding]
model = "facebook/dinov3-vits16-pretrain-lvd1689m"
revision = "main"                 # resolves to a recorded immutable revision
size = 256
batch_size = 32
workers = 4
device = "auto"

[reference]                       # optional existing image corpus
root = "../../existing-dataset"

[review]
blur_samples_per_source = 60
pair_samples_per_category = 36
```

Source names must be unique. `existing` is reserved for reference images.
In `pilot` mode, each source's quota is allocated evenly across recordings, with
shortages redistributed. Short bursts are spread across each recording so
the pilot includes both adjacent scenes and broader variation.

Image sequences use the configured export rate to group consecutive PNGs,
detect filename gaps, and retain the sharper frame in sampled groups. MCAP
sources use capture-time windows and retain the sharpest left-camera image
in each window. Slower recordings retain their available frames. Timestamp
discontinuities start new segments; unfinished tails are recorded in the
index audit. MCAP support is specifically for ROS-Z CDR Image and
StereoImagePair messages with NV12 payloads, not arbitrary ROS image schemas.

Original PNG bytes are preserved; MCAP images are converted with limited-range
BT.709 at native resolution. Indexes are reused only when input metadata and
sampling settings match. Raw inputs are read-only.

## Embeddings and review

The default DINOv3 configuration produces normalized 384-dimensional CLS
embeddings. Full images are resized and padded to a square, preserving field
edges. The run records the resolved model revision, preprocessing, manifest
hash, throughput, and peak GPU memory. Candidate and reference images receive
identical preprocessing. Existing labels are never read.

Neighbor searches use exact dot products in query blocks, both within the
candidate pool and against the reference corpus. They never retain an entire
candidate-by-candidate matrix. Runtime remains quadratic in candidate count.
Reference folders named `train`, `val`, and `test` are identified in the review
pairs; no reference labels are read.

Open `OUTPUT/review.html` locally to review image quality, optional source
eligibility, and similar-image pairs. The report also works before embeddings
exist; regenerate it after embedding to add pairs. Export review JSON to
preserve decisions outside browser storage. A changed manifest or model
revision creates a new review identity, preventing accidental reuse of stale
decisions.

Outputs:

- `manifest.jsonl`: image identity, provenance, sampling, quality, and hashes.
- `indexes/` and `summary.json`: recording coverage and extraction diagnostics.
- `embeddings/`: vectors, ordered row identities, and benchmark metadata.
- `neighbors.json` and `pairs.json`: similarity review inputs.
- `review.html` and `contact-sheets/`: offline review materials.

## Full-pool selection

Use a separate output with `sampling.mode = "all"` for the full temporal pool.
Keep the pilot and its review export unchanged. `timeline` creates a sparse
MCAP contact sheet for identifying unusable intervals; inspect every boundary
at the full candidate cadence before declaring an interval unusable.

```bash
uv run python -m dataset_curation timeline --config /path/to/timeline.toml --step-seconds 10
uv run python -m dataset_curation filter --config /path/to/candidates.toml --policy /path/to/policy.json
uv run python -m dataset_curation select --config /path/to/candidates.toml --policy /path/to/policy.json
uv run python -m dataset_curation audit --config /path/to/shortlist.toml
uv run python -m dataset_curation report --config /path/to/shortlist.toml
```

Policy JSON uses `schema_version: 1`. Paths `review`, `review_manifest`, and
`review_pairs` identify the immutable user review and original pilot artifacts,
relative to the dataset root. Review-manifest hashes are checked. Optional
`exclude_intervals` contain `source_path`, `source_signature: [size, mtime_ns]`,
inclusive `first`/`last` image ordinals, and a `reason`. Optional `exclude_images`
maps individual IDs to exclusion reasons. Source signatures guard against
applying an interval to a replaced recording. Hard scene exclusions take
precedence over earlier usable-image labels.

`selection` specifies a separate `output`, a `quality` object with
`neighbor_cosine`, `relative_sharpness`, and `minimum_sharpness`, and `groups`.
Each group gives `name`, `sources`, `count`, `reserve`, and `max_cosine`.
Optional `recording_cap` limits domination by one recording; this is not a
match-level cap across multiple robots. Optional `require_target` uses known
visible/absent examples to propose target-bearing images and requires
`minimum_target_cosine`. These are proposals requiring visual confirmation.

The selector favors sharper alternatives among similar eligible images,
preserves explicit usable quality decisions, then uses greedy farthest-point
selection with a small sharpness preference. Direct similarity to retained
images enforces the configured cap; connected-component chaining is avoided.
Exact pixel duplicates and explicitly redundant reviewed pairs are excluded.
Optional group `preferred_ids` preserve earlier eligible shortlist choices
while filling vacancies; ineligible preferences are ignored and conflicting
eligible preferences cause an error. Counts can fall short when insufficient
eligible diversity exists. The selector never relaxes a cap to fill a quota.

Selection writes a manifest, reserves, embeddings, summaries, and a review
gallery. Matching user image labels carry forward into the new gallery;
assistant screening is stored separately in job data. Stale generated images
from this candidate inventory are removed from the shortlist image folder.
Only manifest entries define the selection.

`audit` verifies selected file hashes, checks decoded-pixel overlap against
all reference images, and writes `audit.json` plus `audit-pairs.json`. The
latter prioritizes the closest pairs in each source/cross-source/reference-
split category. `report` uses these only when their manifest hash matches.
Shortlist quality queues prioritize the lowest sharpness scores; pilot queues
continue to cover the full score range. Prior browser decisions are isolated
by manifest and model revision.

Raw data and the reference dataset are read-only. No command imports into an
annotation service or writes empty ground-truth labels. `select` outputs are
review drafts; `export` requires an explicitly frozen plan. Neither command
certifies absolute semantic uniqueness. The pixel audit
cannot detect re-encoded or cropped versions of a scene. Sharpness depends on
texture, and global embeddings can miss small-object changes. Inspect the
focused review queue before freezing and exporting final labelling batches.

## Freeze and export

After review, store the exact selected IDs in a JSON export plan. It contains
`schema_version: 1`, hashes `manifest_sha256`, `policy_sha256`, `review_sha256`,
and `pairs_sha256`, paths `images` and `output` relative to the dataset root,
a fixed `seed`, and `groups`. Each group gives `name`, `ids`, one or two
`batches`, and `max_cosine`; optional `require_target: true` requires a reviewed
visible-target decision for every member.

```bash
uv run python -m dataset_curation export --config /path/to/candidates.toml --policy /path/to/final-policy.json --plan /path/to/export-plan.json
```

Export verifies exclusion rules, reviewed redundancy, target eligibility,
similarity caps, exact pixel uniqueness, and reference overlap. It copies
native image bytes into empty destinations, checks every copied file hash,
and stages all files before publishing. Existing nonempty output directories
are refused. Metadata, plan, policy, reviews, and embeddings are written
separately from image folders. No empty annotation files are created.

For two labelling batches, each source/recording contributes equally or with
at most one image difference. Nearby views within a recording are paired
across the batches to distribute visual coverage. A fixed seed makes the
assignment reproducible. These are labelling batches, not evaluation splits.
After export, `report` resolves images from their manifest paths, including
images outside the report directory. Serve their common dataset root when
viewing that report over HTTP.

## Development

```bash
uv run pytest
uv run ruff check dataset_curation tests
uv run ruff format --check dataset_curation tests
```

Tests cover temporal sampling, quota allocation, CDR parsing, NV12 conversion,
blocked similarity search, eligibility, selection, content hashing, and a
separate synthetic dataset's extraction/report workflow. They require
neither model access nor a GPU.

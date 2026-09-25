"""Validate a frozen selection and export native images without annotations."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .audit import reference_overlaps
from .batches import assign_batches
from .common import Record, write_json
from .config import Config
from .eligibility import read_policy, reasons
from .retrieval import exact_topk


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _outputs(config: Config, plan: Record) -> tuple[Path, Path]:
    paths = [
        (config.root / plan[key]).resolve() for key in ("images", "output")
    ]
    for path in paths:
        overlaps = any(
            path.is_relative_to(p) or p.is_relative_to(path)
            for p in (config.raw, config.output)
        )
        if not path.is_relative_to(config.root) or overlaps:
            message = "Export outputs must be separate from raw and candidates"
            raise ValueError(message)
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            message = f"Export requires an empty destination: {path}"
            raise ValueError(message)
    if paths[0].is_relative_to(paths[1]) or paths[1].is_relative_to(paths[0]):
        message = "Image and metadata outputs must be separate"
        raise ValueError(message)
    return paths[0], paths[1]


def _selected_rows(
    rows: list[Record], plan: Record, policy: Record, review: Record
) -> list[Record]:
    lookup = {row["id"]: row for row in rows}
    identifiers = [i for group in plan["groups"] for i in group["ids"]]
    if len(identifiers) != len(set(identifiers)) or not identifiers:
        message = "Export IDs must be unique and nonempty"
        raise ValueError(message)
    selected = []
    batch_names = []
    for group in plan["groups"]:
        batch_names.extend(group["batches"])
        for identifier in group["ids"]:
            row = lookup[identifier]
            decision = review["images"].get(identifier, {})
            if (
                reasons(row, policy, review)
                or decision.get("quality") == "unsure"
            ):
                message = f"Excluded or unresolved export image: {identifier}"
                raise ValueError(message)
            if (
                group.get("require_target")
                and decision.get("eligibility") != "visible"
            ):
                message = f"Target visibility is not confirmed: {identifier}"
                raise ValueError(message)
            selected.append(dict(row, group=group["name"], review=decision))
    if len(batch_names) != len(set(batch_names)) or any(
        Path(name).name != name or name in {"", ".", ".."}
        for name in batch_names
    ):
        message = "Batch names must be unique directory names"
        raise ValueError(message)
    return selected


def _validate_pairs(
    config: Config, policy: Record, review: Record, rows: list[Record]
) -> None:
    selected = {row["id"] for row in rows}
    pairs = json.loads((config.root / policy["review_pairs"]).read_text())
    for pair in pairs:
        if review["pairs"].get(pair["id"]) != "redundant":
            continue
        members = [pair["a"], pair["b"]]
        if any(i in selected for i in members) and all(
            i in selected or i.startswith("existing__") for i in members
        ):
            message = f"Selection retains a reviewed redundancy: {pair['id']}"
            raise ValueError(message)


def _assign(
    rows: list[Record], vectors: NDArray[np.float32], plan: Record
) -> tuple[dict[str, str], Record]:
    assignments, groups = {}, {}
    for group in plan["groups"]:
        indices = [
            i for i, row in enumerate(rows) if row["group"] == group["name"]
        ]
        sample = [rows[i] for i in indices]
        matrix = vectors[indices]
        _, scores = exact_topk(matrix, matrix, 1, same_rows=True)
        maximum = float(scores.max()) if scores.size else None
        if maximum is not None and maximum >= group["max_cosine"]:
            message = f"Selection violates similarity cap: {group['name']}"
            raise ValueError(message)
        assignments.update(
            assign_batches(sample, matrix, group["batches"], plan["seed"])
        )
        groups[group["name"]] = {"count": len(sample), "max_cosine": maximum}
    return assignments, groups


def export(config: Config, policy_path: Path, plan_path: Path) -> None:
    policy, review = read_policy(config, policy_path)
    plan = json.loads(plan_path.read_text())
    manifest = config.output / "manifest.jsonl"
    if plan.get("schema_version") != 1 or plan["manifest_sha256"] != _hash(
        manifest
    ):
        message = "Export plan must match the candidate manifest"
        raise ValueError(message)
    if plan["policy_sha256"] != _hash(policy_path) or plan[
        "review_sha256"
    ] != _hash(config.root / policy["review"]):
        message = "Export policy or review changed after the plan was frozen"
        raise ValueError(message)
    if plan["pairs_sha256"] != _hash(config.root / policy["review_pairs"]):
        message = "Reviewed pair identities changed after the plan was frozen"
        raise ValueError(message)
    image_output, output = _outputs(config, plan)
    candidates = [
        json.loads(line) for line in manifest.read_text().splitlines()
    ]
    rows = _selected_rows(candidates, plan, policy, review)
    _validate_pairs(config, policy, review, rows)
    metadata = json.loads(
        (config.output / "embeddings/metadata.json").read_text()
    )
    stored = json.loads((config.output / "embeddings/rows.json").read_text())
    if metadata["manifest_sha256"] != _hash(manifest) or [
        r["id"] for r in stored[: len(candidates)]
    ] != [r["id"] for r in candidates]:
        message = "Candidate embeddings do not match their manifest"
        raise ValueError(message)
    vectors = np.load(config.output / "embeddings/vectors.npy", mmap_mode="r")
    lookup = {row["id"]: i for i, row in enumerate(candidates)}
    selected_vectors = vectors[[lookup[row["id"]] for row in rows]]
    assignments, groups = _assign(rows, selected_vectors, plan)
    content: dict[tuple[tuple[int, int], str], list[str]] = defaultdict(list)
    for row in rows:
        content[((row["width"], row["height"]), row["pixel_sha256"])].append(
            row["id"]
        )
    old = stored[len(candidates) :]
    if any(len(ids) > 1 for ids in content.values()) or reference_overlaps(
        config, old, content
    ):
        message = (
            "Export contains exact duplicate or reference-overlapping pixels"
        )
        raise ValueError(message)
    with tempfile.TemporaryDirectory(
        prefix=".curation-export-", dir=config.root
    ) as temporary:
        staging = Path(temporary)
        staged_images = staging / "images"
        staged_output = staging / "metadata"
        staged_output.mkdir()
        _copy_images(config, rows, assignments, staged_images, image_output)
        _write_metadata(
            config,
            rows,
            selected_vectors,
            vectors[len(candidates) :],
            old,
            metadata,
            plan,
            groups,
            staged_output,
        )
        shutil.copyfile(plan_path, staged_output / "export-plan.json")
        shutil.copyfile(policy_path, staged_output / "selection-policy.json")
        shutil.copyfile(
            config.root / policy["review"], staged_output / "review.json"
        )
        image_output.parent.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        staged_images.replace(image_output)
        staged_output.replace(output)
    print(
        json.dumps(
            {
                "images": str(image_output),
                "metadata": str(output),
                "batches": dict(Counter(assignments.values())),
            },
            indent=2,
        )
    )


def _copy_images(
    config: Config,
    rows: list[Record],
    assignments: dict[str, str],
    staging: Path,
    destination: Path,
) -> None:
    for row in rows:
        original = config.root / row["path"]
        relative = Path(assignments[row["id"]]) / original.name
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            message = f"Duplicate export filename: {relative}"
            raise ValueError(message)
        shutil.copyfile(original, target)
        if _hash(target) != row["file_sha256"]:
            message = f"Export image content changed: {row['id']}"
            raise ValueError(message)
        row["candidate_path"] = row["path"]
        row["path"] = str((destination / relative).relative_to(config.root))
        row["batch"] = assignments[row["id"]]


def _write_metadata(
    config: Config,
    rows: list[Record],
    selected_vectors: NDArray[np.float32],
    old_vectors: NDArray[np.float32],
    old: list[Record],
    metadata: Record,
    plan: Record,
    groups: Record,
    output: Path,
) -> None:
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    folder = output / "embeddings"
    folder.mkdir()
    np.save(
        folder / "vectors.npy", np.concatenate([selected_vectors, old_vectors])
    )
    write_json(folder / "rows.json", rows + old)
    write_json(
        folder / "metadata.json",
        dict(
            metadata,
            manifest_sha256=_hash(manifest),
            pilot_count=len(rows),
            embedding_origin=str(config.output.relative_to(config.root)),
        ),
    )
    write_json(
        output / "initial-review.json",
        {
            "images": {r["id"]: r["review"] for r in rows if r["review"]},
            "pairs": {},
            "manifest_sha256": _hash(manifest),
        },
    )
    write_json(
        output / "summary.json",
        {
            "pilot_count": len(rows),
            "recording_count": len({r["recording_id"] for r in rows}),
            "by_source": dict(Counter(r["source"] for r in rows)),
            "batches": {
                name: dict(
                    Counter(r["source"] for r in rows if r["batch"] == name)
                )
                for name in sorted({r["batch"] for r in rows})
            },
            "groups": groups,
            "seed": plan["seed"],
            "final_labelling_images_selected": True,
            "manifest_sha256": _hash(manifest),
            "exact_pixel_duplicates": 0,
            "exact_reference_overlaps": 0,
            "reference_images_checked": len(old),
            "note": (
                "Frozen labelling batches. User reviews and documented "
                "replacement screening applied; these are not "
                "train/validation/test splits."
            ),
        },
    )

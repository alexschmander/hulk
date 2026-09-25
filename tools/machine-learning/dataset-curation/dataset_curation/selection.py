"""Diverse review shortlists; all budgets and source rules come from a job."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from .common import Record, write_json
from .config import Config
from .eligibility import read_policy, reasons
from .embeddings import _device, neighbors
from .report import build, contact_sheet
from .retrieval import exact_topk


def quality_ranks(rows: list[Record]) -> NDArray[np.float32]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["recording_id"]].append(index)
    ranks = np.zeros(len(rows), dtype=np.float32)
    for members in groups.values():
        ordered = sorted(members, key=lambda i: rows[i]["quality"]["laplacian"])
        ranks[ordered] = np.linspace(0, 1, len(ordered))
    return ranks


def diverse_indices(
    vectors: NDArray[np.float32],
    rows: list[Record],
    count: int,
    *,
    max_cosine: float,
    device: str,
    conflicts: dict[str, set[str]],
    recording_cap: int,
    preferred_ids: list[str] | None = None,
) -> list[int]:
    if not rows or count <= 0:
        return []
    matrix = torch.as_tensor(vectors, device=device)
    quality = torch.as_tensor(quality_ranks(rows), device=device)
    available = torch.ones(len(rows), dtype=torch.bool, device=device)
    similarities = torch.full((len(rows),), -1.0, device=device)
    lookup = {row["id"]: index for index, row in enumerate(rows)}
    preferred = [
        lookup[identifier]
        for identifier in (preferred_ids or [])
        if identifier in lookup
    ]
    recordings: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        recordings[row["recording_id"]].append(index)
    counts: Counter[str] = Counter()
    selected = []
    pixel_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        pixel_groups[row["pixel_sha256"]].append(index)
    seed_scores = matrix @ torch.nn.functional.normalize(matrix.mean(0), dim=0)
    with torch.inference_mode():
        for step in range(min(count, len(rows))):
            scores = (
                1 - similarities if step else seed_scores
            ) + 0.025 * quality
            scores[~available] = -torch.inf
            index = (
                preferred[step]
                if step < len(preferred)
                else int(scores.argmax())
            )
            if not bool(available[index]):
                if step < len(preferred):
                    message = (
                        "Preferred selection violates diversity constraints"
                    )
                    raise ValueError(message)
                break
            selected.append(index)
            row = rows[index]
            counts[row["recording_id"]] += 1
            similarities = torch.maximum(similarities, matrix @ matrix[index])
            available &= similarities < max_cosine
            available[pixel_groups[row["pixel_sha256"]]] = False
            for identifier in conflicts.get(row["id"], set()):
                if identifier in lookup:
                    available[lookup[identifier]] = False
            if counts[row["recording_id"]] >= recording_cap:
                available[recordings[row["recording_id"]]] = False
    return selected


def target_candidates(
    pool: list[int],
    rows: list[Record],
    vectors: NDArray[np.float32],
    review: Record,
    device: str,
    minimum_cosine: float,
) -> list[int]:
    positives, negatives = [], []
    for index in pool:
        decision = review["images"].get(rows[index]["id"], {})
        if decision.get("eligibility") == "visible":
            positives.append(index)
    for index, row in enumerate(rows):
        if review["images"].get(row["id"], {}).get("eligibility") == "absent":
            negatives.append(index)
    if not positives:
        message = "Target shortlist requires reviewed positive examples"
        raise ValueError(message)
    _, positive_scores = exact_topk(
        vectors[pool], vectors[positives], 1, device=device
    )
    if not negatives:
        return [
            i
            for n, i in enumerate(pool)
            if positive_scores[n, 0] >= minimum_cosine
        ]
    _, negative_scores = exact_topk(
        vectors[pool], vectors[negatives], 1, device=device
    )
    return [
        index
        for offset, index in enumerate(pool)
        if positive_scores[offset, 0]
        >= max(minimum_cosine, negative_scores[offset, 0])
    ]


def quality_candidates(
    pool: list[int],
    rows: list[Record],
    neighbor_rows: list[Record],
    review: Record,
    settings: Record,
) -> list[int]:
    """Prefer sharper similar scenes; preserve explicit usable decisions."""
    lookup = {rows[i]["id"]: rows[i] for i in pool}
    neighbors_by_id = {
        row["id"]: row["pilot_neighbors"] for row in neighbor_rows
    }
    result = []
    for index in pool:
        row = rows[index]
        if review["images"].get(row["id"], {}).get("quality") == "usable":
            result.append(index)
            continue
        matches = neighbors_by_id[row["id"]]
        alternatives = [
            lookup[m["id"]]["quality"]["laplacian"]
            for m in matches
            if m["cosine"] >= settings["neighbor_cosine"] and m["id"] in lookup
        ]
        sharp = row["quality"]["laplacian"]
        if (
            alternatives
            and sharp >= settings["relative_sharpness"] * max(alternatives)
            and sharp > settings["minimum_sharpness"]
        ):
            result.append(index)
    return result


def _conflicts(
    config: Config, policy: Record, review: Record
) -> dict[str, set[str]]:
    pairs = json.loads((config.root / policy["review_pairs"]).read_text())
    result: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        if review["pairs"].get(pair["id"]) == "redundant":
            result[pair["a"]].add(pair["b"])
            result[pair["b"]].add(pair["a"])
    return result


def select(config: Config, path: Path) -> None:
    policy, review = read_policy(config, path)
    job = policy["selection"]
    output = (config.root / job["output"]).resolve()
    if (
        not output.is_relative_to(config.root)
        or output.is_relative_to(config.output)
        or config.output.is_relative_to(output)
        or (
            output.is_relative_to(config.raw)
            or config.raw.is_relative_to(output)
        )
    ):
        message = (
            "Selection output must be separate from raw and candidate inputs"
        )
        raise ValueError(message)
    manifest = config.output / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    metadata = json.loads(
        (config.output / "embeddings/metadata.json").read_text()
    )
    if (
        metadata["manifest_sha256"]
        != hashlib.sha256(manifest.read_bytes()).hexdigest()
    ):
        message = "Embeddings do not match candidate manifest"
        raise ValueError(message)
    stored_rows = json.loads(
        (config.output / "embeddings/rows.json").read_text()
    )
    if [r["id"] for r in stored_rows[: len(rows)]] != [r["id"] for r in rows]:
        message = "Embedding rows have a different order"
        raise ValueError(message)
    vectors = np.load(config.output / "embeddings/vectors.npy")
    allowed = [not reasons(row, policy, review) for row in rows]
    conflicts = _conflicts(config, policy, review)
    neighbor_rows = json.loads((config.output / "neighbors.json").read_text())
    device = _device(config.embedding.device)
    chosen, reserves, groups = [], [], {}
    for group in job["groups"]:
        pool = [
            i
            for i, row in enumerate(rows)
            if allowed[i] and row["source"] in group["sources"]
        ]
        pool = quality_candidates(
            pool, rows, neighbor_rows, review, job["quality"]
        )
        if group.get("require_target", False):
            pool = target_candidates(
                pool,
                rows,
                vectors,
                review,
                device,
                group["minimum_target_cosine"],
            )
        selected = diverse_indices(
            vectors[pool],
            [rows[i] for i in pool],
            group["count"] + group["reserve"],
            max_cosine=group["max_cosine"],
            device=device,
            conflicts=conflicts,
            recording_cap=group.get("recording_cap", len(pool)),
            preferred_ids=group.get("preferred_ids"),
        )
        global_indices = [pool[i] for i in selected]
        chosen.extend(global_indices[: group["count"]])
        reserves.extend(global_indices[group["count"] :])
        groups[group["name"]] = {
            "eligible_proposals": len(pool),
            "selected": min(len(selected), group["count"]),
            "reserve": max(0, len(selected) - group["count"]),
            "max_cosine": group["max_cosine"],
        }
    if len(set(chosen + reserves)) != len(chosen + reserves):
        message = "Selection groups overlap"
        raise ValueError(message)
    _write_shortlist(
        config,
        output,
        rows,
        stored_rows[len(rows) :],
        vectors,
        chosen,
        reserves,
        groups,
        metadata,
        policy,
        path,
    )


def _write_shortlist(
    config: Config,
    output: Path,
    rows: list[Record],
    old: list[Record],
    vectors: NDArray[np.float32],
    chosen: list[int],
    reserves: list[int],
    groups: Record,
    metadata: Record,
    policy: Record,
    policy_path: Path,
) -> None:
    (output / "images").mkdir(parents=True, exist_ok=True)
    result = []
    for index in chosen:
        row = dict(rows[index])
        original = config.root / row["path"]
        destination = output / "images" / original.name
        shutil.copyfile(original, destination)
        row["path"] = str(destination.relative_to(config.root))
        result.append(row)
    generated = {Path(row["path"]).name for row in rows}
    retained = {Path(row["path"]).name for row in result}
    for previous in (output / "images").iterdir():
        if previous.name in generated - retained and previous.is_file():
            previous.unlink()
    result.sort(
        key=lambda row: (
            row["source"],
            row["recording_id"],
            row.get("image_ordinal", row.get("filename_tick")),
        )
    )
    by_id = {row["id"]: i for i, row in enumerate(rows)}
    ordered = [by_id[row["id"]] for row in result]
    matrix = vectors[ordered + list(range(len(rows), len(vectors)))]
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in result))
    (output / "reserve.jsonl").write_text(
        "".join(json.dumps(rows[i]) + "\n" for i in reserves)
    )
    folder = output / "embeddings"
    folder.mkdir(exist_ok=True)
    np.save(folder / "vectors.npy", matrix)
    write_json(folder / "rows.json", result + old)
    copied = dict(
        metadata,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        pilot_count=len(result),
        embedding_origin=str(config.output.relative_to(config.root)),
    )
    write_json(folder / "metadata.json", copied)
    summary = {
        "pilot_count": len(result),
        "recording_count": len({r["recording_id"] for r in result}),
        "by_source": dict(Counter(r["source"] for r in result)),
        "groups": groups,
        "final_labelling_images_selected": False,
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "selection": policy["selection"],
        "note": (
            "Draft shortlist; target proposals and image quality "
            "need visual confirmation."
        ),
    }
    write_json(output / "summary.json", summary)
    original_review = json.loads((config.root / policy["review"]).read_text())
    carried = {
        "images": {
            row["id"]: original_review["images"][row["id"]]
            for row in result
            if row["id"] in original_review["images"]
        },
        "pairs": {},
        "origin": policy["review"],
        "manifest_sha256": copied["manifest_sha256"],
    }
    write_json(output / "initial-review.json", carried)
    shortlist_config = replace(
        config, output=output, title="Dataset selection shortlist"
    )
    neighbors(shortlist_config, result, old, matrix)
    build(shortlist_config)
    for number, row in enumerate(result, 1):
        row["review_number"] = number
    for source in sorted({r["source"] for r in result}):
        sample = [r for r in result if r["source"] == source]
        for page, start in enumerate(range(0, len(sample), 24), 1):
            contact_sheet(
                config.root,
                sample[start : start + 24],
                output / "contact-sheets" / f"selection-{source}-{page:02}.jpg",
                f"{source}: selection review {page}",
            )
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key != "selection"
            },
            indent=2,
        ),
        flush=True,
    )

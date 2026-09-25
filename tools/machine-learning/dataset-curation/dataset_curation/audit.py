"""Exact-content overlap checks and focused nearest-neighbor review queues."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from .common import Record, identity, write_json
from .config import Config


def content_hash(path: Path) -> tuple[tuple[int, int], str]:
    with Image.open(path) as original:
        image = original.convert("RGB")
        return image.size, hashlib.sha256(image.tobytes()).hexdigest()


def focused_pairs(
    rows: list[Record], neighbors: list[Record], per_category: int
) -> list[Record]:
    sources = {row["id"]: row["source"] for row in rows}
    categories: dict[str, dict[tuple[str, str], Record]] = defaultdict(dict)
    for entry in neighbors:
        for match in entry["pilot_neighbors"]:
            category = (
                sources[entry["id"]]
                if sources[entry["id"]] == sources[match["id"]]
                else "cross-source"
            )
            pair = tuple(sorted((entry["id"], match["id"])))
            categories[category][pair] = {
                "a": pair[0],
                "b": pair[1],
                "cosine": match["cosine"],
            }
        for match in entry["existing_neighbors"]:
            category = "existing-" + match["split"]
            pair = (entry["id"], match["id"])
            categories[category][pair] = {
                "a": pair[0],
                "b": pair[1],
                "cosine": match["cosine"],
            }
    result = []
    for category, pairs in sorted(categories.items()):
        ranked = sorted(pairs.values(), key=lambda p: -p["cosine"])
        for pair in ranked[:per_category]:
            pair["kind"] = category
            pair["id"] = "audit-" + identity(pair["a"] + ":" + pair["b"])
            pair["calibration_partition"] = "review"
            result.append(pair)
    return result


def audit(config: Config) -> None:
    manifest = config.output / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    metadata = json.loads(
        (config.output / "embeddings/metadata.json").read_text()
    )
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if metadata["manifest_sha256"] != manifest_hash:
        message = "Audit embeddings belong to a different manifest"
        raise ValueError(message)
    by_content: dict[tuple[tuple[int, int], str], list[str]] = defaultdict(list)
    for row in rows:
        path = config.root / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["file_sha256"]:
            message = f"Image bytes changed: {row['id']}"
            raise ValueError(message)
        by_content[((row["width"], row["height"]), row["pixel_sha256"])].append(
            row["id"]
        )
    old = [
        row
        for row in json.loads(
            (config.output / "embeddings/rows.json").read_text()
        )
        if row["source"] == "existing"
    ]
    overlaps = reference_overlaps(config, old, by_content)
    neighbors = json.loads((config.output / "neighbors.json").read_text())
    pairs = focused_pairs(rows, neighbors, config.pair_samples_per_category)
    write_json(
        config.output / "audit-pairs.json",
        {"manifest_sha256": manifest_hash, "pairs": pairs},
    )
    summary = {
        "manifest_sha256": manifest_hash,
        "images_checked": len(rows),
        "reference_images_checked": len(old),
        "duplicate_pixel_groups": [
            v for v in by_content.values() if len(v) > 1
        ],
        "exact_reference_overlaps": overlaps,
        "focused_pairs": len(pairs),
        "note": (
            "Exact-content checks do not establish visual uniqueness. "
            "Review the similar-image pairs."
        ),
    }
    write_json(config.output / "audit.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def reference_overlaps(
    config: Config,
    old: list[Record],
    by_content: dict[tuple[tuple[int, int], str], list[str]],
) -> list[Record]:
    if not old:
        return []
    if config.reference is None:
        message = "Reference images require a reference root"
        raise ValueError(message)
    overlaps = []
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        hashes = pool.map(
            content_hash, [config.reference / r["path"] for r in old]
        )
        for row, key in zip(old, hashes, strict=True):
            for identifier in by_content.get(key, []):
                overlaps.append(
                    {"a": identifier, "b": row["id"], "split": row["split"]}
                )
    return overlaps

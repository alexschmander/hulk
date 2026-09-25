"""Sparse MCAP contact sheets for locating unusable intervals."""

from __future__ import annotations

import json

from .common import Record, write_json
from .config import Config
from .extraction import export_mcap, index_mcap
from .report import contact_sheet


def sample_timeline(rows: list[Record], step_seconds: float) -> list[Record]:
    if step_seconds <= 0:
        message = "Timeline step must be positive"
        raise ValueError(message)
    chosen = {}
    origins = {}
    previous = None
    for row in rows:
        segment = row["segment"]
        origin = origins.setdefault(segment, row["timestamp_ns"])
        bucket = int((row["timestamp_ns"] - origin) / (step_seconds * 1e9))
        key = segment, bucket
        if key != previous:
            chosen[row["image_ordinal"]] = row
        previous = key
    if rows:
        chosen[rows[-1]["image_ordinal"]] = rows[-1]
    return list(chosen.values())


def build(config: Config, step_seconds: float = 10) -> None:
    (config.output / "images").mkdir(parents=True, exist_ok=True)
    (config.output / "indexes").mkdir(exist_ok=True)
    sheets = config.output / "contact-sheets"
    sheets.mkdir(exist_ok=True)
    all_rows = []
    for source in config.sources:
        if source.kind != "ros_z_mcap":
            continue
        for path in sorted(config.raw.glob(source.pattern)):
            index = index_mcap(config, source, path)
            selected = sample_timeline(index["candidates"], step_seconds)
            sample = dict(
                index, candidates=selected, candidate_count=len(selected)
            )
            rows = export_mcap(config, source, path, sample, len(selected))
            for number, row in enumerate(rows, len(all_rows) + 1):
                row["review_number"] = number
                row["recording_name"] = (
                    f"frame {row['image_ordinal']} segment {row['segment']}"
                )
            for page, start in enumerate(range(0, len(rows), 24), 1):
                contact_sheet(
                    config.root,
                    rows[start : start + 24],
                    sheets / f"{index['recording_id']}-{page:02}.jpg",
                    f"{source.name}: {path.parent.name}; page {page}",
                )
            all_rows.extend(rows)
    manifest = config.output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in all_rows))
    write_json(
        config.output / "timeline.json",
        {
            "step_seconds": step_seconds,
            "image_count": len(all_rows),
            "purpose": "Locate intervals; inspect full-cadence boundaries next",
        },
    )
    print(f"Timeline: {len(all_rows)} images; {sheets}", flush=True)

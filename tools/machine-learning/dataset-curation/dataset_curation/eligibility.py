"""Auditable interval exclusions and explicit image-review decisions."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .common import Record, write_json
from .config import Config


def read_policy(config: Config, path: Path) -> tuple[Record, Record]:
    policy = json.loads(path.read_text())
    if policy.get("schema_version") != 1:
        message = "Unsupported selection policy schema"
        raise ValueError(message)
    for interval in policy.get("exclude_intervals", []):
        if not 1 <= interval["first"] <= interval["last"]:
            message = "Interval bounds must be inclusive positive ordinals"
            raise ValueError(message)
        source = (config.raw / interval["source_path"]).resolve()
        if not source.is_relative_to(config.raw):
            message = "Interval source must stay within raw inputs"
            raise ValueError(message)
        stat = source.stat()
        if [stat.st_size, stat.st_mtime_ns] != interval["source_signature"]:
            message = f"Interval belongs to different source data: {source}"
            raise ValueError(message)
    review = json.loads((config.root / policy["review"]).read_text())
    manifest = config.root / policy["review_manifest"]
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != review.get(
        "manifest_sha256"
    ):
        message = "Review and its original manifest do not match"
        raise ValueError(message)
    known = {
        json.loads(line)["id"] for line in manifest.read_text().splitlines()
    }
    if not set(review["images"]).issubset(known):
        message = "Review contains unknown image identities"
        raise ValueError(message)
    return policy, review


def reasons(row: Record, policy: Record, review: Record) -> list[str]:
    excluded = []
    for interval in policy.get("exclude_intervals", []):
        if row["source_path"] == interval["source_path"] and (
            interval["first"] <= row.get("image_ordinal", 0) <= interval["last"]
        ):
            excluded.append("interval: " + interval["reason"])
    decision = review["images"].get(row["id"], {})
    if decision.get("quality") in {"blur", "scene"}:
        excluded.append("review: " + decision["quality"])
    if decision.get("eligibility") == "absent":
        excluded.append("review: target absent")
    if row["id"] in policy.get("exclude_images", {}):
        excluded.append("image: " + policy["exclude_images"][row["id"]])
    return excluded


def apply(config: Config, path: Path) -> None:
    policy, review = read_policy(config, path)
    manifest = config.output / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    audit = []
    for row in rows:
        excluded = reasons(row, policy, review)
        decision = review["images"].get(row["id"], {})
        audit.append(
            {
                "id": row["id"],
                "source": row["source"],
                "excluded": bool(excluded),
                "reasons": excluded,
                "quality_review": decision.get("quality"),
                "eligibility_review": decision.get("eligibility"),
            }
        )
    destination = config.output / "eligibility.jsonl"
    temporary = destination.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in audit))
    temporary.replace(destination)
    summary = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "review_sha256": hashlib.sha256(
            (config.root / policy["review"]).read_bytes()
        ).hexdigest(),
        "remaining_by_source": dict(
            Counter(row["source"] for row in audit if not row["excluded"])
        ),
        "excluded_by_source": dict(
            Counter(row["source"] for row in audit if row["excluded"])
        ),
        "note": "Remaining images still require quality and eligibility checks",
    }
    write_json(config.output / "eligibility-summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)

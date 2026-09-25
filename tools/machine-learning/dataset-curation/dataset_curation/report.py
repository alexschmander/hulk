"""Offline calibration gallery and contact sheets."""

from __future__ import annotations

import hashlib
import html
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import Record, write_json
from .config import Config


def contact_sheet(
    root: Path, rows: list[Record], destination: Path, title: str
) -> None:
    columns, cell_w, cell_h = 4, 340, 315
    canvas = Image.new(
        "RGB",
        (
            columns * cell_w,
            55 + ((len(rows) + columns - 1) // columns) * cell_h,
        ),
        "#151b23",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=13)
    draw.text((12, 15), title, fill="white", font=font)
    for index, row in enumerate(rows):
        x, y = index % columns * cell_w, 55 + index // columns * cell_h
        with Image.open(root / row["path"]) as original:
            image = original.convert("RGB")
            image.thumbnail((336, 277))
            canvas.paste(image, (x, y))
        label = (
            f"{row['review_number']:04} {row['source']} "
            f"sharp {row['quality']['laplacian']:.1f}"
        )
        draw.text((x + 3, y + 278), label, fill="white", font=font)
        name = row.get("recording_name", Path(row["source_path"]).parent.name)
        draw.text((x + 3, y + 296), name[:44], fill="#adbacc", font=font)
    canvas.save(destination, quality=92)


def read_pairs(output: Path, manifest_hash: str) -> list[Record]:
    audit_path = output / "audit-pairs.json"
    if audit_path.exists():
        audited = json.loads(audit_path.read_text())
        if audited["manifest_sha256"] == manifest_hash:
            return audited["pairs"]
    pair_path = output / "pairs.json"
    return json.loads(pair_path.read_text()) if pair_path.exists() else []


def build(config: Config) -> None:
    output = config.output
    manifest = output / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    summary = json.loads((output / "summary.json").read_text())
    questions = {
        source.name: source.eligibility_question for source in config.sources
    }
    for index, row in enumerate(rows):
        row["review_number"] = index + 1
        row["url"] = Path(
            os.path.relpath(config.root / row["path"], output)
        ).as_posix()
        row["eligibility_question"] = questions.get(row["source"], "")
    blur = []
    sheets = output / "contact-sheets"
    sheets.mkdir(exist_ok=True)
    for source in sorted({row["source"] for row in rows}):
        candidates = sorted(
            [row for row in rows if row["source"] == source],
            key=lambda row: row["quality"]["laplacian"],
        )
        indices = np.linspace(
            0,
            len(candidates) - 1,
            min(config.blur_samples_per_source, len(candidates)),
        ).astype(int)
        if "selection" in summary:
            indices = np.arange(
                min(config.blur_samples_per_source, len(candidates))
            )
        blur.extend(candidates[index]["id"] for index in indices)
        examples = [
            candidates[index]
            for index in np.linspace(
                0, len(candidates) - 1, min(24, len(candidates))
            ).astype(int)
        ]
        contact_sheet(
            config.root,
            examples,
            sheets / f"{source}-blur-range.jpg",
            f"{source}: increasing sharpness; "
            "texture is not annotation usability",
        )
    eligible = [row for row in rows if row["eligibility_question"]]
    for page, start in enumerate(range(0, len(eligible), 24), 1):
        contact_sheet(
            config.root,
            eligible[start : start + 24],
            sheets / f"eligibility-overview-{page:02}.jpg",
            f"Eligibility review {page}; numbers match gallery",
        )
    pairs = read_pairs(output, manifest_hash)
    metadata_path = output / "embeddings/metadata.json"
    metadata = (
        json.loads(metadata_path.read_text())
        if metadata_path.exists()
        else None
    )
    if metadata and metadata["manifest_sha256"] != manifest_hash:
        message = (
            "Embeddings belong to a different pilot manifest; regenerate them"
        )
        raise ValueError(message)
    by_id = {row["id"]: row for row in rows}
    if pairs:
        references = {
            row["id"]: row
            for row in json.loads((output / "embeddings/rows.json").read_text())
            if row["source"] == "existing"
        }
        required = {
            pair[key]
            for pair in pairs
            for key in ["a", "b"]
            if pair[key].startswith("existing__")
        }
        folder = output / "reference-images"
        folder.mkdir(exist_ok=True)
        for identifier in required:
            if config.reference is None:
                message = "Review pairs require a reference directory"
                raise ValueError(message)
            row = references[identifier]
            original = config.reference / row["path"]
            destination = folder / (identifier + original.suffix)
            shutil.copyfile(original, destination)
            row["url"] = "reference-images/" + destination.name
            by_id[identifier] = row
    initial_path = output / "initial-review.json"
    initial = (
        json.loads(initial_path.read_text()) if initial_path.exists() else {}
    )
    if initial and initial["manifest_sha256"] != manifest_hash:
        message = "Initial review belongs to a different manifest"
        raise ValueError(message)
    data = {
        "title": config.title,
        "sources": [source.name for source in config.sources],
        "rows": list(by_id.values()),
        "blur": blur,
        "pairs": pairs,
        "summary": summary,
        "model": metadata,
        "manifest_sha256": manifest_hash,
        "initial_review": initial,
    }
    template = Path(__file__).with_name("review-template.html").read_text()
    content = template.replace(
        "__DATA__", json.dumps(data).replace("<", "\\u003c")
    )
    (output / "review.html").write_text(
        content.replace("__TITLE__", html.escape(config.title))
    )
    write_json(
        output / "review-sample.json",
        {
            "blur_ids": blur,
            "pair_ids": [pair["id"] for pair in pairs],
            "manifest_sha256": manifest_hash,
        },
    )
    print(
        f"Review gallery: {output / 'review.html'}; "
        f"{len(blur)} blur examples; {len(pairs)} pairs"
    )

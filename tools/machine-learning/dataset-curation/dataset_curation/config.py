"""Validated job configuration stored beside the dataset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import tomllib


@dataclass(frozen=True)
class Source:
    name: str
    kind: str
    pattern: str
    pilot_count: int = 0
    frame_rate: float = 0
    filename_pattern: str = r".*_(\d+)"
    topic: str = ""
    stereo: bool = False
    eligibility_question: str = ""


@dataclass(frozen=True)
class Embedding:
    model: str
    revision: str = "main"
    size: int = 256
    batch_size: int = 32
    workers: int = 4
    device: str = "auto"


@dataclass(frozen=True)
class Config:
    path: Path
    title: str
    root: Path
    raw: Path
    output: Path
    sources: tuple[Source, ...]
    embedding: Embedding
    reference: Path | None = None
    fps: float = 3
    burst_size: int = 3
    timestamp_gap_seconds: float = 10
    workers: int = 4
    blur_samples_per_source: int = 60
    pair_samples_per_category: int = 36
    mode: str = "pilot"


def load_config(path: Path) -> Config:
    path = path.expanduser().resolve()
    with path.open("rb") as file:
        document = tomllib.load(file)
    if document.get("schema_version") != 1:
        message = "Expected configuration schema_version = 1"
        raise ValueError(message)
    dataset = document["dataset"]
    root = (path.parent / dataset["root"]).resolve()
    raw = (root / dataset.get("raw", "raw-source")).resolve()
    output = (root / dataset.get("output", "curation/pilot")).resolve()
    if not raw.is_relative_to(root) or not output.is_relative_to(root):
        message = "Raw and output directories must be within the dataset root"
        raise ValueError(message)
    if output.is_relative_to(raw) or raw.is_relative_to(output):
        message = "Raw inputs and generated output directories must be separate"
        raise ValueError(message)
    sources = tuple(Source(**source) for source in document["sources"])
    if not sources or len({source.name for source in sources}) != len(sources):
        message = "Sources must have distinct names and cannot be empty"
        raise ValueError(message)
    _validate_sources(sources)
    sampling = document.get("sampling", {})
    review = document.get("review", {})
    reference = document.get("reference", {}).get("root")
    result = Config(
        path=path,
        title=dataset.get("title", "Dataset selection pilot"),
        root=root,
        raw=raw,
        output=output,
        sources=sources,
        embedding=Embedding(**document["embedding"]),
        reference=(path.parent / reference).resolve() if reference else None,
        fps=sampling.get("fps", 3),
        burst_size=sampling.get("burst_size", 3),
        timestamp_gap_seconds=sampling.get("timestamp_gap_seconds", 10),
        workers=sampling.get("workers", 4),
        blur_samples_per_source=review.get("blur_samples_per_source", 60),
        pair_samples_per_category=review.get("pair_samples_per_category", 36),
        mode=sampling.get("mode", "pilot"),
    )
    positive = [
        result.fps,
        result.burst_size,
        result.timestamp_gap_seconds,
        result.workers,
        result.embedding.size,
        result.embedding.batch_size,
        result.blur_samples_per_source,
        result.pair_samples_per_category,
    ]
    if min(positive) <= 0 or result.embedding.workers < 0:
        message = (
            "Rates, sizes and counts must be positive; embedding "
            "workers can be zero"
        )
        raise ValueError(message)
    if result.embedding.device not in {"auto", "cpu", "cuda"}:
        message = "Embedding device must be auto, cpu or cuda"
        raise ValueError(message)
    if result.mode not in {"pilot", "all"}:
        message = "Sampling mode must be pilot or all"
        raise ValueError(message)
    return result


def _validate_sources(sources: tuple[Source, ...]) -> None:
    for source in sources:
        if source.name == "existing" or not re.fullmatch(
            r"[A-Za-z0-9_-]+", source.name
        ):
            message = "Source names must contain only letters, digits, _ or -"
            raise ValueError(message)
        if source.pilot_count < 0:
            message = "Pilot counts cannot be negative"
            raise ValueError(message)
        if (
            Path(source.pattern).is_absolute()
            or ".." in Path(source.pattern).parts
        ):
            message = "Source patterns must stay within raw inputs"
            raise ValueError(message)
        if source.kind not in {"image_sequence", "ros_z_mcap"}:
            message = f"Unsupported source kind: {source.kind}"
            raise ValueError(message)
        if source.kind == "image_sequence":
            if source.frame_rate <= 0:
                message = "Image sequences require a positive frame_rate"
                raise ValueError(message)
            if re.compile(source.filename_pattern).groups != 1:
                message = (
                    "filename_pattern must capture exactly one numeric frame "
                    "identifier"
                )
                raise ValueError(message)
        elif not source.topic:
            message = "MCAP sources require an image topic"
            raise ValueError(message)

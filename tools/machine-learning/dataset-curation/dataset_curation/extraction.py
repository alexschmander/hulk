"""Source adapters and deterministic pilot extraction."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .common import Record, identity, write_json
from .config import Config, Source
from .mcap_io import image_messages
from .quality import luma, quality, rgb709, sharpness
from .sampling import TimeWindows, allocate, spread_bursts

INDEX_VERSION = 3


def index_mcap(config: Config, source: Source, path: Path) -> Record:
    relative = str(path.relative_to(config.raw))
    recording = identity(relative)
    cache = config.output / "indexes" / f"{recording}.json"
    signature = {
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "version": INDEX_VERSION,
        "fps": config.fps,
        "topic": source.topic,
        "stereo": source.stereo,
        "source": source.name,
        "timestamp_gap_seconds": config.timestamp_gap_seconds,
    }
    if cache.is_file():
        previous = json.loads(cache.read_text())
        if previous["signature"] == signature:
            return previous
    audit: Record = {}
    candidates: list[Record] = []
    pending: Record | None = None
    window: tuple[int, int] | None = None
    clock = TimeWindows(config.fps, config.timestamp_gap_seconds)
    duplicate_timestamps = fallbacks = 0
    started = time.monotonic()
    for image, metadata in image_messages(
        path, audit, source.topic, stereo=source.stereo
    ):
        use_capture = image["valid_timestamp"] and image["capture_ns"] > 0
        timestamp = image["capture_ns"] if use_capture else metadata["log_ns"]
        clock_name = "capture" if use_capture else "log"
        fallbacks += int(not use_capture)
        key = clock.assign(timestamp, clock_name)
        if key is None:
            duplicate_timestamps += 1
            continue
        if window != key:
            if pending is not None:
                candidates.append(pending)
            pending = None
            window = key
        score = sharpness(luma(image))
        if pending is None or score > pending["selection_sharpness"]:
            pending = dict(
                **metadata,
                capture_ns=image["capture_ns"],
                timestamp_ns=timestamp,
                timestamp_source=clock_name,
                recording_id=recording,
                source_path=relative,
                source=source.name,
                segment=key[0],
                window=key[1],
                selection_sharpness=score,
                width=image["width"],
                height=image["height"],
                camera="left",
            )
    if pending is not None:
        candidates.append(pending)
    result = {
        "signature": signature,
        "source_path": relative,
        "recording_id": recording,
        "audit": audit,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "timestamp_resets": clock.segment,
        "duplicate_timestamps": duplicate_timestamps,
        "timestamp_fallbacks": fallbacks,
        "discontinuities": clock.discontinuities,
        "median_frame_interval_ms": float(np.median(clock.intervals)) / 1e6
        if clock.intervals
        else None,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(cache, result)
    print(
        f"Indexed {relative}: {len(candidates)} candidates; "
        f"tail status: {audit['stream_error']}",
        flush=True,
    )
    return result


def sequence_groups(
    config: Config, source: Source, directory: Path
) -> tuple[list[Record], int]:
    pattern = re.compile(source.filename_pattern)
    frames = []
    for path in directory.iterdir():
        match = pattern.fullmatch(path.stem)
        if (
            path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            and match
        ):
            frames.append((int(match.group(1)), path))
    frames.sort()
    ticks = [tick for tick, _ in frames]
    if len(set(ticks)) != len(ticks):
        message = f"Repeated numeric image identifiers in {directory}"
        raise ValueError(message)
    gap = (
        float(np.median(np.diff(ticks))) * 1.8
        if len(ticks) > 1
        else float("inf")
    )
    rate = Fraction(str(config.fps)) / Fraction(str(source.frame_rate))
    groups: list[Record] = []
    segment = position = 0
    previous_tick = None
    key = None
    for ordinal, (tick, path) in enumerate(frames):
        if previous_tick is not None and tick - previous_tick > gap:
            segment += 1
            position = 0
        bucket = position * rate.numerator // rate.denominator
        if key != (segment, bucket):
            groups.append({"segment": segment, "window": bucket, "frames": []})
            key = segment, bucket
        groups[-1]["frames"].append((ordinal, tick, path))
        position += 1
        previous_tick = tick
    return groups, len(frames)


def save_candidate(
    config: Config,
    row: Record,
    rgb: NDArray[np.uint8],
    original: Path | None = None,
) -> Record:
    row = dict(row)
    frame = row.get("image_ordinal", row.get("filename_tick"))
    key = f"{row['recording_id']}:{frame}"
    row["id"] = f"{row['source']}__{identity(key)}"
    suffix = original.suffix.lower() if original else ".png"
    destination = config.output / "images" / (row["id"] + suffix)
    if original:
        shutil.copyfile(original, destination)
    else:
        Image.fromarray(rgb).save(destination, compress_level=2)
    row["path"] = str(destination.relative_to(config.root))
    row["quality"] = quality(rgb)
    row["pixel_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
    row["file_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    return row


def export_sequence(
    config: Config,
    source: Source,
    directory: Path,
    groups: list[Record],
    target: int,
) -> list[Record]:
    rows = []
    recording = identity(str(directory.relative_to(config.raw)))
    for index in spread_bursts(len(groups), target, config.burst_size):
        group = groups[index]
        choices = []
        for ordinal, tick, path in group["frames"]:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
            score = sharpness(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
            choices.append((score, ordinal, tick, path, rgb))
        score, ordinal, tick, path, rgb = max(
            choices, key=lambda value: value[0]
        )
        row = {
            "source": source.name,
            "source_path": str(path.relative_to(config.raw)),
            "recording_id": recording,
            "recording_name": directory.name,
            "camera": "exported",
            "frame_index": ordinal,
            "filename_tick": tick,
            "cadence_fps": source.frame_rate,
            "segment": group["segment"],
            "window": group["window"],
            "selection_sharpness": score,
            "pair_paths": [
                str(value[3].relative_to(config.raw)) for value in choices
            ],
            "pair_scores": [value[0] for value in choices],
            "width": rgb.shape[1],
            "height": rgb.shape[0],
        }
        rows.append(save_candidate(config, row, rgb, original=path))
    print(
        f"Exported {len(rows)} images: {source.name}/{directory.name}",
        flush=True,
    )
    return rows


def export_mcap(
    config: Config, source: Source, path: Path, index: Record, target: int
) -> list[Record]:
    selected = {
        index["candidates"][i]["image_ordinal"]: index["candidates"][i]
        for i in spread_bursts(
            index["candidate_count"], target, config.burst_size
        )
    }
    rows = []
    if not selected:
        return rows
    needed = set(selected)
    for image, metadata in image_messages(
        path, {}, source.topic, stereo=source.stereo
    ):
        ordinal = metadata["image_ordinal"]
        if ordinal in selected:
            rows.append(
                save_candidate(config, selected[ordinal], rgb709(image))
            )
            needed.remove(ordinal)
        if not needed:
            break
    if needed:
        message = f"Indexed images missing from {path}: {sorted(needed)}"
        raise RuntimeError(message)
    print(
        f"Exported {len(rows)} images: {source.name}/{path.parent.name}",
        flush=True,
    )
    return rows


def extract(config: Config) -> None:
    cv2.setNumThreads(1)
    (config.output / "images").mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows: list[Record] = []
    inventories: list[Record] = []
    indexes: list[Record] = []
    seen: set[Path] = set()
    for source in config.sources:
        paths = sorted(
            path
            for path in config.raw.glob(source.pattern)
            if path.is_dir() == (source.kind == "image_sequence")
        )
        if not paths:
            message = f"Source {source.name} matched no inputs"
            raise ValueError(message)
        resolved = {path.resolve() for path in paths}
        if seen & resolved:
            message = f"Source {source.name} overlaps another source's inputs"
            raise ValueError(message)
        seen.update(resolved)
        if source.kind == "ros_z_mcap":
            source_indexes = [
                index_mcap(config, source, path) for path in paths
            ]
            quotas = allocate(
                [index["candidate_count"] for index in source_indexes],
                sum(index["candidate_count"] for index in source_indexes)
                if config.mode == "all"
                else source.pilot_count,
            )
            for path, index, quota in zip(
                paths, source_indexes, quotas, strict=True
            ):
                rows.extend(export_mcap(config, source, path, index, quota))
            indexes.extend(source_indexes)
        else:
            sequences = [
                sequence_groups(config, source, path) for path in paths
            ]
            quotas = allocate(
                [len(groups) for groups, _ in sequences],
                sum(len(groups) for groups, _ in sequences)
                if config.mode == "all"
                else source.pilot_count,
            )
            with ThreadPoolExecutor(max_workers=config.workers) as pool:
                futures = [
                    pool.submit(
                        export_sequence, config, source, path, groups, quota
                    )
                    for path, (groups, _), quota in zip(
                        paths, sequences, quotas, strict=True
                    )
                ]
                for future in futures:
                    rows.extend(future.result())
            inventories.extend(
                {
                    "source": source.name,
                    "recording": path.name,
                    "image_count": count,
                    "candidate_count": len(groups),
                    "pilot_count": quota,
                }
                for path, (groups, count), quota in zip(
                    paths, sequences, quotas, strict=True
                )
            )
    rows.sort(
        key=lambda row: (
            row["source"],
            row["recording_id"],
            row["segment"],
            row["window"],
        )
    )
    if not rows or len({row["id"] for row in rows}) != len(rows):
        message = "Pilot must be nonempty and have unique image identities"
        raise ValueError(message)
    manifest = config.output / "manifest.jsonl"
    temporary = manifest.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    temporary.replace(manifest)
    summary = {
        "version": INDEX_VERSION,
        "sampling": "all temporal candidates"
        if config.mode == "all"
        else "deterministic spread bursts",
        "mode": config.mode,
        "fps": config.fps,
        "pilot_count": len(rows),
        "by_source": dict(Counter(row["source"] for row in rows)),
        "recording_count": len({row["recording_id"] for row in rows}),
        "elapsed_seconds": time.monotonic() - started,
        "image_sequences": inventories,
        "mcaps": [
            {key: value for key, value in index.items() if key != "candidates"}
            for index in indexes
        ],
        "config_sha256": hashlib.sha256(config.path.read_bytes()).hexdigest(),
        "final_labelling_images_selected": False,
        "color_conversion": (
            "NV12 limited-range BT.709; nearest chroma; RGB PNG"
        ),
    }
    write_json(config.output / "summary.json", summary)
    print(
        f"Pilot ready: {len(rows)} images from "
        f"{summary['recording_count']} recordings",
        flush=True,
    )

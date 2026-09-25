"""Frozen image embeddings, benchmark metadata, and pilot neighbor audits."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

from .common import Record, identity, write_json
from .config import Config
from .retrieval import exact_topk


class Images(Dataset):
    def __init__(self, paths: list[Path], size: int) -> None:
        self.paths, self.size = paths, size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as source:
            image = source.convert("RGB")
            ratio = self.size / max(image.size)
            shape = tuple(max(1, round(value * ratio)) for value in image.size)
            image = image.resize(shape, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (self.size, self.size), (124, 116, 104))
            canvas.paste(
                image,
                ((self.size - shape[0]) // 2, (self.size - shape[1]) // 2),
            )
            return torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1)


def reference_images(directory: Path | None) -> list[Record]:
    if directory is None:
        return []
    if not directory.is_dir():
        message = f"Reference directory does not exist: {directory}"
        raise ValueError(message)
    root = (
        directory / "images" if (directory / "images").is_dir() else directory
    )
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {
            ".png",
            ".jpg",
            ".jpeg",
        }:
            continue
        relative = str(path.relative_to(directory))
        group = path.relative_to(root).parts[0]
        rows.append(
            {
                "id": "existing__" + identity(relative),
                "source": "existing",
                "split": group
                if group in {"train", "val", "test"}
                else "unspecified",
                "path": relative,
                "filename": path.name,
            }
        )
    return rows


def embed(config: Config) -> None:
    settings = config.embedding
    device = _device(settings.device)
    dtype = torch.float16 if device == "cuda" else torch.float32
    manifest = config.output / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    old = reference_images(config.reference)
    paths = [config.root / row["path"] for row in rows]
    if config.reference is not None:
        paths.extend(config.reference / row["path"] for row in old)
    if not rows:
        message = "Cannot embed an empty pilot"
        raise ValueError(message)
    revision = (
        HfApi().model_info(settings.model, revision=settings.revision).sha
    )
    print(f"Loading {settings.model} revision {revision}", flush=True)
    model = (
        AutoModel.from_pretrained(
            settings.model,
            revision=revision,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        .eval()
        .to(device)
    )
    patch = model.config.patch_size
    if settings.size % patch:
        message = f"Input size must be divisible by model patch size {patch}"
        raise ValueError(message)
    loader = DataLoader(
        Images(paths, settings.size),
        batch_size=settings.batch_size,
        num_workers=settings.workers,
        pin_memory=device == "cuda",
        shuffle=False,
        persistent_workers=settings.workers > 0,
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[
        None, :, None, None
    ]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[
        None, :, None, None
    ]
    vectors = []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    inference_seconds = 0.0
    progress = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = (
                batch.to(device, non_blocking=True).float() / 255 - mean
            ) / std
            if device == "cuda":
                torch.cuda.synchronize()
            before = time.perf_counter()
            output = model(pixel_values=inputs.to(dtype))
            vector = torch.nn.functional.normalize(
                output.last_hidden_state[:, 0].float(), dim=1
            )
            if device == "cuda":
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - before
            vectors.append(vector.cpu().numpy())
            progress += len(batch)
            if progress % 512 < len(batch):
                print(f"Embedded {progress}/{len(paths)}", flush=True)
    elapsed = time.perf_counter() - started
    matrix = np.concatenate(vectors)
    if not np.isfinite(matrix).all() or not np.allclose(
        np.linalg.norm(matrix, axis=1), 1, atol=1e-5
    ):
        message = "Non-finite or non-normalized embeddings"
        raise ValueError(message)
    folder = config.output / "embeddings"
    folder.mkdir(exist_ok=True)
    np.save(folder / "vectors.npy", matrix)
    write_json(folder / "rows.json", rows + old)
    metadata = {
        "model": settings.model,
        "revision": revision,
        "feature": "normalized last_hidden_state CLS token",
        "image_size": settings.size,
        "preprocessing": (
            "PIL Lanczos aspect-preserving resize, centered "
            "RGB(124,116,104) padding, ImageNet mean/std; no crop"
        ),
        "dtype": str(dtype),
        "patch_size": patch,
        "batch_size": settings.batch_size,
        "pilot_count": len(rows),
        "reference_count": len(old),
        "dimensions": matrix.shape[1],
        "wall_seconds": elapsed,
        "inference_seconds": inference_seconds,
        "wall_images_per_second": len(paths) / elapsed,
        "inference_images_per_second": len(paths) / inference_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
        if device == "cuda"
        else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20
        if device == "cuda"
        else None,
        "device": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    write_json(folder / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    neighbors(config, rows, old, matrix)


def neighbors(
    config: Config,
    rows: list[Record],
    old: list[Record],
    vectors: NDArray[np.float32],
) -> None:
    count = len(rows)
    device = _device(config.embedding.device)
    pilot_indices, pilot_scores = exact_topk(
        vectors[:count], vectors[:count], 8, same_rows=True, device=device
    )
    old_indices, old_scores = exact_topk(
        vectors[:count], vectors[count:], 5, device=device
    )
    neighbor_rows = []
    for index, row in enumerate(rows):
        neighbor_rows.append(
            {
                "id": row["id"],
                "pilot_neighbors": [
                    {"id": rows[j]["id"], "cosine": float(score)}
                    for j, score in zip(
                        pilot_indices[index], pilot_scores[index], strict=True
                    )
                ],
                "existing_neighbors": [
                    {
                        "id": old[j]["id"],
                        "cosine": float(score),
                        "split": old[j]["split"],
                    }
                    for j, score in zip(
                        old_indices[index], old_scores[index], strict=True
                    )
                ],
            }
        )
    write_json(config.output / "neighbors.json", neighbor_rows)
    categories: dict[str, list[Record]] = defaultdict(list)
    seen = set()
    for index in range(count):
        for other, score in zip(
            pilot_indices[index, :5], pilot_scores[index, :5], strict=True
        ):
            key = tuple(sorted((index, int(other))))
            if key in seen:
                continue
            seen.add(key)
            left, right = rows[index], rows[other]
            category = (
                left["source"]
                if left["source"] == right["source"]
                else "cross-source"
            )
            categories[category].append(
                {
                    "a": left["id"],
                    "b": right["id"],
                    "cosine": float(score),
                    "kind": category,
                }
            )
    pairs = []
    for _, items in sorted(categories.items()):
        items.sort(key=lambda pair: pair["cosine"])
        pairs.extend(
            items[i]
            for i in np.linspace(
                0,
                len(items) - 1,
                min(config.pair_samples_per_category, len(items)),
            ).astype(int)
        )
    old_pairs = []
    for entry in neighbor_rows:
        if entry["existing_neighbors"]:
            match = entry["existing_neighbors"][0]
            old_pairs.append(
                {
                    "a": entry["id"],
                    "b": match["id"],
                    "cosine": match["cosine"],
                    "kind": "existing-" + match["split"],
                }
            )
    old_pairs.sort(key=lambda pair: pair["cosine"])
    pairs.extend(
        old_pairs[i]
        for i in np.linspace(
            0,
            len(old_pairs) - 1,
            min(config.pair_samples_per_category, len(old_pairs)),
        ).astype(int)
    )
    for index, pair in enumerate(pairs):
        pair["id"] = f"pair-{index + 1:03}"
        pair["calibration_partition"] = (
            "check" if index % 5 == 0 else "calibrate"
        )
    write_json(config.output / "pairs.json", pairs)
    print(f"Prepared {len(pairs)} calibration pairs", flush=True)


def _device(requested: str) -> str:
    device = requested
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        message = "CUDA was requested but is unavailable"
        raise RuntimeError(message)
    return device

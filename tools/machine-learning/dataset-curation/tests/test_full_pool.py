import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from dataset_curation.eligibility import reasons
from dataset_curation.extraction import extract
from dataset_curation.retrieval import exact_topk
from dataset_curation.timeline import sample_timeline


def test_blocked_search_matches_dense_across_block_boundaries() -> None:
    rng = np.random.default_rng(19)
    vectors = rng.normal(size=(13, 7)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    indices, scores = exact_topk(
        vectors, vectors, 4, same_rows=True, block_size=3
    )
    dense = vectors @ vectors.T
    np.fill_diagonal(dense, -np.inf)
    expected = np.argsort(dense, axis=1)[:, -4:][:, ::-1]
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_allclose(
        scores, np.take_along_axis(dense, expected, axis=1), atol=1e-6
    )


def test_interval_matches_recording_and_inclusive_frame_numbers() -> None:
    policy = {
        "exclude_intervals": [
            {"source_path": "a.mcap", "first": 5, "last": 9, "reason": "seated"}
        ]
    }
    review = {"images": {"sample": {"quality": "usable"}}}
    row = {"id": "sample", "source_path": "a.mcap", "image_ordinal": 5}
    assert reasons(row, policy, review) == ["interval: seated"]
    assert reasons(dict(row, image_ordinal=9), policy, review)
    assert not reasons(dict(row, image_ordinal=10), policy, review)
    assert not reasons(dict(row, source_path="b.mcap"), policy, review)


def test_timeline_resets_at_clock_changes_and_keeps_last_frame() -> None:
    rows = [
        {"image_ordinal": 1, "segment": 0, "timestamp_ns": 100_000_000_000},
        {"image_ordinal": 2, "segment": 0, "timestamp_ns": 101_000_000_000},
        {"image_ordinal": 3, "segment": 1, "timestamp_ns": 5_000_000_000},
        {"image_ordinal": 4, "segment": 1, "timestamp_ns": 6_000_000_000},
    ]
    assert [r["image_ordinal"] for r in sample_timeline(rows, 10)] == [1, 3, 4]


def test_all_mode_uses_every_temporal_group(tmp_path: Path) -> None:
    from dataset_curation.config import Config, Embedding, Source
    from PIL import Image

    directory = tmp_path / "raw/session"
    directory.mkdir(parents=True)
    config_path = tmp_path / "job.toml"
    config_path.write_text("test")
    for frame in range(10):
        Image.new("RGB", (8, 8), (frame, 0, 0)).save(
            directory / f"frame_{frame}.png"
        )
    config = Config(
        path=config_path,
        title="Example",
        root=tmp_path,
        raw=tmp_path / "raw",
        output=tmp_path / "output",
        sources=(Source("camera", "image_sequence", "*", 2, 6),),
        embedding=Embedding(model="unused"),
    )
    extract(replace(config, mode="all"))
    summary = json.loads((config.output / "summary.json").read_text())
    assert summary["pilot_count"] == 5
    assert summary["mode"] == "all"

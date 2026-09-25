from pathlib import Path

import numpy as np
import pytest
from dataset_curation.config import Config, Embedding, Source
from dataset_curation.extraction import sequence_groups
from dataset_curation.sampling import TimeWindows, allocate, spread_bursts
from PIL import Image


def test_time_windows_boundaries_duplicates_and_slow_sources() -> None:
    windows = TimeWindows(3, 10)
    assert windows.assign(0, "capture") == (0, 0)
    assert windows.assign(0, "capture") is None
    assert windows.assign(333_333_333, "capture") == (0, 0)
    assert windows.assign(333_333_334, "capture") == (0, 1)
    assert windows.assign(1_000_000_000, "capture") == (0, 3)


def test_clock_resets_and_large_forward_jumps_start_new_segments() -> None:
    windows = TimeWindows(3, 10)
    assert windows.assign(2_000_000_000, "capture") == (0, 0)
    assert windows.assign(1_000_000_000, "capture") == (1, 0)
    assert windows.assign(1_000_000_001, "log") == (2, 0)
    assert windows.assign(30_000_000_000, "log") == (3, 0)
    assert len(windows.discontinuities) == 3
    assert windows.intervals == []


@pytest.mark.parametrize(
    ("count", "target"), [(0, 10), (1, 1), (12, 8), (200, 36), (4, 20)]
)
def test_spread_bursts_respect_capacity_and_cover_the_sequence(
    count: int, target: int
) -> None:
    selected = spread_bursts(count, target)
    assert len(selected) == min(count, target)
    assert selected == sorted(set(selected))
    assert all(0 <= index < count for index in selected)
    if count >= 12:
        assert selected[0] == 0
        assert selected[-1] >= count - 3


def test_small_recording_quota_is_redistributed() -> None:
    assert allocate([2, 20, 20], 15) == [2, 7, 6]
    assert allocate([2, 3], 20) == [2, 3]
    assert allocate([], 20) == []


def test_filename_gap_does_not_pair_distant_frames(tmp_path: Path) -> None:
    for tick in [1000, 1167, 1334, 3000, 3167]:
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
            tmp_path / f"color_{tick}.png"
        )
    source = Source(
        "camera",
        "image_sequence",
        "*",
        10,
        frame_rate=6,
        filename_pattern=r"color_(\d+)",
    )
    config = Config(
        tmp_path / "config.toml",
        "Test",
        tmp_path,
        tmp_path,
        tmp_path / "out",
        (source,),
        Embedding("unused"),
    )
    groups, count = sequence_groups(config, source, tmp_path)
    assert count == 5
    assert [[frame[1] for frame in group["frames"]] for group in groups] == [
        [1000, 1167],
        [1334],
        [3000, 3167],
    ]
    assert [(group["segment"], group["window"]) for group in groups] == [
        (0, 0),
        (0, 1),
        (1, 0),
    ]

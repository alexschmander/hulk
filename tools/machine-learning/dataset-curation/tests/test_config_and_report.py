import json
from pathlib import Path

import numpy as np
import pytest
from dataset_curation.config import load_config
from dataset_curation.extraction import extract
from dataset_curation.report import build
from PIL import Image

JOB = r"""schema_version = 1
[dataset]
root = "."
title = "Independent camera dataset"
raw = "raw"
output = "out"
[sampling]
fps = 3
[[sources]]
name = "different-camera"
kind = "image_sequence"
pattern = "session-*"
filename_pattern = 'frame_(\d+)'
frame_rate = 6
pilot_count = 4
eligibility_question = "Is the target visible?"
[embedding]
model = "unused"
"""


def test_independent_dataset_extracts_and_reports_without_gpu(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(JOB)
    source = tmp_path / "raw/session-a"
    source.mkdir(parents=True)
    for index in range(8):
        Image.fromarray(np.full((16, 16, 3), index * 20, dtype=np.uint8)).save(
            source / f"frame_{index}.png"
        )
    config = load_config(config_path)
    extract(config)
    build(config)
    rows = [
        json.loads(line)
        for line in (config.output / "manifest.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4
    assert len({row["id"] for row in rows}) == 4
    assert len(list(source.glob("*.png"))) == 8
    report = (config.output / "review.html").read_text()
    assert "Independent camera dataset" in report
    assert "Is the target visible?" in report
    assert "__DATA__" not in report
    assert "hslvision" not in report.lower()


def test_output_cannot_overwrite_raw_inputs(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        JOB.replace('output = "out"', 'output = "raw/generated"')
    )
    with pytest.raises(ValueError, match="separate"):
        load_config(config_path)

import json
from pathlib import Path

import numpy as np
from dataset_curation.config import Config, Embedding
from dataset_curation.embeddings import neighbors


def test_neighbor_audit_keeps_identity_and_held_out_split(
    tmp_path: Path,
) -> None:
    config = Config(
        path=tmp_path / "config.toml",
        title="Example",
        root=tmp_path,
        raw=tmp_path / "raw",
        output=tmp_path,
        sources=(),
        embedding=Embedding(model="unused"),
    )
    rows = [
        {"id": "first", "source": "camera"},
        {"id": "second", "source": "camera"},
        {"id": "third", "source": "other-camera"},
    ]
    references = [{"id": "existing__held-out", "split": "test"}]
    vectors = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    neighbors(config, rows, references, vectors)
    audit = json.loads((tmp_path / "neighbors.json").read_text())
    assert [row["id"] for row in audit] == ["first", "second", "third"]
    assert audit[0]["pilot_neighbors"][0] == {"id": "second", "cosine": 1.0}
    assert audit[2]["existing_neighbors"] == [
        {"id": "existing__held-out", "cosine": 1.0, "split": "test"}
    ]
    for row in audit:
        assert row["id"] not in {
            match["id"] for match in row["pilot_neighbors"]
        }
    pairs = json.loads((tmp_path / "pairs.json").read_text())
    assert len({tuple(sorted((p["a"], p["b"]))) for p in pairs}) == len(pairs)


def test_single_image_without_reference_has_no_neighbors(
    tmp_path: Path,
) -> None:
    config = Config(
        path=tmp_path / "config.toml",
        title="Example",
        root=tmp_path,
        raw=tmp_path / "raw",
        output=tmp_path,
        sources=(),
        embedding=Embedding(model="unused"),
    )
    neighbors(
        config,
        [{"id": "only", "source": "camera"}],
        [],
        np.array([[1, 0]], dtype=np.float32),
    )
    audit = json.loads((tmp_path / "neighbors.json").read_text())
    assert audit == [
        {"id": "only", "pilot_neighbors": [], "existing_neighbors": []}
    ]
    assert json.loads((tmp_path / "pairs.json").read_text()) == []

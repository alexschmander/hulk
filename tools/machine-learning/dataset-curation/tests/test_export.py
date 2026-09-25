import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from dataset_curation.batches import assign_batches
from dataset_curation.config import Config, Embedding
from dataset_curation.export import export
from PIL import Image


def test_nearby_views_are_split_and_sources_and_recordings_are_balanced() -> (
    None
):
    rows = []
    for source, recording, count in [
        ("a", "one", 5),
        ("a", "two", 9),
        ("b", "three", 7),
        ("b", "four", 3),
    ]:
        rows.extend(
            {
                "id": f"{recording}-{i}",
                "source": source,
                "recording_id": recording,
            }
            for i in range(count)
        )
    rng = np.random.default_rng(5)
    vectors = rng.normal(size=(len(rows), 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    names = ["first", "second"]
    assignment = assign_batches(rows, vectors, names, 17)
    assert assignment == assign_batches(rows, vectors, names, 17)
    assert Counter(assignment.values()) == {"first": 12, "second": 12}
    for field in ["source", "recording_id"]:
        for value in {row[field] for row in rows}:
            counts = Counter(
                assignment[r["id"]] for r in rows if r[field] == value
            )
            assert abs(counts["first"] - counts["second"]) <= 1
    pairs = [
        {"id": str(i), "source": "a", "recording_id": "one"} for i in range(4)
    ]
    matrix = np.array(
        [[1, 0], [0.99, 0.01], [0, 1], [0.01, 0.99]], dtype=np.float32
    )
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    assignment = assign_batches(pairs, matrix, names, 3)
    assert assignment["0"] != assignment["1"]
    assert assignment["2"] != assignment["3"]


def _job(root: Path) -> tuple[Config, Path, Path]:
    candidates = root / "candidates"
    (candidates / "images").mkdir(parents=True)
    (candidates / "embeddings").mkdir()
    (root / "raw").mkdir()
    (root / "source").mkdir()
    rows = []
    for i in range(4):
        path = candidates / "images" / f"frame-{i}.png"
        image = Image.new("RGB", (12, 12), (i * 30, 5, 10))
        image.save(path)
        rows.append(
            {
                "id": str(i),
                "source": "camera",
                "recording_id": "one",
                "source_path": f"frame-{i}",
                "path": str(path.relative_to(root)),
                "width": 12,
                "height": 12,
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            }
        )
    manifest = candidates / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    fingerprint = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (candidates / "embeddings/metadata.json").write_text(
        json.dumps({"manifest_sha256": fingerprint})
    )
    (candidates / "embeddings/rows.json").write_text(json.dumps(rows))
    np.save(candidates / "embeddings/vectors.npy", np.eye(4, dtype=np.float32))
    review_path = root / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "manifest_sha256": fingerprint,
                "images": {
                    "3": {"quality": "usable", "eligibility": "visible"}
                },
                "pairs": {},
            }
        )
    )
    (root / "pairs.json").write_text("[]")
    policy_path = root / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review": "review.json",
                "review_manifest": "candidates/manifest.jsonl",
                "review_pairs": "pairs.json",
            }
        )
    )
    plan_path = root / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_sha256": fingerprint,
                "policy_sha256": hashlib.sha256(
                    policy_path.read_bytes()
                ).hexdigest(),
                "review_sha256": hashlib.sha256(
                    review_path.read_bytes()
                ).hexdigest(),
                "pairs_sha256": hashlib.sha256(
                    (root / "pairs.json").read_bytes()
                ).hexdigest(),
                "images": "source",
                "output": "final",
                "seed": 11,
                "groups": [
                    {
                        "name": "camera",
                        "ids": ["0", "1", "2"],
                        "batches": ["one", "two"],
                        "max_cosine": 0.9,
                    },
                    {
                        "name": "target",
                        "ids": ["3"],
                        "batches": ["target"],
                        "max_cosine": 0.9,
                        "require_target": True,
                    },
                ],
            }
        )
    )
    config = Config(
        path=root / "config.toml",
        title="Test",
        root=root,
        raw=root / "raw",
        output=candidates,
        sources=(),
        embedding=Embedding(model="unused"),
    )
    return config, policy_path, plan_path


def test_export_preserves_bytes_and_refuses_to_overwrite_previous_output(
    tmp_path: Path,
) -> None:
    config, policy, plan = _job(tmp_path)
    export(config, policy, plan)
    manifest = tmp_path / "final/manifest.jsonl"
    before = manifest.read_bytes()
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert Counter(r["batch"] for r in rows) == {
        "one": 2,
        "two": 1,
        "target": 1,
    }
    for row in rows:
        assert (tmp_path / row["path"]).read_bytes() == (
            tmp_path / row["candidate_path"]
        ).read_bytes()
    assert len(list((tmp_path / "source").rglob("*.png"))) == 4
    assert not list((tmp_path / "source").rglob("*.txt"))
    with pytest.raises(ValueError, match="empty destination"):
        export(config, policy, plan)
    assert manifest.read_bytes() == before


def test_changed_image_aborts_without_partial_export(tmp_path: Path) -> None:
    config, policy, plan = _job(tmp_path)
    (config.output / "images/frame-2.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="content changed"):
        export(config, policy, plan)
    assert not list((tmp_path / "source").iterdir())
    assert not (tmp_path / "final").exists()
    assert not list(tmp_path.glob(".curation-export-*"))

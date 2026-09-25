import json
from pathlib import Path

from dataset_curation.audit import content_hash, focused_pairs
from dataset_curation.report import read_pairs
from PIL import Image


def test_pixel_identity_ignores_lossless_encoding(tmp_path: Path) -> None:
    image = Image.new("RGB", (20, 30), "red")
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    image.save(first, compress_level=0)
    image.save(second, compress_level=9)
    assert first.read_bytes() != second.read_bytes()
    assert content_hash(first) == content_hash(second)


def test_focused_pairs_prioritize_similarity_and_have_stable_ids() -> None:
    rows = [{"id": name, "source": "camera"} for name in "abc"]
    neighbors = [
        {
            "id": "a",
            "pilot_neighbors": [
                {"id": "b", "cosine": 0.8},
                {"id": "c", "cosine": 0.95},
            ],
            "existing_neighbors": [],
        },
        {
            "id": "c",
            "pilot_neighbors": [{"id": "a", "cosine": 0.95}],
            "existing_neighbors": [],
        },
    ]
    pairs = focused_pairs(rows, neighbors, 1)
    assert [(p["a"], p["b"]) for p in pairs] == [("a", "c")]
    assert focused_pairs(rows, list(reversed(neighbors)), 1) == pairs


def test_stale_audit_pairs_are_not_shown_for_new_manifest(
    tmp_path: Path,
) -> None:
    (tmp_path / "audit-pairs.json").write_text(
        json.dumps({"manifest_sha256": "previous", "pairs": [{"id": "old"}]})
    )
    assert read_pairs(tmp_path, "current") == []
    assert read_pairs(tmp_path, "previous") == [{"id": "old"}]

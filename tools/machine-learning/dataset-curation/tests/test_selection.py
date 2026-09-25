import numpy as np
from dataset_curation.selection import diverse_indices, quality_candidates


def test_selection_uses_direct_representative_similarity() -> None:
    angles = np.deg2rad([0, 15, 30])
    vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(
        np.float32
    )
    rows = [
        {
            "id": str(i),
            "recording_id": str(i),
            "pixel_sha256": str(i),
            "quality": {"laplacian": 10},
        }
        for i in range(3)
    ]
    # Seed is the middle view; both neighbors must stay out at this threshold.
    selected = diverse_indices(
        vectors,
        rows,
        3,
        max_cosine=0.95,
        device="cpu",
        conflicts={},
        recording_cap=3,
    )
    assert selected == [1]
    # With only the endpoint views, both are distinct enough to survive.
    selected = diverse_indices(
        vectors[[0, 2]],
        [rows[0], rows[2]],
        3,
        max_cosine=0.95,
        device="cpu",
        conflicts={},
        recording_cap=3,
    )
    assert len(selected) == 2


def test_explicit_redundancy_and_pixel_duplicates_override_cosine() -> None:
    vectors = np.eye(4, dtype=np.float32)
    rows = [
        {
            "id": str(i),
            "recording_id": str(i),
            "pixel_sha256": "same" if i < 2 else str(i),
            "quality": {"laplacian": 10},
        }
        for i in range(4)
    ]
    selected = diverse_indices(
        vectors,
        rows,
        4,
        max_cosine=0.95,
        device="cpu",
        conflicts={"2": {"3"}, "3": {"2"}},
        recording_cap=4,
    )
    assert len(selected) == 2
    assert len(set(selected) & {0, 1}) == 1
    assert len(set(selected) & {2, 3}) == 1


def test_quality_uses_eligible_alternatives_and_preserves_user_decisions() -> (
    None
):
    rows = [
        {"id": "reviewed", "quality": {"laplacian": 5}},
        {"id": "candidate", "quality": {"laplacian": 50}},
        {"id": "rejected", "quality": {"laplacian": 5000}},
        {"id": "peer", "quality": {"laplacian": 60}},
    ]
    neighbors = [
        {
            "id": row["id"],
            "pilot_neighbors": [
                {"id": "rejected", "cosine": 0.99},
                {"id": "peer", "cosine": 0.95},
            ],
        }
        for row in rows
    ]
    selected = quality_candidates(
        [0, 1, 3],
        rows,
        neighbors,
        {"images": {"reviewed": {"quality": "usable"}}},
        {
            "neighbor_cosine": 0.9,
            "relative_sharpness": 0.6,
            "minimum_sharpness": 1,
        },
    )
    assert selected == [0, 1, 3]

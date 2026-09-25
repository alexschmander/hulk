"""Balance labelling batches across sources, recordings, and nearby views."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from numpy.typing import NDArray

from .common import Record


def assign_batches(
    rows: list[Record],
    vectors: NDArray[np.float32],
    names: list[str],
    seed: int,
) -> dict[str, str]:
    """Put nearby views from each recording into opposite batches."""
    if len(names) not in {1, 2} or len(set(names)) != len(names):
        message = "A selection group requires one or two distinct batch names"
        raise ValueError(message)
    rng = np.random.default_rng(seed)
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        strata[(row["source"], row["recording_id"])].append(index)
    totals: Counter[str] = Counter()
    source_counts: Counter[tuple[str, str]] = Counter()
    result = {}
    for (source, _), indices in sorted(strata.items()):
        remaining = sorted(indices, key=lambda i: rows[i]["id"])
        while remaining:
            anchor = remaining.pop(int(rng.integers(len(remaining))))
            chunk = [anchor]
            if remaining and len(names) == 2:
                scores = vectors[remaining] @ vectors[anchor]
                chunk.append(remaining.pop(int(scores.argmax())))
            available = list(rng.permutation(names))
            for index in chunk:
                batch = min(
                    available,
                    key=lambda b: (source_counts[(source, b)], totals[b]),
                )
                available.remove(batch)
                result[rows[index]["id"]] = str(batch)
                source_counts[(source, batch)] += 1
                totals[batch] += 1
    if (
        max((totals[n] for n in names), default=0)
        - min((totals[n] for n in names), default=0)
        > 1
    ):
        message = "Could not balance the requested batches"
        raise ValueError(message)
    return result

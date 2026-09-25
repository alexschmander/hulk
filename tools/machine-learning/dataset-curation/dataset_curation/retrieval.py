"""Exact cosine neighbors with memory bounded by the query block size."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray


def exact_topk(
    queries: NDArray[np.float32],
    reference: NDArray[np.float32],
    count: int,
    *,
    same_rows: bool = False,
    device: str = "cpu",
    block_size: int = 256,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Inputs must be normalized; self exclusion uses matching row positions."""
    if count < 0 or block_size < 1:
        message = "Neighbor count must be nonnegative and block size positive"
        raise ValueError(message)
    if queries.ndim != 2 or reference.ndim != 2:
        message = "Embedding matrices must have two dimensions"
        raise ValueError(message)
    if queries.shape[1] != reference.shape[1] or (
        same_rows and len(queries) != len(reference)
    ):
        message = "Incompatible embedding shapes"
        raise ValueError(message)
    count = min(count, max(0, len(reference) - int(same_rows)))
    indices = np.empty((len(queries), count), dtype=np.int64)
    values = np.empty((len(queries), count), dtype=np.float32)
    if not count:
        return indices, values
    corpus = torch.as_tensor(reference, device=device, dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, len(queries), block_size):
            stop = min(start + block_size, len(queries))
            batch = torch.as_tensor(
                queries[start:stop], device=device, dtype=torch.float32
            )
            scores = batch @ corpus.T
            if same_rows:
                local = torch.arange(stop - start, device=device)
                scores[local, local + start] = -torch.inf
            best = torch.topk(scores, count, dim=1, sorted=True)
            indices[start:stop] = best.indices.cpu().numpy()
            values[start:stop] = best.values.cpu().numpy()
    return indices, values

"""Matrix characterization: structure, load balance, and x-vector reuse."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def characterize(A: sp.csr_matrix) -> dict:
    nnz = np.diff(A.indptr)
    rows = np.repeat(np.arange(A.shape[0]), nnz)
    dist = np.abs(A.indices.astype(np.int64) - rows) if A.nnz else np.array([0])
    mean = float(nnz.mean()) if nnz.size else 0.0
    return {
        "rows": int(A.shape[0]),
        "cols": int(A.shape[1]),
        "nnz": int(A.nnz),
        "density": A.nnz / (A.shape[0] * A.shape[1]),
        "nnz_per_row_mean": mean,
        "nnz_per_row_max": int(nnz.max()) if nnz.size else 0,
        "nnz_per_row_cv": float(nnz.std() / mean) if mean else 0.0,  # load imbalance
        "empty_rows": int((nnz == 0).sum()),
        "mean_abs_col_minus_row": float(dist.mean()),
        "x_footprint_bytes": int(np.unique(A.indices).size * 4),
    }


def window_reuse(A: sp.csr_matrix, order: np.ndarray, window: int = 32,
                 block_cols: int = 32, sample: int = 4000, seed: int = 0) -> float:
    """Fraction of x blocks a row needs that the previous `window` rows already touched.

    A schedule-only locality metric (no cache model): high values mean the
    schedule places rows with overlapping column footprints close together.
    Computed on a random sample of positions to stay fast on large matrices.
    """
    n = order.size
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    positions = rng.choice(np.arange(1, n), size=min(sample, n - 1), replace=False)
    ip, ix = A.indptr, A.indices

    def blocks(r: int) -> set:
        return set((ix[ip[r]:ip[r + 1]] // block_cols).tolist())

    hit = total = 0
    for p in positions:
        cur = blocks(order[p])
        if not cur:
            continue
        prev: set = set()
        for q in range(max(0, p - window), p):
            prev |= blocks(order[q])
        hit += len(cur & prev)
        total += len(cur)
    return hit / total if total else 0.0

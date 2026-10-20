"""Matrix loading, synthetic generators, and binary export for the CUDA harness.

Binary CSR format (little-endian), read by cuda/csr_io.h:
    int64 rows, int64 cols, int64 nnz
    int32 rowptr[rows + 1]
    int32 colidx[nnz]
    float32 values[nnz]

Binary schedule format:
    int64 n
    int32 order[n]      # order[i] = row processed by work unit i
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp


def _canonical(A) -> sp.csr_matrix:
    A = sp.csr_matrix(A, dtype=np.float32)
    A.sum_duplicates()
    A.sort_indices()
    return A


def load_matrix(path: str | Path) -> sp.csr_matrix:
    """Load a Matrix Market (.mtx) or SciPy (.npz) sparse matrix as canonical CSR."""
    path = Path(path)
    if path.suffix == ".npz":
        return _canonical(sp.load_npz(path))
    return _canonical(scipy.io.mmread(str(path)))


def row_slice(A: sp.csr_matrix, max_rows: int | None) -> sp.csr_matrix:
    """Take a contiguous leading row slice (used to keep the cache simulator tractable)."""
    if max_rows is None or max_rows >= A.shape[0]:
        return A
    return _canonical(A[:max_rows])


# --------------------------------------------------------------------------- #
# Synthetic matrices: each one stresses a different locality regime.
# --------------------------------------------------------------------------- #
def synthetic(kind: str, n: int = 20000, nnz_per_row: int = 16, seed: int = 0) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    if kind == "banded":
        # Strong natural locality: row i touches columns near i.
        half = nnz_per_row // 2
        offsets = np.arange(-half, half)
        rows = np.repeat(np.arange(n), len(offsets))
        cols = np.clip(rows + np.tile(offsets, n), 0, n - 1)
    elif kind == "uniform":
        # No exploitable locality: columns are uniformly random.
        rows = np.repeat(np.arange(n), nnz_per_row)
        cols = rng.integers(0, n, size=rows.size)
    elif kind == "block_shuffled":
        # Hidden locality: block-diagonal structure whose rows were randomly permuted.
        # Row order destroys reuse; a locality-aware schedule can recover it.
        block = 256
        rows = np.repeat(np.arange(n), nnz_per_row)
        base = (rows // block) * block
        cols = np.minimum(base + rng.integers(0, block, size=rows.size), n - 1)
        perm = rng.permutation(n)
        rows = perm[rows]
    elif kind == "powerlaw":
        # Skewed row lengths (load imbalance) with mild locality.
        lengths = np.clip(rng.zipf(1.8, size=n), 1, 2000)
        rows = np.repeat(np.arange(n), lengths)
        centers = rng.integers(0, n, size=n)
        spread = rng.normal(0, 2000, size=rows.size).astype(np.int64)
        cols = np.clip(centers[rows] + spread, 0, n - 1)
    else:
        raise ValueError(f"unknown synthetic kind: {kind}")
    vals = rng.uniform(-1.0, 1.0, size=rows.size).astype(np.float32)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n))
    return _canonical(A)


SYNTHETIC_KINDS = ("banded", "uniform", "block_shuffled", "powerlaw")


# --------------------------------------------------------------------------- #
# Export for the CUDA benchmark
# --------------------------------------------------------------------------- #
def export_csr_bin(A: sp.csr_matrix, path: str | Path) -> None:
    A = _canonical(A)
    with open(path, "wb") as f:
        np.array([A.shape[0], A.shape[1], A.nnz], dtype=np.int64).tofile(f)
        A.indptr.astype(np.int32).tofile(f)
        A.indices.astype(np.int32).tofile(f)
        A.data.astype(np.float32).tofile(f)


def export_order_bin(order: np.ndarray, path: str | Path) -> None:
    with open(path, "wb") as f:
        np.array([order.size], dtype=np.int64).tofile(f)
        order.astype(np.int32).tofile(f)

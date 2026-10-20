"""Row-scheduling policies for CSR SpMV.

A schedule is a permutation `order` of the row indices: work unit i (a thread in
the scalar kernel, a warp in the vector kernel) processes row `order[i]`. The
matrix itself is never reordered; y is written back at the original row index,
so the result is identical to A @ x for every schedule.

Policies
--------
row_static         identity order (conventional row-based processing)
adaptive_grouping  rows binned by log2(nnz) so warps get similar-length rows
                   (targets SIMD divergence / load imbalance, not locality)
locality_aware     rows ordered by the x-vector cache blocks they touch, so
                   consecutive work units reuse the same x cache lines
locality_balanced  locality_aware, then rows sorted by length inside small
                   windows to recover some load balance without losing reuse
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp


@dataclass
class Schedule:
    name: str
    order: np.ndarray        # int32 permutation of range(n_rows)
    preprocess_s: float      # wall-clock cost of building the schedule


def _row_ids(A: sp.csr_matrix) -> np.ndarray:
    return np.repeat(np.arange(A.shape[0], dtype=np.int64), np.diff(A.indptr))


def row_static(A: sp.csr_matrix) -> np.ndarray:
    return np.arange(A.shape[0], dtype=np.int32)


def adaptive_grouping(A: sp.csr_matrix) -> np.ndarray:
    nnz = np.diff(A.indptr)
    bins = np.floor(np.log2(np.maximum(nnz, 1))).astype(np.int64)
    # Stable sort keeps original order inside a bin (preserves any natural locality).
    return np.argsort(bins, kind="stable").astype(np.int32)


def _block_features(A: sp.csr_matrix, block_cols: int):
    """Per-row min block and nnz-weighted centroid block of the x entries it reads."""
    n = A.shape[0]
    nnz = np.diff(A.indptr)
    blocks = (A.indices // block_cols).astype(np.int64)
    rows = _row_ids(A)
    centroid = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    min_blk = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    nonempty = nnz > 0
    if blocks.size:
        sums = np.bincount(rows, weights=blocks, minlength=n)
        centroid[nonempty] = np.rint(sums[nonempty] / nnz[nonempty]).astype(np.int64)
        # indices are sorted per row, so the first entry of each row is its min block
        min_blk[nonempty] = blocks[A.indptr[:-1][nonempty]]
    return centroid, min_blk, nnz


def locality_aware(A: sp.csr_matrix, block_cols: int = 32) -> np.ndarray:
    """Order rows so that rows reading the same x blocks run back-to-back.

    block_cols = 32 float32 entries = one 128-byte cache line of x.
    Primary key: centroid block (rows with similar column footprints cluster).
    Secondary: first block touched. Tertiary: original index (stability).
    Empty rows get the max key and go to the end.
    """
    centroid, min_blk, _ = _block_features(A, block_cols)
    order = np.lexsort((np.arange(A.shape[0]), min_blk, centroid))
    return order.astype(np.int32)


def locality_balanced(A: sp.csr_matrix, block_cols: int = 32, window: int = 256) -> np.ndarray:
    """Locality-aware order, then sort by row length within windows of `window` rows.

    Rows inside one window already share x blocks, so reordering inside it keeps
    most of the reuse while grouping similar-length rows into the same warp.
    """
    order = locality_aware(A, block_cols).astype(np.int64)
    nnz = np.diff(A.indptr)[order]
    win = np.arange(order.size) // window
    inner = np.lexsort((nnz, win))
    return order[inner].astype(np.int32)


POLICIES: dict[str, Callable[[sp.csr_matrix], np.ndarray]] = {
    "row_static": row_static,
    "adaptive_grouping": adaptive_grouping,
    "locality_aware": locality_aware,
    "locality_balanced": locality_balanced,
}


def build(A: sp.csr_matrix, name: str) -> Schedule:
    fn = POLICIES[name]
    t0 = time.perf_counter()
    order = fn(A)
    dt = time.perf_counter() - t0
    assert is_permutation(order, A.shape[0]), f"{name} did not return a permutation"
    return Schedule(name=name, order=order, preprocess_s=dt)


def is_permutation(order: np.ndarray, n: int) -> bool:
    if order.size != n:
        return False
    seen = np.zeros(n, dtype=bool)
    seen[order] = True
    return bool(seen.all())


def spmv_scheduled(A: sp.csr_matrix, order: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Reference SpMV that processes rows in `order` (functional correctness check)."""
    y = np.zeros(A.shape[0], dtype=np.float64)
    # Permute rows, multiply, scatter results back to their original row index.
    P = A[order].astype(np.float64)
    y[order] = P @ x.astype(np.float64)
    return y

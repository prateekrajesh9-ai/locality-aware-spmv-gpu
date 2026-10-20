import numpy as np
import scipy.sparse as sp

from spmvsched import cachesim, io, schedule

TINY = cachesim.GPUConfig(name="tiny", n_sm=4, warps_per_block=4,
                          l1_bytes=4 * 1024, l2_bytes=32 * 1024)


def test_coalescing_one_x_sector_for_contiguous_warp_access():
    # One row with 8 contiguous float32 columns = 32 bytes = exactly one x sector.
    A = sp.csr_matrix(np.ones((1, 8), dtype=np.float32))
    tr = cachesim.build_trace(A, np.array([0]), "vector", TINY, uses_order=False)
    assert int(np.sum((tr.key >> 40) == cachesim.R_X)) == 1


def test_repeated_row_hits_in_l1():
    # Every row reads the same 8 columns: after the first access, x should hit in cache.
    A = sp.csr_matrix(np.ones((64, 8), dtype=np.float32))
    res = cachesim.run(A, schedule.row_static(A), "vector", TINY, uses_order=False)
    assert res["x_dram_sectors"] == 1
    assert res["x_l1_hit_rate"] > 0.9


def test_simd_efficiency_scalar_imbalance():
    lengths = np.array([32] + [1] * 31)
    rows = np.repeat(np.arange(32), lengths)
    cols = np.concatenate([np.arange(n) for n in lengths])
    A = sp.csr_matrix((np.ones(rows.size, np.float32), (rows, cols)), shape=(32, 32))
    tr = cachesim.build_trace(A, schedule.row_static(A), "scalar", TINY, uses_order=False)
    assert tr.n_instructions == 32
    assert abs(tr.simd_efficiency - lengths.sum() / (32 * 32)) < 1e-12


def test_locality_schedule_recovers_hidden_block_locality():
    # Model sanity check, not a performance claim: when block structure is hidden
    # by a row shuffle and x exceeds L2, locality-aware ordering must cut x DRAM traffic.
    A = io.synthetic("block_shuffled", n=20000, seed=0)
    base = cachesim.run(A, schedule.row_static(A), "vector", TINY, uses_order=False)
    loc = cachesim.run(A, schedule.locality_aware(A), "vector", TINY)
    assert loc["x_dram_sectors"] < 0.5 * base["x_dram_sectors"]


def test_static_schedule_skips_order_loads():
    A = io.synthetic("banded", n=1000)
    res = cachesim.run(A, schedule.row_static(A), "scalar", TINY, uses_order=False)
    assert res["order_requests"] == 0

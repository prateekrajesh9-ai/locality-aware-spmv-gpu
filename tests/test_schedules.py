import numpy as np
import pytest

from spmvsched import io, schedule


@pytest.mark.parametrize("kind", io.SYNTHETIC_KINDS)
@pytest.mark.parametrize("policy", list(schedule.POLICIES))
def test_schedule_is_permutation_and_spmv_is_exact(kind, policy):
    A = io.synthetic(kind, n=3000, seed=3)
    sch = schedule.build(A, policy)
    assert schedule.is_permutation(sch.order, A.shape[0])
    x = np.random.default_rng(0).uniform(-1, 1, A.shape[1]).astype(np.float32)
    y = schedule.spmv_scheduled(A, sch.order, x)
    np.testing.assert_allclose(y, A.astype(np.float64) @ x.astype(np.float64), rtol=1e-12, atol=1e-12)


def test_empty_rows_go_last_in_locality_order():
    A = io.synthetic("uniform", n=500, seed=1).tolil()
    A[10, :] = 0
    A[20, :] = 0
    A = io._canonical(A.tocsr())
    A.eliminate_zeros()
    order = schedule.locality_aware(A)
    assert set(order[-2:].tolist()) == {10, 20}


def test_adaptive_grouping_sorts_by_length_bin():
    A = io.synthetic("powerlaw", n=2000, seed=2)
    order = schedule.adaptive_grouping(A)
    bins = np.floor(np.log2(np.maximum(np.diff(A.indptr)[order], 1)))
    assert np.all(np.diff(bins) >= 0)


def test_export_roundtrip(tmp_path):
    A = io.synthetic("banded", n=100)
    io.export_csr_bin(A, tmp_path / "a.bin")
    raw = np.fromfile(tmp_path / "a.bin", dtype=np.uint8)
    rows, cols, nnz = np.frombuffer(raw[:24].tobytes(), dtype=np.int64)
    assert (rows, cols, nnz) == (100, 100, A.nnz)
    assert raw.size == 24 + 4 * (rows + 1) + 8 * nnz

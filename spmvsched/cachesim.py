"""Trace-driven model of a simplified GPU memory hierarchy for CSR SpMV.

What is modeled
---------------
* Two kernels, matching cuda/kernels.cuh:
    scalar : one thread per row, 32 rows per warp
    vector : one warp per row, lanes stride over the row's nonzeros
* Every global load the kernel issues: schedule array (order[]), rowptr,
  colidx, values, and the x vector. Each region gets its own address space.
* Coalescing: within one warp instruction, lanes touching the same 32-byte
  sector produce a single request.
* Per-SM L1 and a shared L2, both fully associative LRU at sector granularity.
* Thread blocks are assigned round-robin to SMs. Blocks in the same "wave"
  (one block per SM) advance in lockstep, one instruction step at a time.
* y stores bypass L1 (write-through, no-allocate) and are counted as sectors.

What is NOT modeled (deliberately; see docs/methodology.md)
----------------------------------------------------------
Set associativity and conflict misses, sector vs. line fill granularity,
MSHRs/latency/bandwidth limits, warp scheduler interleaving inside an SM,
DRAM row-buffer effects, and prefetching. The model is for comparing schedules
on the same matrix, not for predicting absolute runtime. Validate trends
against Nsight Compute counters on real hardware (scripts/profile_ncu.sh).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass

import numpy as np
import scipy.sparse as sp

REGIONS = ("order", "rowptr", "colidx", "values", "x")
R_ORDER, R_ROWPTR, R_COLIDX, R_VALS, R_X = range(5)
_SHIFT = 40  # region id lives above bit 40 of the cache key


@dataclass
class GPUConfig:
    name: str = "small"
    n_sm: int = 16
    warp_size: int = 32
    warps_per_block: int = 8
    sector_bytes: int = 32
    l1_bytes: int = 16 * 1024    # per SM, the part used as data cache
    l2_bytes: int = 256 * 1024   # shared

    @property
    def threads_per_block(self) -> int:
        return self.warp_size * self.warps_per_block


# "small" is scaled down so locality effects show up on matrices that fit a
# Python simulation. "a100_like" uses approximate A100-class sizes; check your
# GPU's documentation before quoting any of these numbers.
PRESETS = {
    "small": GPUConfig(),
    "a100_like": GPUConfig(name="a100_like", n_sm=108, l1_bytes=128 * 1024,
                           l2_bytes=40 * 1024 * 1024),
}


@dataclass
class Trace:
    sm: np.ndarray          # SM id of each request, in issue order
    key: np.ndarray         # (region << 40) | sector
    n_instructions: int     # warp-level memory-instruction steps over nonzeros
    simd_efficiency: float  # useful lanes / (warp_size * instructions)
    y_store_sectors: int


def _sector(index: np.ndarray, elem_bytes: int, sector_bytes: int) -> np.ndarray:
    return (index.astype(np.int64) * elem_bytes) // sector_bytes


def build_trace(A: sp.csr_matrix, order: np.ndarray, kernel: str, cfg: GPUConfig,
                uses_order: bool = True) -> Trace:
    """Generate the coalesced, issue-ordered request stream for one SpMV."""
    if kernel not in ("scalar", "vector"):
        raise ValueError("kernel must be 'scalar' or 'vector'")
    W, S = cfg.warp_size, cfg.sector_bytes
    n = A.shape[0]
    order = order.astype(np.int64)
    rowlen = np.diff(A.indptr).astype(np.int64)[order]
    orig_start = A.indptr[:-1].astype(np.int64)[order]

    # ---- one event per nonzero, in work-unit order -------------------------
    unit = np.repeat(np.arange(n, dtype=np.int64), rowlen)          # work unit i
    first = np.concatenate(([0], np.cumsum(rowlen)[:-1]))
    k = np.arange(unit.size, dtype=np.int64) - np.repeat(first, rowlen)
    orig_j = np.repeat(orig_start, rowlen) + k                         # index into colidx/values
    col = A.indices.astype(np.int64)[orig_j]

    if kernel == "vector":
        warp = unit
        step = k // W + 1
        n_instr = int(np.sum((rowlen + W - 1) // W))
    else:
        warp = unit // W
        step = k + 1
        n_warps = (n + W - 1) // W
        padded = np.zeros(n_warps * W, dtype=np.int64)
        padded[:n] = rowlen
        n_instr = int(padded.reshape(n_warps, W).max(axis=1).sum())
    simd_eff = float(unit.size) / (W * n_instr) if n_instr else 0.0

    parts = []  # (warp, step, region, sector)
    for region, sec in ((R_COLIDX, _sector(orig_j, 4, S)),
                        (R_VALS, _sector(orig_j, 4, S)),
                        (R_X, _sector(col, 4, S))):
        parts.append((warp, step, np.full(warp.size, region, np.int64), sec))

    # ---- per-work-unit header loads: order[i], rowptr[row], rowptr[row+1] --
    units = np.arange(n, dtype=np.int64)
    hwarp = units if kernel == "vector" else units // W
    zero = np.zeros(n, dtype=np.int64)
    if uses_order:
        parts.append((hwarp, zero, np.full(n, R_ORDER, np.int64), _sector(units, 4, S)))
    for off in (0, 1):
        parts.append((hwarp, zero, np.full(n, R_ROWPTR, np.int64), _sector(order + off, 4, S)))

    warp_a = np.concatenate([p[0] for p in parts])
    step_a = np.concatenate([p[1] for p in parts])
    reg_a = np.concatenate([p[2] for p in parts])
    sec_a = np.concatenate([p[3] for p in parts])

    block = warp_a // cfg.warps_per_block
    wave = block // cfg.n_sm
    idx = np.lexsort((sec_a, reg_a, warp_a, block, step_a, wave))
    warp_a, step_a, reg_a, sec_a, block = (a[idx] for a in (warp_a, step_a, reg_a, sec_a, block))

    # Coalesce: drop requests identical to the previous one within the same instruction.
    keep = np.ones(warp_a.size, dtype=bool)
    keep[1:] = ((warp_a[1:] != warp_a[:-1]) | (step_a[1:] != step_a[:-1]) |
                (reg_a[1:] != reg_a[:-1]) | (sec_a[1:] != sec_a[:-1]))
    sm = (block[keep] % cfg.n_sm).astype(np.int64)
    key = (reg_a[keep] << _SHIFT) | sec_a[keep]

    # ---- y stores ----------------------------------------------------------
    ysec = _sector(order, 4, S)
    if kernel == "vector":
        y_store = n  # lane 0 of each warp stores one element -> one sector each
    else:
        y_store = int(sum(np.unique(ysec[i:i + W]).size for i in range(0, n, W)))

    return Trace(sm=sm, key=key, n_instructions=n_instr,
                 simd_efficiency=simd_eff, y_store_sectors=y_store)


def simulate(trace: Trace, cfg: GPUConfig) -> dict:
    """Run the request stream through per-SM L1 + shared L2 LRU caches."""
    l1_cap = cfg.l1_bytes // cfg.sector_bytes
    l2_cap = cfg.l2_bytes // cfg.sector_bytes
    l1 = [OrderedDict() for _ in range(cfg.n_sm)]
    l2: OrderedDict = OrderedDict()
    req = np.zeros(len(REGIONS), np.int64)
    h1 = np.zeros(len(REGIONS), np.int64)
    h2 = np.zeros(len(REGIONS), np.int64)
    dram = np.zeros(len(REGIONS), np.int64)

    for s, key in zip(trace.sm.tolist(), trace.key.tolist()):
        r = key >> _SHIFT
        req[r] += 1
        c1 = l1[s]
        if key in c1:
            c1.move_to_end(key)
            h1[r] += 1
            continue
        if key in l2:
            l2.move_to_end(key)
            h2[r] += 1
        else:
            dram[r] += 1
            l2[key] = None
            if len(l2) > l2_cap:
                l2.popitem(last=False)
        c1[key] = None
        if len(c1) > l1_cap:
            c1.popitem(last=False)

    tot_req, tot_h1 = int(req.sum()), int(h1.sum())
    l2_lookups = tot_req - tot_h1
    out = {
        "requests": tot_req,
        "l1_hit_rate": tot_h1 / tot_req if tot_req else 0.0,
        "l2_requests": l2_lookups,  # L1 misses = L1->L2 traffic
        "l2_hit_rate": int(h2.sum()) / l2_lookups if l2_lookups else 0.0,
        "dram_read_sectors": int(dram.sum()),
        "y_store_sectors": trace.y_store_sectors,
        "dram_bytes": (int(dram.sum()) + trace.y_store_sectors) * cfg.sector_bytes,
        "instructions": trace.n_instructions,
        "simd_efficiency": trace.simd_efficiency,
    }
    for i, name in enumerate(REGIONS):
        out[f"{name}_requests"] = int(req[i])
        out[f"{name}_dram_sectors"] = int(dram[i])
    out["x_l1_hit_rate"] = h1[R_X] / req[R_X] if req[R_X] else 0.0
    return out


def run(A: sp.csr_matrix, order: np.ndarray, kernel: str, cfg: GPUConfig,
        uses_order: bool = True) -> dict:
    res = simulate(build_trace(A, order, kernel, cfg, uses_order), cfg)
    res["gpu_config"] = cfg.name
    return res


def config_dict(cfg: GPUConfig) -> dict:
    return asdict(cfg)

# Methodology

## 1. Baseline and correctness
`row_static` with the scalar and vector CSR kernels is the baseline, plus cuSPARSE
(`cusparseSpMV`, `CUSPARSE_SPMV_ALG_DEFAULT`) as an established implementation.
Every schedule is checked two ways: `tests/` asserts the Python reference matches
`A @ x` exactly in float64, and `cuda/bench.cu` reports the max relative error of each
GPU run against a float64 CPU reference.

## 2. Matrix characterization (`spmvsched/analyze.py`)
| Metric | Why it matters |
|---|---|
| nnz/row mean, max, CV | Load imbalance and lane divergence |
| mean abs(col - row) | Natural (diagonal) locality |
| x footprint | Whether x fits in L2 at all |
| window_reuse(order) | Share of a row's x blocks already touched by the previous 32 rows in the schedule |

## 3. Schedules (`spmvsched/schedule.py`)
| Policy | Targets | Mechanism |
|---|---|---|
| row_static | - | identity |
| adaptive_grouping | divergence | stable sort by floor(log2(nnz)) |
| locality_aware | x reuse | lexsort by (centroid block, first block, row) |
| locality_balanced | both | locality order, then length sort in windows of 256 |

Preprocessing time is recorded for each schedule (`preprocess_s`) and must be amortized
over the number of SpMV calls (e.g. iterations of an iterative solver) to be worthwhile.

## 4. Memory model (`spmvsched/cachesim.py`)
Assumptions, chosen to be simple and stated up front:
- 32-byte sectors; a warp instruction issues one request per distinct sector.
- Regions modeled: order, rowptr, colidx, values, x. y stores bypass L1 and are counted.
- Per-SM L1 and shared L2, fully associative LRU.
- Thread blocks assigned round-robin to SMs; blocks of one wave advance in lockstep.
- Presets: `small` (scaled so effects appear on simulation-sized matrices) and
  `a100_like` (approximate sizes; verify against vendor documentation before quoting).

Not modeled: set conflicts, line vs. sector fills, MSHRs, latency, DRAM bandwidth,
intra-SM warp interleaving, prefetching.

## 5. Hardware measurement
`scripts/profile_ncu.sh` collects, per kernel launch:
`l1tex__t_sector_hit_rate.pct`, `lts__t_sector_hit_rate.pct`, `dram__bytes_read.sum`,
`dram__bytes_write.sum`, global load sectors, and threads-per-instruction (divergence).

## 6. Evaluation protocol
1. Same x, same matrix layout, 3 warm-up launches, mean over N timed launches.
2. Report time, GFLOP/s, effective bandwidth, and ncu counters per schedule.
3. Classify matrices by characterization metrics and report where each schedule wins,
   ties, or regresses, including regressions.

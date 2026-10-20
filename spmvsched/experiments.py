"""Experiment runner.

Examples
--------
python -m spmvsched.experiments --synthetic all
python -m spmvsched.experiments --matrix data/12month1/12month1.mtx --max-rows 3000
python -m spmvsched.experiments --synthetic block_shuffled --export build/
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from . import analyze, cachesim, io, schedule

FIELDS = ["matrix", "schedule", "kernel", "gpu_config", "preprocess_s", "sim_s",
          "max_rel_err", "window_reuse", "requests", "l1_hit_rate", "x_l1_hit_rate",
          "l2_requests", "l2_hit_rate", "dram_read_sectors", "x_dram_sectors", "colidx_dram_sectors",
          "values_dram_sectors", "order_dram_sectors", "rowptr_dram_sectors",
          "y_store_sectors", "dram_bytes", "instructions", "simd_efficiency"]


def run_matrix(tag, A, policies, kernels, cfg, export_dir=None):
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, A.shape[1]).astype(np.float32)
    y_ref = A.astype(np.float64) @ x.astype(np.float64)
    scale = max(np.abs(y_ref).max(), 1e-30)

    if export_dir:
        export_dir.mkdir(parents=True, exist_ok=True)
        io.export_csr_bin(A, export_dir / f"{tag}.csr.bin")

    rows = []
    for name in policies:
        sch = schedule.build(A, name)
        y = schedule.spmv_scheduled(A, sch.order, x)
        err = float(np.abs(y - y_ref).max() / scale)
        reuse = analyze.window_reuse(A, sch.order)
        if export_dir:
            io.export_order_bin(sch.order, export_dir / f"{tag}.{name}.order.bin")
        for kern in kernels:
            t0 = time.perf_counter()
            res = cachesim.run(A, sch.order, kern, cfg, uses_order=(name != "row_static"))
            row = {"matrix": tag, "schedule": name, "kernel": kern,
                   "preprocess_s": sch.preprocess_s, "sim_s": time.perf_counter() - t0,
                   "max_rel_err": err, "window_reuse": reuse, **res}
            rows.append({f: row.get(f) for f in FIELDS})
            print(f"  {name:18s} {kern:6s} dram_sectors={res['dram_read_sectors']:>10d} "
                  f"L1={res['l1_hit_rate']:.3f} L2={res['l2_hit_rate']:.3f} "
                  f"simd={res['simd_efficiency']:.3f} reuse={reuse:.3f} err={err:.1e}")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--matrix", nargs="+", help=".mtx or .npz files")
    src.add_argument("--synthetic", nargs="+", help=f"{io.SYNTHETIC_KINDS} or 'all'")
    p.add_argument("--n", type=int, default=100000, help="rows for synthetic matrices")
    p.add_argument("--max-rows", type=int, default=None, help="leading-row slice for large matrices")
    p.add_argument("--policies", nargs="+", default=list(schedule.POLICIES))
    p.add_argument("--kernels", nargs="+", default=["scalar", "vector"])
    p.add_argument("--gpu", default="small", choices=list(cachesim.PRESETS))
    p.add_argument("--out", default="results/sim_results.csv")
    p.add_argument("--export", default=None, help="directory for CUDA .bin inputs")
    a = p.parse_args(argv)

    cfg = cachesim.PRESETS[a.gpu]
    export_dir = Path(a.export) if a.export else None
    inputs = []
    if a.synthetic:
        kinds = io.SYNTHETIC_KINDS if "all" in a.synthetic else a.synthetic
        inputs = [(k, lambda k=k: io.synthetic(k, n=a.n)) for k in kinds]
    else:
        inputs = [(Path(m).stem, lambda m=m: io.load_matrix(m)) for m in a.matrix]

    all_rows, chars = [], {}
    for tag, loader in inputs:
        A = io.row_slice(loader(), a.max_rows)
        chars[tag] = analyze.characterize(A)
        print(f"[{tag}] {A.shape[0]}x{A.shape[1]} nnz={A.nnz}")
        all_rows += run_matrix(tag, A, a.policies, a.kernels, cfg, export_dir)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    with open(out.with_suffix(".matrices.json"), "w") as f:
        json.dump({"gpu_config": cachesim.config_dict(cfg), "matrices": chars}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

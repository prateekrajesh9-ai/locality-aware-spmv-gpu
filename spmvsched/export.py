"""Export a matrix and every schedule as .bin files for the CUDA benchmark.

python -m spmvsched.export --matrix data/12month1/12month1.mtx --out build/
Also appends schedule preprocessing times to results/preprocess_times.csv.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from . import io, schedule


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", required=True)
    p.add_argument("--out", default="build/")
    p.add_argument("--policies", nargs="+", default=list(schedule.POLICIES))
    p.add_argument("--times", default="results/preprocess_times.csv")
    a = p.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = Path(a.matrix).stem
    A = io.load_matrix(a.matrix)
    io.export_csr_bin(A, out / f"{tag}.csr.bin")

    times = Path(a.times)
    times.parent.mkdir(parents=True, exist_ok=True)
    new = not times.exists()
    with open(times, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["matrix", "schedule", "rows", "nnz", "preprocess_s"])
        for name in a.policies:
            sch = schedule.build(A, name)
            io.export_order_bin(sch.order, out / f"{tag}.{name}.order.bin")
            w.writerow([tag, name, A.shape[0], A.nnz, f"{sch.preprocess_s:.6f}"])
            print(f"{tag}: {name:18s} preprocess {sch.preprocess_s * 1e3:8.2f} ms")


if __name__ == "__main__":
    main()

"""Plot simulator results: python -m spmvsched.plots results/sim_results.csv"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .schedule import POLICIES  # noqa: E402

POLICY_ORDER = list(POLICIES)


def normalized_bars(df, metric, kernel, out, title, baseline="row_static"):
    d = df[df.kernel == kernel].pivot(index="matrix", columns="schedule", values=metric)
    d = d[[c for c in POLICY_ORDER if c in d] + [c for c in d if c not in POLICY_ORDER]]
    if baseline in d:
        d = d.div(d[baseline], axis=0)
    ax = d.plot.bar(figsize=(9, 4), rot=0)
    ax.legend(title="schedule", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_ylabel(f"{metric} (normalized to {baseline})")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def main(path="results/sim_results.csv"):
    df = pd.read_csv(path)
    outdir = Path(path).parent / "figures"
    outdir.mkdir(exist_ok=True)
    for kern in df.kernel.unique():
        normalized_bars(df, "dram_bytes", kern, outdir / f"dram_bytes_{kern}.png",
                        f"Modeled DRAM traffic ({kern} kernel)")
        normalized_bars(df, "x_dram_sectors", kern, outdir / f"x_dram_{kern}.png",
                        f"Modeled x-vector DRAM sectors ({kern} kernel)")
        normalized_bars(df, "simd_efficiency", kern, outdir / f"simd_{kern}.png",
                        f"SIMD lane efficiency ({kern} kernel)")
    print(f"figures in {outdir}")


if __name__ == "__main__":
    main(*sys.argv[1:])

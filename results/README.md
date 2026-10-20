# results/

Generated files land here; commit the CSVs and figures you report.

| File | Produced by |
|---|---|
| `sim_results.csv`, `*.matrices.json` | `python -m spmvsched.experiments ...` |
| `figures/*.png` | `python -m spmvsched.plots results/sim_results.csv` |
| `preprocess_times.csv` | `python -m spmvsched.export ...` |
| `gpu_results.csv` | `scripts/run_gpu_bench.sh` |
| `ncu_metrics.csv` | `scripts/profile_ncu.sh` |

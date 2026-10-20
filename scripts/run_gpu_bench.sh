#!/usr/bin/env bash
# Export schedules with the Python package, then time them on the GPU.
#   scripts/run_gpu_bench.sh data/12month1/12month1.mtx [more.mtx ...]
set -euo pipefail
cd "$(dirname "$0")/.."
make -C cuda >/dev/null
mkdir -p build results
OUT=results/gpu_results.csv
echo "matrix,schedule,kernel,avg_ms,gflops,eff_bw_gbs,max_rel_err" > "$OUT"
for mtx in "$@"; do
  tag="$(basename "${mtx%.*}")"
  python -m spmvsched.export --matrix "$mtx" --out build/
  args=()
  for f in build/"${tag}".*.order.bin; do
    s="${f#build/${tag}.}"; s="${s%.order.bin}"
    [[ "$s" == "row_static" ]] && continue
    args+=(--order "${s}=${f}")
  done
  ./cuda/spmv_bench "build/${tag}.csr.bin" "${args[@]}" --iters 200 | tail -n +2 | sed "s/^/${tag},/" >> "$OUT"
done
echo "wrote $OUT"

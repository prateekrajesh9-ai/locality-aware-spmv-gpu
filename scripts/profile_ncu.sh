#!/usr/bin/env bash
# Measure real cache behaviour with Nsight Compute (needs profiling permissions).
#   scripts/profile_ncu.sh build/12month1.csr.bin locality_aware=build/12month1.locality_aware.order.bin
set -euo pipefail
cd "$(dirname "$0")/.."
BIN="$1"; shift
args=(); for o in "$@"; do args+=(--order "$o"); done
METRICS=l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__thread_inst_executed_per_inst_executed.ratio,gpu__time_duration.sum
mkdir -p results
ncu --kernel-name regex:spmv_csr --metrics "$METRICS" --csv \
    ./cuda/spmv_bench "$BIN" "${args[@]}" --iters 1 > results/ncu_metrics.csv
echo "wrote results/ncu_metrics.csv"

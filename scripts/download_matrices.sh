#!/usr/bin/env bash
# Download SuiteSparse matrices into data/. Edit MATRICES to change the suite.
# Browse/verify names at https://sparse.tamu.edu
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
MATRICES=(
  "Buss/12month1"        # wide LP matrix with very long rows
  "Williams/webbase-1M"  # web graph, irregular
  "SNAP/web-Google"      # power-law graph
  "Boeing/pwtk"          # FEM, strong natural locality
  "Williams/cant"        # FEM, banded
)
for m in "${MATRICES[@]}"; do
  name="${m##*/}"
  if [[ -f "data/${name}/${name}.mtx" ]]; then echo "have ${name}"; continue; fi
  echo "fetching ${m}"
  if curl -fL "https://sparse.tamu.edu/MM/${m}.tar.gz" -o "data/${name}.tar.gz"; then
    tar -xzf "data/${name}.tar.gz" -C data && rm "data/${name}.tar.gz"
  else
    echo "  failed: ${m} (check the name on sparse.tamu.edu)"
  fi
done

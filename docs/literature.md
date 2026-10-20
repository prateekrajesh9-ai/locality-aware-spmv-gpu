# Related work (notes)

> Verify every citation (authors, venue, year, DOI) against the original source before
> reusing these notes in a report.

- **The Sparsity-Aware LazyGPU Architecture** (ISCA 2025). Lazy memory-request issue and a
  Zero Cache that eliminate requests which do not affect the result. Hardware mechanisms,
  evaluated on sparse neural-network workloads. *This project differs:* it changes the
  software schedule on unmodified hardware rather than the memory pipeline.
- **NetCrafter: Tailoring Network Traffic for Non-Uniform Bandwidth Multi-GPU Systems**
  (ISCA 2025). How communication constraints shape GPU performance; relevant background for
  extending scheduling across GPUs.
- **cuSPARSE** `cusparseSpMV`: vendor baseline used in `cuda/bench.cu`.

Useful comparisons to add: CSR-Adaptive / CSR-Stream row binning, merge-based CSR SpMV,
SELL-C-σ, and graph reorderings such as RCM.

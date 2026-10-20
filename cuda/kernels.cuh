// kernels.cuh - CSR SpMV kernels with an optional row schedule.
//
// order == nullptr  -> work unit i processes row i (row_static, no indirection)
// order != nullptr  -> work unit i processes row order[i]; y is written at the
//                      original row index, so every schedule computes y = A x.
#pragma once
#include <cuda_runtime.h>

// One thread per row. Simple, but neighbouring lanes read different rows of A,
// so loads of colidx/vals are uncoalesced and long rows cause lane divergence.
__global__ void spmv_csr_scalar(int n_rows,
                                const int* __restrict__ rowptr,
                                const int* __restrict__ colidx,
                                const float* __restrict__ vals,
                                const float* __restrict__ x,
                                float* __restrict__ y,
                                const int* __restrict__ order) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) return;
    int row = order ? order[i] : i;
    float sum = 0.0f;
    for (int j = rowptr[row]; j < rowptr[row + 1]; ++j)
        sum += vals[j] * __ldg(&x[colidx[j]]);
    y[row] = sum;
}

// One warp per row. Lanes stride across the row's nonzeros (coalesced A loads),
// then reduce with warp shuffles. Better for long rows, wasteful for short ones.
__global__ void spmv_csr_vector(int n_rows,
                                const int* __restrict__ rowptr,
                                const int* __restrict__ colidx,
                                const float* __restrict__ vals,
                                const float* __restrict__ x,
                                float* __restrict__ y,
                                const int* __restrict__ order) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= n_rows) return;  // whole warp exits together, shuffle mask stays full
    const int row = order ? order[warp] : warp;
    const int start = rowptr[row], end = rowptr[row + 1];
    float sum = 0.0f;
    for (int j = start + lane; j < end; j += 32)
        sum += vals[j] * __ldg(&x[colidx[j]]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) y[row] = sum;
}

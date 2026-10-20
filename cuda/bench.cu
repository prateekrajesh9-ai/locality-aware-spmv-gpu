// bench.cu - time scheduled CSR SpMV kernels against cuSPARSE.
//
// Usage:
//   ./spmv_bench <matrix.csr.bin> [--order name=path.order.bin ...]
//                [--kernel scalar|vector|both] [--iters N] [--block 256]
//
// Output: one CSV line per (schedule, kernel) on stdout:
//   schedule,kernel,avg_ms,gflops,eff_bw_gbs,max_rel_err
// eff_bw_gbs uses the minimum bytes an ideal kernel must move (A, x, y, order),
// so it is a lower bound on achieved traffic, not a hardware counter.
// Use scripts/profile_ncu.sh for measured cache hit rates and DRAM bytes.
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "csr_io.h"
#include "kernels.cuh"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t e_ = (call);                                                  \
        if (e_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_),         \
                    __FILE__, __LINE__);                                          \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)
#define CUSPARSE_CHECK(call)                                                      \
    do {                                                                          \
        cusparseStatus_t s_ = (call);                                             \
        if (s_ != CUSPARSE_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuSPARSE error %d at %s:%d\n", (int)s_, __FILE__,    \
                    __LINE__);                                                    \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

struct DeviceCSR {
    int rows, cols, nnz;
    int *rowptr, *colidx;
    float* vals;
};

static std::vector<double> cpu_reference(const HostCSR& A, const std::vector<float>& x) {
    std::vector<double> y(A.rows, 0.0);
    for (int64_t r = 0; r < A.rows; ++r)
        for (int j = A.rowptr[r]; j < A.rowptr[r + 1]; ++j)
            y[r] += (double)A.vals[j] * (double)x[A.colidx[j]];
    return y;
}

static double max_rel_err(const std::vector<float>& y, const std::vector<double>& ref) {
    double scale = 1e-30, err = 0.0;
    for (double v : ref) scale = std::max(scale, std::fabs(v));
    for (size_t i = 0; i < y.size(); ++i) err = std::max(err, std::fabs(y[i] - ref[i]));
    return err / scale;
}

template <typename Launch>
static float time_ms(Launch launch, int iters) {
    for (int i = 0; i < 3; ++i) launch();  // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / iters;
}

static void report(const char* sched, const char* kern, float ms, const DeviceCSR& A,
                   bool has_order, double err) {
    double bytes = (double)A.nnz * (sizeof(int) + sizeof(float)) +
                   (double)(A.rows + 1) * sizeof(int) + (double)A.cols * sizeof(float) +
                   (double)A.rows * sizeof(float) + (has_order ? (double)A.rows * sizeof(int) : 0.0);
    double gflops = 2.0 * A.nnz / (ms * 1e6);
    double gbs = bytes / (ms * 1e6);
    printf("%s,%s,%.5f,%.3f,%.2f,%.3e\n", sched, kern, ms, gflops, gbs, err);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s matrix.csr.bin [--order name=file] [--kernel k] [--iters N] [--block B]\n", argv[0]);
        return 1;
    }
    std::string mat_path = argv[1], kernel = "both";
    int iters = 100, block = 256;
    std::vector<std::pair<std::string, std::string>> orders;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--order" && i + 1 < argc) {
            std::string s = argv[++i];
            size_t eq = s.find('=');
            if (eq == std::string::npos) { fprintf(stderr, "--order expects name=path\n"); return 1; }
            orders.push_back({s.substr(0, eq), s.substr(eq + 1)});
        } else if (a == "--kernel" && i + 1 < argc) kernel = argv[++i];
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--block" && i + 1 < argc) block = std::atoi(argv[++i]);
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }

    HostCSR h = load_csr_bin(mat_path);
    std::vector<float> hx(h.cols);
    for (int64_t i = 0; i < h.cols; ++i) hx[i] = 1.0f + 0.001f * (float)(i % 997);
    std::vector<double> ref = cpu_reference(h, hx);

    DeviceCSR A{(int)h.rows, (int)h.cols, (int)h.nnz, nullptr, nullptr, nullptr};
    float *dx, *dy;
    CUDA_CHECK(cudaMalloc(&A.rowptr, (h.rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&A.colidx, h.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&A.vals, h.nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dx, h.cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, h.rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(A.rowptr, h.rowptr.data(), (h.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A.colidx, h.colidx.data(), h.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A.vals, h.vals.data(), h.nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), h.cols * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> hy(h.rows);
    auto fetch_err = [&]() {
        CUDA_CHECK(cudaMemcpy(hy.data(), dy, h.rows * sizeof(float), cudaMemcpyDeviceToHost));
        return max_rel_err(hy, ref);
    };

    printf("schedule,kernel,avg_ms,gflops,eff_bw_gbs,max_rel_err\n");

    // ---- cuSPARSE baseline --------------------------------------------------
    {
        cusparseHandle_t handle;
        CUSPARSE_CHECK(cusparseCreate(&handle));
        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vx, vy;
        CUSPARSE_CHECK(cusparseCreateCsr(&matA, h.rows, h.cols, h.nnz, A.rowptr, A.colidx, A.vals,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vx, h.cols, dx, CUDA_R_32F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vy, h.rows, dy, CUDA_R_32F));
        float alpha = 1.f, beta = 0.f;
        size_t buf_size = 0;
        void* buf = nullptr;
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vx,
                                               &beta, vy, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size));
        CUDA_CHECK(cudaMalloc(&buf, std::max<size_t>(buf_size, 1)));
        float ms = time_ms([&] {
            CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vx, &beta, vy,
                                        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buf));
        }, iters);
        report("cusparse", "cusparse", ms, A, false, fetch_err());
        cudaFree(buf);
        cusparseDestroySpMat(matA);
        cusparseDestroyDnVec(vx);
        cusparseDestroyDnVec(vy);
        cusparseDestroy(handle);
    }

    // ---- custom kernels, static order + each supplied schedule ------------
    std::vector<std::pair<std::string, int*>> scheds = {{"row_static", nullptr}};
    for (auto& o : orders) {
        std::vector<int> ho = load_order_bin(o.second, h.rows);
        int* d_order;
        CUDA_CHECK(cudaMalloc(&d_order, h.rows * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_order, ho.data(), h.rows * sizeof(int), cudaMemcpyHostToDevice));
        scheds.push_back({o.first, d_order});
    }

    for (auto& s : scheds) {
        if (kernel == "scalar" || kernel == "both") {
            int grid = (A.rows + block - 1) / block;
            CUDA_CHECK(cudaMemset(dy, 0, h.rows * sizeof(float)));
            float ms = time_ms([&] {
                spmv_csr_scalar<<<grid, block>>>(A.rows, A.rowptr, A.colidx, A.vals, dx, dy, s.second);
            }, iters);
            report(s.first.c_str(), "scalar", ms, A, s.second != nullptr, fetch_err());
        }
        if (kernel == "vector" || kernel == "both") {
            long long threads = (long long)A.rows * 32;
            int grid = (int)((threads + block - 1) / block);
            CUDA_CHECK(cudaMemset(dy, 0, h.rows * sizeof(float)));
            float ms = time_ms([&] {
                spmv_csr_vector<<<grid, block>>>(A.rows, A.rowptr, A.colidx, A.vals, dx, dy, s.second);
            }, iters);
            report(s.first.c_str(), "vector", ms, A, s.second != nullptr, fetch_err());
        }
    }

    for (auto& s : scheds) if (s.second) cudaFree(s.second);
    cudaFree(A.rowptr); cudaFree(A.colidx); cudaFree(A.vals); cudaFree(dx); cudaFree(dy);
    return 0;
}

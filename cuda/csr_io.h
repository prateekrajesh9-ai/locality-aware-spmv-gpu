// csr_io.h - readers for the binary formats written by spmvsched/io.py
#pragma once
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

struct HostCSR {
    int64_t rows = 0, cols = 0, nnz = 0;
    std::vector<int> rowptr, colidx;
    std::vector<float> vals;
};

inline void read_exact(FILE* f, void* dst, size_t bytes, const std::string& what) {
    if (fread(dst, 1, bytes, f) != bytes) throw std::runtime_error("short read: " + what);
}

inline HostCSR load_csr_bin(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    HostCSR m;
    int64_t hdr[3];
    read_exact(f, hdr, sizeof(hdr), "header");
    m.rows = hdr[0]; m.cols = hdr[1]; m.nnz = hdr[2];
    if (m.nnz > INT32_MAX) throw std::runtime_error("nnz exceeds int32 indexing");
    m.rowptr.resize(m.rows + 1);
    m.colidx.resize(m.nnz);
    m.vals.resize(m.nnz);
    read_exact(f, m.rowptr.data(), m.rowptr.size() * sizeof(int), "rowptr");
    read_exact(f, m.colidx.data(), m.colidx.size() * sizeof(int), "colidx");
    read_exact(f, m.vals.data(), m.vals.size() * sizeof(float), "values");
    fclose(f);
    return m;
}

inline std::vector<int> load_order_bin(const std::string& path, int64_t expected_rows) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    int64_t n;
    read_exact(f, &n, sizeof(n), "order header");
    if (n != expected_rows) throw std::runtime_error("order length does not match matrix rows");
    std::vector<int> order(n);
    read_exact(f, order.data(), n * sizeof(int), "order");
    fclose(f);
    return order;
}

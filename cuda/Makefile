# Build: make            (set ARCH for your GPU, e.g. ARCH=sm_86 for RTX 30xx, sm_80 for A100)
NVCC  ?= nvcc
ARCH  ?= sm_80
FLAGS ?= -O3 -std=c++17 -lineinfo

spmv_bench: bench.cu kernels.cuh csr_io.h
	$(NVCC) $(FLAGS) -arch=$(ARCH) bench.cu -o $@ -lcusparse

clean:
	rm -f spmv_bench
.PHONY: clean

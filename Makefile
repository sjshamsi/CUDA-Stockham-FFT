# Build the benchmark: make            (final version, V100 / sm_70)
#                      make ARCH=sm_80 (another GPU)
#                      make SRC=versions/v1.3_smem_twiddles.cu
# Run:                 ./fft 67108864  (N must be a power of two)
NVCC  ?= nvcc
ARCH  ?= sm_70
SRC   ?= src/fft.cu
FLAGS ?= -O3 -lineinfo -Xptxas=-v

fft: $(SRC)
	$(NVCC) $(FLAGS) -arch=$(ARCH) -o $@ $< -lcufft

clean:
	rm -f fft

.PHONY: clean

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


/* ================= CUDA Checking Macros  ================= */
#define CUDA_CHECK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(_e)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

#define CUFFT_CHECK(call) do { \
  cufftResult _e = (call); \
  if (_e != CUFFT_SUCCESS) { \
    fprintf(stderr, "cuFFT error %s:%d: code %d\n", \
            __FILE__, __LINE__, (int)_e); \
    exit(EXIT_FAILURE); \
  } \
} while (0)


/* ================= Geneal Host Helpers  ================= */
int isPowerof2(unsigned long long x) {
  return x != 0 && (x & (x - 1)) == 0;
}

float elapsed_time_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  return ms;
}

/* ================= FFT Kernels ================= */

// Atomic add for double (needed for older archs like Kepler K80 if not compiled with arch flags, 
// though native on Pascal+)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
__device__ double myAtomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#else 
__device__ double myAtomicAdd(double* address, double val) {
  return atomicAdd(address, val);
}
#endif

__global__ void init_signal_kernel(const unsigned long long fftArrLen, 
                                   cuDoubleComplex *fftArr_d) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < fftArrLen) {
    double t = (double)i / fftArrLen;
    double v = sin(2 * M_PI * 5 * t) + 0.5 * sin(2 * M_PI * 10 * t);
    fftArr_d[i] = make_cuDoubleComplex(v, 0.0);
  }
}

__device__ __forceinline__ unsigned long long device_bit_reverse(unsigned long long int x, const int log2fftArrLen) {
  unsigned long long int result = 0;
  #pragma unroll
  for (int i = 0; i < log2fftArrLen; i++) {
    result = (result << 1) | (x & 1ull);
    x >>= 1;
  }
  return result;
}

__global__ void bit_reverse_kernel(const unsigned long long fftArrLen,
                                   const int log2fftArrLen,
                                   cuDoubleComplex* __restrict__ myfftArr_d) {
  const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < fftArrLen) {
    const unsigned long long j = device_bit_reverse(i, log2fftArrLen);
    if (i < j) {
      cuDoubleComplex tmp = myfftArr_d[i];
      myfftArr_d[i] = myfftArr_d[j];
      myfftArr_d[j] = tmp;
    }
  }
}

__global__ void fft_stage_kernel(const unsigned long long fftArrLen,
                                 cuDoubleComplex __restrict__ *output_d,
                                 const int m, const int m2) {
  extern __shared__ cuDoubleComplex twiddle_s[];
  const unsigned long long ti = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  int twiddleCacheCount = (m2 < (int)blockDim.x) ? m2 : (int)blockDim.x;

  // Cache small-j twiddles per block; large-j values fall back to direct eval.
  if (threadIdx.x < (unsigned int)twiddleCacheCount) {
    double theta_cached = -2.0 * M_PI * (double)threadIdx.x / (double)m;
    twiddle_s[threadIdx.x] = make_cuDoubleComplex(cos(theta_cached), sin(theta_cached));
  }
  __syncthreads();

  // We launch N/2 threads in total per stage for the butterflies
  if (ti < fftArrLen/2) {
    unsigned long long group_id = ti / m2;
    int j = ti % m2;
    unsigned long long k = group_id * m; 

    cuDoubleComplex w;
    if (j < twiddleCacheCount) {
      w = twiddle_s[j];
    } else {
      double theta = -2.0 * M_PI * (double)j / (double)m;
      w = make_cuDoubleComplex(cos(theta), sin(theta));
    }

    // Use output_d, not data
    cuDoubleComplex t = cuCmul(w, output_d[k + j + m2]);
    cuDoubleComplex u = output_d[k + j];

    output_d[k + j] = cuCadd(u, t);
    output_d[k + j + m2] = cuCsub(u, t);
  }
}

void init_input_signal(const unsigned long long fftArrLen, cuDoubleComplex **fftArr_d, cudaStream_t stream) {
  unsigned long long int fftArrSize = sizeof(cuDoubleComplex) * fftArrLen;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));

  // Allocate on device
  CUDA_CHECK(cudaMalloc((void**)fftArr_d, fftArrSize));
  
  // Launch kernel to initialize data on device
  int blockSize = 256;
  int numBlocks = (fftArrLen + blockSize - 1) / blockSize;
  init_signal_kernel<<<numBlocks, blockSize, 0, stream>>>(fftArrLen, *fftArr_d);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  printf("Device allocation and initialization (launch) took: %10.8f ms\n", elapsed_time_ms(start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

__global__ void check_accuracy_kernel(const unsigned long long fftArrLen,
                                      cuDoubleComplex *cufftArr_d, 
                                      cuDoubleComplex *myfftArr_d, 
                                      double *diff_d, 
                                      double *norm_d) {
  __shared__ double diff_s[256];
  __shared__ double norm_s[256];

  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  double local_diff = 0.0;
  double local_norm = 0.0;

  if (i < fftArrLen) {
    cuDoubleComplex d = cuCsub(cufftArr_d[i], myfftArr_d[i]);
    double d_mag = cuCabs(d);
    double ref_mag = cuCabs(cufftArr_d[i]);

    local_diff = d_mag * d_mag;
    local_norm = ref_mag * ref_mag;
  }

  diff_s[threadIdx.x] = local_diff;
  norm_s[threadIdx.x] = local_norm;
  __syncthreads();

  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      diff_s[threadIdx.x] += diff_s[threadIdx.x + stride];
      norm_s[threadIdx.x] += norm_s[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    myAtomicAdd(diff_d, diff_s[0]);
    myAtomicAdd(norm_d, norm_s[0]);
  }
}

template <typename Kernel>
size_t max_smem_without_reducing_occupancy(Kernel k, int blockSize) {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  int hwMaxPerBlock = (prop.sharedMemPerBlockOptin > 0)
                          ? prop.sharedMemPerBlockOptin
                          : prop.sharedMemPerBlock;
  CUDA_CHECK(cudaFuncSetAttribute(k,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  hwMaxPerBlock));

  int blocksPerSM0 = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocksPerSM0, k, blockSize, /*dynamicSMemBytes=*/0));
  if (blocksPerSM0 <= 0) return 0;

  size_t lo = 0;
  size_t hi = (size_t)prop.sharedMemPerMultiprocessor;
  size_t best = 0;

  while (lo <= hi) {
    size_t mid = lo + ((hi - lo) >> 1);
    int blocksPerSMMid = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSMMid, k, blockSize, (int)mid);

    if (err != cudaSuccess) {
      // mid exceeds per-block limits or violates launch constraints.
      (void)cudaGetLastError();
      if (mid == 0) break;
      hi = mid - 1;
      continue;
    }

    if (blocksPerSMMid >= blocksPerSM0) {
      best = mid;
      lo = mid + 1;
    } else {
      if (mid == 0) break;
      hi = mid - 1;
    }
  }

  return best;
}

double check_accuracy(const unsigned long long fftArrLen,
                      cuDoubleComplex *cufftArr_d, 
                      cuDoubleComplex *myfftArr_d,
                      cudaStream_t stream) {
  double *d_diff, *d_norm;
  double h_diff, h_norm;
    
  CUDA_CHECK(cudaMalloc((void**)&d_diff, sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_norm, sizeof(double)));
  CUDA_CHECK(cudaMemsetAsync(d_diff, 0, sizeof(double), stream));
  CUDA_CHECK(cudaMemsetAsync(d_norm, 0, sizeof(double), stream));

  int blockSize = 256;
  int numBlocks = (fftArrLen + blockSize - 1) / blockSize;
  check_accuracy_kernel<<<numBlocks, blockSize, 0, stream>>>(fftArrLen, cufftArr_d, myfftArr_d, d_diff, d_norm);
  CUDA_CHECK(cudaGetLastError());
    
  CUDA_CHECK(cudaMemcpyAsync(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(&h_norm, d_norm, sizeof(double), cudaMemcpyDeviceToHost, stream));
  
  cudaStreamSynchronize(stream);
    
  cudaFree(d_diff);
  cudaFree(d_norm);

  return sqrt(h_diff / h_norm);
}

void cufftRun(const unsigned long long fftArrLen,
              cuDoubleComplex *cufftArr_d,
              cudaStream_t stream) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Make the cuFFT plan 
  cufftHandle plan;
  CUFFT_CHECK(cufftPlan1d(&plan, fftArrLen, CUFFT_Z2Z, 1));
  CUFFT_CHECK(cufftSetStream(plan, stream));
  printf("cuFFT Plan Creation Time: n/a (host-side API work)\n");

  // Running the cuFFT plan 
  CUDA_CHECK(cudaEventRecord(start, stream));
  CUFFT_CHECK(cufftExecZ2Z(plan, (cufftDoubleComplex*)cufftArr_d, (cufftDoubleComplex*)cufftArr_d, CUFFT_FORWARD));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  printf("cuFFT Plan Execution Time: %10.8f ms\n", elapsed_time_ms(start, stop));
  
  CUFFT_CHECK(cufftDestroy(plan));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}


/* ================= GPU FFT Function ================= */
void fft_1d_gpu(const unsigned long long fftArrLen,
                cuDoubleComplex *myfftArr_d,
                int nthreadsPerBlock,
                int nblocks,
                cudaStream_t stream) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, stream));
  int log2fftArrLen = 0;
  for (unsigned long long i = fftArrLen; i > 1; i >>= 1) log2fftArrLen++;

  // 1. Bit-reversal permutation
  bit_reverse_kernel<<<nblocks, nthreadsPerBlock, 0, stream>>>(fftArrLen, log2fftArrLen, myfftArr_d);
  CUDA_CHECK(cudaGetLastError());
  // cudaDeviceSynchronize();

  size_t tunedDynSmemBytes =
      max_smem_without_reducing_occupancy(fft_stage_kernel, nthreadsPerBlock);
  printf("Dynamic shared-memory budget (max without occupancy drop): %zu bytes\n",
         tunedDynSmemBytes);

  // 2. Cooley-Tukey stages
  for (int s = 1; s <= log2fftArrLen; s++) {
    int m = 1 << s;
    int m2 = m >> 1;
     
    // We have N/2 butterflies to process this stage
    int numButterflies = fftArrLen / 2;
    int stageBlocks = (numButterflies + nthreadsPerBlock - 1) / nthreadsPerBlock;

    size_t stageRequestedSmem = (size_t)((m2 < nthreadsPerBlock) ? m2 : nthreadsPerBlock) *
                                sizeof(cuDoubleComplex);
    size_t stageDynSmem = (stageRequestedSmem < tunedDynSmemBytes) ?
                          stageRequestedSmem : tunedDynSmemBytes;

    fft_stage_kernel<<<stageBlocks, nthreadsPerBlock, stageDynSmem, stream>>>(
        fftArrLen, myfftArr_d, m, m2);
    CUDA_CHECK(cudaGetLastError());
  }

  // Check for kernel errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(err));
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  printf("Execution Time:%10.8f ms\n", elapsed_time_ms(start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}


/* ================= main ================= */
int main(int argc, char **argv) {
  // Argparsing
  if (argc != 2) {
    printf("Usage: %s <number>\n", argv[0]);
    return EXIT_FAILURE;
  }

  unsigned long long fftArrLen = strtoull(argv[1], NULL, 10);

  if (! isPowerof2(fftArrLen)) {
    printf("Argument %8llu is not a power of 2. Exiting...\n", fftArrLen);
    return EXIT_FAILURE;
  }

  // Print the banner
  printf("=== GPU FFT v1.2 vs cuFFT Benchmark ===\n"
         "FFT Array Length: %12llu\n", fftArrLen);

  // Pointers to the arrays on host and the device
  cuDoubleComplex *cufftArr_d = NULL;
  cuDoubleComplex *myfftArr_d = NULL;
  cudaStream_t stream_cufft, stream_myfft;
  CUDA_CHECK(cudaStreamCreate(&stream_cufft));
  CUDA_CHECK(cudaStreamCreate(&stream_myfft));

  cudaEvent_t cufft_done;
  CUDA_CHECK(cudaEventCreate(&cufft_done));


  /* cuFFT Run */
  printf("\n--- cuFFT Run ---\n");
  
  // Allocate the host and device arrays
  init_input_signal(fftArrLen, &cufftArr_d, stream_cufft);

  // Run cuFFT
  cufftRun(fftArrLen, cufftArr_d, stream_cufft);
  
  // Mark event
  CUDA_CHECK(cudaEventRecord(cufft_done, stream_cufft));


  /* myFFT Run */
  printf("\n--- myFFT Run ---\n");
  
  // Begin testing with various numbers of nthreadsPerBlock
  int nthreadsPerBlockList[] = {32, 64, 128, 256, 512, 1024};
  int nthreadsPerBlockListLen = sizeof(nthreadsPerBlockList) / sizeof(nthreadsPerBlockList[0]);

  int nthreadsPerBlock, nblocks;

  for (int i = 0; i < nthreadsPerBlockListLen; i++) {
    nthreadsPerBlock = nthreadsPerBlockList[i];
    nblocks = (fftArrLen + nthreadsPerBlock - 1) / nthreadsPerBlock;

    printf("\n# nThreadsPerBlock:%5d, nBlocks:%8d\n", nthreadsPerBlock, nblocks);

    // Allocate the host and device arrays
    if (myfftArr_d) cudaFree(myfftArr_d);
    init_input_signal(fftArrLen, &myfftArr_d, stream_myfft);

    // run myFFT
    fft_1d_gpu(fftArrLen, myfftArr_d, nthreadsPerBlock, nblocks, stream_myfft);
  }

  // Find difference between cufftArr and myfftArr
  // Wait for cufft to be done
  CUDA_CHECK(cudaStreamWaitEvent(stream_myfft, cufft_done, 0));
  
  double err = check_accuracy(fftArrLen, cufftArr_d, myfftArr_d, stream_myfft);
  printf("\nDiff:%e\n", err);
  
  if (cufftArr_d) cudaFree(cufftArr_d);
  if (myfftArr_d) cudaFree(myfftArr_d);

  cudaStreamDestroy(stream_cufft);
  cudaStreamDestroy(stream_myfft);
  cudaEventDestroy(cufft_done);

  return 0;
}

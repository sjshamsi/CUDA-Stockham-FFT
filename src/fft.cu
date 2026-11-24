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

__global__ void init_twiddle_lut_kernel(const unsigned long long fftArrLen,
                                        cuDoubleComplex *twiddleLut_d) {
  const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long halfN = fftArrLen >> 1;
  if (i < halfN) {
    const double theta = -2.0 * M_PI * (double)i / (double)fftArrLen;
    double s, c;
    sincos(theta, &s, &c);
    twiddleLut_d[i] = make_cuDoubleComplex(c, s);
  }
}

__global__ void fft_stockham_stage_kernel(const unsigned long long fftArrLen,
                                          const cuDoubleComplex* __restrict__ input_d,
                                          cuDoubleComplex* __restrict__ output_d,
                                          const cuDoubleComplex* __restrict__ twiddleLut_d,
                                          const unsigned long long p) {
  extern __shared__ cuDoubleComplex twiddle_s[];
  const unsigned long long ti = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long halfN = fftArrLen >> 1;
  const unsigned long long twiddleStride = halfN / p;
  const unsigned long long twiddleCount = (p <= (unsigned long long)blockDim.x) ? p : 0ull;

  // For smaller stages, twiddles repeat often; cache block-local values in shared memory.
  if (twiddleCount > 0ull && (unsigned long long)threadIdx.x < twiddleCount) {
    const unsigned long long lutIdx = (unsigned long long)threadIdx.x * twiddleStride;
    twiddle_s[threadIdx.x] = twiddleLut_d[lutIdx];
  }
  if (twiddleCount > 0ull) __syncthreads();

  if (ti < halfN) {
    const unsigned long long j = ti & (p - 1ull);
    const unsigned long long i = ((ti - j) << 1) + j;
    const unsigned long long lutIdx = j * twiddleStride;
    const cuDoubleComplex w = (twiddleCount > 0ull) ? twiddle_s[j] : twiddleLut_d[lutIdx];

    const cuDoubleComplex a = input_d[ti];
    const cuDoubleComplex b = input_d[ti + halfN];
    const cuDoubleComplex t = cuCmul(w, b);

    output_d[i] = cuCadd(a, t);
    output_d[i + p] = cuCsub(a, t);
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
  (void)nblocks;

  cuDoubleComplex *scratch_d = NULL;
  cuDoubleComplex *twiddleLut_d = NULL;
  const size_t fftArrSize = sizeof(cuDoubleComplex) * fftArrLen;
  CUDA_CHECK(cudaMalloc((void**)&scratch_d, fftArrSize));
  const unsigned long long halfN = fftArrLen >> 1;
  if (halfN > 0ull) {
    CUDA_CHECK(cudaMalloc((void**)&twiddleLut_d, sizeof(cuDoubleComplex) * halfN));
    const int lutBlockSize = 256;
    const int lutBlocks = (int)((halfN + (unsigned long long)lutBlockSize - 1ull) /
                                (unsigned long long)lutBlockSize);
    init_twiddle_lut_kernel<<<lutBlocks, lutBlockSize, 0, stream>>>(fftArrLen, twiddleLut_d);
    CUDA_CHECK(cudaGetLastError());
  }

  const cuDoubleComplex *stageIn_d = myfftArr_d;
  cuDoubleComplex *stageOut_d = scratch_d;

  const unsigned long long numButterflies = halfN;
  const int stageBlocks = (int)((numButterflies + (unsigned long long)nthreadsPerBlock - 1ull) /
                                (unsigned long long)nthreadsPerBlock);

  // Stockham autosort radix-2 Cooley-Tukey: no explicit bit-reversal pass.
  for (int s = 0; s < log2fftArrLen; s++) {
    const unsigned long long p = 1ull << s;
    const size_t sharedBytes =
        (p <= (unsigned long long)nthreadsPerBlock) ? (size_t)(p * sizeof(cuDoubleComplex)) : 0ull;
    fft_stockham_stage_kernel<<<stageBlocks, nthreadsPerBlock, sharedBytes, stream>>>(
        fftArrLen, stageIn_d, stageOut_d, twiddleLut_d, p);
    CUDA_CHECK(cudaGetLastError());

    const cuDoubleComplex *tmpIn = stageIn_d;
    stageIn_d = stageOut_d;
    stageOut_d = (cuDoubleComplex*)tmpIn;
  }

  if (stageIn_d != myfftArr_d) {
    CUDA_CHECK(cudaMemcpyAsync(myfftArr_d, stageIn_d, fftArrSize, cudaMemcpyDeviceToDevice, stream));
  }
  if (twiddleLut_d) CUDA_CHECK(cudaFree(twiddleLut_d));
  CUDA_CHECK(cudaFree(scratch_d));

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
  printf("=== GPU FFT v1.4 vs cuFFT Benchmark ===\n"
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

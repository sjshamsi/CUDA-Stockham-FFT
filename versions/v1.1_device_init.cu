#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
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

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
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
  const unsigned long long ti = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

  // We launch N/2 threads in total per stage for the butterflies
  if (ti < fftArrLen/2) {
    unsigned long long group_id = ti / m2;
    int j = ti % m2;
    unsigned long long k = group_id * m; 
        
    // Calculate theta for w = exp(-i * 2 * PI * j / m)
    double theta = -2.0 * M_PI * (double)j / (double)m;
     
    cuDoubleComplex w = make_cuDoubleComplex(cos(theta), sin(theta));

    // Use output_d, not data
    cuDoubleComplex t = cuCmul(w, output_d[k + j + m2]);
    cuDoubleComplex u = output_d[k + j];

    output_d[k + j] = cuCadd(u, t);
    output_d[k + j + m2] = cuCsub(u, t);
  }
}

void init_input_signal(const unsigned long long fftArrLen, cuDoubleComplex **fftArr_d) {
  unsigned long long int fftArrSize = sizeof(cuDoubleComplex) * fftArrLen;
  double start, stop;

  // Allocate on device
  start = get_time();
  CUDA_CHECK(cudaMalloc((void**)fftArr_d, fftArrSize));
  
  // Launch kernel to initialize data on device
  int blockSize = 256;
  int numBlocks = (fftArrLen + blockSize - 1) / blockSize;
  init_signal_kernel<<<numBlocks, blockSize>>>(fftArrLen, *fftArr_d);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  
  stop = get_time();
  printf("Device allocation and initialization took: %10.8f ms\n", (stop - start) * 1000.0);
}

__global__ void check_accuracy_kernel(const unsigned long long fftArrLen,
                                      cuDoubleComplex *cufftArr_d, 
                                      cuDoubleComplex *myfftArr_d, 
                                      double *diff_d, 
                                      double *norm_d) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < fftArrLen) {
    cuDoubleComplex d = cuCsub(cufftArr_d[i], myfftArr_d[i]);
    double d_mag = cuCabs(d);
    double ref_mag = cuCabs(cufftArr_d[i]);

    myAtomicAdd(diff_d, d_mag * d_mag);
    myAtomicAdd(norm_d, ref_mag * ref_mag);
  }
}

double check_accuracy(const unsigned long long fftArrLen,
                      cuDoubleComplex *cufftArr_d, 
                      cuDoubleComplex *myfftArr_d) {
  double *d_diff, *d_norm;
  double h_diff, h_norm;
    
  CUDA_CHECK(cudaMalloc((void**)&d_diff, sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_norm, sizeof(double)));
  CUDA_CHECK(cudaMemset(d_diff, 0, sizeof(double)));
  CUDA_CHECK(cudaMemset(d_norm, 0, sizeof(double)));

  int blockSize = 256;
  int numBlocks = (fftArrLen + blockSize - 1) / blockSize;
  check_accuracy_kernel<<<numBlocks, blockSize>>>(fftArrLen, cufftArr_d, myfftArr_d, d_diff, d_norm);
  CUDA_CHECK(cudaGetLastError());
    
  CUDA_CHECK(cudaMemcpy(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_norm, d_norm, sizeof(double), cudaMemcpyDeviceToHost));
    
  cudaFree(d_diff);
  cudaFree(d_norm);

  return sqrt(h_diff / h_norm);
}

/* ================= cuFFT Run Function ================= */
void cufftRun(const unsigned long long fftArrLen,
              cuDoubleComplex* __restrict__ cufftArr_d) {
  double start, stop;

  // Make the cuFFT plan 
  start = get_time();
  cufftHandle plan;
  CUFFT_CHECK(cufftPlan1d(&plan, fftArrLen, CUFFT_Z2Z, 1));
  stop = get_time();
  printf("cuFFT Plan Creation Time: %10.8f ms\n", (stop - start) * 1000.0);

  // Running the cuFFT plan 
  start = get_time();
  CUFFT_CHECK(cufftExecZ2Z(plan, (cufftDoubleComplex*)cufftArr_d, (cufftDoubleComplex*)cufftArr_d, CUFFT_FORWARD));
  cudaDeviceSynchronize();
  stop = get_time();
  printf("cuFFT Plan Excution Time: %10.8lf ms\n", (stop - start) * 1000.0);
  
  CUFFT_CHECK(cufftDestroy(plan));
}


/* ================= GPU FFT Function ================= */
void fft_1d_gpu(const unsigned long long fftArrLen,
                cuDoubleComplex* __restrict__ myfftArr_d,
                const int nthreadsPerBlock,
                const int nblocks) {
  double start, stop;

  start = get_time();
  int log2fftArrLen = 0;
  for (unsigned long long i = fftArrLen; i > 1; i >>= 1) log2fftArrLen++;

  // 1. Bit-reversal permutation
  bit_reverse_kernel<<<nblocks, nthreadsPerBlock>>>(fftArrLen, log2fftArrLen, myfftArr_d);
  CUDA_CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  // 2. Cooley-Tukey stages
  for (int s = 1; s <= log2fftArrLen; s++) {
    int m = 1 << s;
    int m2 = m >> 1;
     
    // We have N/2 butterflies to process this stage
    int numButterflies = fftArrLen / 2;
    int stageBlocks = (numButterflies + nthreadsPerBlock - 1) / nthreadsPerBlock;
     
    fft_stage_kernel<<<stageBlocks, nthreadsPerBlock>>>(fftArrLen, myfftArr_d, m, m2);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
  }

  // Check for kernel errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(err));
  }
  stop = get_time();

  printf("Execution Time:%10.8lf ms\n", (stop - start) * 1000.0);
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
  printf("=== GPU FFT v1.1 vs cuFFT Benchmark ===\n"
         "FFT Array Length: %12llu\n", fftArrLen);

  // Pointers to the arrays on host and the device
  cuDoubleComplex *cufftArr_d = NULL;
  cuDoubleComplex *myfftArr_d = NULL;


  /* cuFFT Run */
  printf("\n--- cuFFT Run ---\n");
  
  // Allocate the host and device arrays
  init_input_signal(fftArrLen, &cufftArr_d);

  // Run cuFFT
  cufftRun(fftArrLen, cufftArr_d);


  /* myFFT Run */
  printf("\n--- myFFT Run ---\n");
  
  // Begin testing with various numbers of nthreadsPerBlock
  int nthreadsPerBlockList[] = {32, 64, 128, 256, 512, 1024};
  int nthreadsPerBlockListLen = sizeof(nthreadsPerBlockList) / sizeof(nthreadsPerBlockList[0]);

  for (int i = 0; i < nthreadsPerBlockListLen; i++) {
    int nthreadsPerBlock = nthreadsPerBlockList[i];
    int nblocks = (fftArrLen + nthreadsPerBlock - 1) / nthreadsPerBlock;

    printf("\n# nThreadsPerBlock:%5d, nBlocks:%8d\n", nthreadsPerBlock, nblocks);

    // Allocate the host and device arrays
    if (myfftArr_d) cudaFree(myfftArr_d);
    init_input_signal(fftArrLen, &myfftArr_d);

    // run myFFT
    fft_1d_gpu(fftArrLen, myfftArr_d, nthreadsPerBlock, nblocks);

    // Find difference between cufftArr and myfftArr
    double err = check_accuracy(fftArrLen, cufftArr_d, myfftArr_d);
    printf("Diff:%e\n", err);
  }

  if (cufftArr_d) cudaFree(cufftArr_d);
  if (myfftArr_d) cudaFree(myfftArr_d);

  return 0;
}

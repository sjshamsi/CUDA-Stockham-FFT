#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cufft.h>

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


/* ================= Complex num def ================= */
typedef struct {double re; double im;} Complex;

__host__ __device__ Complex c_make(double re, double im) {
    Complex c = { re, im };
    return c;
}

__host__ __device__ Complex c_add(Complex a, Complex b) {
    return c_make(a.re + b.re, a.im + b.im);
}

__host__ __device__ Complex c_sub(Complex a, Complex b) {
    return c_make(a.re - b.re, a.im - b.im);
}

__host__ __device__ Complex c_mul(Complex a, Complex b) {
    return c_make(a.re * b.re - a.im * b.im,
                  a.re * b.im + a.im * b.re);
}

__host__ __device__ double c_abs(Complex a) {
    return sqrt(a.re * a.re + a.im * a.im);
}


/* ================= Geneal Host Helpers  ================= */
int isPowerof2(unsigned long long x) {
  return x != 0 && (x & (x - 1)) == 0;
}

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

void * ualloc(unsigned long long size) {
  void *ptr = malloc(size);
  if (ptr == NULL) {
    fprintf(stderr, "malloc() returned null.\nExiting...\n");
    exit(EXIT_FAILURE);
  }
  return ptr;
}

void init_input_signal(const unsigned long long fftArrLen,
                       Complex **fftArr_h,
                       Complex **fftArr_d) {
  unsigned long long int fftArrSize = sizeof(Complex) * fftArrLen;
  double start, stop;

  // Allocate on host and fill it with values
  start = get_time();
  *fftArr_h = (Complex *)ualloc(fftArrSize);
  Complex *h_data = *fftArr_h;
  
  for (unsigned long long i = 0; i < fftArrLen; i++) {
    double t = (double)i / fftArrLen;
    double v = sin(2 * M_PI * 5 * t) + 0.5 * sin(2 * M_PI * 10 * t);
    h_data[i] = c_make(v, 0.0);
  }
  stop = get_time();
  printf("Host allocation and input creation took: %10.8f ms\n", (stop - start) * 1000.0);

  // Allocate on device and copy host information there
  start = get_time();
  CUDA_CHECK(cudaMalloc((void**)fftArr_d, fftArrSize));
  CUDA_CHECK(cudaMemcpy(*fftArr_d, h_data, fftArrSize, cudaMemcpyHostToDevice)); 
  stop = get_time();
  printf("Device allocation and H2D copy took: %10.8f ms\n", (stop - start) * 1000.0);
}

double findDiff(const unsigned long long fftArrLen,
                Complex *cufftArr_h, Complex *myfftArr_h) {
  double diff = 0;
  double norm = 0;

  for (unsigned long long j = 0; j < fftArrLen; j++) {
    Complex d = c_sub(cufftArr_h[j], myfftArr_h[j]);
    double d_mag = c_abs(d);
    double ref_mag = c_abs(cufftArr_h[j]);
     
    diff += d_mag * d_mag;
    norm += ref_mag * ref_mag;
  }
  double err = sqrt(diff / norm);
  return err;
}

/* ================= cuFFT Run Function ================= */
void cufftRun(const unsigned long long fftArrLen,
              Complex* __restrict__ cufftArr_h,
              Complex* __restrict__ cufftArr_d) {
  double start, stop;
  size_t fftArrSize = fftArrLen * sizeof(Complex);

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

  // Do a D2H copy
  start = get_time();
  CUDA_CHECK(cudaMemcpy(cufftArr_h, cufftArr_d, fftArrSize, cudaMemcpyDeviceToHost)); 
  stop = get_time();
  printf("cuFFT D2H took: %10.8lf ms\n", (stop - start) * 1000.0);
  
  CUFFT_CHECK(cufftDestroy(plan));
}


/* ================= FFT Kernels ================= */
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
                                   Complex* __restrict__ myfftArr_d) {
  const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < fftArrLen) {
    const unsigned long long j = device_bit_reverse(i, log2fftArrLen);
    if (i < j) {
      Complex tmp = myfftArr_d[i];
      myfftArr_d[i] = myfftArr_d[j];
      myfftArr_d[j] = tmp;
    }
  }
}

__global__ void fft_stage_kernel(const unsigned long long fftArrLen,
                                 Complex __restrict__ *output_d,
                                 const int m, const int m2) {
  const unsigned long long ti = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

  // We launch N/2 threads in total per stage for the butterflies
  if (ti < fftArrLen/2) {
    unsigned long long group_id = ti / m2;
    int j = ti % m2;
    unsigned long long k = group_id * m; 
        
    // Calculate theta for w = exp(-i * 2 * PI * j / m)
    double theta = -2.0 * M_PI * (double)j / (double)m;
     
    Complex w = c_make(cos(theta), sin(theta));

    // Use output_d, not data
    Complex t = c_mul(w, output_d[k + j + m2]);
    Complex u = output_d[k + j];

    output_d[k + j] = c_add(u, t);
    output_d[k + j + m2] = c_sub(u, t);
  }
}


/* ================= GPU FFT Function ================= */
void fft_1d_gpu(const unsigned long long fftArrLen,
                Complex* __restrict__ myfftArr_h,
                Complex* __restrict__ myfftArr_d,
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

  // Copy device to host
  const unsigned long long size = fftArrLen * sizeof(Complex);
  CUDA_CHECK(cudaMemcpy(myfftArr_h, myfftArr_d, size, cudaMemcpyDeviceToHost));
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
  printf("=== GPU FFT v1 vs cuFFT Benchmark ===\n"
         "FFT Array Length: %12llu\n", fftArrLen);

  // Pointers to the arrays on host and the device
  Complex *cufftArr_h = NULL, *cufftArr_d = NULL;
  Complex *myfftArr_h = NULL, *myfftArr_d = NULL;


  /* cuFFT Run */
  printf("\n--- cuFFT Run ---\n");
  
  // Allocate the host and device arrays
  init_input_signal(fftArrLen, &cufftArr_h, &cufftArr_d);

  // Run cuFFT
  cufftRun(fftArrLen, cufftArr_h, cufftArr_d);


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
    if (myfftArr_h) free(myfftArr_h);
    if (myfftArr_d) cudaFree(myfftArr_d);
    init_input_signal(fftArrLen, &myfftArr_h, &myfftArr_d);

    // run myFFT
    fft_1d_gpu(fftArrLen, myfftArr_h, myfftArr_d, nthreadsPerBlock, nblocks);

    // Find difference between cufftArr and mtfftArr
    double err = findDiff(fftArrLen, cufftArr_h, myfftArr_h);
    printf("Diff:%e\n", err);
  }

  if (cufftArr_h) free(cufftArr_h);
  if (cufftArr_d) cudaFree(cufftArr_d);
  if (myfftArr_h) free(myfftArr_h);
  if (myfftArr_d) cudaFree(myfftArr_d);

  return 0;
}
#pragma once

#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
	    printf("Error at %s:%d\n",__FILE__,__LINE__);\
	    exit(EXIT_FAILURE);}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
	    printf("Error at %s:%d\n",__FILE__,__LINE__);\
	    exit(EXIT_FAILURE);}} while(0)
#define CUBLAS_CALL(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { \
	    printf("Error at %s:%d\n",__FILE__,__LINE__);\
	    exit(EXIT_FAILURE);}} while(0)


namespace bcpnn {

namespace helpers {

namespace cuda {

template<typename REAL>
cublasStatus_t
cublasgemm(
  cublasHandle_t handle,
  cublasOperation_t transa,
  cublasOperation_t transb,
  int m, int n, int k,
  const REAL *alpha,
  const REAL *A, int lda,
  const REAL *B, int ldb,
  const REAL *beta,
  REAL *C, int ldc
)
{ }

template<>
cublasStatus_t
cublasgemm<float>(
  cublasHandle_t handle,
  cublasOperation_t transa,
  cublasOperation_t transb,
  int m, int n, int k,
  const float *alpha,
  const float *A, int lda,
  const float *B, int ldb,
  const float *beta,
  float *C, int ldc
);

template<>
cublasStatus_t
cublasgemm<double>(
  cublasHandle_t handle,
  cublasOperation_t transa,
  cublasOperation_t transb,
  int m, int n, int k,
  const double *alpha,
  const double *A, int lda,
  const double *B, int ldb,
  const double *beta,
  double *C, int ldc
);

template<typename T>
curandStatus_t
CURANDAPI
TcurandGenerateUniform(curandGenerator_t generator,T* outputPtr, size_t num) {}

template<>
curandStatus_t
CURANDAPI
TcurandGenerateUniform<float>(curandGenerator_t generator, float* outputPtr, size_t num);

template<>
curandStatus_t
CURANDAPI
TcurandGenerateUniform<double>(curandGenerator_t generator, double* outputPtr, size_t num);

template<typename T>
curandStatus_t
CURANDAPI
TcurandGenerateNormal( curandGenerator_t generator, T* outputPtr, size_t n, T  mean, T  stddev ) {}

template<>
curandStatus_t
CURANDAPI
TcurandGenerateNormal<float>( curandGenerator_t generator, float* outputPtr, size_t n, float  mean, float  stddev );

template<>
curandStatus_t
CURANDAPI
TcurandGenerateNormal<double>( curandGenerator_t generator, double* outputPtr, size_t n, double  mean, double  stddev );


template<typename REAL>
__global__
void
kernel_initialize(REAL * p, REAL value, size_t n)
{
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    p[i] = value;
  }
}

template<typename REAL>
void
cuda_initialize_array(REAL * p, REAL value, size_t n)
{
  kernel_initialize<<<(n + 256 - 1)/256, 256>>>(p, value, n);
}

template<typename REAL>
__global__
void
kernel_scale_array(REAL * p, REAL factor, size_t n)
{
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    p[i] *= factor;
  }
}

template<typename REAL>
void
cuda_scale_array(REAL * p, REAL factor, size_t n)
{
  kernel_scale_array<<<(n + 256 - 1)/256, 256>>>(p, factor, n);
}

template<typename REAL>
__global__
void
kernel_correct_predictions(int * correct_count, REAL * predictions, REAL * one_hot_labels, size_t n, size_t m)
{
  int correct = 0;

  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    size_t jmax = 0;
    REAL  vmax = predictions[i * m + 0];
    for (size_t j = 1; j < m; ++j) {
      if (predictions[i * m + j] > vmax) {
        jmax = j;
	vmax = predictions[i * m + j];
      }
    }

    if (one_hot_labels[i * m + jmax] > 0.5) {
//    if (one_hot_labels[i * m + jmax] == 1) {
      correct += 1;
    }
  }

  for (size_t d = 16; d > 0; d /= 2) {
    correct += __shfl_down_sync(~0, correct, d);
  }

  if (threadIdx.x % 32 == 0) {
    atomicAdd(correct_count, correct);
  }
}

template<typename REAL>
void
cuda_correct_predictions(int * correct_count, REAL * predictions, REAL * one_hot_labels, size_t n, size_t m)
{
  kernel_correct_predictions<<<(n + 64 - 1)/64, 64>>>(correct_count, predictions, one_hot_labels, n, m);
}

} // namespace cuda

} // namespace helpers

} // namespace bcpnn

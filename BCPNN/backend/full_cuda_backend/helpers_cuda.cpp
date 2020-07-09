#include "helpers_cuda.h"

namespace bcpnn {

namespace helpers {

namespace cuda {

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
)
{
  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

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
)
{
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


template<>
curandStatus_t
CURANDAPI
TcurandGenerateUniform<float>(curandGenerator_t generator,float* outputPtr, size_t num) {
  return curandGenerateUniform(generator, outputPtr, num);
}

template<>
curandStatus_t
CURANDAPI
TcurandGenerateUniform<double>(curandGenerator_t generator,double* outputPtr, size_t num) {
  return curandGenerateUniformDouble(generator, outputPtr, num);
}


template<>
curandStatus_t
CURANDAPI
TcurandGenerateNormal<float>( curandGenerator_t generator, float* outputPtr, size_t n, float  mean, float  stddev )
{
  return curandGenerateNormal(generator, outputPtr, n, mean, stddev);
}

template<>
curandStatus_t
CURANDAPI
TcurandGenerateNormal<double>( curandGenerator_t generator, double* outputPtr, size_t n, double  mean, double  stddev )
{
  return curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
}

} // namespace cuda

} // namespace helpers

} // namespace bcpnn

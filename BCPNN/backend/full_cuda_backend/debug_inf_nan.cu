//#define CHECK_INF_NAN(a, s) do { int x; cuda_check_inf_nan(inf_nan_res, (a),
//(s)); CUDA_CALL(cudaMemcpy(&x, inf_nan_res, sizeof(int),
//cudaMemcpyDeviceToHost)); if (x) { printf("inf or nan at
//%s:%d\n",__FILE__,__LINE__); exit(EXIT_FAILURE);} } while(0)
#define CHECK_INF_NAN(a, s)

int *inf_nan_res;

void init() {
  CUDA_CALL(cudaMalloc((void **)&inf_nan_res, sizeof(int)));
  CUDA_CALL(cudaMemset(inf_nan_res, 0, sizeof(int)));
}

__global__ void kernel_check_inf_nan(int *r, float *a, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    if (isinf(a[i]) || isnan(a[i])) {
      *r = 1;
    }
  }
}

void cuda_check_inf_nan(int *r, float *a, size_t n) {
  kernel_check_inf_nan<<<(n + 256 - 1) / 256, 256>>>(r, a, n);
}

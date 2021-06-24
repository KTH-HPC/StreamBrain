#include <cuda.h>
#include <cuda_runtime.h>

namespace bcpnn {

namespace kernels {

namespace cuda {

template <typename REAL>
__global__ void kernel_add_bias(REAL *matrix, size_t n, size_t m, REAL *bias) {
  size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n && j < m) {
    matrix[i * m + j] += bias[j];
  }
}

template <typename REAL>
void cuda_add_bias(REAL *matrix, size_t n, size_t m, REAL *bias) {
  dim3 block;
  block.x = 16;
  block.y = 16;

  dim3 grid;
  grid.x = (m + block.x - 1) / block.x;
  grid.y = (n + block.y - 1) / block.y;

  kernel_add_bias<REAL><<<grid, block>>>(matrix, n, m, bias);
}

template <typename REAL>
__global__ void kernel_softmax_medium(REAL *matrix, size_t n, size_t m) {
  size_t j = threadIdx.x;  // Only a single block
  size_t i = blockIdx.y * blockDim.y + threadIdx.y;

  REAL max_;
  REAL sum;

  if (i < n) {
    // Find max
    if (j < m) {
      max_ = matrix[i * m + j];
    } else {
      max_ = matrix[i * m + m - 1];
    }

    for (size_t jj = j + blockDim.x; jj < m; jj += blockDim.x) {
      max_ = fmaxf(max_, matrix[i * m + jj]);
    }

    for (size_t d = 16; d > 0; d /= 2) {
      REAL up = __shfl_down_sync(~0, max_, d);
      max_ = fmaxf(max_, up);
    }

    max_ = __shfl_sync(~0, max_, 0);

    // Max found, now subtract max and exponentiate entries while tracking the
    // sum

    sum = 0;
    for (size_t jj = j; jj < m; jj += blockDim.x) {
      sum += (matrix[i * m + jj] = expf(matrix[i * m + jj] - max_));
    }

    for (size_t d = 16; d > 0; d /= 2) {
      REAL up = __shfl_down_sync(~0, sum, d);
      sum += up;
    }

    sum = __shfl_sync(~0, sum, 0);

    for (size_t jj = j; jj < m; jj += blockDim.x) {
      matrix[i * m + jj] /= sum;
    }
  }
}

template <typename REAL>
void cuda_softmax(REAL *matrix, size_t n, size_t m) {
  // cuda_subtract_max(matrix, n, m);
  // cuda_exponentiate(matrix, n, m);
  // cuda_normalize(matrix, n, m);
  dim3 grid;
  dim3 block;

  // Each warp operates on exactly one row of the matrix
  block.x = 32;
  block.y = 8;
  grid.x = 1;
  grid.y = (n + 8 - 1) / 8;

  kernel_softmax_medium<REAL><<<grid, block>>>(matrix, n, m);
}

template <typename REAL>
__global__ void kernel_update_1d_counter(REAL *C, REAL *activations,
                                         size_t batch_size, size_t n,
                                         REAL taupdt) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  REAL tmp = 0;

  if (i < n) {
    for (size_t b = 0; b < batch_size; ++b) {
      tmp += activations[b * n + i];
    }

    C[i] = fmaf(1 - taupdt, C[i], tmp / batch_size * taupdt);
  }
}

template <typename REAL>
__global__ void kernel_update_2d_counter(REAL *C, REAL *activations_1,
                                         REAL *activations_2, size_t batch_size,
                                         size_t n, size_t m, REAL taupdt) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  REAL tmp = 0;

  if (i < n && j < m) {
    for (size_t b = 0; b < batch_size; ++b) {
      tmp += (activations_1[b * n + i] * activations_2[b * m + j]);
    }

    C[i * m + j] = fmaf(1 - taupdt, C[i * m + j], tmp / batch_size * taupdt);
  }
}

template <typename REAL>
void cuda_update_counters(REAL *Ci, REAL *Cj, REAL *Cij, REAL *inputs,
                          REAL *outputs, size_t batch_size, size_t n, size_t m,
                          REAL taupdt) {
  dim3 block_Ci(256);
  dim3 grid_Ci((n + 256 - 1) / 256);

  dim3 block_Cj(256);
  dim3 grid_Cj((m + 256 - 1) / 256);

  dim3 block_Cij(16, 16);
  dim3 grid_Cij((n + 16 - 1) / 16, (m + 16 - 1) / 16);

  cudaStream_t stream_Ci, stream_Cj;
  cudaEvent_t event_Ci, event_Cj;

  cudaStreamCreate(&stream_Ci);
  cudaStreamCreate(&stream_Cj);
  cudaEventCreate(&event_Ci);
  cudaEventCreate(&event_Cj);

  kernel_update_1d_counter<REAL>
      <<<grid_Ci, block_Ci, 0, stream_Ci>>>(Ci, inputs, batch_size, n, taupdt);
  kernel_update_1d_counter<REAL>
      <<<grid_Cj, block_Cj, 0, stream_Cj>>>(Cj, outputs, batch_size, m, taupdt);
  kernel_update_2d_counter<REAL>
      <<<grid_Cij, block_Cij>>>(Cij, inputs, outputs, batch_size, n, m, taupdt);

  cudaEventRecord(event_Ci, stream_Ci);
  cudaEventRecord(event_Cj, stream_Cj);

  cudaStreamWaitEvent(0, event_Ci, 0);
  cudaStreamWaitEvent(0, event_Cj, 0);

  cudaEventDestroy(event_Ci);
  cudaEventDestroy(event_Cj);
  cudaStreamDestroy(stream_Ci);
  cudaStreamDestroy(stream_Cj);
}

template <typename REAL>
__global__ void kernel_update_weights(REAL *weights, REAL *Ci, REAL *Cj,
                                      REAL *Cij, REAL cthr, size_t n,
                                      size_t m) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  size_t idx = i * m + j;

  if (i < n && j < m) {
    if (Ci[i] < cthr || Cj[j] < cthr) {
      weights[idx] = 0;
    } else {
#if 0
      // Original
      weights[idx] = logf(Cij[idx] / (Ci[i] * Cj[j]));
#else
      // Try to get rid of nans
      REAL x = fmaxf(Cij[idx], cthr * cthr * cthr * cthr);
      weights[idx] = logf(x / (Ci[i] * Cj[j]));
#endif
    }
  }
}

template <typename REAL>
void cuda_update_weights(REAL *weights, REAL *Ci, REAL *Cj, REAL *Cij,
                         REAL cthr, size_t n, size_t m) {
  dim3 block(16, 16);
  dim3 grid((n + 16 - 1) / 16, (m + 16 - 1) / 16);

  kernel_update_weights<REAL>
      <<<grid, block>>>(weights, Ci, Cj, Cij, cthr, n, m);
}

template <typename REAL>
__global__ void kernel_update_bias(REAL *bias, REAL *Cj, REAL cthr, size_t m) {
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j < m) {
    if (Cj[j] < cthr) {
      bias[j] = logf(2 * cthr);
    } else {
      bias[j] = logf(Cj[j]);
    }
  }
}

template <typename REAL>
void cuda_update_bias(REAL *bias, REAL *Cj, REAL cthr, size_t m) {
  dim3 block(256);
  dim3 grid((m + 256 - 1) / 256);

  kernel_update_bias<REAL><<<grid, block>>>(bias, Cj, cthr, m);
}

template <typename REAL>
__global__ void kernel_update_bias_regularized(REAL *bias, REAL *kbi, REAL *Cj,
                                               REAL cthr, REAL khalf, REAL pmin,
                                               REAL taubdt, size_t m) {
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j < m) {
    REAL k = (khalf - 1) * (pmin / 4) * (pmin / 4);
    REAL pj = Cj[j];

    REAL kb = 1 + k / ((pj - pmin / 4) * (pj - pmin / 4));
    if (pj <= pmin / 4 || kb < khalf) {
      kb = khalf;
    }

    kbi[j] = (1 - taubdt) * kbi[j] + taubdt * kb;

    if (Cj[j] < cthr) {
      bias[j] = kbi[j] * logf(2 * cthr);
    } else {
      bias[j] = kbi[j] * logf(Cj[j]);
    }
  }
}

template <typename REAL>
void cuda_update_bias_regularized(REAL *bias, REAL *kbi, REAL *Cj, REAL cthr,
                                  REAL khalf, REAL pmin, REAL taubdt,
                                  size_t m) {
  dim3 block(256);
  dim3 grid((m + 256 - 1) / 256);

  kernel_update_bias_regularized<REAL>
      <<<grid, block>>>(bias, kbi, Cj, cthr, khalf, pmin, taubdt, m);
}

template <typename REAL>
__global__ void kernel_compute_wmask_score_nominator(
    REAL *wmask_score_nominator, REAL *weights, REAL *Ci, REAL *Cj, REAL *Cij,
    REAL cthr, size_t n, size_t m, size_t h, size_t hypercolumns,
    size_t minicolumns) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    REAL score = 0;
    for (size_t j = h * minicolumns; j < (h + 1) * minicolumns; ++j) {
      if (Ci[i] >= cthr && Cj[j] >= cthr) {
        REAL pi = Ci[i];
        REAL pj = Cj[j];
        REAL pij = Cij[i * m + j];

        REAL x = (1 - pi) * pj;
        REAL y = (pj - pij) / x;
        REAL WijC = logf(y);

        score += pij * weights[i * m + j] + (pj - pij) * WijC;
      }
      wmask_score_nominator[i] = score;
    }
  }
}

template <typename REAL>
__global__ void kernel_compute_wmask_csum(int *wmask_csum, uint8_t *wmask,
                                          size_t n, size_t m,
                                          size_t hypercolumns,
                                          size_t minicolumns) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    int sum = 0;
    for (size_t j = 0; j < hypercolumns; ++j) {
      sum += wmask[i * hypercolumns + j];
    }
    wmask_csum[i] = sum;
  }
}

constexpr size_t BANK_SIZE = 1;

template <typename REAL>
__global__ void kernel_update_wmask(REAL *wmask_score_nominator,
                                    int *wmask_csum, uint8_t *wmask, size_t n,
                                    size_t m, size_t h, size_t hypercolumns,
                                    size_t minicolumns, size_t iterations) {
  extern __shared__ uint8_t swmask[];
  int cntnue = 1;

  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    swmask[i * BANK_SIZE] = wmask[i * hypercolumns + h];
  }

  for (size_t iter = 0; cntnue && iter < iterations; ++iter) {
    size_t imax0 = n;
    REAL vmax0 = 0;

    size_t imin1 = n;
    REAL vmin1 = 0;

    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
      REAL score = wmask_score_nominator[i] / (1 + wmask_csum[i]);
      if (swmask[i * BANK_SIZE] == 0 && (imax0 == n || score >= vmax0)) {
        imax0 = i;
        vmax0 = score;
      } else if (swmask[i * BANK_SIZE] == 1 && (imin1 == n || score <= vmin1)) {
        imin1 = i;
        vmin1 = score;
      }
    }

    for (size_t d = 16; d > 0; d /= 2) {
      size_t iup = __shfl_down_sync(~0, imax0, d);
      REAL vup = __shfl_down_sync(~0, vmax0, d);

      if (imax0 == n || (iup != n && vup >= vmax0)) {
        imax0 = iup;
        vmax0 = vup;
      }

      iup = __shfl_down_sync(~0, imin1, d);
      vup = __shfl_down_sync(~0, vmin1, d);

      if (imin1 == n || (iup != n && vup <= vmin1)) {
        imin1 = iup;
        vmin1 = vup;
      }
    }

    // Insert shared memory stuff if we want larger block

    if (threadIdx.x == 0) {
      if (imax0 == n || imin1 == n || vmax0 < vmin1) {
        cntnue = 0;
      } else {
        // printf("GPU: Swapping %ld (%f) with %ld (%f)\n", imax0, vmax0, imin1,
        // vmin1);
        swmask[imax0 * BANK_SIZE] = 1;
        swmask[imin1 * BANK_SIZE] = 0;

        wmask_csum[imax0] += 1;
        wmask_csum[imin1] -= 1;
      }
    }

    cntnue = __shfl_sync(~0, cntnue, 0);
  }

  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    wmask[i * hypercolumns + h] = swmask[i * BANK_SIZE];
  }
}

template <typename REAL>
void cuda_update_mask(uint8_t *wmask, REAL *weights, REAL *Ci, REAL *Cj,
                      REAL *Cij, REAL cthr, size_t n, size_t m, size_t h,
                      size_t hypercolumns, size_t minicolumns,
                      size_t iterations) {
  dim3 block_compute(256);
  dim3 grid_compute((n + 256 - 1) / 256);

  dim3 block_update(32);
  dim3 grid_update(1);

  REAL *wmask_score_nominator;
  int *wmask_csum;

  CUDA_CALL(cudaMalloc((void **)&wmask_score_nominator, n * sizeof(REAL)));
  CUDA_CALL(cudaMalloc((void **)&wmask_csum, n * sizeof(int)));

  kernel_compute_wmask_score_nominator<REAL><<<grid_compute, block_compute>>>(
      wmask_score_nominator, weights, Ci, Cj, Cij, cthr, n, m, h, hypercolumns,
      minicolumns);
  kernel_compute_wmask_csum<REAL><<<grid_compute, block_compute>>>(
      wmask_csum, wmask, n, m, hypercolumns, minicolumns);

  kernel_update_wmask<REAL><<<grid_update, block_update, n * BANK_SIZE>>>(
      wmask_score_nominator, wmask_csum, wmask, n, m, h, hypercolumns,
      minicolumns, iterations);

  CUDA_CALL(cudaFree(wmask_score_nominator));
  CUDA_CALL(cudaFree(wmask_csum));
}

template <typename REAL>
__global__ void kernel_apply_mask(REAL *weight, uint8_t *wmask, size_t n,
                                  size_t m, size_t hypercolumns,
                                  size_t minicolumns) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  size_t j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < n && j < m) {
    size_t h = j / minicolumns;

    if (!wmask[i * hypercolumns + h]) {
      weight[i * m + j] = 0;
    }
  }
}

template <typename REAL>
void cuda_apply_mask(REAL *weight, uint8_t *wmask, size_t n, size_t m,
                     size_t hypercolumns, size_t minicolumns) {
  // TOODO: Assert m == hypercolumns * minicolumns
  dim3 block(16, 16);
  dim3 grid((n + 16 - 1) / 16, (m + 16 - 1) / 16);

  kernel_apply_mask<REAL>
      <<<grid, block>>>(weight, wmask, n, m, hypercolumns, minicolumns);
}

}  // namespace cuda

}  // namespace kernels

}  // namespace bcpnn

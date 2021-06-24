#include <cuda.h>

#include "helpers_cuda.h"
#include "helpers_random.h"
#include "kernels_cuda.h"
#include "kernels_naive_cpu.h"

using REAL = float;

using namespace bcpnn::kernels::cuda;
using namespace bcpnn::kernels::naive_cpu;

float test_add_bias() {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> d_size(1, 1024);
  std::uniform_real_distribution<float> d_entry(0, 1);

  bcpnn::helpers::random::seed_generator(generator);

  size_t n = d_size(generator);
  size_t m = d_size(generator);

  float *matrix = (float *)malloc(n * m * sizeof(float));
  float *bias = (float *)malloc(m * sizeof(float));

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      matrix[i * m + j] = d_entry(generator);
    }
  }

  for (size_t j = 0; j < m; ++j) {
    bias[j] = d_entry(generator);
  }

  float *d_matrix;
  float *d_bias;

  CUDA_CALL(cudaMalloc((void **)&d_matrix, n * m * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&d_bias, m * sizeof(float)));

  CUDA_CALL(cudaMemcpy(d_matrix, matrix, n * m * sizeof(float),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(
      cudaMemcpy(d_bias, bias, m * sizeof(float), cudaMemcpyHostToDevice));

  naive_add_bias<REAL>(matrix, n, m, bias);
  cuda_add_bias<REAL>(d_matrix, n, m, d_bias);

  float *h_matrix = (float *)malloc(n * m * sizeof(float));
  float *h_bias = (float *)malloc(m * sizeof(float));

  CUDA_CALL(cudaMemcpy(h_matrix, d_matrix, n * m * sizeof(float),
                       cudaMemcpyDeviceToHost));
  CUDA_CALL(
      cudaMemcpy(h_bias, d_bias, m * sizeof(float), cudaMemcpyDeviceToHost));

  float delta_max = 0;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      size_t idx = i * m + j;
      delta_max = fmaxf(delta_max, fabsf(matrix[idx] - h_matrix[idx]));
    }
  }

  free(matrix);
  free(bias);

  CUDA_CALL(cudaFree(d_matrix));
  CUDA_CALL(cudaFree(d_bias));

  free(h_matrix);
  free(h_bias);

  return delta_max;
}

float test_softmax() {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> d_size(1, 1024);
  std::uniform_real_distribution<float> d_entry(-100, 100);

  bcpnn::helpers::random::seed_generator(generator);

  size_t n = d_size(generator);
  size_t m = d_size(generator);
  // std::cout << "Size: " << n << ", " << m << std::endl;

  float *matrix = (float *)malloc(n * m * sizeof(float));

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      matrix[i * m + j] = d_entry(generator);

#if 0
      std::cout << matrix[i * m + j] << "\t";
#endif
    }
#if 0
    std::cout << std::endl;
#endif
  }

  float *d_matrix;

  CUDA_CALL(cudaMalloc((void **)&d_matrix, n * m * sizeof(float)));

  CUDA_CALL(cudaMemcpy(d_matrix, matrix, n * m * sizeof(float),
                       cudaMemcpyHostToDevice));

  naive_softmax<REAL>(matrix, n, m);
  cuda_softmax<REAL>(d_matrix, n, m);

  float *h_matrix = (float *)malloc(n * m * sizeof(float));

  CUDA_CALL(cudaMemcpy(h_matrix, d_matrix, n * m * sizeof(float),
                       cudaMemcpyDeviceToHost));

  float delta_max = 0;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      size_t idx = i * m + j;
      delta_max = fmaxf(delta_max, fabsf(matrix[idx] - h_matrix[idx]));
    }
  }

#if 0
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      size_t idx = i * m + j;
      std::cout << matrix[idx] << "\t";
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;
  std::cout << std::endl;

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      size_t idx = i * m + j;
      std::cout << h_matrix[idx] << "\t";
    }
    std::cout << std::endl;
  }
#endif

  free(matrix);

  CUDA_CALL(cudaFree(d_matrix));

  free(h_matrix);

  return delta_max;
}

float test_update_counters() {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> d_size(1, 1024);
  std::uniform_real_distribution<float> d_entry(0.001, 1);
  std::uniform_int_distribution<int> d_zero(0, 9);
  std::uniform_int_distribution<int> d_taupdt(1e-7, 0.1);

  bcpnn::helpers::random::seed_generator(generator);

  size_t n = d_size(generator);
  size_t m = d_size(generator);
  size_t batch_size = d_size(generator);
  float taupdt = d_taupdt(generator);
  // std::cout << "Size: " << n << ", " << m << std::endl;

  float *Ci = (float *)malloc(n * sizeof(float));
  float *Cj = (float *)malloc(m * sizeof(float));
  float *Cij = (float *)malloc(n * m * sizeof(float));
  float *inputs = (float *)malloc(batch_size * n * sizeof(float));
  float *outputs = (float *)malloc(batch_size * m * sizeof(float));

  for (size_t i = 0; i < n; ++i) {
    if (d_zero(generator) == 0) {
      Ci[i] = 0;
    } else {
      Ci[i] = d_entry(generator);
    }
  }

  for (size_t j = 0; j < m; ++j) {
    if (d_zero(generator) == 0) {
      Cj[j] = 0;
    } else {
      Cj[j] = d_entry(generator);
    }
  }

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      if (d_zero(generator) == 0) {
        Cij[i * m + j] = 0;
      } else {
        Cij[i * m + j] = d_entry(generator);
      }
    }
  }

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t i = 0; i < n; ++i) {
      inputs[b * n + i] = d_entry(generator);
    }

    for (size_t j = 0; j < m; ++j) {
      outputs[b * m + j] = d_entry(generator);
    }
  }

  float *d_Ci;
  float *d_Cj;
  float *d_Cij;
  float *d_inputs;
  float *d_outputs;

  CUDA_CALL(cudaMalloc((void **)&d_Ci, n * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&d_Cj, m * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&d_Cij, n * m * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&d_inputs, batch_size * n * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&d_outputs, batch_size * m * sizeof(float)));

  CUDA_CALL(cudaMemcpy(d_Ci, Ci, n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_Cj, Cj, m * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(
      cudaMemcpy(d_Cij, Cij, n * m * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_inputs, inputs, batch_size * n * sizeof(float),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_outputs, outputs, batch_size * m * sizeof(float),
                       cudaMemcpyHostToDevice));

  naive_update_counters<REAL>(Ci, Cj, Cij, inputs, outputs, batch_size, n, m,
                              taupdt);
  cuda_update_counters<REAL>(d_Ci, d_Cj, d_Cij, d_inputs, d_outputs, batch_size,
                             n, m, taupdt);

  float *h_Ci = (float *)malloc(n * sizeof(float));
  float *h_Cj = (float *)malloc(m * sizeof(float));
  float *h_Cij = (float *)malloc(n * m * sizeof(float));
  float *h_inputs = (float *)malloc(batch_size * n * sizeof(float));
  float *h_outputs = (float *)malloc(batch_size * m * sizeof(float));

  CUDA_CALL(cudaMemcpy(h_Ci, d_Ci, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(h_Cj, d_Cj, m * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(
      cudaMemcpy(h_Cij, d_Cij, n * m * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(h_inputs, d_inputs, batch_size * n * sizeof(float),
                       cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(h_outputs, d_outputs, batch_size * m * sizeof(float),
                       cudaMemcpyDeviceToHost));

  float delta_max = 0;
  for (size_t i = 0; i < n; ++i) {
    delta_max = fmaxf(delta_max, fabsf(Ci[i] - h_Ci[i]));
  }

  for (size_t j = 0; j < m; ++j) {
    delta_max = fmaxf(delta_max, fabsf(Cj[j] - h_Cj[j]));
  }

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      size_t idx = i * m + j;
      delta_max = fmaxf(delta_max, fabsf(Cij[idx] - h_Cij[idx]));
    }
  }

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t i = 0; i < n; ++i) {
      size_t idx = b * n + i;
      delta_max = fmaxf(delta_max, fabsf(inputs[idx] - h_inputs[idx]));
    }

    for (size_t j = 0; j < m; ++j) {
      size_t idx = b * m + j;
      delta_max = fmaxf(delta_max, fabsf(outputs[idx] - h_outputs[idx]));
    }
  }

#if 0
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      size_t idx = i * m + j;
      std::cout << matrix[idx] << "\t";
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;
  std::cout << std::endl;

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      size_t idx = i * m + j;
      std::cout << h_matrix[idx] << "\t";
    }
    std::cout << std::endl;
  }
#endif

  free(Ci);
  free(Cj);
  free(Cij);
  free(inputs);
  free(outputs);

  CUDA_CALL(cudaFree(d_Ci));
  CUDA_CALL(cudaFree(d_Cj));
  CUDA_CALL(cudaFree(d_Cij));
  CUDA_CALL(cudaFree(d_inputs));
  CUDA_CALL(cudaFree(d_outputs));

  free(h_Ci);
  free(h_Cj);
  free(h_Cij);
  free(h_inputs);
  free(h_outputs);

  return delta_max;
}

float test_update_weights() {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> d_size(1, 1024);
  std::uniform_real_distribution<float> d_entry(0.001, 1);
  std::uniform_int_distribution<int> d_zero(0, 9);
  std::uniform_int_distribution<int> d_taupdt(1e-3, 0.1);

  bcpnn::helpers::random::seed_generator(generator);

  size_t n = d_size(generator);
  size_t m = d_size(generator);
  float cthr = d_taupdt(generator) / 2;
  // std::cout << "Size: " << n << ", " << m << std::endl;

  float *weights = (float *)malloc(n * m * sizeof(float));
  float *Ci = (float *)malloc(n * sizeof(float));
  float *Cj = (float *)malloc(m * sizeof(float));
  float *Cij = (float *)malloc(n * m * sizeof(float));

  for (size_t i = 0; i < n; ++i) {
    if (d_zero(generator) == 0) {
      Ci[i] = 0;
    } else {
      Ci[i] = d_entry(generator);
    }
  }

  for (size_t j = 0; j < m; ++j) {
    if (d_zero(generator) == 0) {
      Cj[j] = 0;
    } else {
      Cj[j] = d_entry(generator);
    }
  }

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      if (d_zero(generator) == 0) {
        Cij[i * m + j] = 0;
      } else {
        Cij[i * m + j] = d_entry(generator);
      }
    }
  }

  float *d_weights;
  float *d_Ci;
  float *d_Cj;
  float *d_Cij;

  CUDA_CALL(cudaMalloc((void **)&d_weights, n * m * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&d_Ci, n * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&d_Cj, m * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&d_Cij, n * m * sizeof(float)));

  CUDA_CALL(cudaMemcpy(d_Ci, Ci, n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_Cj, Cj, m * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(
      cudaMemcpy(d_Cij, Cij, n * m * sizeof(float), cudaMemcpyHostToDevice));

  naive_update_weights<REAL>(weights, Ci, Cj, Cij, cthr, n, m);
  cuda_update_weights<REAL>(d_weights, d_Ci, d_Cj, d_Cij, cthr, n, m);

  float *h_weights = (float *)malloc(n * m * sizeof(float));

  CUDA_CALL(cudaMemcpy(h_weights, d_weights, n * m * sizeof(float),
                       cudaMemcpyDeviceToHost));

  float delta_max = 0;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      size_t idx = i * m + j;
      delta_max = fmaxf(delta_max, fabsf(weights[idx] - h_weights[idx]));
    }
  }

#if 0
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      size_t idx = i * m + j;
      std::cout << matrix[idx] << "\t";
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;
  std::cout << std::endl;

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      size_t idx = i * m + j;
      std::cout << h_matrix[idx] << "\t";
    }
    std::cout << std::endl;
  }
#endif

  free(weights);
  free(Ci);
  free(Cj);
  free(Cij);

  CUDA_CALL(cudaFree(d_weights));
  CUDA_CALL(cudaFree(d_Ci));
  CUDA_CALL(cudaFree(d_Cj));
  CUDA_CALL(cudaFree(d_Cij));

  free(h_weights);

  return delta_max;
}

float test_update_bias() {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> d_size(1, 1024);
  std::uniform_real_distribution<float> d_entry(0.001, 1);
  std::uniform_int_distribution<int> d_zero(0, 9);
  std::uniform_int_distribution<int> d_taupdt(1e-3, 0.1);

  bcpnn::helpers::random::seed_generator(generator);

  size_t m = d_size(generator);
  float cthr = d_taupdt(generator) / 2;
  // std::cout << "Size: " << n << ", " << m << std::endl;

  float *bias = (float *)malloc(m * sizeof(float));
  float *Cj = (float *)malloc(m * sizeof(float));

  for (size_t j = 0; j < m; ++j) {
    if (d_zero(generator) == 0) {
      Cj[j] = 0;
    } else {
      Cj[j] = d_entry(generator);
    }
  }

  float *d_bias;
  float *d_Cj;

  CUDA_CALL(cudaMalloc((void **)&d_bias, m * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&d_Cj, m * sizeof(float)));

  CUDA_CALL(cudaMemcpy(d_Cj, Cj, m * sizeof(float), cudaMemcpyHostToDevice));

  naive_update_bias<REAL>(bias, Cj, cthr, m);
  cuda_update_bias<REAL>(d_bias, d_Cj, cthr, m);

  float *h_bias = (float *)malloc(m * sizeof(float));

  CUDA_CALL(
      cudaMemcpy(h_bias, d_bias, m * sizeof(float), cudaMemcpyDeviceToHost));

  float delta_max = 0;
  for (size_t j = 0; j < m; ++j) {
    delta_max = fmaxf(delta_max, fabsf(bias[j] - h_bias[j]));
  }

#if 0
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      size_t idx = i * m + j;
      std::cout << matrix[idx] << "\t";
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;
  std::cout << std::endl;

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      size_t idx = i * m + j;
      std::cout << h_matrix[idx] << "\t";
    }
    std::cout << std::endl;
  }
#endif

  free(bias);
  free(Cj);

  CUDA_CALL(cudaFree(d_bias));
  CUDA_CALL(cudaFree(d_Cj));

  free(h_bias);

  return delta_max;
}

float test_update_bias_regularized() {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> d_size(1, 1024);
  std::uniform_real_distribution<float> d_entry(0.001, 1);
  std::uniform_real_distribution<float> _d_kbi(-10, 10);
  std::uniform_real_distribution<float> d_khalf(-1000, 0);
  std::uniform_real_distribution<float> d_pmin(0.01, 0.5);
  std::uniform_int_distribution<int> d_zero(0, 9);
  std::uniform_int_distribution<int> d_taupdt(1e-3, 0.1);

  bcpnn::helpers::random::seed_generator(generator);

  size_t m = d_size(generator);
  float cthr = d_taupdt(generator) / 2;
  float khalf = d_khalf(generator);
  float pmin = d_pmin(generator);
  float taubdt = d_taupdt(generator);
  // std::cout << "Size: " << n << ", " << m << std::endl;

  float *bias = (float *)malloc(m * sizeof(float));
  float *kbi = (float *)malloc(m * sizeof(float));
  float *Cj = (float *)malloc(m * sizeof(float));

  for (size_t j = 0; j < m; ++j) {
    if (d_zero(generator) == 0) {
      Cj[j] = 0;
    } else {
      Cj[j] = d_entry(generator);
    }

    kbi[j] = _d_kbi(generator);
  }

  float *d_bias;
  float *d_kbi;
  float *d_Cj;

  CUDA_CALL(cudaMalloc((void **)&d_bias, m * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&d_kbi, m * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&d_Cj, m * sizeof(float)));

  CUDA_CALL(cudaMemcpy(d_kbi, kbi, m * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_Cj, Cj, m * sizeof(float), cudaMemcpyHostToDevice));

  naive_update_bias_regularized<REAL>(bias, kbi, Cj, cthr, khalf, pmin, taubdt,
                                      m);
  cuda_update_bias_regularized<REAL>(d_bias, d_kbi, d_Cj, cthr, khalf, pmin,
                                     taubdt, m);

  float *h_bias = (float *)malloc(m * sizeof(float));
  float *h_kbi = (float *)malloc(m * sizeof(float));

  CUDA_CALL(
      cudaMemcpy(h_bias, d_bias, m * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(
      cudaMemcpy(h_kbi, d_kbi, m * sizeof(float), cudaMemcpyDeviceToHost));

  float delta_max = 0;
  for (size_t j = 0; j < m; ++j) {
    delta_max = fmaxf(delta_max, fabsf(bias[j] - h_bias[j]));
    delta_max = fmaxf(delta_max, fabsf(kbi[j] - h_kbi[j]));
  }

#if 0
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      size_t idx = i * m + j;
      std::cout << matrix[idx] << "\t";
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;
  std::cout << std::endl;

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      size_t idx = i * m + j;
      std::cout << h_matrix[idx] << "\t";
    }
    std::cout << std::endl;
  }
#endif

  free(bias);
  free(kbi);
  free(Cj);

  CUDA_CALL(cudaFree(d_bias));
  CUDA_CALL(cudaFree(d_kbi));
  CUDA_CALL(cudaFree(d_Cj));

  free(h_bias);
  free(h_kbi);

  return delta_max;
}

float test_update_mask() {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> d_size(1, 1024);
  std::uniform_real_distribution<float> d_counter_entry(0.001, 0.999);
  std::uniform_int_distribution<int> d_mask_entry(0, 4);
  std::uniform_real_distribution<float> d_weight_entry(-10, 10);
  std::uniform_int_distribution<int> d_zero(0, 9);
  std::uniform_real_distribution<float> d_taupdt(1e-7, 0.1);

  bcpnn::helpers::random::seed_generator(generator);

  size_t n = d_size(generator);
  size_t hypercolumns = d_size(generator);
  std::uniform_int_distribution<int> d_minicolumns(1, 1024 / hypercolumns);
  std::uniform_int_distribution<int> d_hypercolumn(0, hypercolumns - 1);
  size_t minicolumns = d_minicolumns(generator);
  size_t h = d_hypercolumn(generator);
  size_t m = hypercolumns * minicolumns;
  size_t batch_size = d_size(generator);
  float cthr = d_taupdt(generator) / 2;
  // std::cout << "cthr: " << cthr << std::endl;
  // std::cout << "Size: " << n << ", " << m << std::endl;

  uint8_t *wmask = (uint8_t *)malloc(n * hypercolumns * sizeof(uint8_t));
  float *weights = (float *)malloc(n * m * sizeof(float));
  float *Ci = (float *)malloc(n * sizeof(float));
  float *Cj = (float *)malloc(m * sizeof(float));
  float *Cij = (float *)malloc(n * m * sizeof(float));

  for (size_t i = 0; i < n; ++i) {
    for (size_t h = 0; h < hypercolumns; ++h) {
      wmask[i * hypercolumns + h] = (d_mask_entry(generator) == 0);
    }
  }

  for (size_t i = 0; i < n; ++i) {
    if (d_zero(generator) == 0) {
      Ci[i] = 0;
    } else {
      Ci[i] = d_counter_entry(generator);
    }
  }

  for (size_t j = 0; j < m; ++j) {
    if (d_zero(generator) == 0) {
      Cj[j] = 0;
    } else {
      Cj[j] = d_counter_entry(generator);
    }
  }

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      if (d_zero(generator) == 0) {
        Cij[i * m + j] = 0;
      } else {
        Cij[i * m + j] =
            fminf(0.9 * Ci[i], fminf(0.9 * Cj[j], d_counter_entry(generator)));
      }
    }
  }

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      weights[i * m + j] = d_weight_entry(generator);
    }
  }

  uint8_t *d_wmask;
  float *d_weights;
  float *d_Ci;
  float *d_Cj;
  float *d_Cij;

  CUDA_CALL(cudaMalloc((void **)&d_wmask, n * hypercolumns * sizeof(uint8_t)));
  CUDA_CALL(cudaMalloc((void **)&d_weights, n * m * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&d_Ci, n * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&d_Cj, m * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&d_Cij, n * m * sizeof(float)));

  CUDA_CALL(cudaMemcpy(d_wmask, wmask, n * hypercolumns * sizeof(uint8_t),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_weights, weights, n * m * sizeof(float),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_Ci, Ci, n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_Cj, Cj, m * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(
      cudaMemcpy(d_Cij, Cij, n * m * sizeof(float), cudaMemcpyHostToDevice));

  naive_update_mask<REAL>(wmask, weights, Ci, Cj, Cij, cthr, n, m, h,
                          hypercolumns, minicolumns, 1);
  cuda_update_mask<REAL>(d_wmask, d_weights, d_Ci, d_Cj, d_Cij, cthr, n, m, h,
                         hypercolumns, minicolumns, 1);

  uint8_t *h_wmask = (uint8_t *)malloc(n * hypercolumns * sizeof(uint8_t));

  CUDA_CALL(cudaMemcpy(h_wmask, d_wmask, n * hypercolumns * sizeof(uint8_t),
                       cudaMemcpyDeviceToHost));

  float delta_max = 0;
  for (size_t i = 0; i < n; ++i) {
    for (size_t h = 0; h < hypercolumns; ++h) {
      size_t idx = i * hypercolumns + h;
      delta_max = fmaxf(delta_max, fabsf(wmask[idx] - h_wmask[idx]));
    }
  }

#if 0
  for (size_t i = 0; i < n; ++i) {
    for (size_t h = 0; h < hypercolumns; ++h) {
      size_t idx = i * hypercolumns + h;
      std::cout << (int)wmask[idx] << "\t";
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;
  std::cout << std::endl;

  for (size_t i = 0; i < n; ++i) {
    for (size_t h = 0; h < hypercolumns; ++h) {
      size_t idx = i * hypercolumns + h;
      std::cout << (int)h_wmask[idx] << "\t";
    }
    std::cout << std::endl;
  }
#endif

  free(wmask);
  free(weights);
  free(Ci);
  free(Cj);
  free(Cij);

  CUDA_CALL(cudaFree(d_wmask));
  CUDA_CALL(cudaFree(d_weights));
  CUDA_CALL(cudaFree(d_Ci));
  CUDA_CALL(cudaFree(d_Cj));
  CUDA_CALL(cudaFree(d_Cij));

  free(h_wmask);

  return delta_max;
}

float test_apply_mask() {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> d_size(1, 1024);
  std::uniform_int_distribution<int> d_mask_entry(0, 4);
  std::uniform_real_distribution<float> d_weight_entry(-10, 10);

  bcpnn::helpers::random::seed_generator(generator);

  size_t n = d_size(generator);
  size_t hypercolumns = d_size(generator);
  std::uniform_int_distribution<int> d_minicolumns(1, 1024 / hypercolumns);
  size_t minicolumns = d_minicolumns(generator);
  size_t m = hypercolumns * minicolumns;

  uint8_t *wmask = (uint8_t *)malloc(n * hypercolumns * sizeof(uint8_t));
  float *weights = (float *)malloc(n * m * sizeof(float));

  for (size_t i = 0; i < n; ++i) {
    for (size_t h = 0; h < hypercolumns; ++h) {
      wmask[i * hypercolumns + h] = (d_mask_entry(generator) == 0);
    }
  }

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      weights[i * m + j] = d_weight_entry(generator);
    }
  }

  uint8_t *d_wmask;
  float *d_weights;

  CUDA_CALL(cudaMalloc((void **)&d_wmask, n * hypercolumns * sizeof(uint8_t)));
  CUDA_CALL(cudaMalloc((void **)&d_weights, n * m * sizeof(float)));

  CUDA_CALL(cudaMemcpy(d_wmask, wmask, n * hypercolumns * sizeof(uint8_t),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_weights, weights, n * m * sizeof(float),
                       cudaMemcpyHostToDevice));

  naive_apply_mask<REAL>(weights, wmask, n, m, hypercolumns, minicolumns);
  cuda_apply_mask<REAL>(d_weights, d_wmask, n, m, hypercolumns, minicolumns);

  float *h_weights = (float *)malloc(n * m * sizeof(float));

  CUDA_CALL(cudaMemcpy(h_weights, d_weights, n * m * sizeof(float),
                       cudaMemcpyDeviceToHost));

  float delta_max = 0;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      size_t idx = i * m + j;
      delta_max = fmaxf(delta_max, fabsf(weights[idx] - h_weights[idx]));
    }
  }

#if 0
  for (size_t i = 0; i < n; ++i) {
    for (size_t h = 0; h < hypercolumns; ++h) {
      size_t idx = i * hypercolumns + h;
      std::cout << (int)wmask[idx] << "\t";
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;
  std::cout << std::endl;

  for (size_t i = 0; i < n; ++i) {
    for (size_t h = 0; h < hypercolumns; ++h) {
      size_t idx = i * hypercolumns + h;
      std::cout << (int)h_wmask[idx] << "\t";
    }
    std::cout << std::endl;
  }
#endif

  free(wmask);
  free(weights);

  CUDA_CALL(cudaFree(d_wmask));
  CUDA_CALL(cudaFree(d_weights));

  free(h_weights);

  return delta_max;
}

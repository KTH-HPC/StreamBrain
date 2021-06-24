#include <iostream>
#include <random>

#include "helpers_cuda.h"
#include "helpers_random.h"

namespace bcpnn {

namespace helpers {

namespace training {

void initialize_wmask(uint8_t *wmask, size_t n, size_t hypercolumns) {
  uint8_t *w = (uint8_t *)malloc(n * hypercolumns * sizeof(uint8_t));

  std::default_random_engine generator;
  bcpnn::helpers::random::seed_generator(generator);
  std::uniform_real_distribution<float> d(0, 1);

  for (size_t i = 0; i < n * hypercolumns; ++i) {
    if (d(generator) < 0.1) {
      w[i] = 1;
    } else {
      w[i] = 0;
    }
  }

  CUDA_CALL(cudaMemcpy(wmask, w, n * hypercolumns * sizeof(uint8_t),
                       cudaMemcpyHostToDevice));

  free(w);
}

void print_wmask(uint8_t *wmask, size_t rows, size_t columns,
                 size_t hypercolumns) {
  size_t n = rows * columns;
  uint8_t *w = (uint8_t *)malloc(n * hypercolumns * sizeof(uint8_t));

  CUDA_CALL(cudaMemcpy(w, wmask, n * hypercolumns * sizeof(uint8_t),
                       cudaMemcpyDeviceToHost));

  for (size_t h = 0; h < hypercolumns; ++h) {
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < columns; ++j) {
        std::cout << (int)w[(i * columns + j) * hypercolumns + h];
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::endl;
  }

  free(w);
}

void cpu_correct_predictions(int *correct_count, float *activation_2,
                             float *batch_labels, size_t batch_size, size_t m) {
  float *h_batch_labels = (float *)malloc(batch_size * m * sizeof(float));
  float *h_activation_2 = (float *)malloc(batch_size * m * sizeof(float));

  CUDA_CALL(cudaMemcpy(h_activation_2, activation_2,
                       batch_size * m * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(h_batch_labels, batch_labels,
                       batch_size * m * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < batch_size; ++i) {
    size_t i_max = 0;
    float v_max = h_activation_2[i * m + 0];
    for (int j = 1; j < m; ++j) {
      float v = h_activation_2[i * m + j];
      if (v_max < v) {
        i_max = j;
        v_max = v;
      }
    }

    bool correct = (h_batch_labels[i * m + i_max] == 1);
    for (size_t j = 0; j < m; ++j) {
      if (j != i_max && h_batch_labels[i * m + j] != 0) {
        correct = false;
      }
    }

    if (correct) {
      *correct_count += 1;
    }
  }

  free(h_batch_labels);
  free(h_activation_2);
}

}  // namespace training

}  // namespace helpers

}  // namespace bcpnn

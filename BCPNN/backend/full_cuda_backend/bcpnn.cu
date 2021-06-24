#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

#include "dataloader.h"
#include "dataset.h"
#include "helpers_cuda.h"
#include "helpers_random.h"
#include "helpers_training.h"
#include "kernels_cuda.h"

//#define ACCURACY_CPU_DOUBLE_CHECK

using REAL = double;

size_t batch_size = 0;
size_t constexpr n_inputs = 28 * 28;
size_t constexpr n_hypercolumns = 30;
size_t constexpr n_minicolumns = 100;
size_t constexpr n_hidden = n_hypercolumns * n_minicolumns;
size_t constexpr n_outputs = 10;

size_t constexpr n_cycles_testing = 5;

std::default_random_engine generator;
curandGenerator_t gen;
cublasHandle_t handle;

using namespace bcpnn::helpers::cuda;
using namespace bcpnn::helpers::training;
using namespace bcpnn::helpers::random;
using namespace bcpnn::kernels::cuda;
using namespace bcpnn;

void train_layer_1(dataloader<REAL> &loader, REAL *W1, REAL *B1, REAL taupdt,
                   REAL khalf, REAL pmin, REAL taubdt, size_t epochs);
void train_layer_2(dataloader<REAL> &loader, REAL *W1, REAL *B1, REAL *W2,
                   REAL *B2, REAL taupdt, size_t epochs);
double evaluate_network(dataset_t<REAL, REAL> &test_dataset, REAL *W1, REAL *B1,
                        REAL *W2, REAL *B2);

//#define cudaMalloc cudaMallocManaged

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <batch size>" << std::endl;
    exit(1);
  }

  batch_size = atoi(argv[1]);

  seed_generator(generator);

  dataset_t<uint8_t, uint8_t> _train_dataset;
  dataset_t<uint8_t, uint8_t> _test_dataset;

  _train_dataset =
      read_mnist_dataset("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
  _test_dataset =
      read_mnist_dataset("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

  dataset_t<REAL, REAL> train_dataset = convert_dataset<REAL>(_train_dataset);
  dataset_t<REAL, REAL> test_dataset = convert_dataset<REAL>(_test_dataset);

#if 0
  std::cout << "Training set:" << std::endl;
  std::cout << "=============" << std::endl;
  print_dataset(train_dataset);

  std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "Test set:" << std::endl;
  std::cout << "=========" << std::endl;
  print_dataset(test_dataset);
#endif

  dataloader<REAL> loader(train_dataset, 8, 2);

  CUBLAS_CALL(cublasCreate(&handle));
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

#ifndef BCPNN_FETCH_PARAMETERS
#if 0
  // 94.8% or so, has reached > 95%
  float taupdt = 0.006000765502330436;
  size_t l1_epochs = 325;
  size_t l2_epochs = 347;

  float l1_pmin = 0.07830636855214834;
  float l1_khalf = -81.55156765133142;
  float l1_taubdt = 0.054491579066962004;
#endif
#if 1
  // 94% - 94.5%
  float taupdt = 0.002996755526968425;
  size_t l1_epochs = 23;
  size_t l2_epochs = 298;

  float l1_pmin = 0.3496214817513042;
  float l1_khalf = -435.08426155834593;
  float l1_taubdt = 0.27826430798917945;
#endif
#if 0
  // 93% - 94%
  float taupdt = 0.000861296756773373;
  size_t l1_epochs = 50;
  size_t l2_epochs = 250;

  float l1_pmin = 0.1911360664476474;
  float l1_khalf = -378.52897008489674;
  float l1_taubdt = 0.1395597316674668;
#endif
#else
  float taupdt;
  size_t l1_epochs;
  size_t l2_epochs;

  float l1_pmin;
  float l1_khalf;
  float l1_taubdt;

  char const *BASE_URL = "http://172.17.0.3:5000";
  std::stringstream url;

  url.str("");
  url.clear();
  url << BASE_URL << "/new";

  std::string trial_id = make_request(url.str().c_str());

  std::string parameters;
  while (parameters.size() == 0) {
    url.str("");
    url.clear();
    url << BASE_URL << "/get/" << trial_id;
    parameters = make_request(url.str().c_str());
    usleep(1000000);
  }

  get_parameters(parameters.c_str(), parameters.size(), &taupdt, &l1_epochs,
                 &l2_epochs, &l1_pmin, &l1_khalf, &l1_taubdt);
#endif

  REAL *W1;
  REAL *B1;
  REAL *W2;
  REAL *B2;

  CUDA_CALL(cudaMalloc((void **)&W1, n_inputs * n_hidden * sizeof(REAL)));
  CUDA_CALL(cudaMalloc((void **)&B1, n_hidden * sizeof(REAL)));
  CUDA_CALL(cudaMalloc((void **)&W2, n_hidden * n_outputs * sizeof(REAL)));
  CUDA_CALL(cudaMalloc((void **)&B2, n_outputs * sizeof(REAL)));

  CURAND_CALL(
      TcurandGenerateNormal<REAL>(gen, W1, n_inputs * n_hidden, 0, 0.1));
  CURAND_CALL(TcurandGenerateUniform<REAL>(gen, B1, n_hidden));
  CURAND_CALL(
      TcurandGenerateNormal<REAL>(gen, W2, n_hidden * n_outputs, 0, 0.1));
  CURAND_CALL(TcurandGenerateUniform<REAL>(gen, B2, n_outputs));

  cuda_scale_array<REAL>(B1, 0.1, n_hidden);
  cuda_scale_array<REAL>(B2, 0.1, n_outputs);

  auto training_start = std::chrono::steady_clock::now();
  train_layer_1(loader, W1, B1, taupdt, l1_khalf, l1_pmin, l1_taubdt,
                l1_epochs);
  train_layer_2(loader, W1, B1, W2, B2, taupdt, l2_epochs);
  auto training_end = std::chrono::steady_clock::now();

  auto testing_start = std::chrono::steady_clock::now();
  REAL accuracy = evaluate_network(test_dataset, W1, B1, W2, B2);
  auto testing_end = std::chrono::steady_clock::now();

  std::cout << "Accuracy: " << accuracy << std::endl;
  std::cout << "Training duration: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   training_end - training_start)
                       .count() /
                   1000.0
            << std::endl;
  std::cout << "Testing duration: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   testing_end - testing_start)
                       .count() /
                   1000.0
            << std::endl;

#ifdef BCPNN_FETCH_PARAMETERS
  url.str("");
  url.clear();
  url << BASE_URL << "/complete/" << trial_id << "/" << accuracy;
  make_request(url.str().c_str());
#endif

  // testing();

  CURAND_CALL(curandDestroyGenerator(gen));
  loader.stop();

  return 0;
}

/*
 * Train layers or evaluate the network
 */

void training_step_layer_1(REAL *W1, REAL *B1, size_t batch_size,
                           REAL *batch_images, REAL *batch_labels,
                           REAL *activation_1, REAL *Ci, REAL *Cj, REAL *Cij,
                           uint8_t *wmask, REAL *kbi, REAL taupdt, REAL khalf,
                           REAL pmin, REAL taubdt, size_t hypercolumn);

void training_step_layer_2(REAL *W1, REAL *B1, REAL *W2, REAL *B2,
                           size_t batch_size, REAL *batch_images,
                           REAL *batch_labels, REAL *activation_1, REAL *Ci,
                           REAL *Cj, REAL *Cij, REAL taupdt);

void evaluate_batch(int *correct_count, REAL *W1, REAL *B1, REAL *W2, REAL *B2,
                    size_t batch_size, REAL *batch_images, REAL *batch_labels,
                    REAL *activation_1, REAL *activation_2);

void train_layer_1(dataloader<REAL> &loader, REAL *W1, REAL *B1, REAL taupdt,
                   REAL khalf, REAL pmin, REAL taubdt, size_t epochs) {
  REAL *activation_1;
  REAL *Ci;
  REAL *Cj;
  REAL *Cij;
  uint8_t *wmask;
  REAL *kbi;

  CUDA_CALL(
      cudaMalloc((void **)&activation_1, batch_size * n_hidden * sizeof(REAL)));
  CUDA_CALL(cudaMalloc((void **)&Ci, n_inputs * sizeof(REAL)));
  CUDA_CALL(cudaMalloc((void **)&Cj, n_hidden * sizeof(REAL)));
  CUDA_CALL(cudaMalloc((void **)&Cij, n_inputs * n_hidden * sizeof(REAL)));
  CUDA_CALL(
      cudaMalloc((void **)&wmask, n_inputs * n_hypercolumns * sizeof(uint8_t)));
  CUDA_CALL(cudaMalloc((void **)&kbi, n_hidden * sizeof(REAL)));

  cuda_initialize_array<REAL>(Ci, 1, n_inputs);
  cuda_initialize_array<REAL>(Cj, 1 / ((REAL)n_minicolumns), n_hidden);
  cuda_initialize_array<REAL>(Cij, 1 / ((REAL)n_minicolumns),
                              n_inputs * n_hidden);
  cuda_initialize_array<REAL>(kbi, 1, n_hidden);

  initialize_wmask(wmask, n_inputs, n_hypercolumns);

  std::vector<size_t> hc_shuf;
  for (size_t i = 0; i < n_hypercolumns; ++i) {
    hc_shuf.push_back(i);
  }
  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    std::shuffle(hc_shuf.begin(), hc_shuf.end(), generator);

    std::pair<REAL *, REAL *> p = loader.queue_get_fresh();
    size_t pos = 0;

    size_t n_steps =
        (loader.get_dataset().number_of_examples + batch_size - 1) / batch_size;
    for (size_t step = 0; step < n_steps; ++step) {
      REAL *batch_images = p.first + (pos * n_inputs);
      REAL *batch_labels = p.second + (pos * n_outputs);
      size_t batch_size_step =
          min(batch_size, loader.get_dataset().number_of_examples - pos);

      size_t h = (epoch > 0 && (step % (n_steps / (n_hypercolumns + 1)) == 0))
                     ? step / (n_steps / (n_hypercolumns + 1))
                     : n_hypercolumns;

      // if (h < n_hypercolumns) { printf("Updating hypercolumn: %ld\n", h); }

      training_step_layer_1(W1, B1, batch_size_step, batch_images, batch_labels,
                            activation_1, Ci, Cj, Cij, wmask, kbi, taupdt,
                            khalf, pmin, taubdt,
                            (h < n_hypercolumns ? hc_shuf[h] : n_hypercolumns));

      pos += batch_size_step;
    }

    loader.queue_recycle(p);

    // print_wmask(wmask, 28, 28, 30);
    // printf("\nLayer 1/2 - Epoch : %ld\n\n", epoch);
  }

  CUDA_CALL(cudaFree(Ci));
  CUDA_CALL(cudaFree(Cj));
  CUDA_CALL(cudaFree(Cij));
  CUDA_CALL(cudaFree(wmask));
  CUDA_CALL(cudaFree(kbi));
}

void train_layer_2(dataloader<REAL> &loader, REAL *W1, REAL *B1, REAL *W2,
                   REAL *B2, REAL taupdt, size_t epochs) {
  REAL *activation_1;
  REAL *Ci;
  REAL *Cj;
  REAL *Cij;

  CUDA_CALL(
      cudaMalloc((void **)&activation_1, batch_size * n_hidden * sizeof(REAL)));
  CUDA_CALL(cudaMalloc((void **)&Ci, n_hidden * sizeof(REAL)));
  CUDA_CALL(cudaMalloc((void **)&Cj, n_outputs * sizeof(REAL)));
  CUDA_CALL(cudaMalloc((void **)&Cij, n_hidden * n_outputs * sizeof(REAL)));

  cuda_initialize_array<REAL>(Ci, 1 / ((REAL)n_hidden), n_hidden);
  cuda_initialize_array<REAL>(Cj, 1 / ((REAL)n_outputs), n_outputs);
  cuda_initialize_array<REAL>(Cij, 1 / ((REAL)n_hidden * n_outputs),
                              n_hidden * n_outputs);

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    std::pair<REAL *, REAL *> p = loader.queue_get_fresh();
    size_t pos = 0;

    size_t n_steps =
        (loader.get_dataset().number_of_examples + batch_size - 1) / batch_size;
    for (size_t step = 0; step < n_steps; ++step) {
      REAL *batch_images = p.first + (pos * n_inputs);
      REAL *batch_labels = p.second + (pos * n_outputs);
      size_t batch_size_step =
          min(batch_size, loader.get_dataset().number_of_examples - pos);

      training_step_layer_2(W1, B1, W2, B2, batch_size_step, batch_images,
                            batch_labels, activation_1, Ci, Cj, Cij, taupdt);

      pos += batch_size_step;
    }

    loader.queue_recycle(p);

    // printf("\nLayer 2/2 - Epoch : %ld\n\n", epoch);
  }

  cuda_update_weights(W2, Ci, Cj, Cij, taupdt / 2, n_hidden, n_outputs);
  cuda_update_bias(B2, Cj, taupdt / 2, n_outputs);

  CUDA_CALL(cudaFree(Ci));
  CUDA_CALL(cudaFree(Cj));
  CUDA_CALL(cudaFree(Cij));
}

#ifdef ACCURACY_CPU_DOUBLE_CHECK
int cpu_correct;
#endif

double evaluate_network(dataset_t<REAL, REAL> &test_dataset, REAL *W1, REAL *B1,
                        REAL *W2, REAL *B2) {
  int *correct;
  REAL *activation_1;
  REAL *activation_2;
  REAL *test_images;
  REAL *test_labels;

  CUDA_CALL(
      cudaMalloc((void **)&activation_1, batch_size * n_hidden * sizeof(REAL)));
  CUDA_CALL(cudaMalloc((void **)&activation_2,
                       batch_size * n_outputs * sizeof(REAL)));

  CUDA_CALL(cudaMalloc((void **)&correct, sizeof(int)));
  CUDA_CALL(cudaMemset(correct, 0, sizeof(int)));
#ifdef ACCURACY_CPU_DOUBLE_CHECK
  cpu_correct = 0;
#endif

  CUDA_CALL(cudaMalloc((void **)&test_images, test_dataset.number_of_examples *
                                                  n_inputs * sizeof(REAL)));
  CUDA_CALL(cudaMalloc((void **)&test_labels, test_dataset.number_of_examples *
                                                  n_outputs * sizeof(REAL)));

  CUDA_CALL(
      cudaMemcpy(test_images, test_dataset.images,
                 test_dataset.number_of_examples * n_inputs * sizeof(REAL),
                 cudaMemcpyHostToDevice));
  CUDA_CALL(
      cudaMemcpy(test_labels, test_dataset.labels,
                 test_dataset.number_of_examples * n_outputs * sizeof(REAL),
                 cudaMemcpyHostToDevice));

  size_t pos = 0;

  for (size_t step = 0;
       step < (test_dataset.number_of_examples + batch_size - 1); ++step) {
    REAL *batch_images = test_images + (pos * n_inputs);
    REAL *batch_labels = test_labels + (pos * n_outputs);
    size_t batch_size_step =
        min(batch_size, test_dataset.number_of_examples - pos);

    evaluate_batch(correct, W1, B1, W2, B2, batch_size_step, batch_images,
                   batch_labels, activation_1, activation_2);

    batch_images += batch_size * n_inputs;
    batch_labels += batch_size * n_outputs;

    pos += batch_size_step;
  }

  /*
  float * h_activation_2 = (float *)malloc(batch_size * n_outputs *
  sizeof(float)); CUDA_CALL(cudaMemcpy(h_activation_2, activation_2, batch_size
  * n_outputs * sizeof(float), cudaMemcpyDeviceToHost));

  for (size_t example = 0; example < 5; ++example) {
    for (size_t i = 0; i < 10; ++i) { std::cout << i << ": " <<
  h_activation_2[example*10 + i] << std::endl; } std::cout << std::endl; for
  (size_t i = 0; i < 10; ++i) { std::cout << i << ": " <<
  test_dataset.labels[(10000 - 100 + example)*10 + i] << std::endl; } std::cout
  << std::endl; std::cout << std::endl;
  }
  */

  int h_correct;
  CUDA_CALL(
      cudaMemcpy(&h_correct, correct, sizeof(int), cudaMemcpyDeviceToHost));
#ifdef ACCURACY_CPU_DOUBLE_CHECK
  if (cpu_correct != h_correct) {
    std::cerr << "CPU and GPU differ on number of correctly predicted images"
              << std::endl;
    exit(1);
  }
#endif

  return ((double)h_correct) / test_dataset.number_of_examples;
}

/*
 *  Individual traning or evaluation steps
 */

void training_step_layer_1(REAL *W1, REAL *B1, size_t batch_size,
                           REAL *batch_images, REAL *batch_labels,
                           REAL *activation_1, REAL *Ci, REAL *Cj, REAL *Cij,
                           uint8_t *wmask, REAL *kbi, REAL taupdt, REAL khalf,
                           REAL pmin, REAL taubdt, size_t hypercolumn) {
  REAL v_one = 1;
  REAL v_zero = 0;

  CUBLAS_CALL(cublasgemm<REAL>(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, n_hidden, batch_size, n_inputs, &v_one,
      W1, n_hidden, batch_images, n_inputs, &v_zero, activation_1, n_hidden));

  cuda_add_bias(activation_1, batch_size, n_hidden, B1);

  cuda_softmax(activation_1, batch_size * n_hypercolumns, n_minicolumns);

  cuda_update_counters(Ci, Cj, Cij, batch_images, activation_1, batch_size,
                       n_inputs, n_hidden, taupdt);
  cuda_update_weights(W1, Ci, Cj, Cij, taupdt / 2, n_inputs, n_hidden);
  cuda_update_bias_regularized(B1, kbi, Cj, taupdt / 2, khalf, pmin, taubdt,
                               n_hidden);
  if (hypercolumn < n_hypercolumns) {
    cuda_update_mask(wmask, W1, Ci, Cj, Cij, taupdt / 2, n_inputs, n_hidden,
                     hypercolumn, n_hypercolumns, n_minicolumns, 16);
  }
  cuda_apply_mask(W1, wmask, n_inputs, n_hidden, n_hypercolumns, n_minicolumns);
}

void training_step_layer_2(REAL *W1, REAL *B1, REAL *W2, REAL *B2,
                           size_t batch_size, REAL *batch_images,
                           REAL *batch_labels, REAL *activation_1, REAL *Ci,
                           REAL *Cj, REAL *Cij, REAL taupdt) {
  REAL v_one = 1;
  REAL v_zero = 0;

  CUBLAS_CALL(cublasgemm<REAL>(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, n_hidden, batch_size, n_inputs, &v_one,
      W1, n_hidden, batch_images, n_inputs, &v_zero, activation_1, n_hidden));

  cuda_add_bias(activation_1, batch_size, n_hidden, B1);

  cuda_softmax(activation_1, batch_size * n_hypercolumns, n_minicolumns);

  cuda_update_counters(Ci, Cj, Cij, activation_1, batch_labels, batch_size,
                       n_hidden, n_outputs, taupdt);
  // Not really needed to update weights during training, right?
  //  cuda_update_weights(W2, Ci, Cj, Cij, taupdt/2, n_hidden, n_outputs);
  //  cuda_update_bias(B1, Cj, taupdt/2, n_outputs);
}

void evaluate_batch(int *correct_count, REAL *W1, REAL *B1, REAL *W2, REAL *B2,
                    size_t batch_size, REAL *batch_images, REAL *batch_labels,
                    REAL *activation_1, REAL *activation_2) {
  REAL v_one = 1;
  REAL v_zero = 0;

  for (size_t i = 0; i < n_cycles_testing; ++i) {
    CUBLAS_CALL(cublasgemm<REAL>(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_hidden,
                                 batch_size, n_inputs, &v_one, W1, n_hidden,
                                 batch_images, n_inputs, &v_zero, activation_1,
                                 n_hidden));

    cuda_add_bias(activation_1, batch_size, n_hidden, B1);

    cuda_softmax(activation_1, batch_size * n_hypercolumns, n_minicolumns);

    CUBLAS_CALL(cublasgemm<REAL>(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_outputs,
                                 batch_size, n_hidden, &v_one, W2, n_outputs,
                                 activation_1, n_hidden, &v_zero, activation_2,
                                 n_outputs));

    cuda_add_bias(activation_2, batch_size, n_outputs, B2);

    cuda_softmax(activation_2, batch_size * 1, n_outputs);
  }

#ifdef ACCURACY_CPU_DOUBLE_CHECK
  cpu_correct_predictions(&cpu_correct, activation_2, batch_labels, batch_size,
                          n_outputs);
#endif
  cuda_correct_predictions(correct_count, activation_2, batch_labels,
                           batch_size, n_outputs);
}

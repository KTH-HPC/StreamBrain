#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <unistd.h>

#include <algorithm>
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

#ifdef USE_CATALYST
#include "CatalystAdaptor.h"
#endif

#define ACCURACY_CPU_DOUBLE_CHECK

std::default_random_engine generator;
curandGenerator_t gen;
cublasHandle_t handle;

using namespace bcpnn::helpers::cuda;
using namespace bcpnn::helpers::training;
using namespace bcpnn::helpers::random;
using namespace bcpnn::kernels::cuda;
using namespace bcpnn;

//#define cudaMalloc cudaMallocManaged

template <typename REAL>
class SupervisedInput {
 public:
  REAL *inputs;
  REAL *labels;
  size_t count;
};

template <typename REAL>
class UnsupervisedInput {
 public:
  REAL *inputs;
  size_t count;
};

template <typename REAL>
class Allocation {
 public:
  virtual ~Allocation() {}

  REAL *activation;
};

template <typename REAL>
class Layer {
 public:
  virtual void compute_batch(Allocation<REAL> *alloc,
                             UnsupervisedInput<REAL> *inputs) = 0;
  virtual void train_batch(Allocation<REAL> *alloc,
                           SupervisedInput<REAL> *inputs) = 0;
  virtual void train_batch(Allocation<REAL> *alloc,
                           UnsupervisedInput<REAL> *inputs) = 0;
  virtual void train_finalize(Allocation<REAL> *alloc) = 0;

  virtual Allocation<REAL> *allocate_compute(size_t maximal_batch_size) = 0;
  virtual Allocation<REAL> *allocate_training(size_t maximal_batch_size) = 0;
};

template <typename REAL>
class MaskedDenseLayer;

template <typename REAL>
class MaskedDenseLayerComputeAllocation;

template <typename REAL>
class MaskedDenseLayerTrainingAllocation;

template <typename REAL>
class MaskedDenseLayer : public Layer<REAL> {
 public:
  MaskedDenseLayer(size_t n_inputs, size_t n_hypercolumns,
                   size_t n_minicolumns);

  void compute_batch(Allocation<REAL> *alloc, UnsupervisedInput<REAL> *inputs);
  void train_batch(Allocation<REAL> *alloc, SupervisedInput<REAL> *inputs);
  void train_batch(Allocation<REAL> *alloc, UnsupervisedInput<REAL> *inputs);
  void train_finalize(Allocation<REAL> *alloc);

  Allocation<REAL> *allocate_compute(size_t maximal_batch_size);
  Allocation<REAL> *allocate_training(size_t maximal_batch_size);

  REAL *weights;
  REAL *bias;

  REAL taupdt;
  REAL khalf;
  REAL pmin;
  REAL taubdt;

  REAL initial_Ci;
  REAL initial_Cj;
  REAL initial_Cij;

  size_t n_inputs_;
  size_t n_outputs_;
  size_t n_hypercolumns_;
  size_t n_minicolumns_;
};

template <typename REAL>
class MaskedDenseLayerComputeAllocation : public Allocation<REAL> {
 public:
};

template <typename REAL>
class MaskedDenseLayerTrainingAllocation : public Allocation<REAL> {
 public:
  MaskedDenseLayerComputeAllocation<REAL> *to_compute_allocation();

  REAL *Ci;
  REAL *Cj;
  REAL *Cij;
  REAL *kbi;
  uint8_t *wmask;

  size_t update_hypercolumn;

  std::vector<size_t> hc_permutation;
  size_t hc_pos;
};

template <typename REAL>
MaskedDenseLayer<REAL>::MaskedDenseLayer(size_t n_inputs, size_t n_hypercolumns,
                                         size_t n_minicolumns)
    : weights(NULL),
      bias(NULL),
      n_inputs_(n_inputs),
      n_outputs_(n_hypercolumns * n_minicolumns),
      n_hypercolumns_(n_hypercolumns),
      n_minicolumns_(n_minicolumns) {
  CUDA_CALL(
      cudaMalloc((void **)&weights, n_inputs_ * n_outputs_ * sizeof(REAL)));
  CUDA_CALL(cudaMalloc((void **)&bias, n_outputs_ * sizeof(REAL)));

  CURAND_CALL(TcurandGenerateNormal<REAL>(gen, weights, n_inputs_ * n_outputs_,
                                          0, 0.1));
  CURAND_CALL(TcurandGenerateUniform<REAL>(gen, bias, n_outputs_));

  cuda_scale_array<REAL>(bias, 0.1, n_outputs_);
}

template <typename REAL>
Allocation<REAL> *MaskedDenseLayer<REAL>::allocate_compute(
    size_t maximal_batch_size) {
  MaskedDenseLayerComputeAllocation<REAL> *alloc =
      new MaskedDenseLayerComputeAllocation<REAL>;

  CUDA_CALL(cudaMalloc((void **)&alloc->activation,
                       maximal_batch_size * n_outputs_ * sizeof(REAL)));

  return alloc;
}

template <typename REAL>
Allocation<REAL> *MaskedDenseLayer<REAL>::allocate_training(
    size_t maximal_batch_size) {
  MaskedDenseLayerTrainingAllocation<REAL> *alloc =
      new MaskedDenseLayerTrainingAllocation<REAL>;

  CUDA_CALL(cudaMalloc((void **)&alloc->activation,
                       maximal_batch_size * n_outputs_ * sizeof(REAL)));
  CUDA_CALL(cudaMalloc((void **)&alloc->Ci, n_inputs_ * sizeof(REAL)));
  CUDA_CALL(cudaMalloc((void **)&alloc->Cj, n_outputs_ * sizeof(REAL)));
  CUDA_CALL(
      cudaMalloc((void **)&alloc->Cij, n_inputs_ * n_outputs_ * sizeof(REAL)));
  CUDA_CALL(cudaMalloc((void **)&alloc->wmask,
                       n_inputs_ * n_hypercolumns_ * sizeof(uint8_t)));
  CUDA_CALL(cudaMalloc((void **)&alloc->kbi, n_outputs_ * sizeof(REAL)));

  cuda_initialize_array<REAL>(alloc->Ci, initial_Ci, n_inputs_);
  cuda_initialize_array<REAL>(alloc->Cj, initial_Cj, n_outputs_);
  cuda_initialize_array<REAL>(alloc->Cij, initial_Cij, n_inputs_ * n_outputs_);
  cuda_initialize_array<REAL>(alloc->kbi, 1, n_outputs_);

  initialize_wmask(alloc->wmask, n_inputs_, n_hypercolumns_);

  alloc->update_hypercolumn = 0;

  for (size_t i = 0; i < n_hypercolumns_; ++i) {
    alloc->hc_permutation.push_back(i);
  }
  std::shuffle(alloc->hc_permutation.begin(), alloc->hc_permutation.end(),
               generator);

  return alloc;
}

template <typename REAL>
void MaskedDenseLayer<REAL>::compute_batch(Allocation<REAL> *alloc_,
                                           UnsupervisedInput<REAL> *inputs) {
  MaskedDenseLayerComputeAllocation<REAL> *alloc =
      (MaskedDenseLayerComputeAllocation<REAL> *)alloc_;
  REAL v_one = 1;
  REAL v_zero = 0;

  CUBLAS_CALL(cublasgemm<REAL>(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_outputs_,
                               inputs->count, n_inputs_, &v_one, this->weights,
                               n_outputs_, inputs->inputs, n_inputs_, &v_zero,
                               alloc->activation, n_outputs_));

  cuda_add_bias(alloc->activation, inputs->count, n_outputs_, this->bias);

  cuda_softmax(alloc->activation, inputs->count * n_hypercolumns_,
               n_minicolumns_);
}

template <typename REAL>
void MaskedDenseLayer<REAL>::train_batch(Allocation<REAL> *alloc,
                                         SupervisedInput<REAL> *inputs) {
  std::cerr << "MaskedDenseLayer supervised training unimplemented"
            << std::endl;
  exit(1);
  /*
float v_one = 1;
float v_zero = 0;

CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_outputs_,
inputs->count, n_inputs_, &v_one, this->weights, n_outputs_, inputs->inputs,
n_inputs_, &v_zero, alloc->activation, n_outputs_));

cuda_add_bias(alloc->activation, inputs->count, n_outputs_, this->bias);

cuda_softmax(alloc->activation, inputs->count * n_hypercolumns, n_minicolumns);

cuda_update_counters(alloc->Ci, alloc->Cj, alloc->Cij, inputs->inputs,
alloc->activation, inputs->count, n_inputs_, n_outputs_, taupdt); if
(hypercolumn < n_hypercolumns) { cuda_update_weights(W1, Ci, Cj, Cij, taupdt/2,
n_inputs_, n_outputs_); cuda_update_bias_regularized(B1, kbi, Cj, taupdt/2,
khalf, pmin, taubdt, n_outputs_); cuda_update_mask(wmask, W1, Ci, Cj, Cij,
taupdt/2, n_inputs_, n_outputs_, hypercolumn, n_hypercolumns, n_minicolumns,
16); cuda_apply_mask(W1, wmask, n_inputs_, n_outputs_, n_hypercolumns,
n_minicolumns);
}
*/
}

template <typename REAL>
void MaskedDenseLayer<REAL>::train_batch(Allocation<REAL> *alloc_,
                                         UnsupervisedInput<REAL> *inputs) {
  MaskedDenseLayerTrainingAllocation<REAL> *alloc =
      (MaskedDenseLayerTrainingAllocation<REAL> *)alloc_;
  REAL v_one = 1;
  REAL v_zero = 0;

  CUBLAS_CALL(cublasgemm<REAL>(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_outputs_,
                               inputs->count, n_inputs_, &v_one, this->weights,
                               n_outputs_, inputs->inputs, n_inputs_, &v_zero,
                               alloc->activation, n_outputs_));

  cuda_add_bias(alloc->activation, inputs->count, n_outputs_, this->bias);

  cuda_softmax(alloc->activation, inputs->count * n_hypercolumns_,
               n_minicolumns_);

  cuda_update_counters(alloc->Ci, alloc->Cj, alloc->Cij, inputs->inputs,
                       alloc->activation, inputs->count, n_inputs_, n_outputs_,
                       taupdt);
  cuda_update_weights(this->weights, alloc->Ci, alloc->Cj, alloc->Cij,
                      taupdt / 2, n_inputs_, n_outputs_);
  cuda_update_bias_regularized(this->bias, alloc->kbi, alloc->Cj, taupdt / 2,
                               khalf, pmin, taubdt, n_outputs_);
  if (alloc->update_hypercolumn < n_hypercolumns_) {
    cuda_update_mask(alloc->wmask, this->weights, alloc->Ci, alloc->Cj,
                     alloc->Cij, taupdt / 2, n_inputs_, n_outputs_,
                     alloc->update_hypercolumn, n_hypercolumns_, n_minicolumns_,
                     16);
  }
  cuda_apply_mask(this->weights, alloc->wmask, n_inputs_, n_outputs_,
                  n_hypercolumns_, n_minicolumns_);
}

template <typename REAL>
void MaskedDenseLayer<REAL>::train_finalize(Allocation<REAL> *alloc) {}

template <typename REAL>
class DenseLayer;

template <typename REAL>
class DenseLayerComputeAllocation;

template <typename REAL>
class DenseLayerTrainingAllocation;

template <typename REAL>
class DenseLayer : public Layer<REAL> {
 public:
  DenseLayer(size_t n_inputs, size_t n_hypercolumns, size_t n_minicolumns);

  void compute_batch(Allocation<REAL> *alloc, UnsupervisedInput<REAL> *inputs);
  void train_batch(Allocation<REAL> *alloc, SupervisedInput<REAL> *inputs);
  void train_batch(Allocation<REAL> *alloc, UnsupervisedInput<REAL> *inputs);
  void train_finalize(Allocation<REAL> *alloc);

  Allocation<REAL> *allocate_compute(size_t maximal_batch_size);
  Allocation<REAL> *allocate_training(size_t maximal_batch_size);

  REAL *weights;
  REAL *bias;

  REAL taupdt;

  REAL initial_Ci;
  REAL initial_Cj;
  REAL initial_Cij;

  size_t n_inputs_;
  size_t n_outputs_;
  size_t n_hypercolumns_;
  size_t n_minicolumns_;
};

template <typename REAL>
class DenseLayerComputeAllocation : public Allocation<REAL> {
 public:
};

template <typename REAL>
class DenseLayerTrainingAllocation : public Allocation<REAL> {
 public:
  DenseLayerComputeAllocation<REAL> *to_compute_allocation();

  REAL *Ci;
  REAL *Cj;
  REAL *Cij;
  REAL *kbi;
  uint8_t *wmask;
};

template <typename REAL>
DenseLayer<REAL>::DenseLayer(size_t n_inputs, size_t n_hypercolumns,
                             size_t n_minicolumns)
    : weights(NULL),
      bias(NULL),
      n_inputs_(n_inputs),
      n_outputs_(n_hypercolumns * n_minicolumns),
      n_hypercolumns_(n_hypercolumns),
      n_minicolumns_(n_minicolumns) {
  CUDA_CALL(
      cudaMalloc((void **)&weights, n_inputs_ * n_outputs_ * sizeof(REAL)));
  CUDA_CALL(cudaMalloc((void **)&bias, n_outputs_ * sizeof(REAL)));
}

template <typename REAL>
Allocation<REAL> *DenseLayer<REAL>::allocate_compute(
    size_t maximal_batch_size) {
  DenseLayerComputeAllocation<REAL> *alloc =
      new DenseLayerComputeAllocation<REAL>;

  CUDA_CALL(cudaMalloc((void **)&alloc->activation,
                       maximal_batch_size * n_outputs_ * sizeof(REAL)));

  return alloc;
}

template <typename REAL>
Allocation<REAL> *DenseLayer<REAL>::allocate_training(
    size_t maximal_batch_size) {
  DenseLayerTrainingAllocation<REAL> *alloc =
      new DenseLayerTrainingAllocation<REAL>;

  CUDA_CALL(cudaMalloc((void **)&alloc->activation,
                       maximal_batch_size * n_outputs_ * sizeof(REAL)));
  CUDA_CALL(cudaMalloc((void **)&alloc->Ci, n_inputs_ * sizeof(REAL)));
  CUDA_CALL(cudaMalloc((void **)&alloc->Cj, n_outputs_ * sizeof(REAL)));
  CUDA_CALL(
      cudaMalloc((void **)&alloc->Cij, n_inputs_ * n_outputs_ * sizeof(REAL)));

  cuda_initialize_array<REAL>(alloc->Ci, initial_Ci, n_inputs_);
  cuda_initialize_array<REAL>(alloc->Cj, initial_Cj, n_outputs_);
  cuda_initialize_array<REAL>(alloc->Cij, initial_Cij, n_inputs_ * n_outputs_);

  return alloc;
}

template <typename REAL>
void DenseLayer<REAL>::compute_batch(Allocation<REAL> *alloc_,
                                     UnsupervisedInput<REAL> *inputs) {
  DenseLayerComputeAllocation<REAL> *alloc =
      (DenseLayerComputeAllocation<REAL> *)alloc_;
  REAL v_one = 1;
  REAL v_zero = 0;

  CUBLAS_CALL(cublasgemm<REAL>(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_outputs_,
                               inputs->count, n_inputs_, &v_one, this->weights,
                               n_outputs_, inputs->inputs, n_inputs_, &v_zero,
                               alloc->activation, n_outputs_));

  cuda_add_bias(alloc->activation, inputs->count, n_outputs_, this->bias);

  cuda_softmax(alloc->activation, inputs->count * n_hypercolumns_,
               n_minicolumns_);
}

template <typename REAL>
void DenseLayer<REAL>::train_batch(Allocation<REAL> *alloc_,
                                   SupervisedInput<REAL> *inputs) {
  DenseLayerTrainingAllocation<REAL> *alloc =
      (DenseLayerTrainingAllocation<REAL> *)alloc_;
  REAL v_one = 1;
  REAL v_zero = 0;

  CUBLAS_CALL(cublasgemm<REAL>(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_outputs_,
                               inputs->count, n_inputs_, &v_one, this->weights,
                               n_outputs_, inputs->inputs, n_inputs_, &v_zero,
                               alloc->activation, n_outputs_));

  cuda_add_bias(alloc->activation, inputs->count, n_outputs_, this->bias);

  cuda_softmax(alloc->activation, inputs->count * n_hypercolumns_,
               n_minicolumns_);

  cuda_update_counters(alloc->Ci, alloc->Cj, alloc->Cij, inputs->inputs,
                       inputs->labels, inputs->count, n_inputs_, n_outputs_,
                       taupdt);
}

template <typename REAL>
void DenseLayer<REAL>::train_batch(Allocation<REAL> *alloc,
                                   UnsupervisedInput<REAL> *inputs) {
  std::cerr << "DenseLayer unsupervised training unimplemented" << std::endl;
  exit(1);
#if 0
  float v_one = 1;
  float v_zero = 0;

  CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_outputs_, inputs->count, n_inputs, &v_one, this->weights, n_outputs_, inputs->inputs, n_inputs, &v_zero, alloc->activation, n_outputs_));

  cuda_add_bias(alloc->activation, inputs->count, n_outputs_, this->bias);

  cuda_softmax(alloc->activation, inputs->count * n_hypercolumns, n_minicolumns);

  cuda_update_counters(alloc->Ci, alloc->Cj, alloc->Cij, inputs->inputs, alloc->activation, inputs->count, n_inputs, n_outputs_, taupdt);
  cuda_update_weights(this->weights, alloc->Ci, alloc->Cj, alloc->Cij, taupdt/2, n_inputs, n_outputs_);
  cuda_update_bias_regularized(this->bias, this->bias, this->bias, taupdt/2, khalf, pmin, taubdt, n_outputs_);
  size_t hypercolumn = n_hypercolumns;
  if (hypercolumn < n_hypercolumns) {
    cuda_update_mask(this->bias, this->weights, alloc->Ci, alloc->Cj, alloc->Cij, taupdt/2, n_inputs, n_outputs_, hypercolumn, n_hypercolumns, n_minicolumns, 16);
  }
  cuda_apply_mask(this->weights, this->weights, n_inputs, n_outputs_, n_hypercolumns, n_minicolumns);
#endif
}

template <typename REAL>
void DenseLayer<REAL>::train_finalize(Allocation<REAL> *alloc_) {
  DenseLayerTrainingAllocation<REAL> *alloc =
      (DenseLayerTrainingAllocation<REAL> *)alloc_;

  cuda_update_weights(this->weights, alloc->Ci, alloc->Cj, alloc->Cij,
                      taupdt / 2, n_inputs_, n_outputs_);
  cuda_update_bias(this->bias, alloc->Cj, taupdt / 2, n_outputs_);
}

template <typename REAL>
class Network {
 public:
  //  void add_layer(Layer * layer);

  void train_layer(dataloader<REAL> &loader, size_t maximal_batch_size,
                   size_t layer, size_t epochs);
  double evaluate(dataset_t<REAL, REAL> &dataset, size_t maximal_batch_size);

  std::vector<Layer<REAL> *> layers_;
};

template <typename REAL>
void Network<REAL>::train_layer(dataloader<REAL> &loader,
                                size_t maximal_batch_size, size_t layer,
                                size_t epochs) {
  std::vector<Allocation<REAL> *> allocs;

  for (size_t i = 0; i <= layer; ++i) {
    if (i < layer) {
      allocs.push_back(layers_[i]->allocate_compute(maximal_batch_size));
    } else {
      allocs.push_back(layers_[i]->allocate_training(maximal_batch_size));
    }
  }

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    std::pair<REAL *, REAL *> p = loader.queue_get_fresh();
    size_t pos = 0;

    size_t n_inputs = loader.get_dataset().rows * loader.get_dataset().cols;
    size_t n_outputs = loader.get_dataset().number_of_classes;
    size_t n_steps =
        (loader.get_dataset().number_of_examples + maximal_batch_size - 1) /
        maximal_batch_size;
    for (size_t step = 0; step < n_steps; ++step) {
      REAL *batch_images = p.first + (pos * n_inputs);
      REAL *batch_labels = p.second + (pos * n_outputs);
      size_t batch_size_step = min(
          maximal_batch_size, loader.get_dataset().number_of_examples - pos);

      for (size_t l = 0; l <= layer; ++l) {
        MaskedDenseLayerTrainingAllocation<REAL> *dense_training_alloc =
            dynamic_cast<MaskedDenseLayerTrainingAllocation<REAL> *>(allocs[l]);
        MaskedDenseLayer<REAL> *dense_layer =
            dynamic_cast<MaskedDenseLayer<REAL> *>(layers_[l]);
        if (dense_layer != nullptr && dense_training_alloc != nullptr) {
          size_t n_hypercolumns = dense_layer->n_hypercolumns_;
          size_t h =
              (epoch > 0 && (step % (n_steps / (n_hypercolumns + 1)) == 0))
                  ? step / (n_steps / (n_hypercolumns + 1))
                  : n_hypercolumns;
          dense_training_alloc->update_hypercolumn = h;
        }

        if (l + 1 != layers_.size()) {
          UnsupervisedInput<REAL> inputs;
          inputs.inputs = l == 0 ? batch_images : allocs[l - 1]->activation;
          inputs.count = batch_size_step;

          if (l == layer) {
            layers_[l]->train_batch(allocs[l], &inputs);
          } else {
            layers_[l]->compute_batch(allocs[l], &inputs);
          }
        } else {
          SupervisedInput<REAL> inputs;
          inputs.inputs = l == 0 ? batch_images : allocs[l - 1]->activation;
          inputs.labels = batch_labels;
          inputs.count = batch_size_step;

          layers_[l]->train_batch(allocs[l], &inputs);
        }
      }

      pos += batch_size_step;
    }

    loader.queue_recycle(p);

    if (layer == 0) {
      MaskedDenseLayer<REAL> *layer =
          dynamic_cast<MaskedDenseLayer<REAL> *>(layers_[0]);
      MaskedDenseLayerTrainingAllocation<REAL> *alloc =
          dynamic_cast<MaskedDenseLayerTrainingAllocation<REAL> *>(allocs[0]);
      if (layer && alloc) {
#ifdef USE_CATALYST
        size_t n = 28 * 28;
        uint8_t *w = (uint8_t *)malloc(n * layer->n_hypercolumns_ * sizeof(uint8_t));
        CUDA_CALL(cudaMemcpy(w, alloc->wmask, n * layer->n_hypercolumns_ * sizeof(uint8_t),
                             cudaMemcpyDeviceToHost));
        Adaptor::CoProcess(epoch, epoch, w);
        free(w);
#endif
        print_wmask(alloc->wmask, 28, 28, layer->n_hypercolumns_);
      }
      printf("\nLayer 1/%lu - Epoch : %ld\n\n", layers_.size(), epoch);
    } else {
      printf("\nLayer %lu/%lu - Epoch : %ld\n\n", layer + 1, layers_.size(),
             epoch);
    }
  }

  layers_[layer]->train_finalize(allocs[layer]);
}

template <typename REAL>
double Network<REAL>::evaluate(dataset_t<REAL, REAL> &dataset,
                               size_t maximal_batch_size) {
  std::vector<Allocation<REAL> *> allocs;
  int *correct;
  REAL *test_images;
  REAL *test_labels;

  CUDA_CALL(cudaMalloc((void **)&correct, sizeof(int)));
  CUDA_CALL(cudaMemset(correct, 0, sizeof(int)));
#ifdef ACCURACY_CPU_DOUBLE_CHECK
  // cpu_correct = 0;
#endif

  size_t n_inputs = dataset.rows * dataset.cols;
  size_t n_outputs = dataset.number_of_classes;

  CUDA_CALL(cudaMalloc((void **)&test_images,
                       dataset.number_of_examples * n_inputs * sizeof(REAL)));
  CUDA_CALL(cudaMalloc((void **)&test_labels,
                       dataset.number_of_examples * n_outputs * sizeof(REAL)));

  CUDA_CALL(cudaMemcpy(test_images, dataset.images,
                       dataset.number_of_examples * n_inputs * sizeof(REAL),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(test_labels, dataset.labels,
                       dataset.number_of_examples * n_outputs * sizeof(REAL),
                       cudaMemcpyHostToDevice));

  for (size_t l = 0; l < layers_.size(); ++l) {
    allocs.push_back(layers_[l]->allocate_compute(maximal_batch_size));
  }

  size_t pos = 0;

  for (size_t step = 0;
       step < (dataset.number_of_examples + maximal_batch_size - 1) /
                  maximal_batch_size;
       ++step) {
    REAL *batch_images = test_images + (pos * n_inputs);
    REAL *batch_labels = test_labels + (pos * n_outputs);
    size_t batch_size_step =
        min(maximal_batch_size, dataset.number_of_examples - pos);

    for (size_t l = 0; l < layers_.size(); ++l) {
      UnsupervisedInput<REAL> inputs;
      inputs.inputs =
          l == 0 ? batch_images
                 : ((MaskedDenseLayerComputeAllocation<REAL> *)allocs[l - 1])
                       ->activation;
      inputs.count = batch_size_step;

      layers_[l]->compute_batch(allocs[l], &inputs);
    }

    cuda_correct_predictions(
        correct,
        ((DenseLayerComputeAllocation<REAL> *)allocs[layers_.size() - 1])
            ->activation,
        batch_labels, batch_size_step, n_outputs);

    pos += batch_size_step;
  }

  int h_correct;
  CUDA_CALL(
      cudaMemcpy(&h_correct, correct, sizeof(int), cudaMemcpyDeviceToHost));
#if 0
#ifdef ACCURACY_CPU_DOUBLE_CHECK
  if (cpu_correct != h_correct) {
    std::cerr << "CPU and GPU differ on number of correctly predicted images" << std::endl;
    exit(1);
  }
#endif
#endif

  return ((double)h_correct) / dataset.number_of_examples;
}

namespace py = pybind11;

template <typename REAL>
class PyNetwork {
 public:
  PyNetwork() {}

  void add_dense_layer(size_t n_inputs, size_t n_hypercolumns,
                       size_t n_minicolumns, REAL taupdt, REAL initial_Ci,
                       REAL initial_Cj, REAL initial_Cij) {
    DenseLayer<REAL> *layer =
        new DenseLayer<REAL>(n_inputs, n_hypercolumns, n_minicolumns);
    layer->taupdt = taupdt;

    layer->initial_Ci = initial_Ci;
    layer->initial_Cj = initial_Cj;
    layer->initial_Cij = initial_Cij;

    network.layers_.push_back(layer);
  }

  void add_plastic_layer(size_t n_inputs, size_t n_hypercolumns,
                         size_t n_minicolumns, REAL taupdt, REAL pmin,
                         REAL khalf, REAL taubdt, REAL initial_Ci,
                         REAL initial_Cj, REAL initial_Cij) {
    MaskedDenseLayer<REAL> *layer =
        new MaskedDenseLayer<REAL>(n_inputs, n_hypercolumns, n_minicolumns);
    layer->taupdt = taupdt;
    layer->pmin = pmin;
    layer->khalf = khalf;
    layer->taubdt = taubdt;

    layer->initial_Ci = initial_Ci;
    layer->initial_Cj = initial_Cj;
    layer->initial_Cij = initial_Cij;

    network.layers_.push_back(layer);
  }

  void initiate_training(py::array_t<REAL> py_images,
                         py::array_t<REAL> py_labels) {
    py::buffer_info images_buffer = py_images.request();
    py::buffer_info labels_buffer = py_labels.request();

    dataset.number_of_examples = images_buffer.shape[0];
    dataset.rows = 1;  // TODO: Currently only occurs as rows * cols in the code
    dataset.cols = images_buffer.shape[1];
    dataset.number_of_classes = labels_buffer.shape[1];
    dataset.one_hot_label = true;

    dataset.images =
        new REAL[dataset.number_of_examples * dataset.rows * dataset.cols];
    dataset.labels =
        new REAL[dataset.number_of_examples * dataset.number_of_classes];

    memcpy(dataset.images, images_buffer.ptr,
           dataset.number_of_examples * dataset.rows * dataset.cols *
               sizeof(REAL));
    memcpy(
        dataset.labels, labels_buffer.ptr,
        dataset.number_of_examples * dataset.number_of_classes * sizeof(REAL));

    loader = new dataloader<REAL>(dataset, 8, 2);
  }

  void train_layer(size_t maximal_batch_size, size_t layer, size_t epochs) {
    network.train_layer(*loader, maximal_batch_size, layer, epochs);
  }

  void training_done() {
    loader->stop();
    delete loader;
    loader = nullptr;

    delete[] dataset.images;
    delete[] dataset.labels;
  }

  double evaluate(py::array_t<REAL> py_images, py::array_t<REAL> py_labels,
                  size_t batch_size) {
    py::buffer_info images_buffer = py_images.request();
    py::buffer_info labels_buffer = py_labels.request();

    dataset_t<REAL, REAL> testset;
    testset.number_of_examples = images_buffer.shape[0];
    testset.rows = 1;  // TODO: Currently only occurs as rows * cols in the code
    testset.cols = images_buffer.shape[1];
    testset.number_of_classes = labels_buffer.shape[1];
    testset.one_hot_label = true;

    testset.images =
        new REAL[testset.number_of_examples * testset.rows * testset.cols];
    testset.labels =
        new REAL[testset.number_of_examples * testset.number_of_classes];

    memcpy(testset.images, images_buffer.ptr,
           testset.number_of_examples * testset.rows * testset.cols *
               sizeof(REAL));
    memcpy(
        testset.labels, labels_buffer.ptr,
        testset.number_of_examples * testset.number_of_classes * sizeof(REAL));

    return network.evaluate(testset, batch_size);
  }

  Network<REAL> network;
  dataset_t<REAL, REAL> dataset;
  dataloader<REAL> *loader;
};

PYBIND11_MODULE(_bcpnn_backend_full_cuda_internals, m) {
  m.def("initialize", []() {
    seed_generator(generator);

    CUBLAS_CALL(cublasCreate(&handle));
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
#ifdef USE_CATALYST
    Adaptor::Initialize("BCPNN/viz/image.py", 28, 28, 15);
#endif
  });

  py::class_<PyNetwork<float>>(m, "PyNetwork_float32")
      .def(py::init<>())
      .def("add_dense_layer", &PyNetwork<float>::add_dense_layer)
      .def("add_plastic_layer", &PyNetwork<float>::add_plastic_layer)
      .def("initiate_training", &PyNetwork<float>::initiate_training)
      .def("train_layer", &PyNetwork<float>::train_layer)
      .def("training_done", &PyNetwork<float>::training_done)
      .def("evaluate", &PyNetwork<float>::evaluate);

  py::class_<PyNetwork<double>>(m, "PyNetwork_float64")
      .def(py::init<>())
      .def("add_dense_layer", &PyNetwork<double>::add_dense_layer)
      .def("add_plastic_layer", &PyNetwork<double>::add_plastic_layer)
      .def("initiate_training", &PyNetwork<double>::initiate_training)
      .def("train_layer", &PyNetwork<double>::train_layer)
      .def("training_done", &PyNetwork<double>::training_done)
      .def("evaluate", &PyNetwork<double>::evaluate);
}

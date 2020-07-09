#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>
#include <vector>

#include "dataset.h"

#include <algorithm>

#include <cuda.h>
#include <cuda_runtime_api.h>

//#include "helpers_cuda.h"
#include "helpers_random.h"

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
	    printf("Error at %s:%d\n",__FILE__,__LINE__);\
	    exit(EXIT_FAILURE);}} while(0)

namespace bcpnn {

namespace internal {

template<typename REAL>
void *
dataloader_queue_worker_thread(void * thread_data);

} // namespace internal

template<typename REAL>
class dataloader {
public:
  dataloader(dataset_t<REAL, REAL> & dataset, unsigned int queue_workers, unsigned int queue_size);

  dataset_t<REAL, REAL> & get_dataset();
  std::pair<REAL *, REAL *> queue_get_fresh();
  void queue_recycle(std::pair<REAL *, REAL *> p);

  void stop();

private:
  dataset_t<REAL, REAL> * dataset_;
  unsigned int queue_size_;
  unsigned int queue_workers_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_fresh_;
  std::condition_variable queue_cv_old_;
  std::queue<std::pair<REAL *, REAL *>> queue_fresh_;
  std::queue<std::pair<REAL *, REAL *>> queue_old_;
  std::atomic<int> running_;
  std::atomic<int> active_threads_;

  friend void * internal::dataloader_queue_worker_thread<REAL>(void * thead_data);
};

} // namespace bcpnn


namespace bcpnn {

namespace internal {

template<typename REAL>
void *
dataloader_queue_worker_thread(void * thread_data)
{
  dataloader<REAL> * loader = (dataloader<REAL> *)thread_data;
  dataset_t<REAL,REAL> * dataset = loader->dataset_;
  std::default_random_engine generator;
  ::bcpnn::helpers::random::seed_generator(generator);
  std::vector<size_t> idx;
  size_t features = dataset->rows * dataset->cols;
  size_t classes = dataset->number_of_classes;
  REAL * images;
  REAL * labels;

  CUDA_CALL(cudaMallocHost((void **)&images, dataset->number_of_examples * features * sizeof(REAL)));
  CUDA_CALL(cudaMallocHost((void **)&labels, dataset->number_of_examples * classes * sizeof(REAL)));

  for (size_t i = 0; i < dataset->number_of_examples; ++i) { idx.push_back(i); }

  std::unique_lock<std::mutex> lock(loader->queue_mutex_, std::defer_lock);
  while (loader->running_) {
    std::shuffle(idx.begin(), idx.end(), generator);

    for (size_t i = 0; i < dataset->number_of_examples; ++i) {
      size_t j = idx[i];

      for (size_t k = 0; k < features; ++k) { images[i * features + k] = dataset->images[j * features + k]; }
      for (size_t k = 0; k < classes;  ++k) { labels[i * classes  + k] = dataset->labels[j * classes  + k]; }
    }

    lock.lock();
    while (loader->running_.load() && loader->queue_old_.size() == 0) { loader->queue_cv_old_.wait(lock); }
    if (!loader->running_.load()) { break; }

    std::pair<REAL *, REAL *> p = loader->queue_old_.front();
    loader->queue_old_.pop();
    lock.unlock();

    CUDA_CALL(cudaMemcpy( p.first, images, dataset->number_of_examples * features * sizeof(REAL), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(p.second, labels, dataset->number_of_examples * classes  * sizeof(REAL), cudaMemcpyHostToDevice));

    lock.lock();
    loader->queue_fresh_.push(p);
    loader->queue_cv_fresh_.notify_one();
    lock.unlock();
  }

  loader->active_threads_ -= 1;
  return NULL;
}

} // namespace internal

template<typename REAL>
dataloader<REAL>::dataloader(dataset_t<REAL, REAL> & dataset, unsigned int queue_workers, unsigned int queue_size)
{
  std::lock_guard<std::mutex> lock(queue_mutex_);

  dataset_ = &dataset;

  active_threads_.store(0);
  running_.store(1);

  size_t n_inputs = dataset.rows * dataset.cols;
  size_t n_outputs = dataset.number_of_classes;
  for (size_t i = 0; i < queue_size; ++i) {
    std::pair<REAL *, REAL *> p;
    CUDA_CALL(cudaMalloc((void **)&p.first,  dataset.number_of_examples * n_inputs  * sizeof(REAL)));
    CUDA_CALL(cudaMalloc((void **)&p.second, dataset.number_of_examples * n_outputs * sizeof(REAL)));

    queue_old_.push(p);
  }

  for (size_t i = 0; i < queue_workers; ++i) {
    pthread_t thread_id = { 0 };
    pthread_attr_t attr;

    pthread_attr_init(&attr);

    active_threads_ += 1;
    pthread_create(&thread_id, &attr, &internal::dataloader_queue_worker_thread<REAL>, (void *)this);
  }
}

template<typename REAL>
dataset_t<REAL, REAL> &
dataloader<REAL>::get_dataset()
{
  return *dataset_;
}

template<typename REAL>
std::pair<REAL *, REAL *>
dataloader<REAL>::queue_get_fresh()
{
  std::unique_lock<std::mutex> lock(queue_mutex_);
  while (running_ && queue_fresh_.size() == 0) { queue_cv_fresh_.wait(lock); }
  if (!running_ && queue_fresh_.size() == 0) { return std::pair<REAL *, REAL *>(NULL, NULL); }

  std::pair<REAL *, REAL *> p = queue_fresh_.front();
  queue_fresh_.pop();

  return p;
}

template<typename REAL>
void
dataloader<REAL>::queue_recycle(std::pair<REAL *, REAL *> p)
{
  std::lock_guard<std::mutex> lock(queue_mutex_);
  queue_old_.push(p);
  queue_cv_old_.notify_one();
}

template<typename REAL>
void
dataloader<REAL>::stop()
{
  std::unique_lock<std::mutex> lock(queue_mutex_);
  running_.store(0);
  queue_cv_fresh_.notify_all();
  queue_cv_old_.notify_all();
  lock.unlock();

  while (active_threads_.load() > 0) {}
}

} // namespace bcpnn


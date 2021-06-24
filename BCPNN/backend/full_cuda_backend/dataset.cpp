#include "dataset.h"

#include <fstream>

namespace bcpnn {

dataset_t<uint8_t, uint8_t> read_mnist_dataset(char const* images_file,
                                               char const* labels_file) {
  dataset_t<uint8_t, uint8_t> dataset;
  uint8_t u8;
  uint32_t u32;
  uint32_t num_images;
  uint32_t num_labels;

  dataset.images = NULL;
  dataset.labels = NULL;
  dataset.number_of_examples = 0;
  dataset.rows = 0;
  dataset.cols = 0;
  dataset.number_of_classes = 10;
  dataset.one_hot_label = false;

  std::ifstream in_images(images_file, std::ios::binary);
  std::ifstream in_labels(labels_file, std::ios::binary);

  if (!in_images || !in_labels) {
    std::cerr << "Could not open image file or label file!" << std::endl;
    exit(1);
  }

  in_images.seekg(4);
  in_labels.seekg(4);

  in_images.read((char*)&num_images, sizeof(uint32_t));
  num_images = __builtin_bswap32(num_images);
  in_labels.read((char*)&num_labels, sizeof(uint32_t));
  num_labels = __builtin_bswap32(num_labels);

  if (num_images != num_labels) {
    exit(1);
  }

  dataset.number_of_examples = num_images;

  in_images.read((char*)&dataset.rows, sizeof(uint32_t));
  dataset.rows = __builtin_bswap32(dataset.rows);
  in_images.read((char*)&dataset.cols, sizeof(uint32_t));
  dataset.cols = __builtin_bswap32(dataset.cols);

  size_t images_len = dataset.number_of_examples * dataset.rows * dataset.cols;
  size_t labels_len = dataset.number_of_examples;

  dataset.images = (uint8_t*)malloc(images_len);
  dataset.labels = (uint8_t*)malloc(labels_len);

  if (dataset.images == NULL || dataset.labels == NULL) {
    return dataset;
  }

  in_images.read((char*)dataset.images, images_len);
  in_labels.read((char*)dataset.labels, labels_len);

  return dataset;
}

}  // namespace bcpnn

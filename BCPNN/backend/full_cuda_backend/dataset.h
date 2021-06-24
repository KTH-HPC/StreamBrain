#pragma once

#include <cstdint>
#include <iostream>

namespace bcpnn {

template <typename D, typename L>
struct dataset {
  D* images;
  L* labels;

  uint32_t number_of_examples;
  uint32_t rows;
  uint32_t cols;
  uint32_t number_of_classes;
  bool one_hot_label;
};

template <typename D, typename L>
using dataset_t = struct dataset<D, L>;

template <typename REAL>
dataset_t<REAL, REAL> convert_dataset(dataset<uint8_t, uint8_t> dataset) {
  dataset_t<REAL, REAL> d;

  d.images = (REAL*)malloc(sizeof(REAL) * dataset.rows * dataset.cols *
                           dataset.number_of_examples);
  d.labels = (REAL*)malloc(sizeof(REAL) * dataset.number_of_classes *
                           dataset.number_of_examples);
  d.number_of_examples = dataset.number_of_examples;
  d.rows = dataset.rows;
  d.cols = dataset.cols;
  d.number_of_classes = dataset.number_of_classes;
  d.one_hot_label = true;

  uint8_t* img_1 = dataset.images;
  REAL* img_2 = d.images;
  for (size_t example = 0; example < d.number_of_examples; ++example) {
    for (size_t row = 0; row < d.rows; ++row) {
      for (size_t col = 0; col < d.cols; ++col) {
        *img_2 = ((REAL)(*img_1)) / 255;

        ++img_1;
        ++img_2;
      }
    }
  }

  uint8_t* label_1 = dataset.labels;
  REAL* label_2 = d.labels;
  for (size_t example = 0; example < d.number_of_examples; ++example) {
    for (size_t c = 0; c < d.number_of_classes; ++c) {
      label_2[c] = 0;
    }
    label_2[*label_1] = 1;

    ++label_1;
    label_2 += d.number_of_classes;
  }

  return d;
}

dataset_t<uint8_t, uint8_t> read_mnist_dataset(char const* images_file,
                                               char const* labels_file);

template <typename D, typename L>
void print_dataset(dataset_t<D, L> const& dataset) {
  std::cout << "Number of examples: " << dataset.number_of_examples
            << std::endl;
  std::cout << "Rows: " << dataset.rows << std::endl;
  std::cout << "Columns: " << dataset.cols << std::endl;
  std::cout << std::endl;
  std::cout << "Example image:" << std::endl;
  std::cout << std::endl;

  size_t example = 19;

  D* p = dataset.images + example * dataset.rows * dataset.cols;
  for (size_t i = 0; i < 28; ++i) {
    for (size_t j = 0; j < 28; ++j) {
      std::cout << std::dec << (*p > 0);
      ++p;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  if (dataset.one_hot_label) {
    std::cout << "Label: ";

    L const* label = dataset.labels + dataset.number_of_classes * example;
    for (size_t l = 0; l < dataset.number_of_classes; ++l) {
      if (label[l]) {
        std::cout << l << " ";
      }
    }

    std::cout << std::endl;
  } else {
    std::cout << "Label: " << ((unsigned int)dataset.labels[example])
              << std::endl;
  }
}

}  // namespace bcpnn

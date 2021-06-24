#pragma once

namespace bcpnn {

namespace helpers {

namespace training {

void initialize_wmask(uint8_t* wmask, size_t n, size_t hypercolumns);

void print_wmask(uint8_t* wmask, size_t rows, size_t columns,
                 size_t hypercolumns);

void cpu_correct_predictions(int* correct_count, float* activation_2,
                             float* batch_labels, size_t batch_size, size_t m);

}  // namespace training

}  // namespace helpers

}  // namespace bcpnn

#include <iostream>
#include <cmath>

#include "testing.h"

int
main()
{
  float delta;

  size_t const iterations = 100;

  delta = 0;
  for (size_t i = 0; i < iterations; ++i) {
    delta = fmaxf(delta, test_add_bias());
  }
  std::cout << "Max delta add_bias(): " << delta << std::endl;

  delta = 0;
  for (size_t i = 0; i < iterations; ++i) {
    delta = fmaxf(delta, test_softmax());
  }
  std::cout << "Max delta softmax(): " << delta << std::endl;

  delta = 0;
  for (size_t i = 0; i < iterations; ++i) {
    delta = fmaxf(delta, test_update_counters());
  }
  std::cout << "Max delta update_counters(): " << delta << std::endl;

  delta = 0;
  for (size_t i = 0; i < iterations; ++i) {
    delta = fmaxf(delta, test_update_weights());
  }
  std::cout << "Max delta update_weights(): " << delta << std::endl;

  delta = 0;
  for (size_t i = 0; i < iterations; ++i) {
    delta = fmaxf(delta, test_update_bias());
  }
  std::cout << "Max delta update_bias(): " << delta << std::endl;

  delta = 0;
  for (size_t i = 0; i < iterations; ++i) {
    delta = fmaxf(delta, test_update_bias_regularized());
  }
  std::cout << "Max delta update_bias_regularized(): " << delta << std::endl;

  delta = 0;
  for (size_t i = 0; i < iterations; ++i) {
    delta = fmaxf(delta, test_update_mask());
  }
  std::cout << "Max delta update_mask(): " << delta <<  "\tThis is usually flakey. Ends up with several inputs with score 0 and thus several different options. There are options in the kernels to print more detailed information." << std::endl;

  delta = 0;
  for (size_t i = 0; i < iterations; ++i) {
    delta = fmaxf(delta, test_apply_mask());
  }
  std::cout << "Max delta apply_mask(): " << delta << std::endl;

  return 0;
}

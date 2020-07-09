#pragma once

#include <chrono>
#include <random>

namespace bcpnn {

namespace helpers {

namespace random {

template<typename Generator>
void
seed_generator(Generator & g)
{
  typedef std::chrono::high_resolution_clock clock;
  clock::time_point p = clock::now();

  auto s = std::chrono::duration_cast<std::chrono::nanoseconds>(p.time_since_epoch()).count();

  g.seed(s);
}

} // namespace random

} // namespace helpers

} // namespace bcpnn

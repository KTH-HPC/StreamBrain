#ifndef ADAPTOR_H
#define ADAPTOR_H

#include <cstddef>
#include <cstdint>

namespace Adaptor
{
void Initialize(const char* script, const size_t rows, const size_t columns, const size_t hypercolumns);

void Finalize();

void CoProcess(double time, unsigned int timeStep, uint8_t *wmask);
}

#endif

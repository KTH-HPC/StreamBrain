cmake_minimum_required(VERSION 3.10)
project(BCPNN CXX)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

add_subdirectory(pybind11)
#set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

if(NOT BLA_VENDOR)
  set(BLA_VENDOR All)
endif()

if(NOT ARCH)
  set(ARCH "-march=native")
else()
  set(ARCH "-march=${ARCH}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")

if(NOT BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS ON)
endif()

find_package(OpenMP)
find_package(BLAS)

if(OpenMP_CXX_FOUND AND BLAS_FOUND)
  pybind11_add_module(_bcpnn_kernels_openmp_internal BCPNN/backend/_kernels/kernels_openmp_internal.cc)
  target_include_directories(_bcpnn_kernels_openmp_internal PRIVATE
    ${BLAS_INCLUDE_DIRS})
  target_link_libraries(_bcpnn_kernels_openmp_internal PRIVATE
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_CXX)
  target_compile_options(_bcpnn_kernels_openmp_internal PUBLIC "-O3")
  target_compile_options(_bcpnn_kernels_openmp_internal PUBLIC "${ARCH}")
endif()

find_package(MPI)

if(MPI_CXX_FOUND AND OpenMP_CXX_FOUND AND BLAS_FOUND)
  pybind11_add_module(_bcpnn_kernels_mpi_internal BCPNN/backend/_kernels/kernels_mpi_internal.cc)
  target_include_directories(_bcpnn_kernels_mpi_internal PRIVATE
    ${BLAS_INCLUDE_DIRS})
  target_link_libraries(_bcpnn_kernels_mpi_internal PRIVATE
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX)
  target_compile_options(_bcpnn_kernels_mpi_internal PUBLIC "-O3")
  target_compile_options(_bcpnn_kernels_mpi_internal PUBLIC "${ARCH}")
endif()

find_package(CUDA)
if(CUDA_FOUND)
  enable_language(CUDA)
  # CUDA Kernel backend
  pybind11_add_module(_bcpnn_kernels_cuda_internal MODULE
    BCPNN/backend/_kernels/kernels_cuda_internal.cu
  )
  set_property(TARGET _bcpnn_kernels_cuda_internal PROPERTY CUDA_ARCHITECTURES 30 35 60 61 70 75 80)
  target_link_libraries(_bcpnn_kernels_cuda_internal PRIVATE curand cublas)
  # Fully offloaded CUDA backend
  pybind11_add_module(_bcpnn_backend_full_cuda_internals MODULE
    BCPNN/backend/full_cuda_backend/backend_full_cuda_internals.cu
    BCPNN/backend/full_cuda_backend/dataset.cpp
    BCPNN/backend/full_cuda_backend/helpers_training.cu
    BCPNN/backend/full_cuda_backend/helpers_cuda.cu
  )
  set_property(TARGET _bcpnn_backend_full_cuda_internals PROPERTY CUDA_ARCHITECTURES 30 35 60 61 70 75 80)
  target_link_libraries(_bcpnn_backend_full_cuda_internals PRIVATE curand cublas)
endif()

# greendog
#nvcc -Xcompiler "-fPIC" -O3 --shared -std=c++14 `python3 -m pybind11 --includes` -o kernels_cuda_internal`python3-config --extension-suffix` kernels_cuda_internal.cu  -lcuda -lcublas

#g++ -fopenmp -fPIC -O3 --shared -std=c++14 `python3 -m pybind11 --includes` -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl kernels_openmp_internal.cc -o kernels_openmp_internal`python3-config --extension-suffix`

# Kebnekaise
#nvcc -arch=sm_70 -Xcompiler "-fPIC" -O3 --shared -std=c++14 -I/pfs/nobackup/home/w/wdchien/kebnekaise_local/src/StreamBrain/pybind11/include -I/hpc2n/eb/software/Compiler/GCCcore/8.2.0/Python/3.7.2/include/python3.7m -o kernels_cuda_internal`python3-config --extension-suffix` kernels_cuda_internal.cu  -lcuda -lcublas

#g++ -fopenmp -fPIC -O3 -march=skylake-avx512 --shared -std=c++14 -I/pfs/nobackup/home/w/wdchien/kebnekaise_local/src/StreamBrain/pybind11/include -I/hpc2n/eb/software/Compiler/GCCcore/8.2.0/Python/3.7.2/include/python3.7m -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl kernels_openmp_internal.cc -o kernels_openmp_internal`python3-config --extension-suffix`

bcpnn.out: bcpnn.cu dataset.cpp dataloader.cpp helpers_training.cu
	nvcc -O3 -o bcpnn.out bcpnn.cu dataset.cpp dataloader.cpp helpers_training.cu -lcuda -lcublas -lcurand

#nvcc -Xcompiler "-fPIC" -O3 --shared -std=c++14 `python3 -m pybind11 --includes` -o backend_full_cuda_internals`python3-config --extension-suffix` backend_full_cuda_internals.cu dataset.cpp  helpers_training.cu helpers_cuda.cu -lcuda -lcublas -lcurand

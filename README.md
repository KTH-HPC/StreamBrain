# StreamBrain
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

StreamBrain is a framework for implementing BCPNN networks in Python3.
## Installation
StreamBrain requires a number of dependencies, including Numpy (MKL), CMake, GCC, CUDA (Optional), and FPGA toolchain (Optional). PyBind11 is included as a Git module and a patch needs to be applied to its CMake module before setting up.
```
git submodule update --init --recursive
cd pybind11
git apply ../pybind11_cmake.patch
cd ..
python setup.py install --user
OR
python setup.py develop --user
```
## Running examples
We include an example training script for MNIST digit and Fashion MNIST classification.
### Setting up the dataset
1) Getting the dataset in the project home directory
```
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```
OR
```
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
```
2) Unpack the datasets
```
gzip -d train-images-idx3-ubyte.gz
gzip -d train-labels-idx1-ubyte.gz
gzip -d t10k-images-idx3-ubyte.gz
gzip -d t10k-labels-idx1-ubyte.gz
```
### Training
Specify a backend using the environment variable BCPNN_BACKEND. There are a number of backends provided:
- Numpy, default backend if the variable is not set (BCPNN_BACKEND=numpy)
- Vectorization and OpenMP threading for CPU (BCPNN_BACKEND=cpu)
- GPU offloaded computation kernels (BCPNN_BACKEND=gpu)
- Fully GPU offloaded full layers (BCPNN_BACKEND=full_cuda)
- FPGA (not available currently) (BCPNN_BACKEND=fpga)

Run the training script and specify the batch sizes, 128 for example.
```
export BCPNN_BACKEND=cpu
python3 train.py 128
```
StreamBrain will display the name of the backend that is being used as well as a training progress. The test accuracy will be displayed after evaluation at the end.
```
Using GPU backend.
Layer - 1/2
Epoch 1/15: 100%|███████████████████████████████████████████████| 469/469 [00:17<00:00, 26.27it/s]
Layer - 1/2
Epoch 2/15:   3%|████                                           | 16/469 [00:01<00:44, 10.09it/s]
.
.
.
.
.
.
Layer - 2/2
Epoch 25/25: 100%|██████████████████████████████████████████████| 469/469 [00:06<00:00, 72.74it/s]
Evaluation: 100%|███████████████████████████████████████████████| 79/79 [00:01<00:00, 76.39it/s]
Training duration: 520.6510210037231
Testing duration: 1.0513603687286377
Accuracy: 0.9326
```
## License
StreamBrain is developed by Stefano Markidis, Martin Svedi, Artur Podobas, and Steven W. D. Chien. The software is released under BSD 2-Clause license. See the LICENSE for details.

# StreamBrain
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

StreamBrain is a framework that enables the practical deployment of neural networks that are based on the brain-like Bayesian Confidence Propagation Neural Network (BCPNN). In particular, StreamBrain is a domain-specific language (DSL), similar in concept to existing machine learning (ML) frameworks, that aims to allow the use of BCPNN networks on High-Performance Computing systems. The framework supports a variety of backends, such as CPUs, GPUs, and even FPGAs. We provide a set of example training scripts to train for the classification of MNIST, Fashion MNIST, and STL-10.

# Installation
StreamBrain requires a number of dependencies, including Numpy (MKL), CMake, GCC, CUDA (Optional), and FPGA toolchain (Optional). PyBind11 is included as a Git module and must be fetched before building StreamBrain.
```bash
git submodule update --init --recursive
pip install -r requirements.txt
python setup.py install --user
```

# Setting up backend environments
To select the backend, set the environment variable `BCPNN_BACKEND`. We currently support the following backends:
- Numpy, default backend if the variable is not set (`export BCPNN_BACKEND=numpy`)
- CPU Backend kernels with vectorization and OpenMP threading (`export BCPNN_BACKEND=cpu`)
- MPI+OpenMP backend kernels with data parallelism on training batch (`export BCPNN_BACKEND=mpi`)
- Fully offloaded GPU backend that runs entirely in C++/CUDA (`export BCPNN_BACKEND=full_cuda`)
- FPGA (not available currently) (`export BCPNN_BACKEND=fpga`)

Furthermore, to select the number of threads set the OpenMP environment variable `export OMP_NUM_THREADS=X` where `X` is the number of threads.

# Getting datasets
We include an example training script for MNIST digit, Fashion MNIST, and STL-10 classification. First, we need to get the datasets

## MNIST
Get the MNIST dataset by downloading and expanding it.
```bash
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gzip -d train-images-idx3-ubyte.gz
gzip -d train-labels-idx1-ubyte.gz
gzip -d t10k-images-idx3-ubyte.gz
gzip -d t10k-labels-idx1-ubyte.gz
```

## Fashion MNIST
Fashion MNIST is a drop-in replacement of MNIST.
```bash
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
gzip -d train-images-idx3-ubyte.gz
gzip -d train-labels-idx1-ubyte.gz
gzip -d t10k-images-idx3-ubyte.gz
gzip -d t10k-labels-idx1-ubyte.gz
```

## STL-10
STL-10 is a color image dataset that is partly labeled and partly unlabelled.
```bash
wget http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz
tar -xvzf --strip-components=1 stl10_binary.tar.gz
```

# Training
To train, run the example training script `train.py` and specify the dataset name, precision, and batch size, such as the following:
```bash
$ python3 -u train.py <mnist/fashion_mnist/stl-10> <single/double> <batch size>
```
To run the MPI backend, one runs the exact command, but prepend `mpirun -np Y ...` where `Y` is the number of MPI processes. Note that the Number of OpenMP threads should be adjusted accordingly.

To run with the CPU backend, we can do the following:
```
$ OMP_NUM_THREADS=8 BCPNN_BACKEND=cpu python -u train.py mnist single 128
Using CPU backend.
Dataset: mnist Batch size: 128 precision: <class 'numpy.float32'>
Layer - 1/2
Epoch 1/30: 100%|█████████████████████████████████████████████| 469/469 [00:15<00:00, 30.05it/s]
Layer - 1/2
Epoch 2/30: 100%|█████████████████████████████████████████████| 469/469 [00:11<00:00, 41.42it/s]
Layer - 1/2
Epoch 3/30: 100%|█████████████████████████████████████████████| 469/469 [00:15<00:00, 30.75it/s]
Layer - 1/2
..........
Epoch 58/60: 100%|████████████████████████████████████████████| 469/469 [00:01<00:00, 415.18it/s]
Layer - 1/2
Epoch 59/60: 100%|████████████████████████████████████████████| 469/469 [00:01<00:00, 414.92it/s]
Layer - 1/2
Epoch 60/60: 100%|████████████████████████████████████████████| 469/469 [00:01<00:00, 414.13it/s]
Evaluation: 100%|█████████████████████████████████████████████| 79/79 [00:00<00:00, 537.69it/s]
0 420.7588653564453 0.15649056434631348 [9518] [10000]
Training duration: 420.7588653564453
Testing duration:  0.15649056434631348
Accuracy:          [0.9518]
```
The MNIST training with the provided hyperparameters should give approximately 95% test accuracy, whereas the Fashion MNIST should give approximately 74% test accuracy.

# Cite us
If you find our work useful, we would appreciate that you cite us:
```bibtex
@article{podobas2021streambrain,
  title={StreamBrain: An HPC Framework for Brain-like Neural Networks on CPUs, GPUs and FPGAs},
  author={Podobas, Artur and Svedin, Martin and Chien, Steven WD and Peng, Ivy B and Ravichandran, Naresh Balaji and Herman, Pawel and Lansner, Anders and Markidis, Stefano},
  journal={arXiv preprint arXiv:2106.05373},
  year={2021}
}
```
StreamBrain is published at the International Symposium on Highly Efficient Accelerators and Reconfigurable Technologies (HEART 2021).

# License
StreamBrain is developed by Martin Svedi, Artur Podobas, and Steven W. D. Chien, and Stefano Markidis. The software is released under the BSD 2-Clause license. See LICENSE for details.

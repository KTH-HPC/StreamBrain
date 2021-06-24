import numpy as np

import _bcpnn_backend_full_cuda_internals as backend_cuda
backend_cuda.initialize()

class DenseLayer:
    def __init__(self, in_features, hypercolumns, minicolumns, taupdt, initial_counters, dtype=None):
        self.in_features = in_features
        self.hypercolumns = hypercolumns
        self.minicolumns = minicolumns
        self.out_features = hypercolumns * minicolumns

        self.taupdt = taupdt

        self.initial_counters = initial_counters

class StructuralPlasticityLayer:
    def __init__(self, in_features, hypercolumns, minicolumns, taupdt, khalf, pmin, taubdt, density, mask_iterations, initial_counters, dtype=None):
        self.in_features = in_features
        self.hypercolumns = hypercolumns
        self.minicolumns = minicolumns
        self.out_features = hypercolumns * minicolumns

        self.taupdt = taupdt
        self.khalf = khalf
        self.pmin = pmin
        self.taubdt = taubdt

        self.density = density
        self.mask_iterations = mask_iterations

        self.initial_counters = initial_counters

class Network:
    def __init__(self, dtype):
        self.dtype = dtype
        self.net = None
        self._layers = []

        if self.dtype == np.float32:
            self.net = backend_cuda.PyNetwork_float32()
        elif self.dtype == np.float64:
            self.net = backend_cuda.PyNetwork_float64()
        else:
            raise Exception("Unsupported dtype")

    def add_layer(self, layer):
        self._layers.append(layer)
        if isinstance(layer, DenseLayer):
            cs = layer.initial_counters
            self.net.add_dense_layer(layer.in_features, layer.hypercolumns, layer.minicolumns, layer.taupdt, cs[0], cs[1], cs[2])
        elif isinstance(layer, StructuralPlasticityLayer):
            cs = layer.initial_counters
            self.net.add_plastic_layer(layer.in_features, layer.hypercolumns, layer.minicolumns, layer.taupdt, layer.pmin, layer.khalf, layer.taubdt, cs[0], cs[1], cs[2])
        else:
            raise Exception("Unknown layer type:", layer)

    def fit(self, training_data, training_labels, maximal_batch_size, schedule):
        training_data   = training_data.astype(self.dtype)
        training_labels = training_labels.astype(self.dtype)
        self.net.initiate_training(training_data, training_labels)

        for layer, epochs in schedule:
            self.net.train_layer(maximal_batch_size, layer, epochs)

    def evaluate(self, images, labels, batch_size):
        images = images.astype(self.dtype)
        labels = labels.astype(self.dtype)

        return self.net.evaluate(images, labels, batch_size) * images.shape[0], images.shape[0]

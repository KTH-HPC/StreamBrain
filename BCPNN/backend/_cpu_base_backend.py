import sys
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext


class DenseLayer:
    _update_state = None
    _softmax_minicolumns = None
    _update_counters = None
    _update_weights = None
    _update_bias = None

    def __init__(
            self,
            in_features,
            hypercolumns,
            minicolumns,
            taupdt,
            initial_counters,
            dtype=np.float64):
        self.in_features = in_features
        self.hypercolumns = hypercolumns
        self.minicolumns = minicolumns
        self.out_features = hypercolumns * minicolumns

        self.taupdt = taupdt

        self.dtype = dtype

        self.weights = (
            0.1 *
            np.random.randn(
                self.in_features,
                self.out_features)).astype(dtype)
        self.bias = (0.1 * np.random.rand(self.out_features)).astype(dtype)

        self.Ci = initial_counters[0] * np.ones([in_features]).astype(dtype)
        self.Cj = initial_counters[1] * \
            np.ones([self.out_features]).astype(dtype)
        self.Cij = initial_counters[2] * \
            np.ones([self.in_features, self.out_features]).astype(dtype)

    def compute_activation(self, inputs):
        activations = np.zeros(
            [inputs.shape[0], self.out_features], dtype=self.dtype)
        activations = self._update_state(
            activations, self.weights, self.bias, inputs)
        activations = self._softmax_minicolumns(
            activations, self.hypercolumns, self.minicolumns)
        return activations

    def convert(self, dtype):
        self.dtype = dtype
        self.weights = self.weights.astype(dtype)
        self.bias = self.bias.astype(dtype)
        self.Ci = self.Ci.astype(dtype)
        self.Cj = self.Cj.astype(dtype)
        self.Cij = self.Cij.astype(dtype)

    def train_step(self, inputs, outputs):
        self.Ci, self.Cj, self.Cij = self._update_counters(
            self.Ci, self.Cj, self.Cij, inputs, outputs, self.taupdt)

    def train_finalize(self):
        self.weights = self._update_weights(
            self.weights, self.Ci, self.Cj, self.Cij, self.taupdt / 2)
        self.bias = self._update_bias(self.bias, self.Cj, self.taupdt / 2)


class StructuralPlasticityLayer:
    _update_state = None
    _softmax_minicolumns = None
    _update_counters = None
    _update_weights = None
    _update_bias = None
    _update_mask = None
    _apply_mask = None

    def __init__(
            self,
            in_features,
            hypercolumns,
            minicolumns,
            taupdt,
            khalf,
            pmin,
            taubdt,
            density,
            mask_iterations,
            initial_counters,
            dtype=np.float64):
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

        self.dtype = dtype

        self.weights = (
            0.1 *
            np.random.randn(
                self.in_features,
                self.out_features)).astype(dtype)
        self.bias = (0.1 * np.random.rand(self.out_features)).astype(dtype)

        self.Ci = initial_counters[0] * np.ones([in_features]).astype(dtype)
        self.Cj = initial_counters[1] * \
            np.ones([self.out_features]).astype(dtype)
        self.Cij = initial_counters[2] * \
            np.ones([self.in_features, self.out_features]).astype(dtype)
        self.kbi = np.ones([self.out_features]).astype(dtype)
        self.wmask = (
            np.random.rand(
                self.in_features,
                self.hypercolumns) < self.density).astype(
            np.uint8)

    def compute_activation(self, inputs):
        activations = np.zeros(
            [inputs.shape[0], self.out_features], dtype=self.dtype)
        activations = self._update_state(
            activations, self.weights, self.bias, inputs)
        activations = self._softmax_minicolumns(
            activations, self.hypercolumns, self.minicolumns)
        return activations

    def convert(self, dtype):
        self.dtype = dtype
        self.weights = self.weights.astype(dtype)
        self.bias = self.bias.astype(dtype)
        self.Ci = self.Ci.astype(dtype)
        self.Cj = self.Cj.astype(dtype)
        self.Cij = self.Cij.astype(dtype)
        self.kbi = self.kbi.astype(dtype)

    def train_step(self, inputs, outputs, hypercolumn=None):
        self.Ci, self.Cj, self.Cij = self._update_counters(
            self.Ci, self.Cj, self.Cij, inputs, outputs, self.taupdt)

        self.weights = self._update_weights(
            self.weights, self.Ci, self.Cj, self.Cij, self.taupdt / 2)
        self.bias, self.kbi = self._update_bias(
            self.bias, self.kbi, self.Cj, self.taupdt / 2, self.khalf, self.pmin, self.taubdt)
        if hypercolumn is not None:
            #print("Updating hypercolumn:", hypercolumn)
            self.wmask = self._update_mask(
                self.wmask,
                self.weights,
                self.Ci,
                self.Cj,
                self.Cij,
                self.taupdt / 2,
                self.hypercolumns,
                self.minicolumns,
                hypercolumn,
                self.mask_iterations)
        self.weights = self._apply_mask(
            self.weights,
            self.wmask,
            self.hypercolumns,
            self.minicolumns)

    def train_finalize(self):
        pass


class Network:
    def __init__(self, dtype):
        self.dtype = dtype
        self._layers = []
        self.world_rank = 0
        self.world_size = 1

    def add_layer(self, layer):
        if layer.dtype != self.dtype:
            layer.convert(self.dtype)

        self._layers.append(layer)

    def fit(
            self,
            training_data,
            training_labels,
            maximal_batch_size,
            schedule):
        training_data = training_data.astype(self.dtype)
        training_labels = training_labels.astype(self.dtype)

        for layer, epochs in schedule:
            self._train_layer(
                layer,
                maximal_batch_size,
                training_data,
                training_labels,
                epochs)

    def evaluate(self, images, labels, maximal_batch_size):
        images = images.astype(self.dtype)
        labels = labels.astype(self.dtype)

        correct = np.array([0])
        total = np.array([0])
        number_of_batches = (
            images.shape[0] + maximal_batch_size - 1) // maximal_batch_size

        if self.world_rank == 0:
            cm = tqdm(total=number_of_batches)
        else:
            cm = nullcontext()

        with cm as pbar:
            if self.world_rank == 0:
                pbar.set_description('Evaluation')
            for i in range(number_of_batches):
                global_start = i * maximal_batch_size
                global_end = global_start + maximal_batch_size if global_start + \
                    maximal_batch_size <= images.shape[0] else images.shape[0]
                local_batch_size = (
                    global_end - global_start) // self.world_size

                start_sample = global_start + self.world_rank * local_batch_size
                end_sample = start_sample + local_batch_size

                batch_images = images[start_sample:end_sample, :]
                batch_labels = labels[start_sample:end_sample, :]

                activations = batch_images
                for layer in self._layers:
                    activations = layer.compute_activation(activations)

                correct += (np.argmax(activations, axis=1) ==
                            np.argmax(batch_labels, axis=1)).sum()
                total += batch_images.shape[0]
                if self.world_rank == 0:
                    pbar.update(1)

        return correct, total

    def _train_layer(
            self,
            layer,
            maximal_batch_size,
            images,
            labels,
            epochs):
        for epoch in range(epochs):
            if self.world_rank == 0:
                print('Layer - %d/%d' %
                      (layer + 1, len(self._layers)), flush=True)
            idx = np.random.permutation(range(images.shape[0]))
            shuffled_images = images[idx, :]
            shuffled_labels = labels[idx, :]

            n_hypercolumns = self._layers[layer].hypercolumns
            hypercolumns_shuffled = np.random.permutation(
                range(n_hypercolumns))

            number_of_batches = (
                images.shape[0] + maximal_batch_size - 1) // maximal_batch_size
            local_batch_size = maximal_batch_size // self.world_size

            if self.world_rank == 0:
                cm = tqdm(total=number_of_batches)
            else:
                cm = nullcontext()

            with cm as pbar:
                if self.world_rank == 0:
                    pbar.set_description('Epoch %d/%d' % (epoch + 1, epochs))
                for i in range(number_of_batches):
                    global_start = i * maximal_batch_size
                    global_end = global_start + maximal_batch_size if global_start + \
                        maximal_batch_size <= images.shape[0] else images.shape[0]
                    local_batch_size = (
                        global_end - global_start) // self.world_size

                    start_sample = global_start + self.world_rank * local_batch_size
                    end_sample = start_sample + local_batch_size
                    batch_images = shuffled_images[start_sample:end_sample, :]
                    batch_labels = shuffled_labels[start_sample:end_sample, :]

                    prev_activation = None
                    activation = batch_images
                    for l in range(layer + 1):
                        prev_activation = activation
                        activation = self._layers[l].compute_activation(
                            prev_activation)

                    if epoch > 0 and i % (
                            number_of_batches // (n_hypercolumns + 1)) == 0:
                        h = i // (number_of_batches // (n_hypercolumns + 1))
                        h = hypercolumns_shuffled[h] if h < n_hypercolumns else None
                    else:
                        h = None

                    if layer + 1 == len(self._layers):
                        self._layers[layer].train_step(
                            prev_activation, batch_labels)
                    else:
                        self._layers[layer].train_step(
                            prev_activation, activation, h)

                    if self.world_rank == 0:
                        pbar.update(1)

                self._layers[layer].train_finalize()

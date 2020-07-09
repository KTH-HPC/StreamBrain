from . import _cpu_base_backend as B
from ._cpu_base_backend import Network

from ._kernels import kernels_numpy, kernels_openmp

class DenseLayer(B.DenseLayer):
    #_update_state        = staticmethod(kernels_numpy.update_state)
    _update_state        = staticmethod(kernels_openmp.update_state)
    _softmax_minicolumns = staticmethod(kernels_numpy.softmax_minicolumns)
    _update_bias         = staticmethod(kernels_numpy.update_bias)

    def train_step(self, inputs, outputs):
        self.Ci, self.Cj, self.Cij, self.weights = kernels_openmp.update_weights(self.Ci, self.Cj, self.Cij, inputs, outputs, self.taupdt, self.weights, self.taupdt/2)

    def train_finalize(self):
        self.bias = self._update_bias(self.bias, self.Cj, self.taupdt/2)

class StructuralPlasticityLayer(B.StructuralPlasticityLayer):
    #_update_state        = staticmethod(kernels_numpy.update_state)
    _update_state        = staticmethod(kernels_openmp.update_state)
    _softmax_minicolumns = staticmethod(kernels_numpy.softmax_minicolumns)
    _update_bias         = staticmethod(kernels_numpy.update_bias_regularized)
    _update_mask         = staticmethod(kernels_numpy.update_mask)
    _apply_mask          = staticmethod(kernels_numpy.apply_mask)

    def train_step(self, inputs, outputs, hypercolumn=None):
        self.Ci, self.Cj, self.Cij, self.weights = kernels_openmp.update_weights(self.Ci, self.Cj, self.Cij, inputs, outputs, self.taupdt, self.weights, self.taupdt/2)

        self.bias, self.kbi = self._update_bias(self.bias, self.kbi, self.Cj, self.taupdt/2, self.khalf, self.pmin, self.taubdt)
        if hypercolumn is not None:
            #print("Updating hypercolumn:", hypercolumn)
            self.wmask = self._update_mask(self.wmask, self.weights, self.Ci, self.Cj, self.Cij, self.taupdt/2, self.hypercolumns, self.minicolumns, hypercolumn, 16)
        self.weights = self._apply_mask(self.weights, self.wmask, self.hypercolumns, self.minicolumns)

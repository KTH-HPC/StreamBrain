from . import _cpu_base_backend as B
from ._cpu_base_backend import Network

from ._kernels import kernels_numpy, kernels_openmp, kernels_mpi
from mpi4py import MPI
import time

class DenseLayer(B.DenseLayer):
    #_update_state        = staticmethod(kernels_numpy.update_state)
    _update_state        = staticmethod(kernels_openmp.update_state)
    _softmax_minicolumns = staticmethod(kernels_numpy.softmax_minicolumns)
    _update_bias         = staticmethod(kernels_mpi.update_bias)

    def train_step(self, inputs, outputs):
        self.Ci, self.Cj, self.Cij, self.weights = kernels_mpi.update_weights(self.Ci, self.Cj, self.Cij, inputs, outputs, self.taupdt, self.weights, self.taupdt/2)

    def train_finalize(self):
        self.bias = self._update_bias(self.bias, self.Cj, self.taupdt/2)

class StructuralPlasticityLayer(B.StructuralPlasticityLayer):
    #_update_state        = staticmethod(kernels_numpy.update_state)
    _update_state        = staticmethod(kernels_openmp.update_state)
    _softmax_minicolumns = staticmethod(kernels_numpy.softmax_minicolumns)
    _update_bias         = staticmethod(kernels_mpi.update_bias_regularized)
    _update_mask         = staticmethod(kernels_mpi.update_mask)
    _apply_mask          = staticmethod(kernels_mpi.apply_mask)

    def train_step(self, inputs, outputs, hypercolumn=None):
        self.Ci, self.Cj, self.Cij, self.weights = kernels_mpi.update_weights(self.Ci, self.Cj, self.Cij, inputs, outputs, self.taupdt, self.weights, self.taupdt/2)

        self.bias, self.kbi = self._update_bias(self.bias, self.kbi, self.Cj, self.taupdt/2, self.khalf, self.pmin, self.taubdt)
        if hypercolumn is not None:
            #print("Updating hypercolumn:", hypercolumn)
            self.wmask = self._update_mask(self.wmask, self.weights, self.Ci, self.Cj, self.Cij, self.taupdt/2, self.hypercolumns, self.minicolumns, hypercolumn, 16)
        self.weights = self._apply_mask(self.weights, self.wmask, self.hypercolumns, self.minicolumns)

class Network(B.Network):
    def __init__(self, dtype):
        super().__init__(dtype)
        self.world_rank = MPI.COMM_WORLD.Get_rank()
        self.world_size = MPI.COMM_WORLD.Get_size()

    def evaluate(self, images, labels, maximal_batch_size):
        comm = MPI.COMM_WORLD
        correct, total = super().evaluate(images, labels, maximal_batch_size)
        print(self.world_rank, correct, total)
        comm.Allreduce(MPI.IN_PLACE, correct, op=MPI.SUM)
        comm.Allreduce(MPI.IN_PLACE, total, op=MPI.SUM)
        return correct, total

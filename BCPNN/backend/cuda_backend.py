from . import _cpu_base_backend as B
from ._cpu_base_backend import Network

from ._kernels import kernels_numpy, kernels_cuda

class DenseLayer(B.DenseLayer):
    _update_state        = staticmethod(kernels_cuda.update_state)
    _softmax_minicolumns = staticmethod(kernels_numpy.softmax_minicolumns)
    _update_counters     = staticmethod(kernels_cuda.update_counters)
    _update_weights      = staticmethod(kernels_cuda.update_weights)
    _update_bias         = staticmethod(kernels_cuda.update_bias)


class StructuralPlasticityLayer(B.StructuralPlasticityLayer):
    _update_state        = staticmethod(kernels_cuda.update_state)
    _softmax_minicolumns = staticmethod(kernels_numpy.softmax_minicolumns)
    _update_counters     = staticmethod(kernels_cuda.update_counters)
    _update_weights      = staticmethod(kernels_cuda.update_weights)
    _update_bias         = staticmethod(kernels_cuda.update_bias_regularized)
    _update_mask         = staticmethod(kernels_cuda.update_mask)
    _apply_mask          = staticmethod(kernels_cuda.apply_mask)


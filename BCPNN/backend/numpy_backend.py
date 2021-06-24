from . import _cpu_base_backend as B
from ._cpu_base_backend import Network

from ._kernels import kernels_numpy


class DenseLayer(B.DenseLayer):
    _update_state = staticmethod(kernels_numpy.update_state)
    _softmax_minicolumns = staticmethod(kernels_numpy.softmax_minicolumns)
    _update_counters = staticmethod(kernels_numpy.update_counters)
    _update_weights = staticmethod(kernels_numpy.update_weights)
    _update_bias = staticmethod(kernels_numpy.update_bias)


class StructuralPlasticityLayer(B.StructuralPlasticityLayer):
    _update_state = staticmethod(kernels_numpy.update_state)
    _softmax_minicolumns = staticmethod(kernels_numpy.softmax_minicolumns)
    _update_counters = staticmethod(kernels_numpy.update_counters)
    _update_weights = staticmethod(kernels_numpy.update_weights)
    _update_bias = staticmethod(kernels_numpy.update_bias_regularized)
    _update_mask = staticmethod(kernels_numpy.update_mask)
    _apply_mask = staticmethod(kernels_numpy.apply_mask)

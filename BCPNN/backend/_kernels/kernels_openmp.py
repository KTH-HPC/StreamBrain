import numpy as np

import _bcpnn_kernels_openmp_internal as _i

def update_state(state, weights, bias, inputs):
    _i.updateState(state, inputs, weights, bias)
    return state

def update_weights(Ci, Cj, Cij, a_i, a_o, taupdt, weights, cthr):
    if Ci.dtype != np.float64:
        raise Exception("Invalid dtype")
    _i.updateWeights(1.0, Ci, Cj, Cij, a_i, a_o, taupdt, weights, cthr)
    return Ci, Cj, Cij, weights

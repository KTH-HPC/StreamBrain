import numpy as np

import _bcpnn_kernels_cuda_internal as _i

_i.initialize()

def update_state(state, weights, bias, inputs):
    if state.dtype == np.float32:
        _i.update_state_float32(state, weights, bias, inputs)
    elif state.dtype == np.float64:
        _i.update_state_float64(state, weights, bias, inputs)
    else:
        raise Exception("Invalid dtype")
    return state

def add_bias(a, x):
    #a = a.copy(); x = x.copy()
    if a.dtype == np.float32:
        _i.add_bias_float32(a, x)
    elif a.dtype == np.float64:
        _i.add_bias_float64(a, x)
    else:
        raise Exception("Invalid dtype")
    return a

def softmax_minicolumns(a, hypercolumns, minicolumns):
    #a = a.copy()
    if a.dtype == np.float32:
        _i.softmax_minicolumns_float32(a, hypercolumns, minicolumns)
    elif a.dtype == np.float64:
        _i.softmax_minicolumns_float64(a, hypercolumns, minicolumns)
    else:
        raise Exception("Invalid dtype")
    return a

def update_counters(Ci, Cj, Cij, a_i, a_o, taupdt):
    #Ci = Ci.copy(); Cj = Cj.copy(); Cij = Cij.copy(); a_i = a_i.copy(); a_o = a_o.copy()
    if Ci.dtype == np.float32:
        _i.update_counters_float32(Ci, Cj, Cij, a_i, a_o, taupdt)
    elif Ci.dtype == np.float64:
        _i.update_counters_float64(Ci, Cj, Cij, a_i, a_o, taupdt)
    else:
        raise Exception("Invalid dtype")
    return Ci, Cj, Cij

def update_weights(weights, Ci, Cj, Cij, cthr):
    #weights = weights.copy(); Ci = Ci.copy(); Cj = Cj.copy(); Cij = Cij.copy()
    if weights.dtype == np.float32:
        _i.update_weights_float32(weights, Ci, Cj, Cij, cthr)
    elif weights.dtype == np.float64:
        _i.update_weights_float64(weights, Ci, Cj, Cij, cthr)
    else:
        raise Exception("Invalid dtype")
    return weights

def update_bias(bias, Cj, cthr):
    #bias = bias.copy(); Cj = Cj.copy()
    if bias.dtype == np.float32:
        _i.update_bias_float32(bias, Cj, cthr)
    elif bias.dtype == np.float64:
        _i.update_bias_float64(bias, Cj, cthr)
    else:
        raise Exception("Invalid dtype")
    return bias

def update_bias_regularized(bias, kbi, Cj, cthr, khalf, pmin, taubdt):
    #bias = bias.copy(); kbi = kbi.copy(); Cj = Cj.copy()
    if bias.dtype == np.float32:
        _i.update_bias_regularized_float32(bias, kbi, Cj, cthr, khalf, pmin, taubdt)
    elif bias.dtype == np.float64:
        _i.update_bias_regularized_float64(bias, kbi, Cj, cthr, khalf, pmin, taubdt)
    else:
        raise Exception("Invalid dtype")
    return bias, kbi

def update_mask(wmask, weights, Ci, Cj, Cij, cthr, hypercolumns, minicolumns, h, iterations):
    #wmask = wmask.copy(); weights = weights.copy(); Ci = Ci.copy(); Cj = Cj.copy(); Cij = Cij.copy()
    if weights.dtype == np.float32:
        _i.update_mask_float32(wmask, weights, Ci, Cj, Cij, cthr, hypercolumns, minicolumns, h, iterations)
    elif weights.dtype == np.float64:
        _i.update_mask_float64(wmask, weights, Ci, Cj, Cij, cthr, hypercolumns, minicolumns, h, iterations)
    else:
        raise Exception("Invalid dtype")
    return wmask

def apply_mask(weights, wmask, hypercolumns, minicolumns):
    #weights = weights.copy(); wmask = wmask.copy()
    if weights.dtype == np.float32:
        _i.apply_mask_float32(weights, wmask, hypercolumns, minicolumns)
    elif weights.dtype == np.float64:
        _i.apply_mask_float64(weights, wmask, hypercolumns, minicolumns)
    else:
        raise Exception("Invalid dtype")
    return weights

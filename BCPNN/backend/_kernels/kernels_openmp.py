import numpy as np

import _bcpnn_kernels_openmp_internal as _i


def update_state(state, weights, bias, inputs):
    if state.dtype == np.float64:
        _i.update_state_float64(state, inputs, weights, bias)
    elif state.dtype == np.float32:
        _i.update_state_float32(state, inputs, weights, bias)
    else:
        raise Exception("Invalid dtype")
    return state


def update_weights(Ci, Cj, Cij, a_i, a_o, taupdt, weights, cthr):
    if Ci.dtype == np.float64:
        _i.update_weights_float64(
            1.0, Ci, Cj, Cij, a_i, a_o, taupdt, weights, cthr)
    elif Ci.dtype == np.float32:
        _i.update_weights_float32(
            1.0, Ci, Cj, Cij, a_i, a_o, taupdt, weights, cthr)
    else:
        raise Exception("Invalid dtype")
    return Ci, Cj, Cij, weights


def update_bias(bias, Cj, cthr):
    if bias.dtype == np.float64:
        _i.update_bias_float64(bias, Cj, cthr)
    elif bias.dtype == np.float32:
        _i.update_bias_float32(bias, Cj, cthr)
    else:
        raise Exception("Invalid dtype")
    return bias


def update_bias_regularized(bias, kbi, Cj, cthr, khalf, pmin, taubdt):
    if bias.dtype == np.float64:
        _i.update_bias_regularized_float64(
            bias, kbi, Cj, cthr, khalf, pmin, taubdt)
    elif bias.dtype == np.float32:
        _i.update_bias_regularized_float32(
            bias, kbi, Cj, cthr, khalf, pmin, taubdt)
    else:
        raise Exception("Invalid dtype")
    return bias, kbi


def update_mask(
        wmask,
        weights,
        Ci,
        Cj,
        Cij,
        cthr,
        hypercolumns,
        minicolumns,
        h,
        iterations):
    #wmask = wmask.copy(); weights = weights.copy(); Ci = Ci.copy(); Cj = Cj.copy(); Cij = Cij.copy()
    if weights.dtype == np.float32:
        _i.update_mask_float32(
            wmask,
            weights,
            Ci,
            Cj,
            Cij,
            cthr,
            hypercolumns,
            minicolumns,
            h,
            iterations)
    elif weights.dtype == np.float64:
        _i.update_mask_float64(
            wmask,
            weights,
            Ci,
            Cj,
            Cij,
            cthr,
            hypercolumns,
            minicolumns,
            h,
            iterations)
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

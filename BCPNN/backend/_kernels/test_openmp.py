import numpy as np
import random
import sys

import kernels_openmp
import kernels_numpy


def test_update_state(k1, k2):
    batch_size = random.randint(1, 1024)
    n = random.randint(1, 1024)
    m = random.randint(1, 1024)

    state1 = np.zeros([batch_size, m])
    state2 = np.zeros([batch_size, m])

    weights = np.random.randn(n, m)
    bias    = np.random.randn(m)
    inputs  = np.random.randn(batch_size, n)

    state1 = k1.update_state(state1, weights, bias, inputs)
    state2 = k2.update_state(state2, weights, bias, inputs)

    diff = np.absolute(state1 - state2).max()

    return diff, (np.zeros([batch_size, m]), weights, bias, inputs)

def test_add_bias(k1, k2):
    n = random.randint(1, 1024)
    m = random.randint(1, 1024)

    b = np.random.randn(m)

    x = np.random.randn(n, m)
    x1 = x.copy()
    x2 = x.copy()

    r1 = k1.add_bias(x1, b)
    r2 = k2.add_bias(x2, b)

    diff = np.absolute(r1 - r2).max()

    return diff, (x, b)

# Cuda has bug when mc = 1
def test_softmax_minicolumns(k1, k2):
    n  = random.randint(1, 1024)
    m  = random.randint(2, 1024)
    mc = random.randint(2, m)
    hc = m // mc
    m  = hc*mc

    x = np.random.randn(n, m)
    x1 = x.copy()
    x2 = x.copy()

    r1 = k1.softmax_minicolumns(x1, hc, mc)
    r2 = k2.softmax_minicolumns(x2, hc, mc)

    diff = np.absolute(r1 - r2).max()

    return diff, (x, hc, mc)

def test_update_counters(k1, k2):
    batch_size = random.randint(1, 100)
    n          = random.randint(1, 1024)
    m          = random.randint(1, 1024)

    Ci      = np.random.rand(n)
    Ci1     = Ci.copy()
    Ci2     = Ci.copy()

    Cj      = np.random.rand(m)
    Cj1     = Cj.copy()
    Cj2     = Cj.copy()

    Cij     = np.random.rand(n, m)
    Cij1    = Cij.copy()
    Cij2    = Cij.copy()

    inputs  = np.random.rand(batch_size, n)
    outputs = np.random.rand(batch_size, m)
    taupdt  = random.random()

    Ci1, Cj1, Cij1 = k1.update_counters(Ci1, Cj1, Cij1, inputs, outputs, taupdt)
    Ci2, Cj2, Cij2 = k2.update_counters(Ci2, Cj2, Cij2, inputs, outputs, taupdt)

    diff = np.absolute(Ci1  - Ci2).max()
    diff = max(diff, np.absolute(Cj1  - Cj2).max())
    diff = max(diff, np.absolute(Cij1 - Cij2).max())

    return diff, (Ci, Cj, Cij, inputs, outputs, taupdt)

def test_update_weights(k1, k2):
    batch_size = random.randint(1, 100)
    n          = random.randint(1, 1024)
    m          = random.randint(1, 1024)

    Ci      = np.random.rand(n)
    Ci1     = Ci.copy()
    Ci2     = Ci.copy()

    Cj      = np.random.rand(m)
    Cj1     = Cj.copy()
    Cj2     = Cj.copy()

    Cij     = np.random.rand(n, m)
    Cij1    = Cij.copy()
    Cij2    = Cij.copy()

    inputs  = np.random.rand(batch_size, n)
    outputs = np.random.rand(batch_size, m)
    taupdt  = random.random()

    weights = np.zeros([n, m])
    weights1 = weights.copy()
    weights2 = weights.copy()

    Ci1, Cj1, Cij1, weights1 = k1.update_weights(Ci1, Cj1, Cij1, inputs, outputs, taupdt, weights1, taupdt/2)
    Ci2, Cj2, Cij2 = k2.update_counters(Ci2, Cj2, Cij2, inputs, outputs, taupdt)
    weights2 = k2.update_weights(weights2, Ci2, Cj2, Cij2, taupdt/2)

    diff = np.absolute(weights1  - weights2).max()
    diff = max(diff, np.absolute(Ci1  - Ci2).max())
    diff = max(diff, np.absolute(Cj1  - Cj2).max())
    diff = max(diff, np.absolute(Cij1  - Cij2).max())

    return diff, (Ci, Cj, Cij, weights, taupdt)

def test_update_bias(k1, k2):
    m = random.randint(1, 1024)

    Cj     = np.random.rand(m)
    taupdt = random.random()
    bias   = np.zeros([m])
    bias1  = bias.copy()
    bias2  = bias.copy()

    bias1 = k1.update_bias(bias1, Cj, taupdt)
    bias2 = k2.update_bias(bias2, Cj, taupdt)

    diff = np.absolute(bias1  - bias2).max()

    return diff, (bias, Cj, taupdt)

def test_update_bias_regularized(k1, k2):
    m = random.randint(1, 1024)

    Cj     = np.random.rand(m)
    taupdt = random.random()
    khalf  = -100 - 900*random.random()
    pmin   = random.random()
    taubdt = random.random()

    bias   = np.zeros([m])
    bias1  = bias.copy()
    bias2  = bias.copy()

    kbi    = 10*np.random.randn(m)
    kbi1   = kbi.copy()
    kbi2   = kbi.copy()

    bias1, kbi1 = k1.update_bias_regularized(bias1, kbi1, Cj, taupdt, khalf, pmin, taubdt)
    bias2, kbi2 = k2.update_bias_regularized(bias2, kbi2, Cj, taupdt, khalf, pmin, taubdt)

    diff = np.absolute(bias1 - bias2).max()
    diff = max(diff, np.absolute( kbi1 -  kbi2).max())

    return diff, (bias, Cj, taupdt)

def test_update_mask(k1, k2):
    n  = random.randint(1, 128)
    m  = random.randint(2, 128)
    mc = random.randint(2, m)
    hc = m // mc
    m  = hc*mc

    wmask = (np.random.rand(n, hc) < 0.1).astype(np.uint8)
    wmask1 = wmask.copy()
    wmask2 = wmask.copy()

    weights = 0.1 * np.random.randn(n, m)
    Ci  = np.random.rand(n)
    Cj  = np.random.rand(m)
    Cij = np.zeros([n, m])
    taupdt = random.random()
    h = random.randint(0, hc-1)
    iterations = 1

    for i in range(n):
        for j in range(m):
            Cij[i, j] = min(0.9 * Ci[i], 0.9 * Cj[j], random.random())

    wmask1 = k1.update_mask(wmask1, weights, Ci, Cj, Cij, taupdt, hc, mc, h, iterations)
    wmask2 = k2.update_mask(wmask2, weights, Ci, Cj, Cij, taupdt, hc, mc, h, iterations)

    diff = np.absolute(wmask1 - wmask2).flatten().max()

    return diff, (wmask, weights, Ci, Cj, Cij, taupdt, hc, mc, h, iterations)
    
def test_apply_mask(k1, k2):
    n  = random.randint(1, 1024)
    m  = random.randint(2, 1024)
    mc = random.randint(2, m)
    hc = m // mc
    m  = hc*mc

    wmask   = (np.random.rand(n, hc) < 0.1).astype(np.int8)
    weights = 0.1 * np.random.randn(n, m)
    weights1 = weights.copy()
    weights2 = weights.copy()

    weights1 = k1.apply_mask(weights1, wmask, hc, mc)
    weights2 = k2.apply_mask(weights2, wmask, hc, mc)

    diff = np.absolute(weights1 - weights2).max()

    return diff, (weights, wmask, hc, mc)




def run_tests(k1, k2):
    iterations = 10;

    _max = 0
    for _ in range(iterations):
        diff, args = test_update_state(_k1, _k2)
        _max = max(_max, diff)
        if diff > 1e-3:
            return diff, args
    print("testing update_state():", _max)

    _max = 0
    for _ in range(iterations):
        diff, args = test_update_weights(_k1, _k2)
        _max = max(_max, diff)
        if diff > 1e-3:
            return diff, args
    print("testing update_weights():", _max)

    _max = 0
    for _ in range(iterations):
        diff, args = test_update_bias(_k1, _k2)
        _max = max(_max, diff)
        if diff > 1e-3:
            return diff, args
    print("testing update_bias():", _max)

    _max = 0
    for _ in range(iterations):
        diff, args = test_update_bias_regularized(_k1, _k2)
        _max = max(_max, diff)
        if diff > 1e-3:
            return diff, args
    print("testing update_bias_regularized():", _max)

    _max = 0
    for _ in range(iterations):
        diff, args = test_update_mask(_k1, _k2)
        _max = max(_max, diff)
        if diff > 1e-3:
            return diff, args
    print("testing update_mask():", _max)

    _max = 0
    for _ in range(iterations):
        diff, args = test_apply_mask(_k1, _k2)
        _max = max(_max, diff)
        if diff > 1e-3:
            return diff, args
    print("testing apply_mask():", _max)

    print('success!')

    return None, None

if __name__ == "__main__":
    _k1 = kernels_openmp
    _k2 = kernels_numpy

    diff, _ = run_tests(_k1, _k2)
    if diff is not None:
        print('faild!')

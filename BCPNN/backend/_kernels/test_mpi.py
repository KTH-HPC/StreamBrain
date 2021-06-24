import numpy as np
import random
import sys

import kernels_mpi
import kernels_numpy

from mpi4py import MPI

comm = MPI.COMM_WORLD
world_rank = comm.Get_rank()
world_size = comm.Get_size()


def test_update_weights(k1, k2, trial=0):
    batch_size = np.array(random.randint(1, 100))
    n = np.array(random.randint(1, 1024))
    m = np.array(random.randint(1, 1024))

    comm.Allreduce(MPI.IN_PLACE, batch_size, op=MPI.MAX)
    comm.Allreduce(MPI.IN_PLACE, m, op=MPI.MAX)
    comm.Allreduce(MPI.IN_PLACE, n, op=MPI.MAX)

    Ci = np.random.rand(n)
    comm.Allreduce(MPI.IN_PLACE, Ci, op=MPI.SUM)
    Ci1 = Ci.copy()
    Ci2 = Ci.copy()

    Cj = np.random.rand(m)
    comm.Allreduce(MPI.IN_PLACE, Cj, op=MPI.SUM)
    Cj1 = Cj.copy()
    Cj2 = Cj.copy()

    Cij = np.random.rand(n, m)
    comm.Allreduce(MPI.IN_PLACE, Cij, op=MPI.SUM)
    Cij1 = Cij.copy()
    Cij2 = Cij.copy()

    local_inputs = np.random.rand(batch_size, n)
    comm.Allreduce(MPI.IN_PLACE, local_inputs, op=MPI.SUM)

    local_outputs = np.random.rand(batch_size, m)
    comm.Allreduce(MPI.IN_PLACE, local_outputs, op=MPI.SUM)

    taupdt = np.array(random.random())
    comm.Allreduce(MPI.IN_PLACE, taupdt, op=MPI.SUM)

    inputs = np.tile(local_inputs, (world_size, 1))
    outputs = np.tile(local_outputs, (world_size, 1))

    weights = np.zeros([n, m])
    weights1 = weights.copy()
    weights2 = weights.copy()

    Ci1, Cj1, Cij1, weights1 = k1.update_weights(
        Ci1, Cj1, Cij1, local_inputs, local_outputs, taupdt, weights1, taupdt / 2)
    Ci2, Cj2, Cij2 = k2.update_counters(
        Ci2, Cj2, Cij2, inputs, outputs, taupdt)
    weights2 = k2.update_weights(weights2, Ci2, Cj2, Cij2, taupdt / 2)

    diff = np.absolute(weights1 - weights2).max()
    diff = max(diff, np.absolute(Ci1 - Ci2).max())
    diff = max(diff, np.absolute(Cj1 - Cj2).max())
    diff = max(diff, np.absolute(Cij1 - Cij2).max())

    return diff, (Ci, Cj, Cij, weights, taupdt)


def test_update_bias(k1, k2):
    m = np.array(random.randint(1, 1024))
    comm.Allreduce(MPI.IN_PLACE, m, op=MPI.MAX)

    Cj = np.random.rand(m)
    taupdt = np.array(random.random())
    bias = np.zeros([m])

    comm.Allreduce(MPI.IN_PLACE, Cj, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, taupdt, op=MPI.SUM)

    bias1 = bias.copy()
    bias2 = bias.copy()

    bias1 = k1.update_bias(bias1, Cj, taupdt)
    bias2 = k2.update_bias(bias2, Cj, taupdt)

    diff = np.absolute(bias1 - bias2).max()

    return diff, (bias, Cj, taupdt)


def test_update_bias_regularized(k1, k2):
    m = np.array(random.randint(1, 1024))
    comm.Allreduce(MPI.IN_PLACE, m, op=MPI.MAX)

    Cj = np.random.rand(m)
    taupdt = np.array(random.random())
    khalf = np.array(-100 - 900 * random.random())
    pmin = np.array(random.random())
    taubdt = np.array(random.random())

    comm.Allreduce(MPI.IN_PLACE, Cj, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, taupdt, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, khalf, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, pmin, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, taubdt, op=MPI.SUM)

    bias = np.zeros([m])
    bias1 = bias.copy()
    bias2 = bias.copy()

    kbi = 10 * np.random.randn(m)
    comm.Allreduce(MPI.IN_PLACE, kbi, op=MPI.SUM)
    kbi1 = kbi.copy()
    kbi2 = kbi.copy()

    bias1, kbi1 = k1.update_bias_regularized(
        bias1, kbi1, Cj, taupdt, khalf, pmin, taubdt)
    bias2, kbi2 = k2.update_bias_regularized(
        bias2, kbi2, Cj, taupdt, khalf, pmin, taubdt)

    diff = np.absolute(bias1 - bias2).max()
    diff = max(diff, np.absolute(kbi1 - kbi2).max())

    return diff, (bias, Cj, taupdt)


def test_update_mask(k1, k2):
    n = np.array(random.randint(1, 128))
    m = np.array(random.randint(2, 128))

    comm.Allreduce(MPI.IN_PLACE, m, op=MPI.MAX)
    comm.Allreduce(MPI.IN_PLACE, n, op=MPI.MAX)

    mc = np.array(random.randint(2, m))
    comm.Allreduce(MPI.IN_PLACE, mc, op=MPI.MAX)

    hc = m // mc
    m = hc * mc

    wmask = (np.random.rand(n, hc) < 0.1).astype(np.uint8)
    comm.Bcast(wmask, root=0)
    wmask1 = wmask.copy()
    wmask2 = wmask.copy()

    Ci = np.random.rand(n)
    comm.Bcast(Ci, root=0)
    #comm.Allreduce(MPI.IN_PLACE, Ci, op=MPI.SUM)
    Ci1 = Ci.copy()
    Ci2 = Ci.copy()

    Cj = np.random.rand(m)
    comm.Bcast(Cj, root=0)
    #comm.Allreduce(MPI.IN_PLACE, Cj, op=MPI.SUM)
    Cj1 = Cj.copy()
    Cj2 = Cj.copy()

    Cij = np.zeros([n, m])

    if world_rank == 0:
        weights = 0.1 * np.random.randn(n, m)
    else:
        weights = np.zeros([n, m])
    comm.Bcast(weights, root=0)

    taupdt = np.array(random.random())
    comm.Allreduce(MPI.IN_PLACE, taupdt, op=MPI.SUM)
    h = np.array(random.randint(0, hc - 1))
    comm.Bcast(h, root=0)

    iterations = 1

    for i in range(n):
        for j in range(m):
            Cij[i, j] = min(0.9 * Ci[i], 0.9 * Cj[j], random.random())

    wmask1 = k1.update_mask(
        wmask1,
        weights,
        Ci,
        Cj,
        Cij,
        taupdt,
        hc,
        mc,
        h,
        iterations)
    wmask2 = k2.update_mask(
        wmask2,
        weights,
        Ci,
        Cj,
        Cij,
        taupdt,
        hc,
        mc,
        h,
        iterations)

    diff = np.absolute(wmask1 - wmask2).flatten().max()

    return diff, (wmask, weights, Ci, Cj, Cij, taupdt, hc, mc, h, iterations)


def test_apply_mask(k1, k2):
    n = np.array(random.randint(1, 128))
    m = np.array(random.randint(2, 128))

    comm.Allreduce(MPI.IN_PLACE, m, op=MPI.MAX)
    comm.Allreduce(MPI.IN_PLACE, n, op=MPI.MAX)

    mc = np.array(random.randint(2, m))
    comm.Allreduce(MPI.IN_PLACE, mc, op=MPI.MAX)

    hc = m // mc
    m = hc * mc

    wmask = (np.random.rand(n, hc) < 0.1).astype(np.int8)
    comm.Bcast(wmask, root=0)
    weights = 0.1 * np.random.randn(n, m)
    comm.Bcast(weights, root=0)
    weights1 = weights.copy()
    weights2 = weights.copy()

    weights1 = k1.apply_mask(weights1, wmask, hc, mc)
    weights2 = k2.apply_mask(weights2, wmask, hc, mc)
    diff = np.absolute(weights1 - weights2).max()

    return diff, (weights, wmask, hc, mc)


def run_tests(k1, k2):
    iterations = 10
#    _max = 0
#    for _ in range(iterations):
#        diff, args = test_add_bias(_k1, _k2)
#        _max = max(_max, diff)
#    print("testing add_bias():", _max)

    _max = 0
    for t in range(iterations):
        diff, args = test_update_weights(_k1, _k2, t)
        _max = max(_max, diff)
        if diff > 1e-3:
            print(world_rank, 'test update_weights() error!', diff)
            return diff, args
    print(world_rank, "testing update_weights():", _max)

    _max = 0
    for _ in range(iterations):
        diff, args = test_update_bias(_k1, _k2)
        _max = max(_max, diff)
        if diff > 1e-3:
            return diff, args
    print(world_rank, "testing update_bias():", _max)

    _max = 0
    for _ in range(iterations):
        diff, args = test_update_bias_regularized(_k1, _k2)
        _max = max(_max, diff)
        if diff > 1e-3:
            return diff, args
    print(world_rank, "testing update_bias_regularized():", _max)

    _max = 0
    for _ in range(iterations):
        diff, args = test_update_mask(_k1, _k2)
        if world_rank == 0:
            _max = max(_max, diff)
            if diff > 1e-3:
                return diff, args
    print(world_rank, "testing update_mask():", _max)

    _max = 0
    for _ in range(iterations):
        diff, args = test_apply_mask(_k1, _k2)
        _max = max(_max, diff)
        if diff > 1e-3:
            return diff, args
    print(world_rank, "testing apply_mask():", _max)


if __name__ == "__main__":
    _k1 = kernels_mpi
    _k2 = kernels_numpy

    run_tests(_k1, _k2)

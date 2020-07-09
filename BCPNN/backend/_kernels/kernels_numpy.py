import numpy as np

def update_state(state, weights, bias, inputs):
    return np.matmul(inputs, weights) + bias

def add_bias(a, x):
    # Assert shapes?
    return a + x

def _softmax(x, axis):
    t = np.exp(x - x.max(axis=axis, keepdims=True))
    return t / np.sum(t, axis=axis, keepdims=True)

def softmax_minicolumns(a, hypercolumns, minicolumns):
  return np.reshape(
          _softmax(
              np.reshape(a, [-1, hypercolumns, minicolumns]),
              2
          ),
          [-1, hypercolumns * minicolumns]
      )

def update_counters(Ci, Cj, Cij, a_i, a_o, taupdt):
    # update counters
    #C = (1 - taupdt)*C + taupdt
    #C  = max(0.0, C) # ensure it is positive: it is probability
    # Ci
    Ci = (1 - taupdt) * Ci + taupdt  * np.mean(a_i, axis=0) # mean op
    Ci = np.maximum(0.0, Ci)
    # Cj
    Cj = (1 - taupdt) * Cj + taupdt  * np.mean(a_o, axis=0) # mean op
    Cj = np.maximum(0.0,Cj)
    # Cij
    ai_resh = np.reshape(a_i, [a_i.shape[0], a_i.shape[1],1]) #reshape: mem op
    ao_resh = np.reshape(a_o, [a_o.shape[0], 1, a_o.shape[1]]) #reshape: mem op
    
    ai_times_ao = np.matmul(ai_resh,ao_resh) # here the matmul: n_features
    
    Cij = (1 - taupdt)*Cij + taupdt*np.mean(ai_times_ao,axis=0) #mean op
    Cij = np.maximum(0.0, Cij)
    
    return Ci, Cj, Cij

def update_weights(weights, Ci, Cj, Cij, cthr):
    n = Ci.shape[0]
    m = Cj.shape[0]
    weights = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            if Ci[i] < cthr or Cj[j] < cthr:
                weights[i, j] = 0.0
            else:
                x = max(Cij[i, j], cthr ** 4)
                weights[i, j] = np.log(x / (Ci[i] * Cj[j]))

    return weights

def update_bias(bias, Cj, cthr):
  bias = np.zeros_like(Cj)
  for i in range(Cj.shape[0]):
    if Cj[i] < cthr:
      bias[i] = np.log(2*cthr)
    else:
      bias[i] = np.log(Cj[i])

  return bias

def update_bias_regularized(bias, kbi, Cj, cthr, khalf, pmin, taubdt):
  _bias = np.zeros_like(Cj)
  _kbi = np.zeros_like(kbi)

  k = (khalf - 1) * (pmin/4)**2
  for i in range(Cj.shape[0]):
    pj = Cj[i]
  
    kb = 1 + k/((pj - pmin/4)**2)
    if pj < pmin/4 or kb < khalf:
      kb = khalf
  
    _kbi[i] = (1 - taubdt) * kbi[i] + taubdt * kb
          
    if Cj[i] < cthr:
      _bias[i] = _kbi[i] * np.log(2*cthr)
    else:
      _bias[i] = _kbi[i] * np.log(pj)
  return _bias, _kbi

def update_mask(wmask, weights, Ci, Cj, Cij, cthr, hypercolumns, minicolumns, hypercolumn, iterations):
    wmask = wmask.copy()
    h = hypercolumn

    a =  h    * minicolumns
    b = (h+1) * minicolumns

    wmask_nominator = np.zeros_like(Ci)

    for i in range(Ci.shape[0]):
        for j in range(a, b):
            if Ci[i] > cthr and Cj[j] > cthr:
                pi  = Ci[i]
                pj  = Cj[j]
                pij = Cij[i, j]

                WijC = np.log((pj - pij) / ((1 - pi)*pj))

                wmask_nominator[i] += pij*weights[i, j] + (pj - pij)*WijC

    for _ in range(iterations):
        wmaskcsum = wmask.sum(axis=1, keepdims=False)
        wmask_score = wmask_nominator / (1.0 + wmaskcsum)

        vmax0, imax0 = None, None
        vmin1, imin1 = None, None

        for i in range(wmask_score.shape[0]):
            score = wmask_score[i]
            if wmask[i, h] == 0 and (vmax0 is None or score >= vmax0):
                imax0 = i
                vmax0 = score
            if wmask[i, h] == 1 and (vmin1 is None or score <= vmin1):
                imin1 = i
                vmin1 = score

        if vmax0 is None or vmin1 is None:
            break

        if vmax0 < vmin1:
            break

        #print("CPU: Swapping {} ({}) with {} ({})".format(imax0, vmax0, imin1, vmin1))

        wmask[imax0, h] = 1
        wmask[imin1, h] = 0

    return wmask
    

def apply_mask(weights, wmask, hypercolumns, minicolumns):
    return (
          weights
        * np.reshape(
              np.broadcast_to(
                  np.reshape(wmask, (-1, hypercolumns, 1)),
                  (weights.shape[0], hypercolumns, minicolumns)
              ),
              (weights.shape[0], hypercolumns * minicolumns)
          )
      )

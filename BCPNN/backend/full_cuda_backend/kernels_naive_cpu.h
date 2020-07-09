namespace bcpnn {

namespace kernels {

namespace naive_cpu {

template<typename REAL>
void
naive_add_bias(REAL * matrix, size_t n, size_t m, REAL * bias)
{
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      matrix[i * m + j] += bias[j];
    }
  }
}


template<typename REAL>
void
naive_softmax(REAL * matrix, size_t n, size_t m)
{
  for (size_t i = 0; i < n; ++i) {
    REAL max_ = matrix[i * m + 0];
    for (size_t j = 0; j < m; ++j) {
      max_ = fmaxf(max_, matrix[i * m + j]);
    }

    REAL sum = 0;
    for (size_t j = 0; j < m; ++j) {
      matrix[i * m + j] = expf(matrix[i * m + j] - max_);
      sum += matrix[i * m + j];
    }

    for (size_t j = 0; j < m; ++j) {
      matrix[i * m + j] /= sum;
    }
  }
}

template<typename REAL>
void
naive_update_counters(REAL * Ci, REAL * Cj, REAL * Cij, REAL * inputs, REAL * outputs, size_t batch_size, size_t n, size_t m, REAL taupdt)
{
  REAL * ti = (REAL *)malloc(n * sizeof(REAL));
  REAL * tj = (REAL *)malloc(m * sizeof(REAL));
  REAL * tij = (REAL *)malloc(n * m * sizeof(REAL));

  for (size_t i = 0; i < n; ++i) { ti[i] = 0; }

  for (size_t j = 0; j < m; ++j) { tj[j] = 0; }

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      tij[i * m + j] = 0;
    }
  }

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t i = 0; i < n; ++i) {
      ti[i] += inputs[b * n + i];
    }
  }

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t j = 0; j < m; ++j) {
      tj[j] += outputs[b * m + j];
    }
  }

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j) {
        tij[i * m + j] += (inputs[b * n + i] * outputs[b * m + j]);
      }
    }
  }

  for (size_t i = 0; i < n; ++i) {
    Ci[i] = fmaf(1 - taupdt, Ci[i], ti[i] / batch_size * taupdt);
  }

  for (size_t j = 0; j < m; ++j) {
    Cj[j] = fmaf(1 - taupdt, Cj[j], tj[j] / batch_size * taupdt);
  }

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      Cij[i * m + j] = fmaf(1 - taupdt, Cij[i * m + j], tij[i * m + j] / batch_size * taupdt);
    }
  }
}

template<typename REAL>
void
naive_update_weights(REAL * weights, REAL * Ci, REAL * Cj, REAL * Cij, REAL cthr, size_t n, size_t m)
{
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      size_t idx = i * m + j;
      if (Ci[i] < cthr || Cj[j] < cthr) {
        weights[idx] = 0;
      } else {
#if 0
        // Original
        weights[idx] = logf(Cij[idx] / (Ci[i] * Cj[j]));
#else
        // Try to get rid of nans
        REAL x = fmaxf(Cij[idx], cthr * cthr * cthr * cthr);
        weights[idx] = logf(x / (Ci[i] * Cj[j]));
#endif
      }
    }
  }
}

template<typename REAL>
void
naive_update_bias(REAL * bias, REAL * Cj, REAL cthr, size_t m)
{
  for (size_t j = 0; j < m; ++j) {
    if (Cj[j] < cthr) {
      bias[j] = logf(2 * cthr);
    } else {
      bias[j] = logf(Cj[j]);
    }
  }
}


template<typename REAL>
void
naive_update_bias_regularized(REAL * bias, REAL * kbi, REAL * Cj, REAL cthr, REAL khalf, REAL pmin, REAL taubdt, size_t m)
{
  for (size_t j = 0; j < m; ++j) {
    REAL k = (khalf - 1) * (pmin/4) * (pmin/4);
    REAL pj = Cj[j];

    REAL kb = 1 + k/((pj - pmin/4)*(pj - pmin/4));
    if (pj <= pmin/4 || kb < khalf) { kb = khalf; }

    kbi[j] = (1 - taubdt)*kbi[j] + taubdt*kb;

    if (Cj[j] < cthr) {
      bias[j] = kbi[j] * logf(2 * cthr);
    } else {
      bias[j] = kbi[j] * logf(Cj[j]);
    }
  }
}


template<typename REAL>
void
naive_update_mask(uint8_t * wmask, REAL * weights, REAL * Ci, REAL * Cj, REAL * Cij, REAL cthr, size_t n, size_t m, size_t h, size_t hypercolumns, size_t minicolumns, size_t iterations)
{
    REAL * wmask_score_nominator = (REAL *)malloc(n * sizeof(REAL));
    int   * wmask_csum = (int *)malloc(n * sizeof(int));

    for (size_t i = 0; i < n; ++i) {
      REAL score = 0;
      for (size_t j = h*minicolumns; j < (h+1)*minicolumns; ++j) {
        if (Ci[i] >= cthr && Cj[j] >= cthr) {
          REAL pi = Ci[i];
          REAL pj = Cj[j];
          REAL pij = Cij[i * m + j];

          REAL x = (1 - pi)*pj;
          REAL y = (pj - pij) / x;
          REAL WijC = logf(y);

          score += pij * weights[i * m + j] + (pj - pij)*WijC;
#if 0
	  printf("%e, %e, %e, %e, %e, %e, %e, %e\n",
			  cthr,
			  pij * weights[i * m + j] + (pj - pij)*WijC,
			  pi,
			  pj,
			  pij,
			  x,
			  y,
			  WijC
			  );
#endif
	}
      }
      wmask_score_nominator[i] = score;
    }

    for (size_t i = 0; i < n; ++i) {
      int sum = 0;
      for (size_t j = 0; j < hypercolumns; ++j) {
        sum += wmask[i * hypercolumns + j];
      }
      wmask_csum[i] = sum;
    }

    for (size_t iter = 0; iter < iterations; ++iter) {
      size_t imax0 = n;
      REAL  vmax0 = 0;

      size_t imin1 = n;
      REAL  vmin1 = 0;

      for (size_t i = 0; i < n; ++i) {
        REAL score = wmask_score_nominator[i] / (1 + wmask_csum[i]);
	if        (wmask[i * hypercolumns + h] == 0 && (imax0 == n || score >= vmax0)) {
          imax0 = i;
	  vmax0 = score;
	} else if (wmask[i * hypercolumns + h] == 1 && (imin1 == n || score <= vmin1)) {
          imin1 = i;
          vmin1 = score;
	}
      }

      if (imax0 == n || imin1 == n) { break; }
      if (vmax0 < vmin1) { break; }

      //printf("CPU: Swapping %ld (%f) with %ld (%f)\n", imax0, vmax0, imin1, vmin1);

      wmask[imax0 * hypercolumns + h] = 1;
      wmask[imin1 * hypercolumns + h] = 0;

      wmask_csum[imax0] += 1;
      wmask_csum[imin1] -= 1;
    }

    free(wmask_score_nominator);
    free(wmask_csum);
}

template<typename REAL>
void
naive_apply_mask(REAL * weight, uint8_t * wmask, size_t n, size_t m, size_t hypercolumns, size_t minicolumns)
{
  // TOODO: Assert m == hypercolumns * minicolumns
  for (size_t i = 0; i < n; ++i) {
    size_t j = 0;
    for (size_t h = 0; h < hypercolumns; ++h) {
      if (!wmask[i * hypercolumns + h]) {
        for (size_t j0 = 0; j0 < minicolumns; ++j0) {
          weight[i * m + j + j0] = 0;
        }
      }
      j += minicolumns;
    }
  }
}

} // namespace naive_cpu

} // namespace kernels

} // namespace bcpnn

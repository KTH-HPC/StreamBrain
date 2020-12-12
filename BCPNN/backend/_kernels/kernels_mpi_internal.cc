#include <iostream>
#include <memory>
#include <omp.h>
#include <math.h>
#include <cblas.h>
#include <mpi.h>

#define MIN(a,b) (((a)<(b))?(a):(b))

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <type_traits>

namespace py = pybind11;

static MPI_Datatype datatype;
static int world_rank = 0;
static int world_size = 0;

/*
    # comment out once update weights implemented in kernel
    #n = Ci.shape[0]
    #m = Cj.shape[0]
    #weights = np.zeros([n, m]).astype(weights.dtype)
    #for i in range(n):
        #for j in range(m):
            #if Ci[i] < cthr or Cj[j] < cthr:
                #weights[i, j] = 0.0
            #else:
                #weights[i, j] = np.log(C * Cij[i, j] / (Ci[i] * Cj[j]))
*/

namespace bcpnn {

namespace helpers {

namespace offsets {

void compute_offsets(size_t size, int world_rank, int world_size, int *stride, int *displs, int *rcounts)
{
    MPI_Request displs_req;

    /* compute stride */
    *stride = size / world_size;

    for (size_t i = 0; i < world_size; i++) {
        /* compute displacements */
        displs[i] = i * (*stride);

        /* compute offsets and tailing offset */
        rcounts[i] = (*stride);
        if (i == world_size - 1 && size % world_size != 0)
            rcounts[i] = size - (*stride) * (world_size - 1);
    }

    MPI_Bcast(displs, world_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(rcounts, world_size, MPI_INT, 0, MPI_COMM_WORLD);
}

} // namespace offsets

namespace blas {

template <typename REAL>
void matmul(REAL *activation, REAL *inputs, REAL *weights, const long inputs_rows, const long inputs_cols, const long weights_cols)
{ }

template<>
void matmul<double>(double *activation, double *inputs, double *weights, const long inputs_rows, const long inputs_cols, const long weights_cols)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                inputs_rows, weights_cols, inputs_cols,
                1.0, inputs, inputs_cols, weights, weights_cols,
                0.0, activation, weights_cols);
}

template<>
void matmul<float>(float *activation, float *inputs, float *weights, const long inputs_rows, const long inputs_cols, const long weights_cols)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                inputs_rows, weights_cols, inputs_cols,
                1.0, inputs, inputs_cols, weights, weights_cols,
                0.0, activation, weights_cols);
}

} // end namespace helpers
} // end namespace blas

namespace kernels {

namespace mpi {

template <typename REAL>
REAL updateWeights(REAL C, REAL *Ci, REAL *Cj, REAL *Cij,
                              REAL *a_i, REAL *a_o,
                              const REAL taupdt, const long batch_size,
                              const long in_features, const long out_features,
                              REAL *weights, const REAL cthr)
{
    size_t BS = 64 / sizeof(REAL);
    typedef REAL VECTOR __attribute__ ((vector_size (64)));

    if (world_size == 0) {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (typeid(double) == typeid(REAL))
        datatype = MPI_DOUBLE;
        else if (typeid(float) == typeid(REAL))
        datatype = MPI_FLOAT;
    }
    
    C = (1 - taupdt) * C + taupdt;
    C = std::max((REAL)0.0, C);
    
    REAL one_minus_taupdt = (1.0 - taupdt);
    REAL *tmp_array = (REAL*)malloc(sizeof(REAL) * in_features * out_features);

#pragma omp parallel firstprivate(one_minus_taupdt,C)
{
#pragma omp for
    for (int j = 0; j < in_features; j+=BS) {
        VECTOR tmp = {0.0};
        for (int i = 0; i < batch_size; i++) {
            VECTOR a_i_ld;
            #pragma unroll
            for (int vl = j; vl < MIN(in_features,j+BS); vl++)
                a_i_ld[vl-j] = a_i[i * in_features + vl];
            tmp += a_i_ld;
        }

        #pragma unroll
        for (int ij = j; ij < MIN(in_features,j+BS); ij++)
            tmp_array[ij] = tmp[ij-j];
    }

#pragma omp single
    MPI_Allreduce(MPI_IN_PLACE, tmp_array, in_features, datatype, MPI_SUM, MPI_COMM_WORLD);

#pragma omp for nowait
    for (int j = 0; j < in_features; j+=BS) {
	register VECTOR C_p;
	register VECTOR tmp;
        #pragma unroll
	for (int ij = j; ij < MIN(in_features,j+BS); ij++)
	  tmp[ij-j] = tmp_array[ij];
        #pragma unroll
	for (int ij = j; ij < MIN(in_features,j+BS); ij++)
	  C_p[ij-j] = Ci[ij];
	tmp /= (REAL) (batch_size * world_size);
	tmp *= taupdt;
	C_p *= one_minus_taupdt;
        C_p += tmp;
        #pragma unroll
	for (int ij = j; ij < MIN(in_features,j+BS); ij++)
	  Ci[ij] = C_p[ij-j];
      }
//Ci[i] = one_minus_taupdt * Ci[i] + (tmp_array[i] / (REAL)(batch_size * world_size) * taupdt);

#pragma omp for schedule(static)
    for (int j = 0; j < out_features; j+=BS) {
        VECTOR tmp = {0.0};
        for (int i = 0; i < batch_size; i++) {
            VECTOR a_i_ld;
            #pragma unroll
            for (int vl = j; vl < MIN(out_features,j+BS); vl++)
                a_i_ld[vl-j] = a_o[i * out_features + vl];
            tmp += a_i_ld;
        }
        #pragma unroll
        for (int ij = j; ij < MIN(out_features,j+BS); ij++)
            tmp_array[ij] = tmp[ij-j];
    }

#pragma omp single
    MPI_Allreduce(MPI_IN_PLACE, tmp_array, out_features, datatype, MPI_SUM, MPI_COMM_WORLD);

#pragma omp for nowait schedule(static)
    for (int j = 0; j < out_features; j+=BS) {
	register VECTOR C_p;
	register VECTOR tmp;
        #pragma unroll
	for (int ij = j; ij < MIN(out_features,j+BS); ij++)
            tmp[ij-j] = tmp_array[ij];
        #pragma unroll
        for (int ij = j; ij < MIN(out_features,j+BS); ij++)
            C_p[ij-j] = Cj[ij];
 
        tmp /= (REAL) (batch_size * world_size);
        tmp *= taupdt;
        C_p *= one_minus_taupdt;
        C_p += tmp;
         
        #pragma unroll
        for (int ij = j; ij < MIN(out_features,j+BS); ij++)
            Cj[ij] = C_p[ij-j];
    }

//Cj[i] = one_minus_taupdt * Cj[i] + (tmp_array[i] / (REAL)(batch_size * world_size) * taupdt);

#pragma omp for
    for (int i = 0; i < in_features; i++) {
        for (int j = 0; j < out_features; j+=BS) {
            VECTOR tmp = {0.0};
            
            for (int batch = 0; batch < batch_size; batch++) {
                REAL a_i_pl = a_i[batch * in_features + i];
                VECTOR a_o_pl;
                
                #pragma unroll
                for (int vl = j; vl < MIN(out_features,j+BS); vl++)
                    a_o_pl[vl-j] = a_o[batch * out_features + vl];
                
                tmp += (a_i_pl * a_o_pl);
            }

            #pragma unroll
            for (int vl = j; vl < MIN(out_features,j+BS); vl++)
                tmp_array[i * out_features + vl] = tmp[vl-j];
        }
    }

#pragma omp single
    MPI_Allreduce(MPI_IN_PLACE, tmp_array, in_features * out_features, datatype, MPI_SUM, MPI_COMM_WORLD);

#pragma omp for
    for (int i = 0; i < in_features; i++) {
        bool skip_weight = (Ci[i] < cthr) ? true : false;	  
        if (skip_weight)
            memset( &weights[i*out_features], 0, sizeof(REAL) * out_features);
        for (int j = 0; j < out_features; j+=BS) {
            register VECTOR tmp = {0.0};
            register VECTOR C_p = {0.0};
            #pragma unroll
            for (int vl = j; vl < MIN(out_features,j+BS); vl++)
                tmp[vl-j] = tmp_array[i * out_features + vl];
            #pragma unroll
            for (int vl = j; vl < MIN(out_features,j+BS); vl++)
                C_p[vl-j] = Cij[i * out_features + vl];

            tmp /= (REAL) (batch_size * world_size);
            tmp *= taupdt;
            C_p *= one_minus_taupdt;
            C_p += tmp;
            #pragma unroll
            for (int vl = j; vl < MIN(out_features,j+BS); vl++)
                Cij[i * out_features + vl] = C_p[vl-j];

            #pragma unroll
            for (int vl = j; vl < MIN(out_features,j+BS); vl++)
                weights[i * out_features + vl] = (Cj[vl] < cthr) ? 0.0 : log((C * C_p[vl-j]) / (Ci[i] * Cj[vl]));
        }
    }
} // end omp parallel

  free(tmp_array);
  return C;
}

template <typename REAL>
void update_bias(REAL *bias, REAL *Cj, REAL cthr, const long out_features)
{
    if (world_size == 0) {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (typeid(double) == typeid(REAL))
            datatype = MPI_DOUBLE;
        else if (typeid(float) == typeid(REAL))
            datatype = MPI_FLOAT;
    }
 
    int displs[world_size];
    int rcounts[world_size];
    int stride;

    bcpnn::helpers::offsets::compute_offsets(out_features, world_rank, world_size, &stride, displs, rcounts);

#pragma omp parallel
{
    //for (long i = 0; i < out_features; i++) {
#pragma omp for
    for (size_t i = displs[world_rank]; i < displs[world_rank] + rcounts[world_rank]; i++) {
        if (Cj[i] < cthr)
            bias[i] = std::log((REAL)2 * cthr);
        else
            bias[i] = std::log(Cj[i]);
    }
} // end omp parallel

    MPI_Allgatherv(MPI_IN_PLACE, rcounts[world_rank], datatype,
                   bias, rcounts, displs,
                   datatype, MPI_COMM_WORLD);
}

template <typename REAL>
void update_bias_regularized(REAL *bias, REAL *kbi, REAL *Cj, REAL cthr, REAL khalf, REAL pmin, REAL taubdt, size_t m)
{
    if (world_size == 0) {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (typeid(double) == typeid(REAL))
            datatype = MPI_DOUBLE;
        else if (typeid(float) == typeid(REAL))
            datatype = MPI_FLOAT;
    }
 
    int displs[world_size];
    int rcounts[world_size];
    int stride;

    bcpnn::helpers::offsets::compute_offsets(m, world_rank, world_size, &stride, displs, rcounts);

    REAL pmin_div_four = pmin / (REAL)4;
    REAL k = (khalf - 1) * (pmin_div_four * pmin_div_four);
    REAL pj;
    REAL kb;

#pragma omp parallel private(pj, kb) shared(pmin_div_four, k)
{
    //for (long i = 0; i < m; i++) {
#pragma omp for
    for (size_t i = displs[world_rank]; i < displs[world_rank] + rcounts[world_rank]; i++) {
        pj = Cj[i];
        kb = 1 + k / ((pj - pmin_div_four) * (pj - pmin_div_four));

        if (pj < pmin_div_four || kb < khalf)
            kb = khalf;
        kbi[i] = (1 - taubdt) * kbi[i] + taubdt * kb;

        if (Cj[i] < cthr)
            bias[i] = kbi[i] * std::log((REAL)2 * cthr);
        else
            bias[i] = kbi[i] * std::log(pj);
    }
} // end omp parallel
    MPI_Allgatherv(MPI_IN_PLACE, rcounts[world_rank], datatype,
                   kbi, rcounts, displs,
                   datatype, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, rcounts[world_rank], datatype,
                   bias, rcounts, displs,
                   datatype, MPI_COMM_WORLD);
}

template<typename REAL>
void update_mask(uint8_t * wmask, REAL * weights,
                 REAL * Ci, REAL * Cj, REAL * Cij,
                 REAL cthr, size_t n, size_t m, size_t h,
                 size_t hypercolumns, size_t minicolumns, size_t iterations)
{
    if (world_size == 0) {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (typeid(double) == typeid(REAL))
            datatype = MPI_DOUBLE;
        else if (typeid(float) == typeid(REAL))
            datatype = MPI_FLOAT;
    }
 
    int displs[world_size];
    int rcounts[world_size];
    int stride;

    bcpnn::helpers::offsets::compute_offsets(n, world_rank, world_size, &stride, displs, rcounts);

    REAL wmask_score_nominator[n] = {0.0};
    int wmask_csum[n] = {0};
    REAL wmask_score[n] = {0.0};

#pragma omp parallel
{
    //for (size_t i = 0; i < n; i++) {
#pragma omp for
    for (size_t i = displs[world_rank]; i < displs[world_rank] + rcounts[world_rank]; i++) {
        REAL score = 0.0;
        for (size_t j = h * minicolumns; j < (h + 1) * minicolumns; j++) {
            if (Ci[i] >= cthr && Cj[j] >= cthr) {
                REAL pi = Ci[i];
                REAL pj = Cj[j];
                REAL pij = Cij[i * m + j];

                REAL x = (1 - pi)*pj;
                REAL y = (pj - pij) / x;
                REAL WijC = std::log(y);

                score += pij * weights[i * m + j] + (pj - pij)*WijC;
            }
        }
        wmask_score_nominator[i] = score;
    }

    //for (size_t i = 0; i < n; i++) {
//        int sum = 0;
#pragma omp for
    for (size_t i = displs[world_rank]; i < displs[world_rank] + rcounts[world_rank]; i++) {
#pragma omp simd
        for (size_t j = 0; j < hypercolumns; j++) {
            wmask_csum[i] += wmask[i * hypercolumns + j];
        }
//        wmask_csum[i] = sum;
    }
} // end omp parallel

    for (size_t i = 0; i < iterations; i++) {
        REAL vmax0 = 0.0;
        size_t imax0 = n;
        REAL vmin1 = 0.0;
        size_t imin1 = n;

        for (size_t j = 0; j < n; j++) {
            REAL score = wmask_score_nominator[j] / (1.0 + (REAL)wmask_csum[j]);
            if (wmask[j * hypercolumns + h] == 0 && (imax0 == n || score >= vmax0)) {
                imax0 = j;
                vmax0 = score;
            }
            else if (wmask[j * hypercolumns + h] == 1 && (imin1 == n || score <= vmin1)) {
                imin1 = j;
                vmin1 = score;
            }
        }

        if (imax0 == n || imin1 == n || vmax0 < vmin1) break;

        //printf("omp: swapping %ld (%f) with %ld (%f)\n", imax0, vmax0, imin1, vmin1);
        wmask[imax0 * hypercolumns + h] = 1;
        wmask[imin1 * hypercolumns + h] = 0;

        wmask_csum[imax0] += 1;
        wmask_csum[imin1] -= 1;
    }
}

template<typename REAL>
void apply_mask(REAL * weight, uint8_t * wmask, size_t n, size_t m, size_t hypercolumns, size_t minicolumns)
{
    if (world_size == 0) {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (typeid(double) == typeid(REAL))
            datatype = MPI_DOUBLE;
        else if (typeid(float) == typeid(REAL))
            datatype = MPI_FLOAT;
    }
 
    int displs[world_size];
    int rcounts[world_size];
    int stride;

    bcpnn::helpers::offsets::compute_offsets(n, world_rank, world_size, &stride, displs, rcounts);

    //for (size_t i = 0; i < n; i++) {
#pragma omp parallel
{
#pragma omp for
    for (size_t i = displs[world_rank]; i < displs[world_rank] + rcounts[world_rank]; i++) {
        for (size_t j = 0; j < m; j++) {
            size_t h = j / minicolumns;

            if (!wmask[i * hypercolumns + h]) {
                weight[i * m + j] = 0;
            }
        }
    }

#pragma omp for
    for (size_t i = 0; i < world_size; i++) {
        displs[i]  = displs[i]  * m;
        rcounts[i] = rcounts[i] * m;
    }
} // end omp parallel

    MPI_Allgatherv(MPI_IN_PLACE, rcounts[world_rank], datatype,
                   weight, rcounts, displs,
                   datatype, MPI_COMM_WORLD);
}

} // end name space cpu
} // end name space kernels
} // end name space bcpnn


/* wrapper to call vectorized_updateWeights */
template <typename REAL>
REAL updateWeights(REAL C, py::array_t<REAL> py_Ci,
                   py::array_t<REAL> py_Cj,
                   py::array_t<REAL> py_Cij,
                   py::array_t<REAL> py_a_i,
                   py::array_t<REAL> py_a_o,
                   const REAL taupdt,
                   py::array_t<REAL> py_weights,
                   const REAL cthr)
{
    py::buffer_info      Ci_buffer = py_Ci.request();
    py::buffer_info      Cj_buffer = py_Cj.request();
    py::buffer_info     Cij_buffer = py_Cij.request();
    py::buffer_info     a_i_buffer = py_a_i.request();
    py::buffer_info     a_o_buffer = py_a_o.request();
    py::buffer_info weights_buffer = py_weights.request();
    
    REAL      *Ci = (REAL*)Ci_buffer.ptr;
    REAL      *Cj = (REAL*)Cj_buffer.ptr;
    REAL     *Cij = (REAL*)Cij_buffer.ptr;
    REAL     *a_i = (REAL*)a_i_buffer.ptr;
    REAL     *a_o = (REAL*)a_o_buffer.ptr;
    REAL *weights = (REAL*)weights_buffer.ptr;
    
    return bcpnn::kernels::mpi::updateWeights(C, Ci, Cj, Cij, a_i, a_o, taupdt, a_i_buffer.shape[0], Ci_buffer.shape[0], Cj_buffer.shape[0], weights, cthr);
}


/* wrapper to call update_bias */
template <typename REAL>
void updateBias(py::array_t<REAL> py_bias, py::array_t<REAL> py_Cj, const REAL &cthr)
{
    py::buffer_info bias_buffer = py_bias.request();
    py::buffer_info   Cj_buffer = py_Cj.request();

    REAL *bias = (REAL*)bias_buffer.ptr;
    REAL   *Cj = (REAL*)Cj_buffer.ptr;

    bcpnn::kernels::mpi::update_bias(bias, Cj, cthr, bias_buffer.shape[0]);
}


/* wrapper to call update_bias_regularized */
template <typename REAL>
void updateBiasRegularized(py::array_t<REAL> py_bias,
                           py::array_t<REAL> py_kbi,
                           py::array_t<REAL> py_Cj,
                           REAL cthr,
                           REAL khalf,
                           REAL pmin,
                           REAL taubdt)
{
    py::buffer_info bias_buffer = py_bias.request();
    py::buffer_info kbi_buffer  = py_kbi.request();
    py::buffer_info Cj_buffer   = py_Cj.request();
 
    REAL * cpu_bias = (REAL*)bias_buffer.ptr;
    REAL * cpu_kbi  = (REAL*)kbi_buffer.ptr;
    REAL * cpu_Cj   = (REAL*)Cj_buffer.ptr;

    size_t m = bias_buffer.shape[0];

    bcpnn::kernels::mpi::update_bias_regularized<REAL>(cpu_bias, cpu_kbi, cpu_Cj, cthr, khalf, pmin, taubdt, m);
}


/* wrapper to call update_mask */
template<typename REAL>
void updateMask(py::array_t<uint8_t> py_wmask,
                py::array_t<REAL> py_weights,
                py::array_t<REAL> py_Ci,
                py::array_t<REAL> py_Cj,
                py::array_t<REAL> py_Cij,
                REAL cthr, size_t hypercolumns,
                size_t minicolumns,
                size_t h,
                size_t iterations)
{
    py::buffer_info wmask_buffer   = py_wmask.request();
    py::buffer_info weights_buffer = py_weights.request();
    py::buffer_info Ci_buffer      = py_Ci.request();
    py::buffer_info Cj_buffer      = py_Cj.request();
    py::buffer_info Cij_buffer     = py_Cij.request();

    uint8_t * cpu_wmask = (uint8_t*)wmask_buffer.ptr;
    REAL * cpu_weights  = (REAL*)weights_buffer.ptr;
    REAL * cpu_Ci       = (REAL*)Ci_buffer.ptr;
    REAL * cpu_Cj       = (REAL*)Cj_buffer.ptr;
    REAL * cpu_Cij      = (REAL*)Cij_buffer.ptr;

    size_t n = weights_buffer.shape[0];
    size_t m = weights_buffer.shape[1];

    bcpnn::kernels::mpi::update_mask<REAL>(cpu_wmask, cpu_weights, cpu_Ci, cpu_Cj, cpu_Cij, cthr, n, m, h, hypercolumns, minicolumns, iterations);
}


/* wrapper to call apply_mask */
template<typename REAL>
void applyMask(py::array_t<REAL> py_weights,
               py::array_t<uint8_t> py_wmask,
               size_t hypercolumns,
               size_t minicolumns)
{
    py::buffer_info weights_buffer = py_weights.request();
    py::buffer_info wmask_buffer   = py_wmask.request();

    REAL * cpu_weights  = (REAL*)weights_buffer.ptr;
    uint8_t * cpu_wmask = (uint8_t*)wmask_buffer.ptr;

    size_t n = weights_buffer.shape[0];
    size_t m = weights_buffer.shape[1];

    bcpnn::kernels::mpi::apply_mask<REAL>(cpu_weights, cpu_wmask, n, m, hypercolumns, minicolumns);
}


PYBIND11_MODULE(_bcpnn_kernels_mpi_internal, m)
{
    m.def("update_weights_float64",          &updateWeights<double>);
    m.def("update_weights_float32",          &updateWeights<float>);

    m.def("update_bias_float64",             &updateBias<double>);
    m.def("update_bias_float32",             &updateBias<float>);

    m.def("update_bias_regularized_float64", &updateBiasRegularized<double>);
    m.def("update_bias_regularized_float32", &updateBiasRegularized<float>);

    m.def("update_mask_float64",             &updateMask<double>);
    m.def("update_mask_float32",             &updateMask<float>);

    m.def("apply_mask_float64",              &applyMask<double>);
    m.def("apply_mask_float32",              &applyMask<float>);
}

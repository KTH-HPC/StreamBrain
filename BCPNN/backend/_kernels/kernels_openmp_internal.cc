#include "mkl_service.h"
#include <iostream>
#include <memory>
#include <omp.h>
#include <math.h>
#include <mkl.h>

#define MIN(a,b) (((a)<(b))?(a):(b))

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define REAL double
#define BS 4

namespace py = pybind11;

typedef REAL VECTOR __attribute__ ((vector_size (BS*sizeof(REAL))));


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

void matmul_add_bias(double *activation, double *inputs, double *weights, double *bias, const long inputs_rows, const long inputs_cols, const long weights_rows, const long weights_cols)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                inputs_rows, weights_cols, inputs_cols,
                1.0, inputs, inputs_cols, weights, weights_cols,
                0.0, activation, weights_cols);

#pragma omp parallel
{
#pragma omp for
    for (long i = 0; i < inputs_rows; i++) {
#pragma omp simd
        for (long j = 0; j < weights_cols; j++) {
            activation[i * weights_cols + j] += bias[j];
        }
    }
} // pragma omp end
}


double vectorized_updateWeights(double C, double *Ci, double *Cj, double *Cij, double *a_i, double *a_o, const double &taupdt,
                                const long batch_size, const long in_features, const long out_features, double *weights, const double cthr)
{

  C = (1 - taupdt) * C + taupdt;
  C = std::max(0.0, C);

  REAL one_minus_taupdt = (1.0 - taupdt);
  
#pragma omp parallel firstprivate(one_minus_taupdt,C)
  {
    
    #pragma omp for nowait
    for (int j = 0; j < in_features; j+=BS)
      {
	VECTOR tmp = {0.0};
	for (int i = 0; i < batch_size; i++)
	  { VECTOR a_i_ld;
	    #pragma unroll
	    for (int vl = j; vl < MIN(in_features,j+BS); vl++)
	      a_i_ld[vl-j] = a_i[i * in_features + vl];
	    tmp += a_i_ld; }

	tmp /= (double) batch_size;
	tmp *= taupdt;
	
	register VECTOR C_p;
        #pragma unroll
	for (int ij = j; ij < MIN(in_features,j+BS); ij++)
	  C_p[ij-j] = Ci[ij];
	C_p *= one_minus_taupdt;
	C_p += tmp;	

        #pragma unroll
	for (int ij = j; ij < MIN(in_features,j+BS); ij++)
	  Ci[ij] = std::max(0.0, C_p[ij-j]);	
      }

#pragma omp for nowait schedule(static)
    for (int j = 0; j < out_features; j+=BS) {
      
      VECTOR tmp = {0.0};
      for (int i = 0; i < batch_size; i++)
	{ VECTOR a_i_ld;
         #pragma unroll
	  for (int vl = j; vl < MIN(out_features,j+BS); vl++)
	    a_i_ld[vl-j] = a_o[i * out_features + vl];
	  tmp += a_i_ld;
	}

	tmp /= (double) batch_size;
	tmp *= taupdt;
	
	register VECTOR C_p;
        #pragma unroll
	for (int ij = j; ij < MIN(out_features,j+BS); ij++)
	  C_p[ij-j] = Cj[ij];
	C_p *= one_minus_taupdt;
	C_p += tmp;	

        #pragma unroll
	for (int ij = j; ij < MIN(out_features,j+BS); ij++)
	  Cj[ij] = std::max(0.0, C_p[ij-j]);	      
    }
    
    
#pragma omp for
    for (int i = 0; i < in_features; i++)
      {
	bool skip_weight = (Ci[i] < cthr) ? true : false;	  
	if (skip_weight)
	  memset( &weights[i*out_features], 0, sizeof(REAL) * out_features);
			    
	for (int j = 0; j < out_features; j+=BS)
	  { VECTOR tmp = {0.0};
	    
	    for (int batch = 0; batch < batch_size; batch++)
	      { REAL a_i_pl = a_i[batch * in_features + i];
		VECTOR a_o_pl;

                #pragma unroll
		for (int vl = j; vl < MIN(out_features,j+BS); vl++)
		  a_o_pl[vl-j] = a_o[batch * out_features + vl];
		
		tmp += (a_i_pl * a_o_pl); }

	    tmp /= (double) batch_size;
	    tmp *= taupdt;
	    
	    VECTOR C_p;
	    #pragma unroll
	    for (int vl = j; vl < MIN(out_features,j+BS); vl++)
	      C_p[vl-j] = Cij[i * out_features + vl];
	    
	    C_p *= one_minus_taupdt;
	    C_p += tmp;
	    #pragma unroll
	    for (int vl = j; vl < MIN(out_features,j+BS); vl++)
	      Cij[i * out_features + vl] = C_p[vl-j];

	    if (!skip_weight)
	    #pragma unroll
	      for (int vl = j; vl < MIN(out_features,j+BS); vl++)
		weights[i * out_features + vl] = (Cj[vl] < cthr) ? 0.0 : log((C * C_p[vl-j]) / (Ci[i] * Cj[vl]));
	  }
      }    
  }
  
  return C;
}
  

PYBIND11_MODULE(_bcpnn_kernels_openmp_internal, m)
{
  m.def("updateWeights", [](REAL C, 
                            py::array_t<REAL> py_Ci, 
                            py::array_t<REAL> py_Cj,
                            py::array_t<REAL> py_Cij,
                            py::array_t<REAL> py_a_i,
                            py::array_t<REAL> py_a_o,
                            const REAL &taupdt,
                            py::array_t<REAL> py_weights,
                            const REAL &cthr) {
  
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
    
    return vectorized_updateWeights(C, Ci, Cj, Cij, a_i, a_o, taupdt, a_i_buffer.shape[0], Ci_buffer.shape[0], Cj_buffer.shape[0], weights, cthr);
  });

  m.def("updateState", [](py::array_t<REAL> py_activation, 
                          py::array_t<REAL> py_inputs,
                          py::array_t<REAL> py_weights,
                          py::array_t<REAL> py_bias) {
 
    py::buffer_info activation_buffer = py_activation.request();
    py::buffer_info     inputs_buffer = py_inputs.request();
    py::buffer_info    weights_buffer = py_weights.request();
    py::buffer_info       bias_buffer = py_bias.request();

    REAL *activation = (REAL*)activation_buffer.ptr;
    REAL     *inputs = (REAL*)inputs_buffer.ptr;
    REAL    *weights = (REAL*)weights_buffer.ptr;
    REAL       *bias = (REAL*)bias_buffer.ptr;

    matmul_add_bias(activation, inputs, weights, bias, inputs_buffer.shape[0], inputs_buffer.shape[1], weights_buffer.shape[0], weights_buffer.shape[1]);
  });

}


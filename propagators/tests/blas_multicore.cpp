
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
// #include <math.h>
#include <cmath>
#include <math.h>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <complex.h>
#include <string>

#ifndef MKL_Complex16
    #define MKL_Complex16 std::complex<double>
#endif

#include "mkl.h"
#include "mkl_vml.h"
#include "mkl_vsl.h"
#include "mkl_lapacke.h"

#include <omp.h>

namespace py = pybind11;

void test_eigenvals(py::array_t<double> A)
{
    int size = A.request().shape[0];
    double *eigenvals = (double*) mkl_malloc(size * sizeof(double), 64);
    double *matrix = (double *) A.request().ptr;

    LAPACKE_dsyev (LAPACK_ROW_MAJOR, 'V', 'U', size, matrix, size, eigenvals);

    for (size_t i = 0; i < size; i++)
    {
        printf("Eigenvalue %d = %f\n", i, eigenvals[i]);
    }
    
    #pragma omp parallel
    {
        printf("Hello from process: %d\n", omp_get_thread_num());
    }

    mkl_free(eigenvals);
}

void test_omp()
{
    #pragma omp parallel
    {
        printf("Hello from process: %d\n", omp_get_thread_num());
    }

}


PYBIND11_MODULE(blas_multicore, m)
{
    m.def("test_eigenvals", &test_eigenvals);
    m.def("test_omp", &test_omp);
}
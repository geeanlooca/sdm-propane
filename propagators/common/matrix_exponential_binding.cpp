
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
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
#include "matrix_exponential.h"

namespace py = pybind11;

py::array_t<MKL_Complex16> expm(py::array_t<MKL_Complex16> A, int N, int power_terms)
{
    py::buffer_info buf_info = A.request();
    int n = buf_info.shape[0];
    int m = buf_info.shape[1];

    py::array_t<MKL_Complex16> result = py::array_t<MKL_Complex16>( n * n);
    MKL_Complex16 *_result = (MKL_Complex16 *)result.request().ptr;
    MKL_Complex16 *_A = (MKL_Complex16 *)buf_info.ptr;

    _expm(_A, _result, n, N, power_terms);
    result.resize({n, n});
    return result;
}

py::tuple symeig(py::array_t<double> A)
{
    py::buffer_info buf_info  = A.request();
    double *_A = (double *) buf_info.ptr;

    int n = buf_info.shape[0];

    py::array_t<double> eigvals = py::array_t<double>(n);
    py::array_t<double> eigvecs = py::array_t<double>(n * n);
    double *_eigvals = (double *) eigvals.request().ptr;
    double *_eigvecs = (double *) eigvecs.request().ptr;

    // first copy the matrix over, since the eigenvectors will overwrite the input matrix
    cblas_dcopy(n * n, _A, 1, _eigvecs, 1);
    sym_eig(_eigvecs, _eigvals, n);

    eigvecs.resize({n, n});

    return py::make_tuple(eigvals, eigvecs);
}

py::array_t<double> symexpm(py::array_t<double> A)
{

    py::buffer_info buf_info  = A.request();
    double *_A = (double *) buf_info.ptr;
    int n = buf_info.shape[0];

    py::array_t<double> expA = py::array_t<double>(n * n);
    double *_expA = (double *) expA.request().ptr;

    // first copy the matrix over, since the eigenvectors will overwrite the input matrix
    sym_eig_expm(_A, _expA, n);

    expA.resize({n, n});

    return expA;
}

py::array_t<MKL_Complex16> linear_operator(py::array_t<double> A, double dz)
{
    int n = A.request().shape[0];
    double *_A = (double *) A.request().ptr;

    py::array_t<MKL_Complex16> expM = py::array_t<MKL_Complex16>(n * n);
    MKL_Complex16* _expM = (MKL_Complex16*) expM.request().ptr;

    _linear_operator(_A, _expM, n, dz);

    expM.resize({n, n});

    return expM;
}



PYBIND11_MODULE(matrix_exponential, m)
{
    m.def("expm", &expm);
    m.def("symeig", &symeig);
    m.def("symexpm", &symexpm);
    m.def("linear_operator", &linear_operator);
}
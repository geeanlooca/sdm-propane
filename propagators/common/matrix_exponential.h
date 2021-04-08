#include <complex.h>
#ifndef MKL_Complex16
    #define MKL_Complex16 std::complex<double>
#endif

#include "mkl.h"
#include <cmath>


void _expm(MKL_Complex16 *_A, MKL_Complex16 *_result, int n, int N, int power_terms);

void sym_eig_expm(double *A, double *expA, int n);

void her_eig_expm(MKL_Complex16 *A, MKL_Complex16 *expA, int n);

void sym_eig(double *A, double *eigenvals, int n);

void _linear_operator(double *A, MKL_Complex16 *expM, int num_modes, double dz);
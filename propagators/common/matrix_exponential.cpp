#include "matrix_exponential.h"

void _expm(MKL_Complex16 *_A, MKL_Complex16 *_result, int n, int N, int power_terms)
{
    MKL_Complex16 scaling = 1.0 / pow(2, N);
    MKL_Complex16 one = 1;
    MKL_Complex16 zero = 0;

    MKL_Complex16* _A_scaled = (MKL_Complex16*) mkl_malloc(n * n * sizeof(MKL_Complex16), 64);
    MKL_Complex16* _A_power = (MKL_Complex16*) mkl_malloc(n * n * sizeof(MKL_Complex16), 64);
    MKL_Complex16* _A_power2 = (MKL_Complex16*) mkl_malloc(n * n * sizeof(MKL_Complex16), 64);
    MKL_Complex16* _scaled_power = (MKL_Complex16*) mkl_malloc(n * n * sizeof(MKL_Complex16), 64);
    MKL_Complex16* m_exp1 = (MKL_Complex16*)mkl_malloc(n * n * sizeof(MKL_Complex16), 64);
	MKL_Complex16* m_exp2 = (MKL_Complex16*)mkl_malloc(n * n * sizeof(MKL_Complex16), 64);

    cblas_zcopy(n * n, _A, 1, _A_scaled, 1);
    cblas_zscal(n * n, (void*) &scaling, _A_scaled, 1);

    cblas_zcopy(n * n, _A_scaled, 1, _A_power, 1);
    cblas_zscal(n * n, (void*) &zero, _result, 1);

    int factorial = 1;
    for (int i = 1; i < power_terms; i++)
    {
        // add the new term of the power series
        factorial *= i;
        MKL_Complex16 fact_scaling = 1.0/factorial;
        cblas_zcopy(n * n, _A_power, 1, _scaled_power, 1);

        cblas_zscal(n * n, (void*) (void*) &fact_scaling, _scaled_power, 1);

        vdAdd(2 * n * n, 
            (double*) &(_result[0]),
            (double *) &(_scaled_power[0]),
            (double*) &(_result[0]));


        // A_power contains the (i+1)-th power of the initial scaled matrix
        cblas_zcopy(n * n, _A_power, 1, _A_power2, 1);
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            n, n, n, (void*)&one, _A_power2, n, _A_scaled, n, (void*) &zero, _A_power, n);
    }

    // sum 1 to the diagonal (first term of the power series is the identity)
    for (int i = 0; i < n * n; i+= (n + 1))
    {
        _result[i] += 1;
    }

    for (int i = 0; i < N; i++)
    {
        cblas_zcopy(n * n, _result, 1, m_exp1, 1);
        cblas_zcopy(n * n, _result, 1, m_exp2, 1);
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            n, n, n, (void*) &one, m_exp1, n, m_exp2, n, (void *)&zero, _result, n);
    }


    mkl_free(_A_scaled);
    mkl_free(_A_power);
    mkl_free(_A_power2);
    mkl_free(_scaled_power);
    mkl_free(m_exp1);
    mkl_free(m_exp2);
} 


void sym_eig_expm(double *A, double *expA, int n)
{
    // 1) first get the eigen-decomposition of the input symmetric matrix
    // A = MDM^T (this is valid for symmetric matrices)
    double* eigenvectors = (double*) mkl_malloc(n * n * sizeof(double), 64);
    double* eigenvals = (double*) mkl_malloc(n * sizeof(double), 64);
    double* D = (double *) mkl_calloc(n  * n, sizeof(double), 64);
    double* intermediate = (double*) mkl_calloc(n * n,  sizeof(double), 64);

    // copy the matrix in the eigenvectors matrix otherwise it will be overwritten
    cblas_dcopy(n * n, A, 1, eigenvectors, 1);
    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, eigenvectors, n, eigenvals);

    // 2) compute M * e^d * M^T

    // make sure the expA matrix is set to 0 first
    cblas_dscal(n * n, 0, expA, 1);

    // a) take the exponent of the diagonal entries
    for (size_t i = 0, j=0; i < n * n; i += (n+1), j++)
    {
        D[i] = exp(eigenvals[j]);
    }

    
    // b) Compute B = e^d * M^T
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
        n, n, n, 1, D, n, eigenvectors, n, 0, intermediate, n);

    // c) compute E = M * B
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        n, n, n, 1, eigenvectors, n, intermediate, n, 0, expA, n);


    mkl_free(eigenvectors);
    mkl_free(eigenvals);
    mkl_free(D);
    mkl_free(intermediate);
}

void sym_eig(double *A, double *eigenvals, int n)
{
    LAPACKE_dsyev (LAPACK_ROW_MAJOR, 'V', 'U', n, A, n, eigenvals);
}

void _linear_operator(double *A, MKL_Complex16 *expM, int num_modes, double dz)
{
    int n = num_modes;

    // 1) first get the eigen-decomposition of the input symmetric matrix
    // A = MDM^T (this is valid for symmetric matrices)
    double* eigenvals = (double*) mkl_malloc(n * sizeof(double), 64);
    double* eigenvectors = (double*) mkl_malloc(n * n * sizeof(double), 64);
    MKL_Complex16* eigenvectors_complex = (MKL_Complex16*) mkl_malloc(n * n * sizeof(MKL_Complex16), 64);
    MKL_Complex16* D = (MKL_Complex16 *) mkl_calloc(n  * n, sizeof(MKL_Complex16), 64);
    MKL_Complex16* intermediate = (MKL_Complex16*) mkl_calloc(n * n,  sizeof(MKL_Complex16), 64);

    // copy the matrix in the eigenvectors matrix otherwise it will be overwritten
    cblas_dcopy(n * n, A, 1, eigenvectors, 1);
    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, eigenvectors, n, eigenvals);

    // copy the eigenvectors to the real part of a complex matrix of the same size
    cblas_dcopy(n * n, eigenvectors, 1, (double*) &(eigenvectors_complex[0]), 2);

    // 2) compute M * e^d * M^T
    // make sure the expA matrix is set to 0 first
    // cblas_dscal(2 * n * n, 0, (double *) &(expM[0]), 1);

    // a) take the exponent of the diagonal entries
    for (size_t i = 0, j=0; i < n * n; i += (n+1), j++)
        D[i] = std::exp( std::complex<double>(0, dz * eigenvals[j]) );

    MKL_Complex16 one = 1;
    MKL_Complex16 zero = 0;
    
    // b) Compute B = e^d * M^T
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasTrans , 
        n, n, n, &one, D, n, eigenvectors_complex, n, &zero, intermediate, n);

    // c) compute E = M * B
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        n, n, n, &one, eigenvectors_complex, n, intermediate, n, &zero, expM, n);


    mkl_free(eigenvectors);
    mkl_free(eigenvectors_complex);
    mkl_free(eigenvals);
    mkl_free(D);
    mkl_free(intermediate);
}
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
#include "matrix_exponential.h"


namespace py = pybind11;

#define C0 299792458.0
#define E0 8.8541878128e-12


int get_linear_index(int row, int col, int cols)
{
    return col + row * cols;
}

void RKRt(double *K, double *R, double *RKRt, int n)
{

    double *KRt = (double*)mkl_malloc(n * n * sizeof(double), 64);
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans, CblasTrans,
                n, n, n,
                1, K, n, R, n,
                0, KRt, n);

    cblas_dgemm(CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                n, n, n,
                1, R, n, KRt, n,
                0, RKRt, n);

    mkl_free(KRt);
}

void _perturbation_rotation_matrix(double *buf_R, double theta, int *indices, int num_groups, int num_modes)
{
    double c = cos(theta);
    double s = sin(theta);

    int current_index = 0;
    for (size_t i = 0; i < num_groups; i++)
    {
        int n = indices[i];
        if (n > 0)
        {
            double cn = cos(n * theta);
            double sn = sin(n * theta);
            // top left
            buf_R[get_linear_index(current_index, current_index, num_modes)] = c * cn;
            buf_R[get_linear_index(current_index, current_index + 1, num_modes)] = -s * cn;
            buf_R[get_linear_index(current_index + 1, current_index, num_modes)] = s * cn;
            buf_R[get_linear_index(current_index + 1, current_index + 1, num_modes)] = c * cn;

            // top right
            buf_R[get_linear_index(current_index, current_index + 2, num_modes)] = -sn * c;
            buf_R[get_linear_index(current_index, current_index + 3, num_modes)] = s * sn;
            buf_R[get_linear_index(current_index + 1, current_index + 2, num_modes)] = -s * sn;
            buf_R[get_linear_index(current_index + 1, current_index + 3, num_modes)] = -c * sn;

            // bottom left
            buf_R[get_linear_index(current_index + 2, current_index, num_modes)] = c * sn;
            buf_R[get_linear_index(current_index + 2, current_index + 1, num_modes)] = -s * sn;
            buf_R[get_linear_index(current_index + 3, current_index, num_modes)] = s * sn;
            buf_R[get_linear_index(current_index + 3, current_index + 1, num_modes)] = c * sn;

            // bottom right
            buf_R[get_linear_index(current_index + 2, current_index + 2, num_modes)] = c * cn;
            buf_R[get_linear_index(current_index + 2, current_index + 3, num_modes)] = -s * cn;
            buf_R[get_linear_index(current_index + 3, current_index + 2, num_modes)] = s * cn;
            buf_R[get_linear_index(current_index + 3, current_index + 3, num_modes)] = c * cn;

            current_index += 4;
        }
        else
        {
            buf_R[get_linear_index(current_index, current_index, num_modes)] = c;
            buf_R[get_linear_index(current_index, current_index + 1, num_modes)] = -s;
            buf_R[get_linear_index(current_index + 1, current_index, num_modes)] = s;
            buf_R[get_linear_index(current_index + 1, current_index + 1, num_modes)] = c;
            current_index += 2;
        }
    }
}

py::array_t<double> perturbation_rotation_matrix(double theta, py::array_t<int> indices)
{
    int num_modes = 0;
    int *indices_buf = (int*) indices.request().ptr;
    int num_groups = indices.request().shape[0];
    for (size_t i = 0; i < num_groups; i++)
        num_modes += (indices_buf[i] > 0) ? 4 : 2;
    
    py::array_t<double> R = py::array_t<double>(num_modes * num_modes);
    double * _R = (double*) R.request().ptr;
    for (size_t i = 0; i < num_modes * num_modes; i++)
        _R[i] = 0;

    _perturbation_rotation_matrix(_R, theta, indices_buf, num_groups, num_modes);
    R.resize({num_modes, num_modes});
    return R;
}


void apply_linear_operator(MKL_Complex16 *y, MKL_Complex16 *y0, MKL_Complex16 *M, int num_modes)
{
    MKL_Complex16 one = 1;
    MKL_Complex16 zero = 0;

    // compute the new field amplitudes after linear propagation
    cblas_zgemv(CblasRowMajor, CblasNoTrans, num_modes, num_modes, (void*) &one, 
        M, num_modes, y0, 1, (void*) &zero, y, 1);
}


void print_matrix(double *A, int rows, int cols)
{
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            printf("%f\t", A[i *  rows + j]);
        }
        printf("\n");
    }
}

void print_matrix_complex(MKL_Complex16 *A, int rows, int cols)
{

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            MKL_Complex16 a = A[i *  rows + j];
            printf("%f+j%f\t", a.real(), a.imag());
        }
        printf("\n");
    }
}

int get_4d_index(int h, int i, int j, int k,int m0, int m1, int m2, int m3)
{
    return m3*m2*m1*h + m2*m3*i + m3*j + k;
}


void apply_losses(MKL_Complex16 *y, double loss_coefficient, double dz, int num_modes)
{
    MKL_Complex16 scaling = std::exp(-loss_coefficient / 2 * dz);
    cblas_zscal(num_modes, (void *) &scaling, y, 1);
}

void compute_linear_operator(double *alpha, double *beta, double *K, MKL_Complex16 *totalK, MKL_Complex16 *expM, int num_modes, double dz)
{
    // copy the coupling matrix in the imaginary part of the total complex matrix
    cblas_dcopy(num_modes * num_modes, K, 1, (double *) &(totalK[0]) + 1, 2);

    // add the propagation constants and fiber losses to the diagonals
    for (int i = 0, j = 0; i < num_modes * num_modes; i+= (num_modes+1), j++)
        totalK[i] = std::complex<double>(-alpha[j]/2 + totalK[i].real(), beta[j]);

    MKL_Complex16 scaling = dz;
    cblas_zscal(num_modes * num_modes, (void *) &scaling, totalK, 1);

    // compute the matrix exponential
    _expm(totalK, expM, num_modes, 5, 10);
}

void compute_linear_operator_eigenvals(double *beta, double alpha, double *K, MKL_Complex16 *expM, int num_modes, double dz)
{
    int n = num_modes;

    // 1) first get the eigen-decomposition of the input symmetric matrix
    // A = MDM^T (this is valid for symmetric matrices)
    double* eigenvals = (double*) mkl_malloc(n * sizeof(double), 64);
    double* eigenvectors = (double*) mkl_malloc(n * n * sizeof(double), 64);
    MKL_Complex16* eigenvectors_complex = (MKL_Complex16*) mkl_calloc(n * n, sizeof(MKL_Complex16), 64);
    MKL_Complex16* D = (MKL_Complex16 *) mkl_calloc(n  * n, sizeof(MKL_Complex16), 64);
    MKL_Complex16* intermediate = (MKL_Complex16*) mkl_calloc(n * n,  sizeof(MKL_Complex16), 64);

    // copy the matrix in the eigenvectors matrix otherwise it will be overwritten
    cblas_dcopy(n * n, K, 1, eigenvectors, 1);
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
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans , 
        n, n, n, &one, D, n, eigenvectors_complex, n, &zero, intermediate, n);

    // c) compute E = M * B
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        n, n, n, &one, eigenvectors_complex, n, intermediate, n, &zero, expM, n);


    // for (size_t i = 0, j=0; i < n * n; i += (n+1), j++)
    //     expM[i] *= std::exp(std::complex<double>(- dz/2 * alpha, dz * beta[j]));


    mkl_free(eigenvectors);
    mkl_free(eigenvectors_complex);
    mkl_free(eigenvals);
    mkl_free(D);
    mkl_free(intermediate);
}


void compute_nonlinear_propagation(
    MKL_Complex16* y0,
    MKL_Complex16* y,
    int num_modes_signal,
    int num_modes_pump, 
    double signal_frequency,
    double pump_frequency,
    double sigma,
    double a0,
    double b0,
    MKL_Complex16 aW,
    MKL_Complex16 bW,
    double *Q_signal[],
    double *Q_pump[],
    bool undepleted_pump
)
{

    MKL_Complex16* y0s;
    MKL_Complex16* y0p;

    y0s = &y0[0];
    y0p = &y0[num_modes_signal];

    double omega_pump = 2 * M_PI * pump_frequency;
    double omega_signal = 2 * M_PI * signal_frequency;
    std::complex<double> imag = std::complex(0.0, 1.0);

    double *Q1p = Q_pump[0];
    double *Q2p = Q_pump[1];
    double *Q3p = Q_pump[2];
    double *Q4p = Q_pump[3];
    double *Q5p = Q_pump[4];
    double *Q1s = Q_signal[0];
    double *Q2s = Q_signal[1];
    double *Q3s = Q_signal[2];
    double *Q4s = Q_signal[3];
    double *Q5s = Q_signal[4];

    // *******************************************************************
    // SPM and XPM contributions: terms 1, 2 of the Nonlinear Polarization
    // *******************************************************************


    for (size_t h = 0; h < num_modes_signal; h++)
    {
        MKL_Complex16 total_interaction = 0;
        for (size_t i = 0; i < num_modes_signal; i++)
            for (size_t j = 0; j < num_modes_signal; j++)
                for (size_t k = 0; k < num_modes_signal; k++)
                {
                    float nu = h;
                    float mu = i;
                    float eta = j;
                    float rho = k;

                    int index = get_4d_index(
                        nu, rho, mu, eta, 
                        num_modes_signal,
                        num_modes_signal,
                        num_modes_signal,
                        num_modes_signal);

                    MKL_Complex16 N1 = Q1s[index] * y0s[i] * y0s[j] * std::conj(y0s[k]);
                    MKL_Complex16 N2 = Q2s[index] * std::conj(y0s[i]) * y0s[j] * y0s[k];

                    total_interaction +=  ( 
                        E0/8 * (sigma + 2.0 * b0) * N1 + 
                        E0/4*(sigma + 2.0 * a0 * b0) * N2
                        );
                }
        y[h] += imag * omega_pump / 4.0 * total_interaction;
    } 

    // ***************************************************************
    // Raman contribution: terms 3, 4, 5 of the Nonlinear Polarization
    // ***************************************************************

    //
    // Pump
    //
    if (!undepleted_pump)
    {
        for (size_t h = 0; h < num_modes_pump; h++)
        {
            MKL_Complex16 total_contribution = 0;
            for (size_t i = 0; i < num_modes_signal; i++)
                for (size_t j = 0; j < num_modes_pump; j++)
                    for (size_t k = 0; k < num_modes_signal; k++)
                    {
                        float nu = h;
                        float mu = i;
                        float eta = j;
                        float rho = k;

                        int index3 = get_4d_index(
                            nu, eta, mu, rho, 
                            num_modes_pump, 
                            num_modes_pump, 
                            num_modes_signal,
                            num_modes_signal);

                        int index4 = get_4d_index(
                            nu, rho, mu, eta, 
                            num_modes_pump,
                            num_modes_signal,
                            num_modes_signal,
                            num_modes_pump);

                        int index5 = get_4d_index(
                            nu, rho, mu, eta,
                            num_modes_pump,
                            num_modes_signal,
                            num_modes_signal,
                            num_modes_pump);

                        MKL_Complex16 N3 = Q3p[index3] * std::conj(y0s[i]) * y0s[j] * y0p[k];
                        MKL_Complex16 N4 = Q4p[index4] * y0s[i] * y0p[j] * std::conj(y0s[k]);
                        MKL_Complex16 N5 = Q5p[index5] * std::conj(y0s[i]) * y0p[j]  * y0s[k];

                        MKL_Complex16 contrib3 = E0 / 4 * (sigma + 2.0 * a0 + std::conj(bW)) * N3;
                        MKL_Complex16 contrib4 = E0 / 4 * (sigma + b0 + std::conj(bW)) * N4;
                        MKL_Complex16 contrib5 = E0 / 4 * (sigma + 2.0 * std::conj(aW) + b0) * N5;
                        total_contribution += (contrib3 + contrib4 + contrib5);
                    }

            y[num_modes_signal + h] += imag * omega_pump / 4.0 * total_contribution;
        }

        for (size_t h = 0; h < num_modes_pump; h++)
        {
            MKL_Complex16 total_interaction = 0;
            for (size_t i = 0; i < num_modes_pump; i++)
                for (size_t j = 0; j < num_modes_pump; j++)
                    for (size_t k = 0; k < num_modes_pump; k++)
                    {
                        float nu = h;
                        float mu = i;
                        float eta = j;
                        float rho = k;

                        int index = get_4d_index(
                            nu, rho, mu, eta, 
                            num_modes_pump,
                            num_modes_pump,
                            num_modes_pump,
                            num_modes_pump);

                        MKL_Complex16 N1 = Q1p[index] * y0p[i] * y0p[j] * std::conj(y0p[k]);
                        MKL_Complex16 N2 = Q2p[index] * std::conj(y0p[i]) * y0p[j] * y0p[k];

                        total_interaction +=  ( 
                            E0/8 * (sigma + 2.0 * b0) * N1 + 
                            E0/4*(sigma + 2.0 * a0 * b0) * N2
                            );
                    }

            y[num_modes_signal + h] += imag * omega_pump / 4.0 * total_interaction;
        }

    }

    //
    // signal
    //
    for (size_t h = 0; h < num_modes_signal; h++) // nu
    {
        MKL_Complex16 total_contribution = 0;
        for (size_t i = 0; i < num_modes_pump; i++) // mu
            for (size_t j = 0; j < num_modes_signal; j++) // eta
                for (size_t k = 0; k < num_modes_pump; k++) // rho
                {

                    float nu = h;
                    float mu = i;
                    float eta = j;
                    float rho = k;

                    int index3 = get_4d_index(
                        nu, eta, mu, rho, 
                        num_modes_signal, 
                        num_modes_signal, 
                        num_modes_pump,
                        num_modes_pump);

                    int index4 = get_4d_index(
                        nu, rho, mu, eta, 
                        num_modes_signal,
                        num_modes_pump,
                        num_modes_pump,
                        num_modes_signal);

                    int index5 = get_4d_index(
                        nu, rho, mu, eta,
                        num_modes_signal,
                        num_modes_pump,
                        num_modes_pump,
                        num_modes_signal);

                    MKL_Complex16 N3 = Q3s[index3] * std::conj(y0p[i]) * y0p[j] * y0s[k];
                    MKL_Complex16 N4 = Q4s[index4] * y0p[i] * y0s[j] * std::conj(y0p[k]);
                    MKL_Complex16 N5 = Q5s[index5] * std::conj(y0p[i]) * y0s[j]  * y0p[k];

                    MKL_Complex16 contrib3 = E0 / 4 * (sigma + 2.0 * a0 + bW) * N3;
                    MKL_Complex16 contrib4 = E0 / 4 * (sigma + b0 + bW) * N4;
                    MKL_Complex16 contrib5 = E0 / 4 * (sigma + 2.0 * aW + b0) * N5;
                    total_contribution += (contrib3 + contrib4 + contrib5);
                }

        y[h] += imag * omega_signal / 4.0 * total_contribution;
    }
}

void nonlinear_RK4(
    MKL_Complex16* y0s,
    MKL_Complex16* y0p,
    MKL_Complex16* ys,
    MKL_Complex16* yp,
    int num_modes_signal,
    int num_modes_pump, 
    double signal_frequency,
    double pump_frequency,
    double step_size,
    double sigma,
    double a0,
    double b0,
    MKL_Complex16 aW,
    MKL_Complex16 bW,
    double *Q_signal[],
    double *Q_pump[],
    bool undepleted_pump
)
{
    MKL_Complex16 *y_new, *y0, *y0tmp1, *y0tmp2, *y0tmp3, *k1, *k2, *k3, *k4;
    y_new = (MKL_Complex16*) mkl_calloc((num_modes_signal + num_modes_pump) , sizeof(MKL_Complex16), 64);
    y0 = (MKL_Complex16*) mkl_calloc(num_modes_signal + num_modes_pump, sizeof(MKL_Complex16), 64);
    y0tmp1 = (MKL_Complex16*) mkl_calloc(num_modes_signal + num_modes_pump, sizeof(MKL_Complex16), 64);
    y0tmp2 = (MKL_Complex16*) mkl_calloc(num_modes_signal + num_modes_pump, sizeof(MKL_Complex16), 64);
    y0tmp3 = (MKL_Complex16*) mkl_calloc(num_modes_signal + num_modes_pump, sizeof(MKL_Complex16), 64);
    k1 = (MKL_Complex16*) mkl_calloc(num_modes_signal + num_modes_pump, sizeof(MKL_Complex16), 64);
    k2 = (MKL_Complex16*) mkl_calloc(num_modes_signal + num_modes_pump, sizeof(MKL_Complex16), 64);
    k3 = (MKL_Complex16*) mkl_calloc(num_modes_signal + num_modes_pump, sizeof(MKL_Complex16), 64);
    k4 = (MKL_Complex16*) mkl_calloc(num_modes_signal + num_modes_pump, sizeof(MKL_Complex16), 64);



    cblas_dcopy(2 * num_modes_signal, (double*) &y0s[0], 1, (double *) &(y0[0]), 1);
    cblas_dcopy(2 * num_modes_pump, (double*) &y0p[0], 1, (double *) &(y0[num_modes_signal]), 1);
    cblas_dcopy(2 * (num_modes_signal + num_modes_pump), (double*) &y0[0], 1, (double *) &(y0tmp1[0]), 1);
    cblas_dcopy(2 * (num_modes_signal + num_modes_pump), (double*) &y0[0], 1, (double *) &(y0tmp2[0]), 1);
    cblas_dcopy(2 * (num_modes_signal + num_modes_pump), (double*) &y0[0], 1, (double *) &(y0tmp3[0]), 1);

    

    //
    // compute k1
    //

    // k1 = f(y0)
    compute_nonlinear_propagation(y0, k1, num_modes_signal, num_modes_pump, signal_frequency, pump_frequency,
                sigma, a0, b0, aW, bW, Q_signal, Q_pump, undepleted_pump);



    //
    // compute k2
    //

    // k2 = f(y0 + h/2 * k1)
    // y0tmp1 = y0 + h/2 * k1
    MKL_Complex16 scaling_k1 = step_size / 2.0;
    cblas_zaxpy(num_modes_pump + num_modes_signal, (void *) &scaling_k1, k1, 1, y0tmp1, 1);

    compute_nonlinear_propagation(y0tmp1, k2, num_modes_signal, num_modes_pump, signal_frequency, pump_frequency,
                sigma, a0, b0, aW, bW, Q_signal, Q_pump, undepleted_pump);

    // compute k3

    // k3 = f(y0 + h/2 * k2)
    // y0tmp2 = y0 + h/2 * k2
    MKL_Complex16 scaling_k2 = step_size / 2.0;
    cblas_zaxpy(num_modes_pump + num_modes_signal, (void *) &scaling_k2, k2, 1, y0tmp2, 1);

    compute_nonlinear_propagation(y0tmp2, k3, num_modes_signal, num_modes_pump, signal_frequency, pump_frequency,
                sigma, a0, b0, aW, bW, Q_signal, Q_pump, undepleted_pump);

    // compute k4
    // k4 = f(y0 + h*k3)
    // y0tmp3 = y0 + h*k3
    MKL_Complex16 scaling_k3 = step_size;
    cblas_zaxpy(num_modes_pump + num_modes_signal, (void *) &scaling_k3, k3, 1, y0tmp3, 1);
    compute_nonlinear_propagation(y0tmp3, k4, num_modes_signal, num_modes_pump, signal_frequency, pump_frequency,
                sigma, a0, b0, aW, bW, Q_signal, Q_pump, undepleted_pump);



    MKL_Complex16 scaling = 1.0/6.0 * step_size;
    MKL_Complex16 scaling2 = scaling * 2.0;
    MKL_Complex16 one = 1;

    cblas_zaxpy(num_modes_pump + num_modes_signal, (void*) &scaling,    k1, 1, y_new, 1);
    cblas_zaxpy(num_modes_pump + num_modes_signal, (void*) &(scaling2), k2, 1, y_new, 1);
    cblas_zaxpy(num_modes_pump + num_modes_signal, (void*) &(scaling2), k3, 1, y_new, 1);
    cblas_zaxpy(num_modes_pump + num_modes_signal, (void*) &scaling,    k4, 1, y_new, 1);
    cblas_zaxpy(num_modes_pump + num_modes_signal, (void*) &one,        y0, 1, y_new, 1);

    // copy results back
    cblas_dcopy(2 * num_modes_signal, (double*) &y_new[0], 1, (double *) &(ys[0]), 1);
    cblas_dcopy(2 * num_modes_pump, (double*) &y_new[num_modes_signal], 1, (double *) &(yp[0]), 1);
    

    mkl_free(y_new);
    mkl_free(k1);
    mkl_free(k2);
    mkl_free(k3);
    mkl_free(k4);
    mkl_free(y0);
    mkl_free(y0tmp1);
    mkl_free(y0tmp2);
    mkl_free(y0tmp3);
}

py::tuple integrate(
    py::array_t<MKL_Complex16> A_s,
    py::array_t<MKL_Complex16> A_p,
    double fiber_length,
    double stepsize,
    py::array_t<int> indices_s,
    py::array_t<int> indices_p,
    double alpha_s,
    double alpha_p,
    py::array_t<double> beta_s,
    py::array_t<double> beta_p,
    py::array_t<double> thetas_array,
    std::optional<py::array_t<double>> K_s,
    std::optional<py::array_t<double>> K_p,
    std::optional<py::dict> nonlinear_params,
    std::optional<bool> undepleted_pump,
    std::optional<bool> signal_coupling,
    std::optional<bool> pump_coupling,
    std::optional<float> filter_percent
    )
{
    double dz = stepsize;
    int step_count = fiber_length / stepsize + 1;

    bool nonlinear_propagation = false;
    double signal_freq, pump_freq, a0, b0, sigma;
    MKL_Complex16 aW, bW;
    double *Q_s[5],*Q_p[5]; 

    if (nonlinear_params.has_value())
    {
        nonlinear_propagation = true;
        py::dict nl_params = nonlinear_params.value();
        signal_freq = py::cast<double>(nl_params["signal_frequency"]);
        pump_freq = py::cast<double>(nl_params["pump_frequency"]);
        a0 = py::cast<double>(nl_params["a0"]);
        b0 = py::cast<double>(nl_params["b0"]);
        sigma = py::cast<double>(nl_params["sigma"]);
        aW = py::cast<MKL_Complex16>(nl_params["aW"]);
        bW = py::cast<MKL_Complex16>(nl_params["bW"]);

        std::string key;
        py::array_t<double> Q;
        for (size_t i = 0; i < 5; i++)
        {
            key = "Q" + std::to_string(i+1) + "_s";
            Q = py::cast<py::array_t<double>>(nl_params[key.c_str()]);
            Q_s[i] = (double *) Q.request().ptr;

            key = "Q" + std::to_string(i+1) + "_p";
            Q = py::cast<py::array_t<double>>(nl_params[key.c_str()]);
            Q_p[i] = (double *) Q.request().ptr;
        }
    }

    bool undepleted = undepleted_pump.has_value() ? undepleted_pump.value() : false;

    py::array_t<double> z = py::array_t<double>(step_count);
    double *z_buf = (double *)z.request().ptr;

    for (size_t i = 0; i < step_count; i++)
    {
        z_buf[i] = i * dz;
    }

    double *thetas = (double *)thetas_array.request().ptr;

    py::buffer_info indices_buf_info_s = indices_s.request();
    int *indices_buf_s = (int *)indices_buf_info_s.ptr;
    int num_groups_s = indices_buf_info_s.size;
    int num_modes_s = 0;

    py::buffer_info indices_buf_info_p = indices_p.request();
    int *indices_buf_p = (int *)indices_buf_info_p.ptr;
    int num_groups_p = indices_buf_info_p.size;
    int num_modes_p = 0;

    for (size_t i = 0; i < num_groups_s; i++)
        num_modes_s += (indices_buf_s[i] > 0) ? 4 : 2;

    for (size_t i = 0; i < num_groups_p; i++)
        num_modes_p += (indices_buf_p[i] > 0) ? 4 : 2;

    // determine if we need to simulate  linear coupling
    // we simulate it if the K matrices have value (not None) and
    // if the *_coupling argument is set to False
    // (if it is not set, assume we want linear coupling)
    bool apply_pump_linear_coupling, apply_signal_linear_coupling;


    apply_pump_linear_coupling = K_p.has_value();
    apply_signal_linear_coupling = K_s.has_value();

    if (signal_coupling.has_value())
        apply_signal_linear_coupling &= signal_coupling.value();
    if (pump_coupling.has_value())
        apply_pump_linear_coupling &= pump_coupling.value();

    double *Kp;
    double *Kp_theta;
    double *Ks;
    double *Ks_theta;
    double *Rs;
    double *Rp;
    MKL_Complex16 *Ktotal_p;
    MKL_Complex16 *Ktotal_s;
    MKL_Complex16 *expMp;
    MKL_Complex16 *expMs;

    // get the pointers to the coupling matrices and allocate the necessary matrices
    if (apply_pump_linear_coupling)
    {
        Rp = (double *)mkl_calloc(num_modes_p * num_modes_p, sizeof(double), 64);
        Kp = (double *)K_p.value().request().ptr;
        Kp_theta = (double *)mkl_calloc(num_modes_p * num_modes_p, sizeof(double), 64);
        Ktotal_p = (MKL_Complex16 *)mkl_calloc(num_modes_s * num_modes_s, sizeof(MKL_Complex16), 64);
        expMp = (MKL_Complex16 *)mkl_calloc(num_modes_p * num_modes_p, sizeof(MKL_Complex16), 64);
    }

    if (apply_signal_linear_coupling)
    {
        Rs = (double *)mkl_calloc(num_modes_s * num_modes_s, sizeof(double), 64);
        Ks = (double *)K_s.value().request().ptr;
        Ks_theta = (double *)mkl_calloc(num_modes_s * num_modes_s, sizeof(double), 64);
        Ktotal_s = (MKL_Complex16 *)mkl_calloc(num_modes_p * num_modes_p, sizeof(MKL_Complex16), 64);
        expMs = (MKL_Complex16 *)mkl_calloc(num_modes_s * num_modes_s, sizeof(MKL_Complex16), 64);
    }





    double *_beta_s = (double *)beta_s.request().ptr;
    double *_beta_p = (double *)beta_p.request().ptr;

    // initialize the matrix for storing the evolution of the fields along the fiber
    py::array_t<MKL_Complex16> Ap = py::array_t<MKL_Complex16>(step_count * num_modes_p);
    py::array_t<MKL_Complex16> As = py::array_t<MKL_Complex16>(step_count * num_modes_s);
    MKL_Complex16 *_Ap = (MKL_Complex16 *)Ap.request().ptr;
    MKL_Complex16 *_As = (MKL_Complex16 *)As.request().ptr;

    // copy the initial conditions in the matrix storing the evolution of field amplitudes        
    MKL_Complex16 *_As0 = (MKL_Complex16*) A_s.request().ptr;
    MKL_Complex16 *_Ap0 = (MKL_Complex16*) A_p.request().ptr;
    cblas_dscal(2 * num_modes_s * step_count, 0, (double *) &(_As[0]), 1);
    cblas_dscal(2 * num_modes_p * step_count, 0, (double *) &(_Ap[0]), 1);
    cblas_dcopy(2 * num_modes_s, (double*) &_As0[0], 1, (double *) &(_As[0]), 1);
    cblas_dcopy(2 * num_modes_p, (double*) &_Ap0[0], 1, (double *) &(_Ap[0]), 1);

    MKL_Complex16 *y0p;
    MKL_Complex16 *y0s;
    MKL_Complex16 *yp;
    MKL_Complex16 *ys;

    MKL_Complex16* ys_tmp, *yp_tmp;
    ys_tmp = (MKL_Complex16 *)mkl_calloc(num_modes_s, sizeof(MKL_Complex16), 64);
    yp_tmp = (MKL_Complex16 *)mkl_calloc(num_modes_p, sizeof(MKL_Complex16), 64);

    for (size_t iteration = 1; iteration < step_count; iteration++)
    {
        double theta = thetas[iteration-1];
        y0p = &_Ap[(iteration-1) * num_modes_p];
        yp = &_Ap[iteration * num_modes_p];
        y0s = &_As[(iteration-1) * num_modes_s];
        ys = &_As[iteration * num_modes_s];

        if (apply_pump_linear_coupling)
        {
            // compute and apply the linear coupling operator
            _perturbation_rotation_matrix(Rp, theta, indices_buf_p, num_groups_p, num_modes_p);
            RKRt(Kp, Rp, Kp_theta, num_modes_p);
            compute_linear_operator_eigenvals(_beta_p, alpha_p, Kp_theta, expMp, num_modes_p, dz);
            apply_linear_operator(yp, y0p, expMp, num_modes_p);
            for (size_t i = 0; i < num_modes_p; i++)
                yp[i] = yp[i] * std::exp( std::complex<double>(-dz * alpha_p/2, dz * _beta_p[i]));
        }
        else
        {
            for (size_t i = 0; i < num_modes_p; i++)
                yp[i] = y0p[i] * std::exp( std::complex<double>(-dz * alpha_p/2, dz * _beta_p[i]));
        }


        if (apply_signal_linear_coupling)
        {
            // compute and apply the linear coupling operator
            _perturbation_rotation_matrix(Rs, theta, indices_buf_s, num_groups_s, num_modes_s);
            RKRt(Ks, Rs, Ks_theta, num_modes_s);
            compute_linear_operator_eigenvals(_beta_s, alpha_s, Ks_theta, expMs, num_modes_s, dz);
            apply_linear_operator(ys, y0s, expMs, num_modes_s);
            for (size_t i = 0; i < num_modes_s; i++)
                ys[i] = ys[i] * std::exp( std::complex<double>(-dz * alpha_s/2, dz * _beta_s[i]));
        }
        else
        {
            // the propagation phase factor and losses
            for (size_t i = 0; i < num_modes_s; i++)
                ys[i] = y0s[i] * std::exp( std::complex<double>(-dz * alpha_s/2, dz * _beta_s[i]));
        }
        

        // copy the result in the vector of initial conditions for the nonlinear step
        cblas_dcopy(2 * num_modes_s, (double*) &ys[0], 1, (double *) &(ys_tmp[0]), 1);
        cblas_dcopy(2 * num_modes_p, (double*) &yp[0], 1, (double *) &(yp_tmp[0]), 1);

        // if required, perform the nonlinear step
        if (nonlinear_propagation)
        {
            nonlinear_RK4(ys_tmp, yp_tmp, ys, yp, num_modes_s, num_modes_p,
                    signal_freq, pump_freq, dz, sigma, a0, b0, aW, bW, Q_s, Q_p, 
                    undepleted);
        }

    }

    // free allocated heap memory
    if (apply_pump_linear_coupling)
    {
        mkl_free(Rp);
        mkl_free(Kp_theta);
        mkl_free(Ktotal_p);
        mkl_free(expMp);
    }

    if (apply_signal_linear_coupling)
    {
        mkl_free(Rs);
        mkl_free(Ks_theta);
        mkl_free(Ktotal_s);
        mkl_free(expMs);
    }

    mkl_free(ys_tmp);
    mkl_free(yp_tmp);
    Ap.resize({step_count, num_modes_p});
    As.resize({step_count, num_modes_s});

    return py::make_tuple(z, Ap, As);
}



PYBIND11_MODULE(raman_linear_coupling, m)
{
    m.def("propagate", &integrate, "Propagator for Raman equations with linear coupling.",
            py::arg("A_s"),
            py::arg("A_p"),
            py::arg("fiber_length"),
            py::arg("stepsize"),
            py::arg("indices_s"),
            py::arg("indices_p"),
            py::arg("alpha_s"),
            py::arg("alpha_p"),
            py::arg("beta_s"),
            py::arg("beta_p"),
            py::arg("theta"),
            py::arg("K_s") = py::none(),
            py::arg("K_p") = py::none(),
            py::arg("nonlinear_params") = py::none(),
            py::arg("undepleted_pump") = py::none(),
            py::arg("signal_coupling") = py::none(),
            py::arg("pump_coupling") = py::none(),
            py::arg("filtering_percent") = py::none()
        );

    m.def("perturbation_rotation_matrix", &perturbation_rotation_matrix);
}
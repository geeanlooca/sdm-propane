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

void random_normal(int size)
{

    // allocate a square matrix of size num_modes x num_modes
    // py::array_t<double> R = py::array_t<double>(size);
    // py::buffer_info buf_R_info = R.request();
    // double *buf_R = (double *)buf_R_info.ptr;
    double *buf_R = (double *)mkl_malloc(size * sizeof(double), 64);

    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, dsecnd());

    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, size, buf_R, 0, 1);
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

// void apply_nonlinear_operator(MKL_Complex16 *y, double double *Q, double coefficient, double frequency, int num_modes, int dz)
// {
//     for (size_t m1 = 0; m1 < num_modes; m1++)
//     {
//         for (size_t m2 = 0; m2 < num_modes; m2++)
//             for (size_t m3 = 0; m3 < num_modes; m3++)
//                 for (size_t m4 = 0; m4 < num_modes; m4++)
//                 {
//                     y[m1] += Q[num_modes * m1]
//                 }

//         y[m1] *= 1/4 * (std::complex(0, 1) * 2 * M_PI * frequency) * coefficient
//     }
// }

void apply_losses(MKL_Complex16 *y, double loss_coefficient, double dz, int num_modes)
{
    MKL_Complex16 scaling = exp(-loss_coefficient / 2 * dz);
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

void compute_linear_operator_eigenvals(double *beta, double *K, MKL_Complex16 *expM, int num_modes, double dz)
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


    mkl_free(eigenvectors);
    mkl_free(eigenvectors_complex);
    mkl_free(eigenvals);
    mkl_free(D);
    mkl_free(intermediate);
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

void compute_perturbation_angles(double correlation_length, double dz, int step_count, double* buffer, std::optional<unsigned MKL_INT64> seed = NULL)
{
    double sigma = 1 / sqrt((2 * correlation_length));

    unsigned MKL_INT64 seed_;

    if (!seed.has_value())
        mkl_get_cpu_clocks(&seed_);
    else
        seed_ = seed.value();

    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed_);
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, step_count, buffer, 0, 1);

    double theta0[1];
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1,theta0, 0, 2 * M_PI);

    // TODO: buffer[0] = i.i.d -> U(0, 2pi)
    buffer[0] = theta0[0];

    for (size_t iteration = 1; iteration < step_count; iteration++)
    {
        double dtheta = sqrt(dz) * sigma * buffer[iteration];
        double new_theta = buffer[iteration - 1] + dtheta;
        buffer[iteration] = new_theta;
    }
}

py::array_t<double> thetas(double correlation_length, double dz, int step_count)
{
    py::array_t<double> theta = py::array_t<double>(step_count);
    double *buff = (double *) theta.request().ptr;
    compute_perturbation_angles(correlation_length, dz, step_count, buff);

    return theta;
}

py::tuple integrate(
    py::array_t<MKL_Complex16> A_s,
    py::array_t<MKL_Complex16> A_p,
    double fiber_length,
    double stepsize,
    py::array_t<int> indices_s,
    py::array_t<int> indices_p,
    double correlation_length,
    py::array_t<double> K_s,
    py::array_t<double> K_p,
    double alpha_s,
    double alpha_p,
    py::array_t<double> beta_s,
    py::array_t<double> beta_p,
    std::optional<int> seed)
{
    int procs = 4;
    mkl_set_num_threads(procs);
    mkl_set_dynamic(1);

    double dz = stepsize;
    int step_count = fiber_length / stepsize + 1;

    py::array_t<double> z = py::array_t<double>(step_count);
    double *z_buf = (double *)z.request().ptr;

    for (size_t i = 0; i < step_count; i++)
    {
        z_buf[i] = i * dz;
    }

    // generate a vector of iid normal random variables
    // it contains the value of the angle of the perturbation at each step

    
    py::array_t<double> thetas_array = py::array_t<double>(step_count);
    double *thetas = (double *)thetas_array.request().ptr;
    compute_perturbation_angles(correlation_length, dz, step_count, thetas, seed);

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

    // allocate memory for the rotation matrices
    double *Rs = (double *)mkl_calloc(num_modes_s * num_modes_s, sizeof(double), 64);
    double *Rp = (double *)mkl_calloc(num_modes_p * num_modes_p, sizeof(double), 64);

    // get the pointers to the coupling matrices
    double *Ks = (double *)K_s.request().ptr;
    double *Kp = (double *)K_p.request().ptr;

    double *Ks_theta = (double *)mkl_calloc(num_modes_s * num_modes_s, sizeof(double), 64);
    double *Kp_theta = (double *)mkl_calloc(num_modes_p * num_modes_p, sizeof(double), 64);
    MKL_Complex16 *Ktotal_p = (MKL_Complex16 *)mkl_calloc(num_modes_s * num_modes_s, sizeof(MKL_Complex16), 64);
    MKL_Complex16 *Ktotal_s = (MKL_Complex16 *)mkl_calloc(num_modes_p * num_modes_p, sizeof(MKL_Complex16), 64);
    MKL_Complex16 *expMs = (MKL_Complex16 *)mkl_calloc(num_modes_s * num_modes_s, sizeof(MKL_Complex16), 64);
    MKL_Complex16 *expMp = (MKL_Complex16 *)mkl_calloc(num_modes_p * num_modes_p, sizeof(MKL_Complex16), 64);

    // get the pointers to attenuation and prop. constants
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
    cblas_dcopy(2 * num_modes_s, (double*) &_As0[0], 1, (double *) &(_As[0]), 1);
    cblas_dcopy(2 * num_modes_p, (double*) &_Ap0[0], 1, (double *) &(_Ap[0]), 1);

    double *identity_p = (double *)mkl_calloc(num_modes_p * num_modes_p, sizeof(double), 64);
    double *identity_s = (double *)mkl_calloc(num_modes_s * num_modes_s, sizeof(double), 64);
    for (size_t i = 0; i < num_modes_p * num_modes_p; i += (num_modes_p+1))
        identity_p[i] = 1; 
    for (size_t i = 0; i < num_modes_s * num_modes_s; i += (num_modes_s+1))
        identity_s[i] = 1; 

    for (size_t iteration = 1; iteration < step_count; iteration++)
    {
        double theta = thetas[iteration-1];

        // compute the perturbation rotation matrices for the two frequencies
        _perturbation_rotation_matrix(Rs, theta, indices_buf_s, num_groups_s, num_modes_s);
        _perturbation_rotation_matrix(Rp, theta, indices_buf_p, num_groups_p, num_modes_p);

        // apply the rotation to the coupling matrix: compute R * K * R^T
        RKRt(Kp, Rp, Kp_theta, num_modes_p);
        RKRt(Ks, Rs, Ks_theta, num_modes_s);

        compute_linear_operator_eigenvals(_beta_p, Kp_theta, expMp, num_modes_p, dz);
        compute_linear_operator_eigenvals(_beta_s, Ks_theta, expMs, num_modes_s, dz);

        // compute the new field amplitudes after linear propagation
        MKL_Complex16 *y0p = &_Ap[(iteration-1) * num_modes_p];
        MKL_Complex16 *yp = &_Ap[iteration * num_modes_p];
        MKL_Complex16 *y0s = &_As[(iteration-1) * num_modes_s];
        MKL_Complex16 *ys = &_As[iteration * num_modes_s];

        apply_linear_operator(yp, y0p, expMp, num_modes_p);
        apply_linear_operator(ys, y0s, expMs, num_modes_s);

        // apply_nonlinear_operator( y0p, num_modes_p);
        // apply_nonlinear_operator( y0s, num_modes_s);

        apply_losses(yp, alpha_p, dz, num_modes_p);
        apply_losses(ys, alpha_s, dz, num_modes_s);
    }

    // free allocated heap memory
    mkl_free(Kp_theta);
    mkl_free(Ks_theta);
    mkl_free(Ktotal_p);
    mkl_free(Ktotal_s);
    mkl_free(Rs);
    mkl_free(Rp);
    mkl_free(expMs);
    mkl_free(expMp);

    Ap.resize({step_count, num_modes_p});
    As.resize({step_count, num_modes_s});

    return py::make_tuple(z, thetas_array, Ap, As);
}

// py::tuple nonlinear_propagation(
//     py::array_t<MKL_Complex16> A_s,
//     py::array_t<MKL_Complex16> A_p,
//     double fiber_length,
//     double stepsize,
//     py::array_t<int> indices_s,
//     py::array_t<int> indices_p,
//     double correlation_length,
//     py::array_t<double> K_s,
//     py::array_t<double> K_p,
//     double alpha_s,
//     double alpha_p,
//     py::array_t<double> beta_s,
//     py::array_t<double> beta_p,
//     py::array_t<double> Q1_s,
//     py::array_t<double> Q2_s,
//     py::array_t<double> Q3_s,
//     py::array_t<double> Q4_s,
//     py::array_t<double> Q5_s,
//     py::array_t<double> Q1_p,
//     py::array_t<double> Q2_p,
//     py::array_t<double> Q3_p,
//     py::array_t<double> Q4_p,
//     py::array_t<double> Q5_p,
//     double sigma,
//     double a0,
//     double b0,
//     MKL_Complex16 aW,
//     MKL_Complex16 bW,
//     double signal_frequency,
//     double pump_frequency)
// // {
// //     int procs = 4;
// //     mkl_set_num_threads(procs);
// //     mkl_set_dynamic(1);

    // double * _Q1_s = (double *) Q1_s.request().ptr;
    // double * _Q2_s = (double *) Q2_s.request().ptr;
    // double * _Q3_s = (double *) Q3_s.request().ptr;
    // double * _Q4_s = (double *) Q4_s.request().ptr;
    // double * _Q5_s = (double *) Q5_s.request().ptr;
    // double * _Q1_p = (double *) Q1_p.request().ptr;
    // double * _Q2_p = (double *) Q2_p.request().ptr;
    // double * _Q3_p = (double *) Q3_p.request().ptr;
    // double * _Q4_p = (double *) Q4_p.request().ptr;
    // double * _Q5_p = (double *) Q5_p.request().ptr;

//     double dz = stepsize;

//     int step_count = fiber_length / stepsize + 1;

//     py::array_t<double> z = py::array_t<double>(step_count);
//     double *z_buf = (double *)z.request().ptr;

//     for (size_t i = 0; i < step_count; i++)
//     {
//         z_buf[i] = i * dz;
//     }

    // MKL_Complex16 aW_signal = aW;
    // MKL_Complex16 bW_signal = aW;
    // MKL_Complex16 aW_pump = std::complex<double>(aW_signal.real(), -aW_signal.imag())
    // MKL_Complex16 bW_pump = std::complex<double>(bW_signal.real(), -bW_signal.imag())


//     // generate a vector of iid normal random variables
//     // it contains the value of the angle of the perturbation at each step
//     py::array_t<double> thetas_array = py::array_t<double>(step_count);
//     double *thetas = (double *)thetas_array.request().ptr;
//     compute_perturbation_angles(correlation_length, dz, step_count, thetas);

//     py::buffer_info indices_buf_info_s = indices_s.request();
//     int *indices_buf_s = (int *)indices_buf_info_s.ptr;
//     int num_groups_s = indices_buf_info_s.size;
//     int num_modes_s = 0;

//     py::buffer_info indices_buf_info_p = indices_p.request();
//     int *indices_buf_p = (int *)indices_buf_info_p.ptr;
//     int num_groups_p = indices_buf_info_p.size;
//     int num_modes_p = 0;

//     for (size_t i = 0; i < num_groups_s; i++)
//         num_modes_s += (indices_buf_s[i] > 0) ? 4 : 2;

//     for (size_t i = 0; i < num_groups_p; i++)
//         num_modes_p += (indices_buf_p[i] > 0) ? 4 : 2;

//     // allocate memory for the rotation matrices
//     double *Rs = (double *)mkl_calloc(num_modes_s * num_modes_s, sizeof(double), 64);
//     double *Rp = (double *)mkl_calloc(num_modes_p * num_modes_p, sizeof(double), 64);

//     // get the pointers to the coupling matrices
//     double *Ks = (double *)K_s.request().ptr;
//     double *Kp = (double *)K_p.request().ptr;

//     double *Ks_theta = (double *)mkl_calloc(num_modes_s * num_modes_s, sizeof(double), 64);
//     double *Kp_theta = (double *)mkl_calloc(num_modes_p * num_modes_p, sizeof(double), 64);
//     MKL_Complex16 *Ktotal_p = (MKL_Complex16 *)mkl_calloc(num_modes_s * num_modes_s, sizeof(MKL_Complex16), 64);
//     MKL_Complex16 *Ktotal_s = (MKL_Complex16 *)mkl_calloc(num_modes_p * num_modes_p, sizeof(MKL_Complex16), 64);
//     MKL_Complex16 *expMs = (MKL_Complex16 *)mkl_calloc(num_modes_s * num_modes_s, sizeof(MKL_Complex16), 64);
//     MKL_Complex16 *expMp = (MKL_Complex16 *)mkl_calloc(num_modes_p * num_modes_p, sizeof(MKL_Complex16), 64);

//     // get the pointers to attenuation and prop. constants
//     double *_beta_s = (double *)beta_s.request().ptr;
//     double *_beta_p = (double *)beta_p.request().ptr;

//     // initialize the matrix for storing the evolution of the fields along the fiber
//     py::array_t<MKL_Complex16> Ap = py::array_t<MKL_Complex16>(step_count * num_modes_p);
//     py::array_t<MKL_Complex16> As = py::array_t<MKL_Complex16>(step_count * num_modes_s);
//     MKL_Complex16 *_Ap = (MKL_Complex16 *)Ap.request().ptr;
//     MKL_Complex16 *_As = (MKL_Complex16 *)As.request().ptr;

//     // copy the initial conditions in the matrix storing the evolution of field amplitudes        
//     MKL_Complex16 *_As0 = (MKL_Complex16*) A_s.request().ptr;
//     MKL_Complex16 *_Ap0 = (MKL_Complex16*) A_p.request().ptr;
//     cblas_dcopy(2 * num_modes_s, (double*) &_As0[0], 1, (double *) &(_As[0]), 1);
//     cblas_dcopy(2 * num_modes_p, (double*) &_Ap0[0], 1, (double *) &(_Ap[0]), 1);

//     double *identity_p = (double *)mkl_calloc(num_modes_p * num_modes_p, sizeof(double), 64);
//     double *identity_s = (double *)mkl_calloc(num_modes_s * num_modes_s, sizeof(double), 64);
//     for (size_t i = 0; i < num_modes_p * num_modes_p; i += (num_modes_p+1))
//         identity_p[i] = 1; 
//     for (size_t i = 0; i < num_modes_s * num_modes_s; i += (num_modes_s+1))
//         identity_s[i] = 1; 

//     for (size_t iteration = 1; iteration < step_count; iteration++)
//     {
//         double theta = thetas[iteration-1];

//         // compute the perturbation rotation matrices for the two frequencies
//         _perturbation_rotation_matrix(Rs, theta, indices_buf_s, num_groups_s, num_modes_s);
//         _perturbation_rotation_matrix(Rp, theta, indices_buf_p, num_groups_p, num_modes_p);

//         // apply the rotation to the coupling matrix: compute R * K * R^T
//         RKRt(Kp, Rp, Kp_theta, num_modes_p);
//         RKRt(Ks, Rs, Ks_theta, num_modes_s);

//         compute_linear_operator_eigenvals(_beta_p, Kp_theta, expMp, num_modes_p, dz);
//         compute_linear_operator_eigenvals(_beta_s, Ks_theta, expMs, num_modes_s, dz);

//         // compute the new field amplitudes after linear propagation
//         MKL_Complex16 *y0p = &_Ap[(iteration-1) * num_modes_p];
//         MKL_Complex16 *yp = &_Ap[iteration * num_modes_p];
//         MKL_Complex16 *y0s = &_As[(iteration-1) * num_modes_s];
//         MKL_Complex16 *ys = &_As[iteration * num_modes_s];

//         apply_linear_operator(yp, y0p, expMp, num_modes_p);
//         apply_linear_operator(ys, y0s, expMs, num_modes_s);

//         // apply_nonlinear_operator( y0p, num_modes_p);
//         // apply_nonlinear_operator( y0s, num_modes_s);

//         apply_losses(yp, alpha_p, dz, num_modes_p);
//         apply_losses(ys, alpha_s, dz, num_modes_s);
//     }

//     // free allocated heap memory
//     mkl_free(Kp_theta);
//     mkl_free(Ks_theta);
//     mkl_free(Ktotal_p);
//     mkl_free(Ktotal_s);
//     mkl_free(Rs);
//     mkl_free(Rp);
//     mkl_free(expMs);
//     mkl_free(expMp);

//     Ap.resize({step_count, num_modes_p});
//     As.resize({step_count, num_modes_s});

//     return py::make_tuple(z, thetas_array, Ap, As);
// }

int optional(int a, std::optional<int> b)
{
    int b_ = 0;
    if (b.has_value())
        b_ = b.value();

    return a + b_;
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
        py::arg("correlation_length"),
        py::arg("K_s"),
        py::arg("K_p"),
        py::arg("alpha_s"),
        py::arg("alpha_p"),
        py::arg("beta_s"),
        py::arg("beta_p"),
        py::arg("seed") = py::none()
        );

    // m.def("nonlinear_propagation", &nonlinear_propagation, "Propagator for Raman equations with linear coupling.");
    m.def("perturbation_rotation_matrix", &perturbation_rotation_matrix);
    m.def("theta", &thetas);
    m.def("optional", &optional, py::arg("a"), py::arg("b") = py::none());
}
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <math.h>
#include <cmath>
#include "mkl.h"
#include "mkl_vml.h"
#include "mkl_vsl.h"
#include "mkl_lapacke.h"
#include <chrono>
#include <cstdlib>
#include <cstdio>

namespace py = pybind11;

void _expm(double *_A, double *_result, int n)
{

    // initialize to 0

    int N = 5;
    int power_terms = 10;
    
    double scaling = 1.0 / pow(2, N);
    double *_A_scaled = (double*) mkl_malloc(n * n * sizeof(double), 64);
    double *_A_power = (double*) mkl_malloc(n * n * sizeof(double), 64);
    double *_A_power2 = (double*) mkl_malloc(n * n * sizeof(double), 64);
    double *_scaled_power = (double*) mkl_malloc(n * n * sizeof(double), 64);
    double* m_exp1 = (double*)mkl_malloc(n * n * sizeof(double), 64);
	double* m_exp2 = (double*)mkl_malloc(n * n * sizeof(double), 64);

    cblas_dcopy(n * n, _A, 1, _A_scaled, 1);
    cblas_dscal(n * n, scaling, _A_scaled, 1);
    cblas_dcopy(n * n, _A_scaled, 1, _A_power, 1);
    cblas_dscal(n * n, 0, _result, 1);

    int factorial = 1;
    for (int i = 1; i < power_terms; i++)
    {
        // add the new term of the power series
        factorial *= i;
        cblas_dcopy(n * n, _A_power, 1, _scaled_power, 1);
        cblas_dscal(n * n, 1.0 / factorial, _scaled_power, 1);
        vdAdd(n * n, _result, _scaled_power, _result);

        // A_power contains the (i+1)-th power of the initial scaled matrix
        cblas_dcopy(n * n, _A_power, 1, _A_power2, 1);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            n, n, n, 1.0, _A_power2, n, _A_scaled, n, 0, _A_power, n);
    }

    // sum 1 to the diagonal (first term of the power series is the identity)
    for (int i = 0; i < n * n; i+= (n + 1))
    {
        _result[i] += 1;
    }

    for (int i = 0; i < N; i++)
    {
        cblas_dcopy(n * n, _result, 1, m_exp1, 1);
        cblas_dcopy(n * n, _result, 1, m_exp2, 1);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            n, n, n, 1.0, m_exp1, n, m_exp2, n, 0, _result, n);
    }


    mkl_free(_A_scaled);
    mkl_free(_A_power);
    mkl_free(_A_power2);
    mkl_free(_scaled_power);
    mkl_free(m_exp1);
    mkl_free(m_exp2);
}

py::array_t<double> expm(py::array_t<double> A)
{
    py::buffer_info buf_info = A.request();
    int n = buf_info.shape[0];
    int m = buf_info.shape[1];

    py::array_t<double> result = py::array_t<double>( n * n);
    double *_result = (double *)result.request().ptr;
    double *_A = (double *)buf_info.ptr;

    _expm(_A, _result, n);
    result.resize({n, n});
    return result;
}

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

py::tuple integrate(
    py::array_t<double> A_s,
    py::array_t<double> A_p,
    double fiber_length,
    double stepsize,
    py::array_t<int> indices_s,
    py::array_t<int> indices_p,
    double correlation_length,
    py::array_t<double> K_s,
    py::array_t<double> K_p,
    py::array_t<double> alpha_s,
    py::array_t<double> alpha_p,
    py::array_t<double> beta_s,
    py::array_t<double> beta_p)
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
    double sigma = 1 / correlation_length;
    py::array_t<double> thetas_array = py::array_t<double>(step_count);
    double *thetas = (double *)thetas_array.request().ptr;

    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, dsecnd());
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, step_count, thetas, 0, 1);
    thetas[0] *= sigma;

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
    double *expMs = (double *)mkl_calloc(num_modes_s * num_modes_s, sizeof(double), 64);
    double *expMp = (double *)mkl_calloc(num_modes_p * num_modes_p, sizeof(double), 64);

    // get the pointers to attenuation and prop. constants
    double *_alpha_s = (double *)alpha_s.request().ptr;
    double *_alpha_p = (double *)alpha_s.request().ptr;
    double *_beta_s = (double *)beta_s.request().ptr;
    double *_beta_p = (double *)beta_p.request().ptr;

    // initialize the matrix for storing the evolution of the fields along the fiber
    py::array_t<double> Ap = py::array_t<double>(step_count * num_modes_p);
    py::array_t<double> As = py::array_t<double>(step_count * num_modes_s);
    double *_Ap = (double *)Ap.request().ptr;
    double *_As = (double *)As.request().ptr;

    for (size_t iteration = 1; iteration < step_count; iteration++)
    {
        // draw the new normal random variable
        // and update the perturbation angle according to the
        // constant modulus model
        double dtheta = sigma * thetas[iteration];
        double new_theta = thetas[iteration - 1] + dz * dtheta;
        // store the updated perturbation angle
        thetas[iteration] = new_theta;

        // compute the perturbation rotation matrices for the two frequencies
        // TODO: optimize by checking if the number of modes is the same: in that case, only compute one matrix
        _perturbation_rotation_matrix(Rs, new_theta, indices_buf_s, num_groups_s, num_modes_s);
        _perturbation_rotation_matrix(Rp, new_theta, indices_buf_p, num_groups_p, num_modes_p);

        // apply the rotation to the coupling matrix
        // R * K * R^T
        // First for the pump frequencies
        RKRt(Kp, Rp, Kp_theta, num_modes_p);

        // Then for the signal frequencies
        RKRt(Ks, Rs, Ks_theta, num_modes_s);

        // add the ideal propagation constants and the
        // fiber losses to the diagonals
        for (size_t i = 0; i < num_modes_s; i++)
            Ks_theta[get_linear_index(i, i, num_modes_s)] += (_beta_s[i] + _alpha_s[i]);
        for (size_t i = 0; i < num_modes_p; i++)
            Kp_theta[get_linear_index(i, i, num_modes_p)] += (_beta_p[i] + _alpha_p[i]);

        // compute the matrix exponential
        _expm(Kp_theta, expMp, num_modes_p);
        _expm(Ks_theta, expMs, num_modes_s);

        double *y0 = &_Ap[(iteration-1) * num_modes_p];
        double *y = &_Ap[iteration * num_modes_p];

        // compute the new field amplitudes after linear propagation
        cblas_dgemv(CblasRowMajor, CblasNoTrans, num_modes_p, num_modes_p, 1.0, 
            expMp, num_modes_p, y0, 1, 0, y, 1);

        y0 = &_As[(iteration-1) * num_modes_s];
        y = &_As[iteration * num_modes_s];

        cblas_dgemv(CblasRowMajor, CblasNoTrans, num_modes_s, num_modes_s, 1.0, 
            expMs, num_modes_s, y0, 1, 0, y, 1);



        // Compute the nonlinear part of the equation

        for (size_t m1 = 0; m1 < num_modes_s; m1++)
            for (size_t m2 = 0; m2 < num_modes_s; m2++)
                for (size_t m3 = 0; m3 < num_modes_s; m3++)
                    for (size_t m4 = 0; m4 < num_modes_s; m4++)
                    {
                        _As[iteration * num_groups_s + m1] += m2 * m3 * m4;
                    }

        for (size_t m1 = 0; m1 < num_modes_p; m1++)
            for (size_t m2 = 0; m2 < num_modes_p; m2++)
                for (size_t m3 = 0; m3 < num_modes_p; m3++)
                    for (size_t m4 = 0; m4 < num_modes_p; m4++)
                    {
                        _Ap[iteration * num_modes_p + m1] += m2 * m3 * m4;
                    }
    }

    Ap.resize({num_modes_p, step_count});
    As.resize({num_modes_s, step_count});
    mkl_free(Kp_theta);
    mkl_free(Ks_theta);
    mkl_free(Rs);
    mkl_free(Rp);
    mkl_free(expMs);
    mkl_free(expMp);

    return py::make_tuple(z, thetas_array, Ap, As);
}


PYBIND11_MODULE(raman_linear_coupling, m)
{
    m.def("propagate", &integrate, "Propagator for Raman equations with linear coupling.");
    m.def("expm", &expm);
}
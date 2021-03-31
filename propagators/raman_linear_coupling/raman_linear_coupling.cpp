#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// #include <math.h>
#include <cmath>
#include "mkl.h"
#include "mkl_vml.h"
#include "mkl_vsl.h"
#include "mkl_lapacke.h"
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <complex.h>

#define MKL_Complex16 std::complex<double>


namespace py = pybind11;

void _expm(MKL_Complex16 *_A, MKL_Complex16 *_result, int n, int N, int power_terms)
{

    // initialize to 0

    // int N = 5;
    // int power_terms = 10;

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

void apply_nonlinear_operator(MKL_Complex16 *y, int num_modes)
{
    for (size_t m1 = 0; m1 < num_modes; m1++)
        for (size_t m2 = 0; m2 < num_modes; m2++)
            for (size_t m3 = 0; m3 < num_modes; m3++)
                for (size_t m4 = 0; m4 < num_modes; m4++)
                {
                    y[m1] += m2 * m3 * m4;
                }
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
    double sigma = sqrt(1 / (2 * correlation_length));
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
    MKL_Complex16 *Ktotal_p = (MKL_Complex16 *)mkl_calloc(num_modes_s * num_modes_s, sizeof(MKL_Complex16), 64);
    MKL_Complex16 *Ktotal_s = (MKL_Complex16 *)mkl_calloc(num_modes_p * num_modes_p, sizeof(MKL_Complex16), 64);
    MKL_Complex16 *expMs = (MKL_Complex16 *)mkl_calloc(num_modes_s * num_modes_s, sizeof(MKL_Complex16), 64);
    MKL_Complex16 *expMp = (MKL_Complex16 *)mkl_calloc(num_modes_p * num_modes_p, sizeof(MKL_Complex16), 64);

    // get the pointers to attenuation and prop. constants
    double *_alpha_s = (double *)alpha_s.request().ptr;
    double *_alpha_p = (double *)alpha_p.request().ptr;
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
        // draw the new normal random variable
        // and update the perturbation angle according to the
        // constant modulus model
        double dtheta = sigma * thetas[iteration];
        double new_theta = thetas[iteration - 1] - dz * dtheta;
        // store the updated perturbation angle
        thetas[iteration] = new_theta;

        // compute the perturbation rotation matrices for the two frequencies
        // TODO: optimize by checking if the number of modes is the same: in that case, only compute one matrix
        _perturbation_rotation_matrix(Rs, new_theta, indices_buf_s, num_groups_s, num_modes_s);
        _perturbation_rotation_matrix(Rp, new_theta, indices_buf_p, num_groups_p, num_modes_p);

        // apply the rotation to the coupling matrix: compute R * K * R^T
        RKRt(Kp, identity_p, Kp_theta, num_modes_p);
        RKRt(Ks, identity_s, Ks_theta, num_modes_s);

        compute_linear_operator(_alpha_p, _beta_p, Kp_theta, Ktotal_p, expMp, num_modes_p, dz);
        compute_linear_operator(_alpha_s, _beta_s, Ks_theta, Ktotal_s, expMs, num_modes_s, dz);
        

        // compute the new field amplitudes after linear propagation
        MKL_Complex16 *y0 = &_Ap[(iteration-1) * num_modes_p];
        MKL_Complex16 *y = &_Ap[iteration * num_modes_p];
        apply_linear_operator(y, y0, expMp, num_modes_p);

        y0 = &_As[(iteration-1) * num_modes_s];
        y = &_As[iteration * num_modes_s];
        apply_linear_operator(y, y0, expMs, num_modes_s);

        // apply_nonlinear_operator( &_Ap[iteration * num_modes_p], num_modes_p);
        // apply_nonlinear_operator( &_As[iteration * num_modes_s], num_modes_s);
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
    As.resize({step_count, num_modes_s });

    return py::make_tuple(z, thetas_array, Ap, As);
}

py::array_t<std::complex<double>> sumComplex(py::array_t<double> alpha, py::array_t<double> beta, py::array_t<double> K)
{
    py::buffer_info buf_info = alpha.request();
    int n = buf_info.shape[0];

    double *_alpha = (double*) alpha.request().ptr;
    double *_beta = (double*) beta.request().ptr;
    double *_K = (double*) K.request().ptr;


    py::array_t<std::complex<double>> result = py::array_t<std::complex<double>>( n * n);
    MKL_Complex16 *_result = (MKL_Complex16*)result.request().ptr;

    cblas_dcopy(n * n, _K, 1, (double *) &(_result[0]), 2);
    cblas_dcopy(n * n, _K, 1, (double *) &(_result[0]) + 1, 2);
    for (int i = 0, j = 0; i < n * n; i+= (n+1), j++)
    {
        printf("Alpha: %f\tBeta: %f\n", _alpha[j], _beta[j]);
        _result[i] += std::complex<double>(-_alpha[j], _beta[j]);
    }

    result.resize({n, n});
    return result;
}


PYBIND11_MODULE(raman_linear_coupling, m)
{
    m.def("propagate", &integrate, "Propagator for Raman equations with linear coupling.");
    m.def("expm", &expm);
    m.def("sumComplex", &sumComplex);
    m.def("perturbation_rotation_matrix", &perturbation_rotation_matrix);
}
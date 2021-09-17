from numpy.core.numeric import ones_like
from scipy.constants import epsilon_0 as e0, speed_of_light as c0
from scipy.integrate import solve_ivp
import scipy.linalg
import numpy as np
import numba
from numba import jit
import tqdm

import time

@jit
def linear_operator(alpha, beta, dz):
    return np.exp((-alpha/2 + 1j*beta)*dz)

def nonlinear_operator():
    pass

@jit
def euler_step(fun, stepsize, y0, *args):
    yn = fun(y0, *args)
    return y0 + stepsize * yn


def rk4_step(fun, stepsize, y0, *args, **kwargs):
    k1 = fun(y0, *args, **kwargs)
    k2 = fun(y0 + stepsize * k1 / 2, *args, **kwargs)
    k3 = fun(y0 + stepsize * k2 / 2, *args, **kwargs)
    k4 = fun(y0 + stepsize * k3, *args, **kwargs)
    return y0 +  1/6 * stepsize * (k1 + 2*k2 + 2*k3 + k4)

@jit(nopython=True, parallel=True)
def raman_interaction(y0l, y0f, frequency, aW, bW, Q3, Q4, Q5):

    y = np.zeros_like(y0l, dtype="complex128")
    num_l = len(y0l)
    num_f = len(y0f)
    omega = 2 * np.pi * frequency

    for h in numba.prange(num_l):
        parallel = 0
        orthogonal = 0

        # compute the raman interaction for each mode
        for i in range(num_f):
            for j in range(num_f):
                for k in range(num_l):
                    fields_3 = np.conj(y0f[i]) * y0f[j] * y0l[k]
                    fields_4 = y0f[i] * y0l[j] * np.conj(y0f[k])
                    fields_5 = np.conj(y0f[i]) * y0l[j] * y0f[k]

                    orthogonal += e0/4 * bW * (fields_3 * Q3[h,i,j,k] + fields_4 * Q4[h,i,j,k])
                    parallel += e0/4 * 2*aW *  fields_5 *  Q5[h,i,j,k]

        y[h] = 1j * omega / 4 * (orthogonal + parallel)

    return y

def spm_xpm_interaction():
    pass

def integrate(y0p, y0s, alpha_p, alpha_s, beta_p, beta_s, nonlinear_params, fiber_length, dz):

    num_p = int(y0p.size)
    num_s = int(y0s.size)
    step_count = int( fiber_length / dz + 1)

    a0 = nonlinear_params['a0']
    b0 = nonlinear_params['b0']
    aW = nonlinear_params['aW']
    bW = nonlinear_params['bW']
    sigma = nonlinear_params['sigma']
    pump_frequency = nonlinear_params['pump_frequency']
    signal_frequency = nonlinear_params['signal_frequency']

    nonlinear_pol_coefficients_signal = e0/4 * np.array([
        1/2 * (sigma + 2*b0),
        sigma + 2*a0 + b0,
        sigma + 2*a0 + bW,
        sigma + b0 + bW,
        sigma + 2*aW * b0])

    nonlinear_pol_coefficients_pump = e0/4 * np.array([
        1/2 * (sigma + 2*b0),
        sigma + 2*a0 + b0,
        sigma + 2*a0 + np.conj(bW),
        sigma + b0 + np.conj(bW),
        sigma + 2*np.conj(aW) + b0])

    # nonlinear_pol_coefficients_pump = nonlinear_pol_coefficients_signal

    Q3_p = nonlinear_params['Q3_p']
    Q4_p = nonlinear_params['Q4_p']
    Q5_p = nonlinear_params['Q5_p']
    Q3_s = nonlinear_params['Q3_s']
    Q4_s = nonlinear_params['Q4_s']
    Q5_s = nonlinear_params['Q5_s']

    z = np.arange(step_count) * dz

    yp = np.zeros((step_count, num_p), dtype="complex128")
    ys = np.zeros((step_count, num_s), dtype="complex128")

    yp[0] = y0p
    ys[0] = y0s

    counter = 0
    for i in tqdm.tqdm(range(1, step_count)):
    # for i in range(1, step_count):
        y0_p = yp[i - 1]
        y0_s = ys[i - 1]


        y_p = y0_p * linear_operator(alpha_p, beta_p, dz/2)
        y_s = y0_s * linear_operator(alpha_s, beta_s, dz/2)

        y_s_tmp = euler_step(raman_interaction, dz, y_s, y_p, signal_frequency, np.conj(aW), np.conj(bW), Q3_s, Q4_s, Q5_s)
        y_p_tmp = euler_step(raman_interaction, dz, y_p, y_s, pump_frequency, aW, bW, Q3_p, Q4_p, Q5_p)



        y_p = y_p * linear_operator(alpha_p, beta_p, dz/2)
        y_s = y_s_tmp * linear_operator(alpha_s, beta_s, dz/2)


        ys[i] = y_s
        yp[i] = y_p

    return yp.T, ys.T, z
        
def propagate(
        As0,
        Ap0,
        fiber_length,
        dz,
        indices_s,
        indices_p,
        correlation_length,
        alpha_s,
        alpha_p,
        beta_s,
        beta_p,
        K_s=None,
        K_p=None,
        seed=0,
        nonlinear_params=None,
        undepleted_pump=True,
        signal_coupling=False,
        pump_coupling=False,
        signal_spm=False,
        pump_spm=False):

    step_count = int(fiber_length // dz + 1)

    print("Fiber length:", fiber_length, "Step count:", step_count)
    z = np.arange(step_count) * dz

    f_p = nonlinear_params['pump_frequency']
    f_s = nonlinear_params['signal_frequency']

    num_modes_s = sum([4 if p > 0 else 2 for p in indices_s])
    num_modes_p = sum([4 if p > 0 else 2 for p in indices_p])


    A0 = np.concatenate((Ap0, As0))

    beta = np.concatenate((beta_p, beta_s))

    # solution = solve_ivp(rhs, (z[0], z[-1]), A0, args=(alpha_s, beta, num_modes_p, num_modes_s, f_p,
    #                      f_s, Q3_p, Q4_p, Q5_p, Q3_s, Q4_s, Q5_s, nonlinear_pol_coefficients_pump, nonlinear_pol_coefficients_signal), t_eval=z)
    # A = solution.y

    # z = solution.t


    Ap, As, z = integrate(Ap0, As0, alpha_p, alpha_s, beta_p, beta_s, nonlinear_params, fiber_length, dz)

    return z, z, Ap.T, As.T

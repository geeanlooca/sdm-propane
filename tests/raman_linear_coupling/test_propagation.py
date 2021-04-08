from pathlib import Path
import os
import sys

dirname = Path(__file__).resolve().parents[2]
sys.path.append(os.path.abspath(dirname))

import numpy as np 
import scipy.linalg
import time 
import pytest
import raman_linear_coupling


def test_propagate():
    import matplotlib.pyplot as plt 

    fiber_length = 1e3
    correlation_length = 10
    dz = correlation_length / 10
    indices_s = np.array([1]).astype("int32")
    indices_p = np.array([1]).astype("int32")

    num_modes_s = 0
    for n in indices_s:
        num_modes_s += 4 if n else 2

    num_modes_p = 0
    for n in indices_p:
        num_modes_p += 4 if n else 2


    Pp0 = 1e-3
    Ps0 = 1e-3
    Ap0 = np.sqrt(Pp0) * np.ones((num_modes_p,)).astype("complex128")
    As0 = np.sqrt(Ps0) * np.ones((num_modes_s,)).astype("complex128")

    As0[2:] = 0
    Ap0[2:] = 0

    pump_attenuation = 0.3 * 1e-3 * np.log(10) / 10 
    signal_attenuation = 0.2* 1e-3 * np.log(10) / 10 

    alpha_s = np.ones((num_modes_s,)) * signal_attenuation 
    alpha_p = np.ones((num_modes_p,)) * pump_attenuation

    Ks = np.random.random((num_modes_s, num_modes_s)) * signal_attenuation * 1000
    Ks = 0.5 * (Ks + Ks.T)
    Kp = np.zeros((num_modes_p, num_modes_p))


    kappa_s = np.linalg.eigvals(Ks)
    deltaK = np.max(kappa_s) - np.min(kappa_s)
    print("DeltaKappa:", deltaK)

    beta_s = np.zeros((num_modes_s,)) * 4e9
    beta_p = np.zeros((num_modes_p,)) * 4.5e9

    curves = 5

    thetas = []
    plt.figure()
    for _ in range(curves):
        start = time.perf_counter()
        z, theta, Ap, As = raman_linear_coupling.propagate(
            As0,
            Ap0,
            fiber_length,
            dz,
            indices_s,
            indices_p,
            correlation_length,
            Ks,
            Kp,
            alpha_s,
            alpha_p,
            beta_s,
            beta_p,
        )
        end = time.perf_counter()
        print("Time: ", (end - start))

        signal_power_s = np.abs(As) ** 2
        signal_power_p = np.abs(Ap) ** 2

        thetas.append(theta)

        plt.plot(z, 10 * np.log10(signal_power_s))


    total_signal_power = np.sum(np.abs(As0) ** 2)
    total_pump_power = np.sum(np.abs(Ap0) ** 2)

    plt.figure()
    plt.plot(z, signal_power_s.sum(axis=1), label="Total power, pump")
    plt.plot(z, signal_power_p.sum(axis=1), label="Total power, signal")
    plt.ylim(0, 1.5 * max(total_signal_power, total_pump_power))
    plt.legend()

    plt.figure()
    for theta in thetas:
        plt.plot(z, theta)
    plt.show()



def perturbation_rotation_matrix(theta, azimuthal_indeces):
    total_modes = 0
    for n in azimuthal_indeces:
        total_modes += 4 if n else 2

    c = np.cos(theta)
    s = np.sin(theta)
    R = np.zeros((total_modes, total_modes))

    current_index = 0
    for m, n in enumerate(azimuthal_indeces):
        if n:
            cn = np.cos(n * theta)
            sn = np.sin(n * theta)
            # top left
            R[current_index, current_index] = cn * c
            R[current_index, current_index + 1] = -s * cn
            R[current_index + 1, current_index] = s * cn
            R[current_index + 1, current_index + 1] = c * cn

            # top right
            R[current_index, current_index + 2] = -sn * c
            R[current_index, current_index + 3] = s * sn
            R[current_index + 1, current_index + 2] = -s * sn
            R[current_index + 1, current_index + 3] = -c * sn

            # bottom left
            R[current_index + 2, current_index] = c * sn
            R[current_index + 2, current_index + 1] = -s * sn
            R[current_index + 3, current_index] = s * sn
            R[current_index + 3, current_index + 1] = c * sn

            # bottom right
            R[current_index + 2, current_index + 2] = c * cn
            R[current_index + 2, current_index + 3] = -s * cn
            R[current_index + 3, current_index + 2] = s * cn
            R[current_index + 3, current_index + 3] = c * cn
        else:
            R[current_index, current_index] = c
            R[current_index, current_index + 1] = -s
            R[current_index + 1, current_index] = s
            R[current_index + 1, current_index + 1] = c

        current_index += 4 if n else 2

    return R


@pytest.mark.parametrize(
    "theta", np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=(3,))
)
@pytest.mark.parametrize("indices", [[0], [1], [0, 1], [1, 1], [0, 1, 0, 1], [1, 1, 0, 1]])
def test_perturbation_rotation(theta, indices):
    print(f"{theta}\t{indices}".center(100, "#"))

    R = perturbation_rotation_matrix(theta, indices)
    R_cpp = raman_linear_coupling.perturbation_rotation_matrix(theta, np.array(indices).astype("int32"))

    max_abs_error = np.max(np.abs(R - R_cpp))
    print("Numpy".center(50, "="))
    print(R)
    print("C++".center(50, "="))
    print(R_cpp)
    # print(f"Max abs. error: {max_abs_error}, theta: {theta}")
    assert np.allclose(R, R_cpp), f"Max abs. error: {max_abs_error}, theta: {theta}"

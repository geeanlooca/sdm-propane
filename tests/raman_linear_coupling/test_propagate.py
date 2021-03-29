import numpy as np 
import time 

import os

import raman_linear_coupling


def test_propagate():
    fiber_length = 100e3
    correlation_length = 1
    dz = correlation_length / 10
    indices_s = np.array([0, 1, 0, 1]).astype("int32")
    indices_p = np.array([0, 1, 0, 1]).astype("int32")

    num_modes_s = 0
    for n in indices_s:
        num_modes_s += 4 if n else 2

    num_modes_p = 0
    for n in indices_p:
        num_modes_p += 4 if n else 2

    Ks = np.random.random((num_modes_s, num_modes_s))
    Kp = np.random.random((num_modes_p, num_modes_p))
    Ap0 = np.random.random((num_modes_p,))
    As0 = np.random.random((num_modes_s,))
    alpha_s = np.ones((num_modes_s,)) * 1e-3
    alpha_p = np.ones((num_modes_p,)) * 1e-3
    beta_s = np.ones((num_modes_s,)) * 1e-3
    beta_p = np.ones((num_modes_p,)) * 1e-3

    start = time.perf_counter()
    z, thetas, Ap, As = raman_linear_coupling.propagate(
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
    assert z[-1] == fiber_length
    assert z[1] - z[0] == dz
    assert Ap.shape[0] == num_modes_p
    assert As.shape[0] == num_modes_s
    assert Ap.shape[1] == len(z)
    assert len(thetas) == len(z)
    print("Number of steps:", len(z))
    print("Number of modes:", num_modes_s, num_modes_p)
    print("A shape:", Ap.shape)
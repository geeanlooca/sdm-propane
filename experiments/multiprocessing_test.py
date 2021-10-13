
import numpy as np 
import scipy.linalg
import time 

import os

import raman_linear_coupling


def test_propagate(_):
    fiber_length = 1e3
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

    return z, thetas



def say_hi(name):
    print("Hi " + name)
    time.sleep(2)

import multiprocessing
cpu_num = multiprocessing.cpu_count()

with multiprocessing.Pool(cpu_num) as pool:
    results = pool.map(test_propagate, 2 * cpu_num * [None] )

pool.join()

import matplotlib.pyplot as plt

plt.figure()

for (z, theta) in results:
    plt.plot(z, theta)

plt.show()
    
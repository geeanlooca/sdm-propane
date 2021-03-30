import numpy as np 
import scipy.linalg
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


def test_matrix_exponential():

    size = 5
    A = 1 * np.random.randint(-1, 1, (3, 3)).astype("float64")
    # A = np.random.randn(3, 3)
    A = A + A.T
    w, v = np.linalg.eig(A)
    expA = scipy.linalg.expm(A)

    # scale and square method
    N = 10
    Asmall = A / (2 ** N)
    acc = 10

    fact = 1
    tmp = np.eye(3)
    Apow = np.copy(Asmall)

    for i in range(1, acc+1):
        fact *= i
        tmp += (Apow) / fact
        Apow = np.matmul(Apow, Asmall)
        

    expB = np.copy(tmp)
    for i in range(N):
        expB = np.matmul(expB, expB)


    expB = raman_linear_coupling.expm(A)


    print(np.max(np.abs(expA-expB)))
    print(expA)
    print(expB)
    print(np.max(np.abs(expA-expB)/expB))




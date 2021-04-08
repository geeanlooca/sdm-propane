
import pytest 
import numpy as np 
import scipy.linalg

import matrix_exponential

# @pytest.mark.parametrize("size", [3, 5, 10, 15, 20, 30])
# @pytest.mark.parametrize("scaling", [1, 1e1, 1e2, 1e4, 1e6 ])
# def test_matrix_exponential(size, scaling):

#     A = scaling * np.ones((size, size)).astype("complex128")
#     A = np.tril(A) + np.triu(A.T, 1)

#     norm = np.abs(A).sum(axis=1).max()

#     w, v = np.linalg.eig(A)

#     s = max(0, int(np.log2(norm) + 1))
#     N = int(s)
#     N = 15
#     power_terms = int(10)

#     expA = scipy.linalg.expm(A)
#     expB = matrix_exponential.expm(A, N, power_terms)


#     assert np.allclose(expA, expB), f"Size: {size}, scaling: {scaling}"


def sort_matrix(A, idx):
    B = np.zeros_like(A)
    for i in range(idx.shape[0]):
        B[:, i] = A[:, idx[i]]

    return B

@pytest.mark.parametrize("size", [3, 5, 7, 10, 12, 15])
# @pytest.mark.parametrize("size", [3])
def test_symmetric_eig(size):

    def vecnorm(a, axis=0):
        return np.sqrt(np.sum(a * a, axis=axis))

    def rebuild_matrix(eva, eve):
        Vm1 = np.linalg.inv(eve)
        return np.matmul(eve, np.matmul(np.diag(eva), Vm1))

    A = np.random.random((size, size))
    A = np.tril(A) + np.triu(A.T, 1)

    eigvals_np, eigvecs_np = np.linalg.eig(A)
    eigvals, eigvecs = matrix_exponential.symeig(A)

    sort_idx_np = np.argsort(eigvals_np)
    sort_idx = np.argsort(eigvals)


    assert np.allclose(A, rebuild_matrix(eigvals_np, eigvecs_np)), "Rebuilt matrix not correct (np)"
    assert np.allclose(A, rebuild_matrix(eigvals, eigvecs)), "Rebuilt matrix not correct (C++)"
    assert np.allclose(eigvals_np[sort_idx_np], eigvals[sort_idx]), "Eigenvalues not matching"
    # print(np.real(np.max(np.abs(expA-expB)/expB)))


@pytest.mark.parametrize("size", [3, 5, 10, 15, 20, 40])
@pytest.mark.parametrize("scaling", [1, 1e1, 1e2])
def test_scalar_mult(size, scaling):
    """Check if e^jB is  the same as M * e^jD * M^T"""
    print(f"Size: {size}".center(50, "="))
    alpha = 0.2
    A = scaling * np.random.random((size, size))
    A = np.tril(A) + np.triu(A.T, 1)
    D, V = np.linalg.eig(A)

    expA = np.matmul(V, np.matmul(np.diag(np.exp(alpha * 1j * D)), V.T))
    expB = scipy.linalg.expm(alpha * 1j * A)

    assert np.allclose(expA, expB), f"Size: {size}, scaling: {scaling}"

def test_reconstruct_hermitian_from_eigen():
    size = 3
    R = 5 * np.random.random((size, size))
    I = 5 * np.random.random((size, size))
    A = np.round(R) + 1j * np.round(I)
    A = np.tril(A) + np.conj(np.triu(A.T))
    assert np.allclose(A, A.conj().T), "Matrix not hermitian"

    D, V = np.linalg.eig(A)
    A_reconstructed = np.matmul(V, np.matmul(np.diag(D), V.conj().T))
    assert np.allclose(A, A_reconstructed)

def test_hermitian_exponential():
    size = 3
    R = 5 * np.random.random((size, size))
    I = 5 * np.random.random((size, size))
    A = np.round(R) + 1j * np.round(I)
    A = np.tril(A) + np.conj(np.triu(A.T))
    assert np.allclose(A, A.conj().T), "Matrix not hermitian"


@pytest.mark.parametrize("size", [3, 5, 10, 15, 20])
@pytest.mark.parametrize("scaling", [1, 1e1, 1e2])
def test_scaling_squaring_eigenvals(size, scaling):
    A = scaling * np.random.random((size, size))
    A = np.tril(A) + np.triu(A.T, 1)
    expA = scipy.linalg.expm(A)

    N = 10
    A_scaled = A / (2 ** N)
    A_exp = scipy.linalg.expm(A_scaled)
    for _ in range(N):
        A_exp = np.matmul(A_exp, A_exp)


    assert np.allclose(expA, A_exp)



@pytest.mark.parametrize("size", [3, 5, 10, 15, 20])
@pytest.mark.parametrize("scaling", [1, 10])
def test_linear_operator(size, scaling):
    print(f"Size: {size}".center(50, "="))

    dz = 0.1

    A = scaling * np.random.random((size, size))
    A = np.tril(A) + np.triu(A.T, 1)
    D, V = np.linalg.eig(A)

    expA = np.matmul(V, np.matmul(np.diag(np.exp(dz * 1j * D)), V.T))
    expB = scipy.linalg.expm(dz * 1j * A)

    expC = matrix_exponential.linear_operator(A, dz)


    assert np.allclose(expA, expB), f"Size: {size}, scaling: {scaling}"
    assert np.allclose(expA, expC), f"Size: {size}, scaling: {scaling}"
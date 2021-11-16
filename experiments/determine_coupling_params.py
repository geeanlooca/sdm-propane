import sys
import os

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cwd))
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np

from scipy.constants import lambda2nu
import scipy.linalg
from scipy.linalg.decomp import eig
from fiber import StepIndexFiber


def assert_error(target, actual, rtol=1e-3, str=""):
    error = np.abs((actual - target) / target)
    assert (
        error < rtol
    ), f"{str} Expected: {target} \tObserved: {actual}\tError: {error}"


def get_coupling_strength(K):
    eigvals = scipy.linalg.eigvals(K)
    coupling_birefringence = eigvals.max() - eigvals.min()
    return coupling_birefringence


parser = argparse.ArgumentParser()
parser.add_argument("-a", default=0.5, type=float)
parser.add_argument("-t", "--tex", action="store_true")

args = parser.parse_args()

if args.tex:
    plt.rcParams.update(
        {
            "text.latex.preamble": r"\usepackage{mathpazo}",
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["palatino"],
        }
    )

fiber = StepIndexFiber(
    clad_index=1.4545, delta=0.005, core_radius=6, clad_radius=60, data_path="fibers"
)

signal_wavelength = 1550
pump_wavelength = 1459.45

signal_freq = lambda2nu(signal_wavelength * 1e-9)
pump_freq = lambda2nu(pump_wavelength * 1e-9)

fiber.load_data(wavelength=signal_wavelength)
fiber.load_data(wavelength=pump_wavelength)
print(fiber.num_groups(wavelength=signal_wavelength))
print(fiber.num_groups(wavelength=pump_wavelength))

beta = fiber.propagation_constants(wavelength=signal_wavelength)
modal_birefringence = beta.max() - beta.min()

Lbeta = 2 * np.pi / modal_birefringence

print(f"Fiber modal beat length: {Lbeta * 1e3:.3f} mm")

# physical parameters for which their perturbation matrix has strength equal to (theoretically) 1
gamma0 = fiber.gamma0[signal_wavelength]
delta_n = fiber.delta_n0[signal_wavelength]

# in reality its more like .99 so I want to obtain the exact values
Kb = fiber.birefringence_coupling_matrix(coupling_strength=1)
Ke = fiber.core_ellipticity_coupling_matrix(coupling_strength=1)

# Check that the matrices are normalized to unit strength
kappa_b = get_coupling_strength(Kb)
kappa_e = get_coupling_strength(Ke)
print("Ke strength: ", get_coupling_strength(Ke))
print("Kb strength: ", get_coupling_strength(Kb))

factors = np.logspace(0, 7)
Lcoupling = factors * Lbeta

delta_ns = []
gammas = []

for L in Lcoupling:
    total_strength = 2 * np.pi / L

    # in theory, birefringence and core ellipticity act
    # together with the same strength, hence the total
    # coupling matrix is the sum of the two, with the
    # same coupling strength

    bire_strength = total_strength * args.a
    ellip_strength = total_strength * (1 - args.a)

    # get the equivalent physical parameters
    delta_n = fiber.birefringence(bire_strength)
    gamma = fiber.core_ellipticity(ellip_strength)

    gammas.append(gamma)
    delta_ns.append(delta_n)

    # double check

    K = bire_strength * Kb / kappa_b + ellip_strength * Ke / kappa_e

    total_strength_2 = get_coupling_strength(K)

    print(2 * np.pi / total_strength, 2 * np.pi / total_strength_2)

    # assert_error(2 * np.pi / bire_strength, 2*np.pi/bire_strength_2, str=f"Birefringence {L}")
    # assert_error(2 * np.pi / ellip_strength, 2*np.pi/ellip_strength_2, str=f"Core ellip. {L}")
    # assert_error(2 * np.pi / total_strength, 2 * np.pi/total_strength_2, str=f"Total {L}")

fig, ax = plt.subplots()


def convert(x):
    return Lbeta * x


def invert(x):
    return x / Lbeta


bir_perc = args.a * 100
ell_perc = (1 - args.a) * 100
sec_ax = ax.secondary_xaxis("top", functions=(convert, invert))
sec_ax.set_xlabel(r"$L_{\kappa}$ [m]", labelpad=10)
ax.loglog(factors, gammas, label=r"$\gamma$")
ax.loglog(factors, delta_ns, label=r"$\delta n$")
ax.set_xlabel(r"$L_{\kappa} / L_{\beta}$")
ax.legend()
# plt.suptitle(f"Biref.: {bir_perc}%   Ellip: {ell_perc}%")
plt.tight_layout()
plt.show()


# %%

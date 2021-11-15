import sys

sys.path.append("/home/gianluca/sdm-propane")

from scipy.constants import epsilon_0 as e0
from perturbation_angles import generate_perturbation_angles
import raman_linear_coupling
import raman_linear_coupling_optim
from fiber import StepIndexFiber
from scipy.optimize import nonlin
from scipy.constants import Planck as hp, lambda2nu
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import os

# %%
import matplotlib


np.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("-S", "--signal-power", default=1e-6, type=float)
parser.add_argument("-P", "--power", default=400.0, type=float)
parser.add_argument("-L", "--fiber-length", default=60, type=float)
parser.add_argument("-d", "--dz", default=5, type=float)
parser.add_argument("-U", "--undepleted-pump", action="store_true")
parser.add_argument("-c", "--coupling", action="store_true")
parser.add_argument("-N", "--nonlinear-propagation", action="store_true")
parser.add_argument("-R", "--raman", action="store_true")
parser.add_argument("-K", "--kerr", action="store_true")
parser.add_argument("-Lk", default=1, type=float)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-Lc", "--correlation-length", default=10, type=float)
parser.add_argument("--tex", action="store_true")


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
fiber_path = "/home/gianluca/sdm-propane/fibers"
fiber = StepIndexFiber(
    clad_index=1.46, delta=0.005, core_radius=6, clad_radius=60, data_path=fiber_path
)
signal_wavelength = 1550
pump_wavelength = 1459.45

signal_freq = lambda2nu(signal_wavelength * 1e-9)
pump_freq = lambda2nu(pump_wavelength * 1e-9)

n2 = 4e-19
gR = 1e-13

fiber.load_data(wavelength=signal_wavelength)
fiber.load_data(wavelength=pump_wavelength)
fiber.load_nonlinear_coefficients(signal_wavelength, pump_wavelength)


nonlinear_params = fiber.get_raman_coefficients(
    n2, gR, signal_wavelength, pump_wavelength, as_dict=True
)


fiber_length = args.fiber_length * 1e3
correlation_length = args.correlation_length
dz = args.dz

indices_s = fiber.group_azimuthal_orders(wavelength=signal_wavelength)
indices_p = fiber.group_azimuthal_orders(wavelength=pump_wavelength)
num_modes_s = fiber.num_modes(signal_wavelength)
num_modes_p = fiber.num_modes(pump_wavelength)

Pp0 = args.power * 1e-3
Ps0 = args.signal_power
Ap0 = np.zeros((num_modes_p,)).astype("complex128")
As0 = np.zeros((num_modes_s,)).astype("complex128")
Ap0[0] = np.sqrt(Pp0)
As0[0] = np.sqrt(Ps0)

pump_attenuation = 0.2 * 1e-3 * np.log(10) / 10
signal_attenuation = 0.2 * 1e-3 * np.log(10) / 10
alpha_s = signal_attenuation
alpha_p = pump_attenuation

beta_s = fiber.propagation_constants(wavelength=signal_wavelength, remove_mean=True)
beta_p = fiber.propagation_constants(wavelength=pump_wavelength, remove_mean=True)

nonlinear_params["aW"] = np.conj(nonlinear_params["aW"])
nonlinear_params["bW"] = np.conj(nonlinear_params["bW"])


thetas = generate_perturbation_angles(correlation_length, dz, fiber_length)


propagation_function = raman_linear_coupling.propagate
propagation_function = raman_linear_coupling_optim.propagate


if not args.kerr:
    nonlinear_params["sigma"] *= 0
    nonlinear_params["a0"] *= 0
    nonlinear_params["b0"] *= 0

if not args.raman:
    nonlinear_params["aW"] *= 0
    nonlinear_params["bW"] *= 0

# nonlinear_params["aW"] = np.imag(nonlinear_params["aW"])
# nonlinear_params["bW"] = np.imag(nonlinear_params["bW"])

Q1 = nonlinear_params["Q3_s"][0, 0, 0, 0]
aW = nonlinear_params["aW"]
bW = nonlinear_params["bW"]

# Q1 = Q1 * np.ones_like(nonlinear_params["Q3_s"])

# nonlinear_params["Q1_p"] *= 0
# nonlinear_params["Q2_p"] *= 0
# nonlinear_params["Q3_p"] = Q1
# nonlinear_params["Q4_p"] = Q1
# nonlinear_params["Q5_p"] = Q1

# nonlinear_params["Q1_s"] *= 0
# nonlinear_params["Q2_s"] *= 0
# nonlinear_params["Q3_s"] = Q1
# nonlinear_params["Q4_s"] = Q1
# nonlinear_params["Q5_s"] = Q1

nonlinear_params = nonlinear_params if (args.kerr or args.raman) else None

start = time.perf_counter()
z, Ap, As = propagation_function(
    As0,
    Ap0,
    fiber_length,
    dz,
    indices_s,
    indices_p,
    alpha_s,
    alpha_p,
    beta_s,
    beta_p,
    thetas,
    nonlinear_params=nonlinear_params,
    undepleted_pump=args.undepleted_pump,
    signal_coupling=False,
    pump_coupling=False,
    filtering_percent=0.00,
)
end = time.perf_counter()
print("Time: ", (end - start))


signal_power_s = np.abs(As[:, 0]) ** 2
signal_power_p = np.abs(Ap[:, 0]) ** 2


Leff = (1 - np.exp(-alpha_p * z)) / alpha_p
G = -2 * np.pi * signal_freq / 4 * e0 * Q1 * np.imag(aW + bW) * Pp0
Ps_t = Ps0 * np.exp(-alpha_s * z) * np.exp(Leff * G)
Pp_t = Pp0 * np.exp(-alpha_p * z)

# %%

plt.figure()
plt.plot(z * 1e-3, signal_power_s * 1e3, label="Signal, numerical")
plt.plot(
    z * 1e-3,
    Ps_t * 1e3,
    marker=".",
    markevery=500,
    linestyle="none",
    label="Signal, analytical",
)
plt.plot(z * 1e-3, signal_power_p * 1e3, label="Pump, numerical")
plt.plot(
    z * 1e-3,
    Pp_t * 1e3,
    marker=".",
    markevery=500,
    linestyle="none",
    label="Pump, analytical",
)
plt.grid(False, which="minor")
plt.grid(False, which="minor")
plt.xlabel("Position [km]")
plt.ylabel("Power [mW]")
plt.legend()
plt.tight_layout()

plt.show()
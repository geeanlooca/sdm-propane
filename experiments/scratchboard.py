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


args = parser.parse_args()


fiber_path = "/home/gianluca/sdm-propane/fibers"
fiber = StepIndexFiber(
    clad_index=1.4545, delta=0.005, core_radius=6, clad_radius=60, data_path=fiber_path
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

Lbeta = fiber.modal_beat_length(wavelength=signal_wavelength)
Lpert = args.Lk
delta_n = fiber.birefringence(2 * np.pi / Lpert, wavelength=signal_wavelength)

K_signal = fiber.birefringence_coupling_matrix(
    delta_n=delta_n, wavelength=signal_wavelength
)
K_pump = fiber.birefringence_coupling_matrix(
    delta_n=delta_n, wavelength=pump_wavelength
)

total_strength = 2 * np.pi / Lpert

# in theory, birefringence and core ellipticity act
# together with the same strength, hence the total
# coupling matrix is the sum of the two, with the
# same coupling strength

bire_strength = total_strength / 2
ellip_strength = total_strength / 2

# get the equivalent physical parameters
delta_n = fiber.birefringence(bire_strength, wavelength=signal_wavelength)
gamma = fiber.core_ellipticity(ellip_strength, wavelength=signal_wavelength)

Ke_s = fiber.core_ellipticity_coupling_matrix(gamma=gamma, wavelength=signal_wavelength)
Kb_s = fiber.birefringence_coupling_matrix(
    delta_n=delta_n, wavelength=signal_wavelength
)
Ke_signal = Ke_s
Kb_signal = Kb_s
Ktot_signal = Kb_s + Ke_s

kappa_signal = np.linalg.eigvals(Ktot_signal)
Lk_signal = 2 * np.pi / (kappa_signal.max() - kappa_signal.min())
print(Lk_signal)

Ke_p = fiber.core_ellipticity_coupling_matrix(gamma=gamma, wavelength=pump_wavelength)
Kb_p = fiber.birefringence_coupling_matrix(delta_n=delta_n, wavelength=pump_wavelength)
Ktot_pump = Kb_p + Ke_p


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


Pp0 = args.power * 1e-3 / (num_modes_p / 2)
Ps0 = args.signal_power
Ap0 = np.zeros((num_modes_p,)).astype("complex128")
As0 = np.zeros((num_modes_s,)).astype("complex128")
Ap0[0::2] = np.sqrt(Pp0)
As0[0::2] = np.sqrt(Ps0)

pump_attenuation = 0.2 * 1e-3 * np.log(10) / 10
signal_attenuation = 0.2 * 1e-3 * np.log(10) / 10
alpha_s = signal_attenuation
alpha_p = pump_attenuation
Kb_s = Ktot_signal
Kb_p = Ktot_pump

beta_s = fiber.propagation_constants(wavelength=signal_wavelength, remove_mean=True)
beta_p = fiber.propagation_constants(wavelength=pump_wavelength, remove_mean=True)

nonlinear_params["aW"] = np.conj(nonlinear_params["aW"])
nonlinear_params["bW"] = np.conj(nonlinear_params["bW"])

print(beta_s)
print(beta_p)

if args.verbose:
    print("sigma", nonlinear_params["sigma"])
    print("a0", nonlinear_params["a0"])
    print("b0", nonlinear_params["b0"])
    print("aW", nonlinear_params["aW"])
    print("bW", nonlinear_params["bW"])


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

nonlinear_params["Q1_p"] *= 1
nonlinear_params["Q2_p"] *= 1
nonlinear_params["Q3_p"] *= 1
nonlinear_params["Q4_p"] *= 1
nonlinear_params["Q5_p"] *= 1

nonlinear_params["Q1_s"] *= 1
nonlinear_params["Q2_s"] *= 1
nonlinear_params["Q3_s"] *= 1
nonlinear_params["Q4_s"] *= 1
nonlinear_params["Q5_s"] *= 1

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
    K_s=Kb_s,
    K_p=Kb_p,
    nonlinear_params=nonlinear_params,
    undepleted_pump=args.undepleted_pump,
    signal_coupling=args.coupling,
    pump_coupling=args.coupling,
    filtering_percent=0.01,
)
end = time.perf_counter()
print("Time: ", (end - start))


signal_power_s = np.abs(As[:, 0::2]) ** 2 + np.abs(As[:, 1::2]) ** 2
signal_power_p = np.abs(Ap[:, 0::2]) ** 2 + np.abs(Ap[:, 1::2]) ** 2

attenuation_signal = np.abs(As0[0]) ** 2 * np.exp(-alpha_s * z)


# %%

plt.figure()
plt.subplot(121)
plt.plot(z * 1e-3, 10 * np.log10(signal_power_s * 1e3))
plt.grid(False, which="minor")
plt.grid(False, which="minor")
plt.legend(loc="best")
plt.xlabel("Position [km]")
plt.ylabel("Power [dBm]")
plt.title("Signal power")


plt.subplot(122)
plt.plot(z * 1e-3, 30 + 10 * np.log10((signal_power_p)))
plt.legend()
plt.xlabel("Position [km]")
plt.ylabel("Power [dBm]")
plt.title("Pump power")


total_signal_power = signal_power_s.sum(axis=1)
total_pump_power = signal_power_p.sum(axis=1)

Pp0_total = total_pump_power[0]
Ps0_total = total_signal_power[0]

pump_attenuation = np.exp(z * pump_attenuation)
signal_attenuation = np.exp(z * signal_attenuation)

total_energy = total_signal_power * signal_attenuation / (
    hp * signal_freq
) + total_pump_power * pump_attenuation / (hp * pump_freq)
initial_energy = Pp0_total / (hp * pump_freq) + Ps0_total / (hp * signal_freq)
normalized_energy = total_energy / initial_energy

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(z * 1e-3, 30 + 10 * np.log10(total_signal_power), label="Total power, signal")
plt.plot(
    z * 1e-3,
    30 + 10 * np.log10(total_pump_power),
    label="Total power, pump",
    linestyle="--",
)
# plt.grid()
plt.xlabel("Position [km]")
plt.ylabel("Power [dBm]")
plt.ylim(-50, 50)
plt.legend()
plt.title("Total power")

plt.subplot(122)
plt.semilogy(z * 1e-3, normalized_energy)
# plt.grid()
plt.xlabel("Position [km]")
plt.ylabel("Energy")
plt.title("Total energy variation")


plt.show()
# %%

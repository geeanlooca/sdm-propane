#%%
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from scipy.constants import Planck as hp, lambda2nu
from scipy.optimize import nonlin

from fiber import StepIndexFiber
import raman_linear_coupling


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sigma", action="store_false")
parser.add_argument("-P", "--power", default=400.0, type=float)
parser.add_argument("-L", "--fiber-length", default=60, type=float)
parser.add_argument("-d", "--dz", default=5, type=float)
parser.add_argument("-U", "--undepleted-pump", action="store_true")
parser.add_argument("-sc", "--signal-coupling", action="store_true")
parser.add_argument("-pc", "--pump-coupling", action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")


args = parser.parse_args(args=[])


fiber = StepIndexFiber(clad_index=1.46, delta=0.005, core_radius=6, clad_radius=60, data_path="fibers")
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
Lpert = 1e5 * Lbeta
delta_n = fiber.birefringence(2 * np.pi / Lpert, wavelength=signal_wavelength)

# fiber.core_ellipticity()
K_signal = fiber.birefringence_coupling_matrix(delta_n=delta_n, wavelength=signal_wavelength)
K_pump = fiber.birefringence_coupling_matrix(delta_n=delta_n, wavelength=pump_wavelength)
# K_signal = fiber.core_ellipticity_coupling_matrix(delta_n=delta_n, wavelength=signal_wavelength)
# K_pump = fiber.core_ellipticity_coupling_matrix(delta_n=delta_n, wavelength=pump_wavelength)

nonlinear_params = fiber.get_raman_coefficients(n2, gR, signal_wavelength, pump_wavelength, as_dict=True) 


fiber_length = args.fiber_length * 1e3
correlation_length = 5
dz = args.dz

indices_s = fiber.group_azimuthal_orders(wavelength=signal_wavelength)
indices_p = fiber.group_azimuthal_orders(wavelength=pump_wavelength)
num_modes_s = fiber.num_modes(signal_wavelength)
num_modes_p = fiber.num_modes(pump_wavelength)

Pp0 = args.power * 1e-3
Ps0 = 3e-3
Ap0 = np.zeros((num_modes_p,)).astype("complex128")
As0 = np.zeros((num_modes_s,)).astype("complex128")
Ap0[0] = np.sqrt(Pp0)
As0[0] = np.sqrt(Ps0)

pump_attenuation = 0.2* 1e-3 * np.log(10) / 10
signal_attenuation = 0.2 * 1e-3 * np.log(10) / 10
alpha_s = signal_attenuation
alpha_p = pump_attenuation
Kb_s = K_signal
Kb_p = K_pump

beta_s = fiber.propagation_constants(wavelength=signal_wavelength, remove_mean=True)
beta_p = fiber.propagation_constants(wavelength=pump_wavelength, remove_mean=True)

if not args.sigma:
    nonlinear_params['sigma'] = 0
# nonlinear_params['a0'] = 0
# nonlinear_params['b0'] = 0
nonlinear_params['aW'] = np.conj(nonlinear_params['aW'])
nonlinear_params['bW'] = np.conj(nonlinear_params['bW'])

if args.verbose:
    print("sigma", nonlinear_params['sigma'])
    print("a0", nonlinear_params['a0'])
    print("b0", nonlinear_params['b0'])
    print("aW", nonlinear_params['aW'])
    print("bW", nonlinear_params['bW'])


nonlinear_params["Q1_s"] *= 0
nonlinear_params["Q2_s"] *= 0
nonlinear_params["Q1_p"] *= 0
nonlinear_params["Q2_p"] *= 0

# Q3 = nonlinear_params["Q3_s"]
# Q30 = Q3[0,0,0,0]
# Q3 = np.zeros_like(Q3)
# Q3[0,0,0,0] = Q30
# nonlinear_params["Q3_s"] = Q3
# nonlinear_params["Q4_s"] = Q3
# nonlinear_params["Q5_s"] = Q3

# Q3p = nonlinear_params["Q3_p"]
# Q30 = Q3p[0,0,0,0]
# Q3p = np.zeros_like(Q3p)
# Q3p[0,0,0,0] = Q30
# nonlinear_params["Q3_p"] = Q3p
# nonlinear_params["Q4_p"] = Q3p
# nonlinear_params["Q5_p"] = Q3p


propagation_function = raman_linear_coupling.propagate
# import python_propagator
# propagation_function = python_propagator.propagate
start = time.perf_counter()
z, theta, Ap, As =propagation_function(
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
    K_s=Kb_s,
    K_p=Kb_p,
    seed=0,
    nonlinear_params=nonlinear_params,
    undepleted_pump=args.undepleted_pump,
    signal_coupling=args.signal_coupling,
    pump_coupling=args.pump_coupling,
    signal_spm=True,
    pump_spm=True,
)
end = time.perf_counter()
print("Time: ", (end - start))


target_points = 1000
# print("Decimation factor:", decimation_factor, "Sample count:", len(z))

# z = z[::10]
# As = As[::10]
# Ap = Ap[::10]


signal_power_s = np.abs(As) ** 2
signal_power_p = np.abs(Ap) ** 2

attenuation_signal = np.abs(As0[0]) ** 2 * np.exp(-alpha_s * z)

markevery = len(z) // 50

from scipy.constants import epsilon_0 as e0
Q3 = nonlinear_params['Q3_s'][0,0,0,0]
aW = nonlinear_params['aW']
bW = nonlinear_params['bW']

Leff = (1 - np.exp(-alpha_p*z))/alpha_p
omega_s = 2 * np.pi * signal_freq
gain_efficienty = - e0 * omega_s / 4 * Q3 * np.imag(aW + bW)
print("Gain efficiency [1/W/km]", gain_efficienty * 1000)
G = gain_efficienty * Pp0 * Leff
P_th = Ps0 * np.exp(-alpha_s * z) * np.exp(G)
Pp_th = Pp0 * np.exp(-alpha_p * z)


#%%
plt.style.use(['science', 'ieee', 'bright'])


plt.figure(figsize=(4, 2.5))
# plt.subplot(121)
plt.plot(z*1e-3, (signal_power_s[:, 0]) * 1e3, label="Signal, simulation")
plt.plot(z*1e-3, P_th * 1e3, marker='.', markevery=markevery, markersize=3, linestyle='none', label="Signal, analytical solution")
plt.plot(z * 1e-3, signal_power_p[:, 0] * 1e3, label="Pump, simulation")
plt.plot(z*1e-3, Pp_th * 1e3, marker='.', markevery=markevery, markersize=3, linestyle='none', label="Pump, analytical solution")
# plt.plot(z*1e-3, attenuation_signal * 1e3, linestyle='dotted', color='black')
plt.grid(False, which='minor')
plt.grid(False, which='minor')
plt.legend(loc="best")
plt.xlabel("Position [km]")
plt.ylabel("Power [mW]")

plt.savefig("/media/sf_onedrive/signal_power_undepleted_smf.eps")
# plt.subplot(122)
# plt.plot(z*1e-3, 30 + 10 * np.log10((signal_power_s)))
# plt.plot(z*1e-3, 30 + 10 * np.log10(P_th), marker='.', markevery=markevery, linestyle='none')
# # plt.plot(z*1e-3, 30 + 10 * np.log10(attenuation_signal), linestyle='dotted', color='black')
# plt.legend(fiber.mode_names(signal_wavelength), loc="best")
# plt.xlabel("Position [km]")
# plt.ylabel("Power [dBm]")
# # plt.grid()

plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(z * 1e-3, signal_power_p * 1e3)
plt.plot(z*1e-3, Pp_th * 1e3, marker='.', markevery=markevery, linestyle='none')
plt.legend(fiber.mode_names(pump_wavelength), loc="best")
plt.xlabel("Position [km]")
plt.ylabel("Power [mW]")
# plt.grid()

plt.subplot(122)
plt.plot(z*1e-3, 30 + 10 * np.log10((signal_power_p)))
plt.plot(z*1e-3, 30 + 10 * np.log10((Pp_th)), marker='.', markevery=markevery, linestyle='none', label="Undepleted pump solution")
plt.legend(fiber.mode_names(pump_wavelength), loc="best")
plt.xlabel("Position [km]")
plt.ylabel("Power [dBm]")
# plt.grid()

plt.suptitle("Pump power")
plt.tight_layout()


total_signal_power =  signal_power_s.sum(axis=1)
total_pump_power =  signal_power_p.sum(axis=1)

Pp0_total = total_pump_power[0]
Ps0_total = total_signal_power[0]

pump_attenuation = np.exp(z * pump_attenuation)
signal_attenuation = np.exp(z * signal_attenuation)

total_energy = total_signal_power * signal_attenuation / (hp * signal_freq) + total_pump_power * pump_attenuation / (hp * pump_freq)
initial_energy = Pp0_total / (hp * pump_freq) + Ps0_total / (hp * signal_freq)
normalized_energy = total_energy / initial_energy

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(z * 1e-3, 30 + 10 * np.log10(total_signal_power),
         label="Total power, signal")
plt.plot(z * 1e-3, 30 + 10 * np.log10(total_pump_power),
         label="Total power, pump", linestyle="--")
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
plt.tight_layout()

plt.show()

# %%

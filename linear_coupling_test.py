import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from fiber import StepIndexFiber

import raman_linear_coupling

fiber = StepIndexFiber(clad_index=1.46, delta=0.005, core_radius=6, clad_radius=60, data_path="fibers")
signal_wavelength = 1550
pump_wavelength = 1459.45

fiber.load_data(wavelength=signal_wavelength)
fiber.load_data(wavelength=pump_wavelength)

fiber_length = 10e3
correlation_length = 10
dz = correlation_length / 100

indices_s = fiber.group_azimuthal_orders(wavelength=signal_wavelength).astype("int32")
indices_p = fiber.group_azimuthal_orders(wavelength=pump_wavelength).astype("int32")
num_modes_s = fiber.num_modes(wavelength=signal_wavelength)
num_modes_p = fiber.num_modes(wavelength=pump_wavelength)

LbetaLkappa = 1e-3
Lbeta = fiber.modal_beat_length(wavelength=signal_wavelength)
Lkappa = Lbeta / LbetaLkappa
perturb_strength = 2 * np.pi / Lkappa

print(f"Lbeta: {Lbeta}", f"Lkappa: {Lkappa}")

delta_n = fiber.birefringence(perturb_strength, wavelength=signal_wavelength)
Kb_s = fiber.birefringence_coupling_matrix(delta_n=delta_n, wavelength=signal_wavelength)
Kb_p = fiber.birefringence_coupling_matrix(delta_n=delta_n, wavelength=pump_wavelength)

gamma = fiber.core_ellipticity(perturb_strength, wavelength=signal_wavelength)
Ke_s = fiber.core_ellipticity_coupling_matrix(gamma=gamma, wavelength=signal_wavelength)
Ke_p = fiber.core_ellipticity_coupling_matrix(gamma=gamma, wavelength=pump_wavelength)

K_s = Ke_s + Kb_s
K_p = Ke_p + Kb_p
K_s = Ke_s
K_p = Ke_p

np.set_printoptions(precision=3)
print(Kb_s)
print(Ke_s)

Pp0 = 1e-3
Ps0 = 1e-3
Ap0 = np.zeros((num_modes_p,)).astype("complex128")
As0 = np.zeros((num_modes_s,)).astype("complex128")

As0[0] = np.sqrt(Ps0)
Ap0[0] = np.sqrt(Pp0)

pump_attenuation = 0.3 * 1e-3 * np.log(10) / 10
signal_attenuation = 0.2 * 1e-3 * np.log(10) / 10
alpha_s = signal_attenuation
alpha_p = pump_attenuation

beta_s = fiber.propagation_constants(wavelength=signal_wavelength, remove_mean=True)
beta_p = fiber.propagation_constants(wavelength=pump_wavelength, remove_mean=True)

thetas = []
start = time.perf_counter()
z, theta, Ap, As = raman_linear_coupling.propagate(
    As0.astype("complex128"),
    Ap0.astype("complex128"),
    fiber_length,
    dz,
    indices_s,
    indices_p,
    correlation_length,
    alpha_s,
    alpha_p,
    beta_s.astype("float64"),
    beta_p.astype("float64"),
    K_s=K_s.astype("float64"),
    K_p=K_p.astype("float64"),
    signal_coupling=True,
    pump_coupling=False,
    # seed=None,
    # nonlinear_opt=None,
    # undepleted_pump=None
)

end = time.perf_counter()
print("Time: ", (end - start))

signal_power_s = np.abs(As) ** 2
signal_power_p = np.abs(Ap) ** 2

mode_names = fiber.mode_names(wavelength=signal_wavelength)

plt.figure()
plt.subplot(121)
plt.plot(z, (signal_power_s) * 1e3)
plt.legend(mode_names, loc="upper right")
plt.xlabel("Position [m]")
plt.ylabel("Power [mW]")

plt.subplot(122)
plt.plot(z, 30 + 10 * np.log10((signal_power_s)))
plt.legend(mode_names, loc="upper right")
plt.xlabel("Position [m]")
plt.ylabel("Power [dBm]")

plt.suptitle("Signal power")

plt.figure()
plt.subplot(121)
plt.plot(z, (signal_power_p) * 1e3)
plt.legend(mode_names, loc="upper right")
plt.xlabel("Position [m]")
plt.ylabel("Power [mW]")

plt.subplot(122)
plt.plot(z, 30 + 10 * np.log10((signal_power_p)))
plt.legend(mode_names, loc="upper right")
plt.xlabel("Position [m]")
plt.ylabel("Power [dBm]")

plt.suptitle("Signal power")

total_signal_power = np.sum(np.abs(As0) ** 2)
total_pump_power = np.sum(np.abs(Ap0) ** 2)

plt.figure()
plt.plot(z, 30 + 10 * np.log10(signal_power_s.sum(axis=1)),
         label="Total power, pump")
plt.plot(z, 30 + 10 * np.log10(signal_power_p.sum(axis=1)),
         label="Total power, signal")

plt.xlabel("Position [m]")
plt.ylabel("Power [dBm]")
plt.legend()

plt.figure()
plt.plot(z, theta / np.pi)
plt.xlabel("Position [m]")
plt.ylabel(r"$\theta / \pi$")

plt.show()

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

import raman_linear_coupling


linear_coupling_filename = "coupling_coefficients-2modes-clad_index=1.4600-Delta=0.005-core_radius=6.00um-clad_radius=60.00um-wavelength=1550.00nm-mesh_size=1.mat"
filepath = os.path.join("fibers", linear_coupling_filename)
data = scipy.io.loadmat(filepath, simplify_cells=True)
data["mode_names"] = data["mode_names"].tolist()

linear_coupling_filename_pump = "coupling_coefficients-2modes-clad_index=1.4600-Delta=0.005-core_radius=6.00um-clad_radius=60.00um-wavelength=1459.45nm-mesh_size=1.mat"
filepath = os.path.join("fibers", linear_coupling_filename_pump)
data_pump = scipy.io.loadmat(filepath, simplify_cells=True)
data_pump["mode_names"] = data_pump["mode_names"].tolist()


fiber_length = 10e3
correlation_length = 10
dz = correlation_length / 100
indices_s = np.array(
    [0 if d == 2 else 1 for d in data["degeneracies"]], dtype="int32")
indices_p = np.array(
    [0 if d == 2 else 1 for d in data_pump["degeneracies"]], dtype="int32")

num_modes_s = data["num_modes"]
num_modes_p = data_pump["num_modes"]

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

Lbeta = data["Lbeta"]
Lkappa = Lbeta * 1e3
deltaKappa = 2 * np.pi / Lkappa
print("Lbeta:", Lbeta, "Lkappa", Lkappa)
deltaN0 = data["deltaN0"]
deltaN = deltaN0 * deltaKappa

deltaKappa_pump = deltaN / data_pump["deltaN0"]

Kb_s = deltaKappa * data["Kb"]
Kb_p = deltaKappa_pump * data_pump["Kb"]


Kb_s = np.real(Kb_s)
Kb_p = np.real(Kb_p)
Kb_s = np.tril(Kb_s) + np.triu(Kb_s.T, 1)
Kb_p = np.tril(Kb_p) + np.triu(Kb_p.T, 1)


beta_s = data["beta"]
beta_p = data_pump["beta"]

beta_s -= beta_s.mean()
beta_p -= beta_p.mean()

beta_s *= 1
beta_p *= 1

print(beta_s, beta_p)
print(Kb_s)
print(Kb_p)


thetas = []
start = time.perf_counter()
z, theta, Ap, As = raman_linear_coupling.propagate(
    As0,
    Ap0,
    fiber_length,
    dz,
    indices_s,
    indices_p,
    correlation_length,
    Kb_s,
    Kb_p,
    alpha_s,
    alpha_p,
    beta_s,
    beta_p,
)
end = time.perf_counter()
print("Time: ", (end - start))

signal_power_s = np.abs(As) ** 2
signal_power_p = np.abs(Ap) ** 2


plt.figure()
plt.plot(z, (signal_power_s) * 1e3)
plt.legend(data["mode_names"], loc="upper right")
plt.xlabel("Position [m]")
plt.ylabel("Power [mW]")

plt.figure()
plt.plot(z, 30 + 10 * np.log10((signal_power_s)))
plt.legend(data["mode_names"], loc="upper right")
plt.xlabel("Position [m]")
plt.ylabel("Power [dBm]")

total_signal_power = np.sum(np.abs(As0) ** 2)
total_pump_power = np.sum(np.abs(Ap0) ** 2)

plt.figure()
plt.plot(z, 30 + 10 * np.log10(signal_power_s.sum(axis=1)),
         label="Total power, pump")
plt.plot(z, 30 + 10 * np.log10(signal_power_p.sum(axis=1)),
         label="Total power, signal")
# plt.ylim(0, 10*np.log10(1.5 * max(total_signal_power, total_pump_power)))
plt.xlabel("Position [m]")
plt.ylabel("Power [dBm]")
plt.legend()

plt.figure()
plt.plot(z, theta / np.pi)
plt.xlabel("Position [m]")
plt.ylabel(r"$\theta / \pi$")

plt.show()

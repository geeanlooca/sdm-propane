import sys

sys.path.append("/home/gianluca/sdm-propane")

from scipy.constants import epsilon_0 as e0
from fiber import StepIndexFiber
from scipy.constants import Planck as hp, lambda2nu
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1)

fiber_path = "/home/gianluca/sdm-propane/fibers"
fiber = StepIndexFiber(
    clad_index=1.4545, delta=0.005, core_radius=6, clad_radius=60, data_path=fiber_path
)
signal_wavelength = 1550

signal_freq = lambda2nu(signal_wavelength * 1e-9)

fiber.load_data(wavelength=signal_wavelength)


Ke = np.real(fiber.core_ellipticity_coupling_matrix())

Ke = Ke / np.max(np.abs(Ke))
Kb = np.real(fiber.birefringence_coupling_matrix())
Kb = Kb / np.max(np.abs(Kb))


# print(fiber.mode_names())
# print(fiber.spatial_mode_names())
# print(fiber.group_names())
print(fiber.group_orders())
print(fiber.group_degeneracies())

plt.set_cmap("inferno")

plt.imshow(np.log10(np.abs(Ke)))
plt.grid(False)
plt.colorbar()

plt.show()
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

print(fiber.mode_names())
print(fiber.spatial_mode_names())
print(fiber.group_names())
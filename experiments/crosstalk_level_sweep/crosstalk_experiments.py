import sys
import math
import os

# add path containing the code and data files
current_path = os.path.dirname(os.path.abspath(__file__))
experiments_path = os.path.dirname(current_path)
root_path = os.path.dirname(experiments_path)
experiments_path = sys.path.append(root_path)

import numpy as np
import scipy.linalg
from scipy.constants import lambda2nu

from fiber import StepIndexFiber
from fibers.available_fibers import SIF2Modes, SIF4Modes
from polarization import hyperstokes_to_jones
import raman_linear_coupling
import raman_linear_coupling_optim

from experiment import Experiment

from typing import Tuple, List


class CrossTalkLevelExperiment(Experiment):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args

        if args.numpy_seed:
            np.random.seed(args.numpy_seed)

        self.fiber = SIF4Modes()

        self.signal_wavelength = 1550

        self.signal_freq = lambda2nu(self.signal_wavelength * 1e-9)

        self.fiber.load_data(wavelength=self.signal_wavelength)

        self.Lbeta = self.fiber.modal_beat_length(wavelength=self.signal_wavelength)
        self.Lpert = args.perturbation_beat_length

        self.Ktot_signal = self.get_coupling_matrix(self.Lpert, self.signal_wavelength)

        self.fiber_length = args.fiber_length * 1e3
        self.correlation_length = args.correlation_length
        self.dz = args.dz

        self.indices_s = self.fiber.group_azimuthal_orders(
            wavelength=self.signal_wavelength
        )
        self.num_modes_s = self.fiber.num_modes(self.signal_wavelength)

        self.Ps0 = args.signal_power_per_mode * 1e-3

        signal_attenuation = args.alpha * 1e-3 * np.log(10) / 10
        self.alpha_s = signal_attenuation

        self.beta_s = self.fiber.propagation_constants(
            wavelength=self.signal_wavelength, remove_mean=True
        )

    def get_coupling_strength(self, K):
        eigvals = scipy.linalg.eigvals(K)
        strength = eigvals.max() - eigvals.min()
        return strength

    def get_coupling_matrix(self, perturbation_beat_length, wavelength):

        total_strength = 2 * np.pi / perturbation_beat_length
        Ke = self.fiber.core_ellipticity_coupling_matrix(wavelength=wavelength)
        Kb = self.fiber.birefringence_coupling_matrix(wavelength=wavelength)

        kappa_e = self.get_coupling_strength(Ke)
        kappa_b = self.get_coupling_strength(Kb)

        bire_strength = total_strength * 0.5
        ellip_strength = total_strength * 0.5

        K_total = bire_strength * Kb / kappa_b + ellip_strength * Ke / kappa_e

        return K_total

    def propagate(self, As0, thetas):

        z, Ap, As = raman_linear_coupling_optim.propagate(
            As0,
            As0,
            self.fiber_length,
            self.dz,
            self.indices_s,
            self.indices_s,
            self.alpha_s,
            self.alpha_s,
            self.beta_s,
            self.beta_s,
            thetas,
            K_s=self.Ktot_signal,
            signal_coupling=True,
        )

        return z, As

    def metadata(self):

        metadata = {
            "signal_wavelength": self.signal_wavelength,
        }

        metadata = {**metadata, **vars(self.args)}
        return metadata

    @staticmethod
    def build_mode_vector(idx, azimuthal_orders):
        """Build a mode vector"""
        num_groups = len(azimuthal_orders)
        modes_per_group = np.array([2 if x > 0 else 1 for x in azimuthal_orders])
        pols_per_group = np.array([2 * x for x in modes_per_group])

        num_modes = 0
        for i in azimuthal_orders:
            num_modes += 2 if i > 0 else 1

        num_pols = num_modes * 2

        v = np.zeros((num_pols, 0))

        offsets = []

        start = 0
        offsets.append(start)

        for i in range(len(azimuthal_orders) - 1):
            ind = azimuthal_orders[i]
            start += pols_per_group[i]
            offsets.append(start)

        offset = offsets[idx]
        order = azimuthal_orders[idx]
        v = np.zeros((num_pols,))
        v[offset] = 1
        if order > 0:
            v[offset + 2] = 1

        return v

    @staticmethod
    def get_polarization_indeces(group_idx, azimuthal_orders):
        modes_per_group = np.array([2 if x > 0 else 1 for x in azimuthal_orders])
        pols_per_group = np.array([2 * x for x in modes_per_group])
        offsets = []

        num_modes = 0
        for i in azimuthal_orders:
            num_modes += 2 if i > 0 else 1

        num_pols = num_modes * 2
        start = 0
        offsets.append(start)

        for i in range(len(azimuthal_orders) - 1):
            ind = azimuthal_orders[i]
            start += pols_per_group[i]
            offsets.append(start)

        offset = offsets[group_idx]
        order = azimuthal_orders[group_idx]
        v = np.zeros((num_pols,))

        indices = [offset + i for i in range(pols_per_group[group_idx])]

        return indices

    def group_power(
        complex_amplitudes: np.ndarray, group_idx: int, azimuthal_orders: List
    ) -> np.ndarray:
        """Return the power in a given mode group"""

        indices = CrossTalkLevelExperiment.get_polarization_indeces(
            group_idx, azimuthal_orders
        )

        P_group = np.zeros((complex_amplitudes.shape[0],))
        for i in indices:
            P_group = P_group + np.abs(complex_amplitudes[:, i]) ** 2

        return P_group

    def run(self, thetas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        signal_jones_vector: np.ndarray
            The hyperjones vector for the signal
        thetas: np.ndarray
            Array of perturbation angles
        """

        signal_power_per_spatial_mode = self.Ps0
        num_groups = len(self.indices_s)
        num_steps = int(self.fiber_length / self.dz) + 1

        # create matrix holding the results, first dimension tells the
        # mode in which power is injected, second dimension is fiber position,
        # third dimension is the power in each group

        P_groups = []

        for group_idx in range(num_groups):
            input_signal = CrossTalkLevelExperiment.build_mode_vector(
                group_idx, self.indices_s
            )
            input_signal *= np.sqrt(signal_power_per_spatial_mode)

            z, As = self.propagate(input_signal, thetas)

            power = np.zeros((z.shape[0], num_groups))

            for i in range(num_groups):
                power[:, i] = CrossTalkLevelExperiment.group_power(
                    As, i, self.indices_s
                )

            power = power / np.reshape(power.sum(axis=-1), (-1, 1))

            P_groups.append(power)

        # downsample data
        P_groups = np.stack(P_groups)

        return z, P_groups

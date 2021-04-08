from abc import ABC
import os
import glob

import numpy as np
import scipy.io


class Fiber(ABC):
    def __init__(self):
        pass


class StepIndexFiber(Fiber):
    def __init__(self, clad_index, delta, core_radius, clad_radius, data_path=None):

        if data_path is None:
            self.path = "."
        else:
            self.path = data_path

        self.clad_index = clad_index 
        self.delta = delta 
        self.core_radius = core_radius
        self.clad_radius = clad_radius
        self.core_index = self.clad_index / np.sqrt(1 - 2 * self.delta)
        self.wavelength = [  ]
        self.Ke = {} 
        self.Kb = {} 
        self.gamma0 = {}
        self.delta_n0 = {}
        self._mode_names = {}
        self._num_groups = {}
        self._num_modes = {}
        self._degeneracies = {}
        self.betas = {}
        

    def load_data(self, wavelength, mesh_size=1, data_path=None, modes=None):
        if data_path is None:
            data_path = self.path

        _modes = modes if modes else "*"

        filename = StepIndexFiber.get_filename(modes=_modes, clad_index=self.clad_index,
                         delta=self.delta, core_radius=self.core_radius,
                         clad_radius=self.clad_radius,
                         wavelength=wavelength, mesh_size=mesh_size)

        filepath = os.path.join(data_path, filename)

        if not modes:
            filepath, *_ = glob.glob(filepath)

        data = scipy.io.loadmat(filepath, simplify_cells=True)
        data["mode_names"] = data["mode_names"].tolist()
        wavelength = round(data["wavelength"], 2)
        self.wavelength.append(wavelength)
        self._mode_names[wavelength] = data["mode_names"]
        self._degeneracies[wavelength] = data["degeneracies"]
        self.Ke[wavelength] = 0.5 * (data["Ke"] + data["Ke"].T)
        self.Kb[wavelength] = 0.5 * (data["Kb"] + data["Ke"].T)
        self.gamma0[wavelength] =  data["gamma0"] 
        self.delta_n0[wavelength] = data["deltaN0"] 
        self._num_groups[wavelength] = data["num_groups"]
        self.betas[wavelength] = np.array(data["beta"])
        self._num_modes[wavelength] = data["num_modes"]



    @property
    def attenuation_coefficient(self):
        return self._alpha

    @attenuation_coefficient.setter
    def attenuation_coefficient(self, value):
        self._alpha =  value

    @property
    def length(self):
        return self._length 

    @length.setter
    def length(self, value):
        self._length = value

    def num_modes(self, wavelength=None):
        return self.get_param("_num_modes", wavelength=wavelength)

    def num_groups(self, wavelength=None):
        return self.get_param("_num_groups", wavelength=wavelength)

    def mode_names(self, wavelength=None):
        return self.get_param("_mode_names", wavelength=wavelength)
    
    def get_param(self, param, wavelength=None):
        if wavelength:
            val = getattr(self, param)[wavelength]
        else:
            _, val = next(iter(getattr(self, param).items()))
        return val

    def degeneracies(self, wavelength=None):
        return self.get_param("_degeneracies", wavelength=wavelength)

    def propagation_constants(self, wavelength=None):
        return self.get_param("betas", wavelength=wavelength)

    def birefringence(self, strength, wavelength=None):
        delta_n0 = self.get_param("delta_n0", wavelength=wavelength)
        return strength * delta_n0

    def birefringence_strength(self, delta_n, wavelength=None):
        delta_n0 = self.get_param("delta_n0", wavelength=wavelength)
        return delta_n / delta_n0

    def core_ellipticity(self, strength, wavelength=None):
        gamma0 = self.get_param("gamma0", wavelength=wavelength)
        return strength * gamma0

    def core_ellipticity_strength(self, gamma, wavelength=None):
        gamma0 = self.get_param("gamma0", wavelength=wavelength)
        return gamma / gamma0

    def modal_beat_length(self, wavelength=None):
        beta = self.propagation_constants(wavelength=wavelength)
        delta_beta = np.max(beta) - np.min(beta)
        return 2 * np.pi / delta_beta

    def birefringence_coupling_matrix(self, coupling_strength=None, delta_n=None, wavelength=None):
        K = self.get_param("Kb", wavelength=wavelength)

        if coupling_strength is None and delta_n is None:
            return K

        if coupling_strength:
            return coupling_strength * K

        if delta_n:
            strength = self.birefringence_strength(delta_n)
            return strength * K

    def core_ellipticity_coupling_matrix(self, coupling_strength=None, gamma=None, wavelength=None):
        K = self.get_param("Ke", wavelength=wavelength)

        if coupling_strength is None and gamma is None:
            return K

        if coupling_strength:
            return coupling_strength * K

        if gamma:
            strength = self.core_ellipticity_strength(gamma, wavelength=wavelength)
            return strength * K

    @property
    def numerical_aperture(self):
        return np.sqrt(self.core_index ** 2 - self.clad_index ** 2)

    @property
    def NA(self):
        return self.numerical_aperture

    def normalized_frequency(self, wavelength):
        return 2 * np.pi * self.core_radius / wavelength * self.numerical_aperture

    @staticmethod
    def get_filename(modes, clad_index, delta, core_radius, clad_radius, wavelength, mesh_size=1):
        filename = (f"coupling_coefficients-{modes}modes-clad_index={clad_index:.4f}-"
                    f"Delta={delta:.3f}-core_radius={core_radius:.2f}um-"
                    f"clad_radius={clad_radius:.2f}um-wavelength={wavelength:.2f}nm-mesh_size={mesh_size}.mat")

        return filename 


if __name__ == "__main__":
    fiber = StepIndexFiber(clad_index=1.46, delta=0.005, core_radius=6, clad_radius=60, data_path="fibers")

    signal_wavelength = 1550
    pump_wavelength = 1459.45

    fiber.load_data(wavelength=signal_wavelength)
    fiber.load_data(wavelength=pump_wavelength)

    Lbeta = fiber.modal_beat_length(wavelength=signal_wavelength)
    Lpert = 1e3 * Lbeta

    delta_n = fiber.birefringence(2 * np.pi / Lpert, wavelength=signal_wavelength)

    K_signal = fiber.birefringence_coupling_matrix(delta_n=delta_n, wavelength=signal_wavelength)
    K_pump = fiber.birefringence_coupling_matrix(delta_n=delta_n, wavelength=pump_wavelength)





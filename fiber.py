from abc import ABC
import os
import glob

import numpy as np
import scipy.io
from scipy.constants import lambda2nu, speed_of_light as c0, epsilon_0 as e0
import scipy.interpolate

import itertools


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
        self._radial_orders = {}
        self._azimuthal_orders = {}
        self._group_azimuthal_orders = {}
        self._orders = {}
        self._group_orders = {}
        self._degeneracies = {}
        self._group_degeneracies = {}
        self.betas = {}
        self.load_raman_data()
        self._Q1 = {}
        self._Q2 = {}
        self._Q3 = {}
        self._Q4 = {}
        self._Q5 = {}
        

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

        orders = self._mode_orders_from_names(data["mode_names"])

        self._radial_orders[wavelength] = [b for a, b in orders]
        self._azimuthal_orders[wavelength] = [a for a, b in orders]
        self._orders[wavelength] = orders
        self._group_orders[wavelength] = list(dict.fromkeys(orders))
        self._group_azimuthal_orders[wavelength] = [a for a, b in self._group_orders[wavelength]]

        group_degen = [ 2 if l==0 else 4 for l in self._group_azimuthal_orders[wavelength]]
        degen = [ 2 if l==0 else 4 for l in self._azimuthal_orders[wavelength]]

        self._degeneracies[wavelength] = degen
        self._group_degeneracies[wavelength] =  group_degen

        self.gamma0[wavelength] =  data["gamma0"] 
        self.delta_n0[wavelength] = data["deltaN0"] 
        self._num_groups[wavelength] = data["num_groups"]
        self.betas[wavelength] = np.array(data["beta"])
        self._num_modes[wavelength] = data["num_modes"]

        Ke = self._force_linear_coupling_simmetry(data["Ke"])
        Kb = self._force_linear_coupling_simmetry(data["Kb"])
        self.Kb[wavelength] = self._clean_birefringence_matrix(Kb)
        self.Ke[wavelength] = self._clean_ellipticity_matrix(Ke, wavelength)

    def _mode_orders_from_name(self, name):
        a, r = name[2:4]
        return int(a), int(r)

    def _mode_orders_from_names(self, names):
        orders = []

        for name in names:
            azimuthal, radial = self._mode_orders_from_name(name)
            orders.append((azimuthal, radial))

        return orders

    def _force_linear_coupling_simmetry(self, K):
        """Remove the imaginary part and make the matrix symmetric."""
        _K  = np.real(K)
        K = np.tril(K) + np.triu(K.T, 1)
        return K

    def _clean_birefringence_matrix(self, K):
        """Set the elements outside of the main diagonal to 0."""
        return np.diag(np.diag(K))

    def _clean_ellipticity_matrix(self, K, wavelength):
        """Clear the blocks outside the diagonal to remove inter-group coupling.

        Selection rules for core ellipticity are those specified in [1]

        References
        ----------
        [1] L. Palmieri, ‘Coupling mechanism in multimode fibers’, in 
            Next-Generation Optical Communication: Components, Sub-Systems,
            and Systems III, Feb. 2014, vol. 9009, p. 90090G, doi: 10.1117/12.2042763.
        """
        group_orders = self.group_orders(wavelength=wavelength)

        _K = np.copy(K)

        for order_a, order_b in itertools.combinations(group_orders, 2):
            azim_a, _ = order_a
            azim_b, _ = order_b

            if not self._coupling_relations_ellipticity(azim_a, azim_b):
                i, j = self._get_block_indeces(order_a, order_b, wavelength)
                _K[i, j] = 0
                _K[j, i] = 0

        return _K

    def _coupling_relations_ellipticity(self, n, m):
        """Implement the selection rule for intra-group coupling according to [1].

        References
        ----------
        [1] L. Palmieri, ‘Coupling mechanism in multimode fibers’,
            in Next-Generation Optical Communication: Components, 
            Sub-Systems, and Systems III, Feb. 2014, vol. 9009,
            p. 90090G, doi: 10.1117/12.2042763.
        """

        if n == m and n == 1:
            return True

        if abs(n - m) == 2:
            return True

        if n == m:
            return True

        if n + m == 4:
            return True

        if abs(n - m) == 4:
            return True

        return False
        

    def _get_block_indeces(self, group_a, group_b, wavelength):
        """Get the slices of the block of the coupling matrix for the two specified groups."""

        group_orders = self.group_orders(wavelength=wavelength)
        group_a_idx = group_orders.index(group_a)
        group_b_idx = group_orders.index(group_b)

        group_degen = self.group_degeneracies(wavelength=wavelength)
        degen_a = group_degen[group_a_idx]
        degen_b = group_degen[group_b_idx]

        start_idx_a = sum(group_degen[:(group_a_idx-1)])
        start_idx_b = sum(group_degen[:(group_b_idx-1)])
        stop_idx_a = start_idx_a + degen_b
        stop_idx_b = start_idx_b + degen_a

        return slice(start_idx_a, stop_idx_a), slice(start_idx_b, stop_idx_b)

    def load_raman_data(self, filename=None):
        if filename is None:
            filename = "raman_data.npz"
        data = np.load(filename)

        self.frequency_shift = data["frequency"]
        self.a_response = data["a"]
        self.b_response = data["b"]


    def load_nonlinear_coefficients(self, signal_wavelength, pump_wavelength, data_path=None, mesh_size=1):
        filename = StepIndexFiber.get_nonlinear_filename(clad_index=self.clad_index,
                         delta=self.delta, core_radius=self.core_radius,
                         clad_radius=self.clad_radius,
                         signal_wavelength=signal_wavelength, pump_wavelength=pump_wavelength,
                         mesh_size=mesh_size)

        if data_path is None:
            data_path = self.path

        filepath = os.path.join(data_path, filename)

        data = scipy.io.loadmat(filepath, simplify_cells=True)
        pump_wavelength = round(data["pump_wavelength"], 2)
        signal_wavelength = round(data["signal_wavelength"], 2)

        s = signal_wavelength
        p = pump_wavelength

        self._Q1[(s, p)] = data["Q1_signal"]
        self._Q2[(s, p)] = data["Q2_signal"]
        self._Q3[(s, p)] = data["Q3_signal"]
        self._Q4[(s, p)] = data["Q4_signal"]
        self._Q5[(s, p)] = data["Q5_signal"]

        self._Q1[(p, s)] = data["Q1_pump"]
        self._Q2[(p, s)] = data["Q2_pump"]
        self._Q3[(p, s)] = data["Q3_pump"]
        self._Q4[(p, s)] = data["Q4_pump"]
        self._Q5[(p, s)] = data["Q5_pump"]

    def Q1(self, wavelength_a, wavelength_b):
        a = wavelength_a
        b = wavelength_b
        return self._Q1[(a, b)]

    def Q2(self, wavelength_a, wavelength_b):
        a = wavelength_a
        b = wavelength_b
        return self._Q2[(a, b)]

    def Q3(self, wavelength_a, wavelength_b):
        a = wavelength_a
        b = wavelength_b
        return self._Q3[(a, b)]

    def Q4(self, wavelength_a, wavelength_b):
        a = wavelength_a
        b = wavelength_b
        return self._Q4[(a, b)]

    def Q5(self, wavelength_a, wavelength_b):
        a = wavelength_a
        b = wavelength_b
        return self._Q5[(a, b)]
        
    def get_raman_coefficients(self, n2, gR, signal_wavelength, pump_wavelength, as_dict=False):
        signal_freq = lambda2nu(signal_wavelength * 1e-9)
        pump_freq = lambda2nu(pump_wavelength * 1e-9)
        n = self.core_index 

        a_interp = scipy.interpolate.interp1d(self.frequency_shift, self.a_response)
        b_interp = scipy.interpolate.interp1d(self.frequency_shift, self.b_response)


        f_shift = (pump_freq - signal_freq) * 1e-12
        aW = a_interp(f_shift)
        bW = b_interp(f_shift)

        scaling = gR / (2 * np.pi * signal_freq) * (n * e0 * c0 ** 2) / np.imag(aW + bW)
        aW *= scaling
        bW *= scaling

        a0 = scaling * np.real(a_interp(0))
        b0 = scaling * np.real(b_interp(0))


        sigma = ( n2 * 4 * c0 * e0 * n**2 - 2 * ( a0 + b0) ) * 2 / 3

        if as_dict:
            d = {
                "n2": n2,
                "gR": gR,
                "sigma": sigma,
                "a0": a0,
                "b0": b0,
                "aW": aW,
                "bW": bW,
                "signal_frequency": signal_freq,
                "pump_frequency": pump_freq,
                "Q1_s": self.Q1(signal_wavelength, pump_wavelength),
                "Q2_s": self.Q2(signal_wavelength, pump_wavelength),
                "Q3_s": self.Q3(signal_wavelength, pump_wavelength),
                "Q4_s": self.Q4(signal_wavelength, pump_wavelength),
                "Q5_s": self.Q5(signal_wavelength, pump_wavelength),
                "Q1_p": self.Q1(pump_wavelength, signal_wavelength),
                "Q2_p": self.Q2(pump_wavelength, signal_wavelength),
                "Q3_p": self.Q3(pump_wavelength, signal_wavelength),
                "Q4_p": self.Q4(pump_wavelength, signal_wavelength),
                "Q5_p": self.Q5(pump_wavelength, signal_wavelength),
            }

            return d
        else:
            return sigma, a0, b0, aW, bW


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

    def group_degeneracies(self, wavelength=None):
        return self.get_param("_group_degeneracies", wavelength=wavelength)

    def group_orders(self, wavelength=None):
        return self.get_param("_group_orders", wavelength=wavelength)

    def orders(self, wavelength=None):
        return self.get_param("_orders", wavelength=wavelength)

    def radial_orders(self, wavelength=None):
        return self.get_param("_radial_orders", wavelength=wavelength)

    def azimuthal_orders(self, wavelength=None):
        return self.get_param("_azimuthal_orders", wavelength=wavelength)

    def group_azimuthal_orders(self, wavelength=None):
        return np.array(self.get_param("_group_azimuthal_orders", wavelength=wavelength)).astype("int32")

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

    def propagation_constants(self, wavelength=None, remove_mean=False):
        betas = self.get_param("betas", wavelength=wavelength)

        if remove_mean:
            betas = betas - np.mean(betas)

        return betas

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

    @staticmethod
    def get_nonlinear_filename(clad_index, delta, core_radius, clad_radius, signal_wavelength, pump_wavelength, mesh_size=1):
        filename = (f"nonlinear_coefficients-clad_index={clad_index:.4f}-"
                    f"Delta={delta:.3f}-core_radius={core_radius:.2f}um-"
                    f"clad_radius={clad_radius:.2f}um-signal_wavelength={signal_wavelength:.2f}nm-"
                    f"pump_wavelength={pump_wavelength:.2f}nm-mesh_size={mesh_size}.mat")

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

    fiber.load_raman_data()

    n2 = 2.18e-18
    gR = 1.5e-11

    sigma, a0, b0, aW, bW = fiber.get_raman_coefficients(n2, gR, signal_wavelength * 1e-9, pump_wavelength * 1e-9) 

    fiber.load_nonlinear_coefficients(signal_wavelength, pump_wavelength)

    print(fiber.group_degeneracies(signal_wavelength))







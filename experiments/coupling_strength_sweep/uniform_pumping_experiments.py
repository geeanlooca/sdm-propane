import sys
import os

# add path containing the code and data files
current_path = os.path.dirname(os.path.abspath(__file__))
experiments_path = os.path.dirname(current_path)
root_path = os.path.dirname(experiments_path)
experiments_path = sys.path.append(root_path)

import numpy as np
from scipy.constants import lambda2nu

from fiber import StepIndexFiber
from polarization import hyperstokes_to_jones
import raman_linear_coupling

from experiment import Experiment

class UniformPumpingExperiment(Experiment):

    def __init__(self, args) -> None:
        super().__init__()

        self.args = args

        fiber_path = os.path.join(root_path, "fibers")

        self.clad_index = 1.46
        self.delta = 0.005
        self.core_radius=6
        self.clad_radius=60

        if args.numpy_seed:
            np.random.seed(args.numpy_seed)

        self.fiber = StepIndexFiber(clad_index=self.clad_index, delta=self.delta,
                                    core_radius=self.core_radius, clad_radius=self.clad_radius, data_path=fiber_path)

        self.signal_wavelength = 1550
        self.pump_wavelength = 1459.45

        self.signal_freq = lambda2nu(self.signal_wavelength * 1e-9)
        self.pump_freq = lambda2nu(self.pump_wavelength * 1e-9)

        n2 = args.n2
        gR = args.gR

        self.fiber.load_data(wavelength=self.signal_wavelength)
        self.fiber.load_data(wavelength=self.pump_wavelength)
        self.fiber.load_nonlinear_coefficients(
            self.signal_wavelength, self.pump_wavelength)

        self.Lbeta = self.fiber.modal_beat_length(
            wavelength=self.signal_wavelength)
        self.Lpert = args.perturbation_beat_length


        total_strength = 2 * np.pi / self.Lpert
        
        # in theory, birefringence and core ellipticity act
        # together with the same strength, hence the total
        # coupling matrix is the sum of the two, with the 
        # same coupling strength

        bire_strength = total_strength / 2
        ellip_strength = total_strength / 2

        # get the equivalent physical parameters
        self.delta_n = self.fiber.birefringence(bire_strength)
        self.gamma = self.fiber.core_ellipticity(ellip_strength)

        Ke_s = self.fiber.core_ellipticity_coupling_matrix(gamma=self.gamma, wavelength=self.signal_wavelength)
        Kb_s = self.fiber.birefringence_coupling_matrix(delta_n=self.delta_n, wavelength=self.signal_wavelength)
        self.Ke_signal = Ke_s
        self.Kb_signal = Kb_s
        self.Ktot_signal = Kb_s + Ke_s

        Ke_p = self.fiber.core_ellipticity_coupling_matrix(gamma=self.gamma, wavelength=self.pump_wavelength)
        Kb_p = self.fiber.birefringence_coupling_matrix(delta_n=self.delta_n, wavelength=self.pump_wavelength)
        self.Ktot_pump = Kb_p + Ke_p

        self.nonlinear_params = self.fiber.get_raman_coefficients(
            n2, gR, self.signal_wavelength, self.pump_wavelength, as_dict=True)

        self.fiber_length = args.fiber_length * 1e3
        self.correlation_length = args.correlation_length
        self.dz = args.dz

        self.indices_s = self.fiber.group_azimuthal_orders(
            wavelength=self.signal_wavelength)
        self.indices_p = self.fiber.group_azimuthal_orders(
            wavelength=self.pump_wavelength)

        self.num_modes_s = self.fiber.num_modes(self.signal_wavelength)
        self.num_modes_p = self.fiber.num_modes(self.pump_wavelength)

        self.Pp0 = args.total_pump_power * 1e-3
        self.Ps0 = args.signal_power_per_mode * 1e-3

        pump_attenuation = 0.2 * 1e-3 * np.log(10) / 10
        signal_attenuation = 0.2 * 1e-3 * np.log(10) / 10
        self.alpha_s = signal_attenuation
        self.alpha_p = pump_attenuation

        self.beta_s = self.fiber.propagation_constants(
            wavelength=self.signal_wavelength, remove_mean=True)
        self.beta_p = self.fiber.propagation_constants(
            wavelength=self.pump_wavelength, remove_mean=True)

        if self.args.sigma:
            self.nonlinear_params['sigma'] = 0

        self.nonlinear_params['aW'] = np.conj(self.nonlinear_params['aW'])
        self.nonlinear_params['bW'] = np.conj(self.nonlinear_params['bW'])

    def propagate(self, As0, Ap0, thetas):
        z, Ap, As = raman_linear_coupling.propagate(
            As0,
            Ap0,
            self.fiber_length,
            self.dz,
            self.indices_s,
            self.indices_p,
            self.alpha_s,
            self.alpha_p,
            self.beta_s,
            self.beta_p,
            thetas,
            K_s=self.Ktot_signal,
            K_p=self.Ktot_pump,
            nonlinear_params=self.nonlinear_params,
            undepleted_pump=False,
            signal_coupling=True,
            pump_coupling=True,
            signal_spm=True,
            pump_spm=True,
        )

        return z, As, Ap

    def metadata(self):

        metadata = {
            "delta_n" : self.delta_n,
            "gamma": self.gamma,
            "signal_wavelength": self.signal_wavelength,
            "pump_wavelength": self.pump_wavelength,
        }

        metadata = {**metadata, **vars(self.args)}
        return metadata
        

    def run(self, signal_jones_vector, pump_jones_vector, thetas):
        """
        Parameters
        ----------
            signal_jones_vector: np.ndarray
                The hyperjones vector for the signal
            thetas: np.ndarray
                The hyperjones vector for the pump
            thetas: np.ndarray
                Array of perturbation angles
        """

        signal_power_per_spatial_mode = self.Ps0


        # inject power equally among the three spatial modes
        pump_power_per_spatial_mode = self.Pp0 / 3
        input_pump = pump_jones_vector * np.sqrt(pump_power_per_spatial_mode)

        input_signal = signal_jones_vector * np.sqrt(signal_power_per_spatial_mode)

        print(input_signal, input_pump)

        z, As, Ap = self.propagate(input_signal, input_pump, thetas)

        # downsample data 
        target_points = int(self.fiber_length // self.args.sampling)
        samples = len(z)
        df = int(samples / target_points)
        z = z[::df]
        As = As[::df]
        Ap = Ap[::df]


        return z, As, Ap

class ParallelLinearPolarizationsUniformPumpingExperiment(UniformPumpingExperiment):
    def run(self, thetas):
        """
        Parameters
        ----------
            thetas: np.ndarray
                Array of perturbation angles
        """

        signal_jones = np.zeros((self.num_modes_s,))
        signal_jones[0::2] = 1

        # inject power equally among the three spatial modes
        pump_jones = np.zeros((self.num_modes_p,))
        pump_jones[0::2] = 1

        return super().run(signal_jones, pump_jones, thetas)

    
class OrthogonalLinearPolarizationsUniformPumpingExperiment(UniformPumpingExperiment):
    def run(self, thetas):
        """
        Parameters
        ----------
            thetas: np.ndarray
                Array of perturbation angles
        """

        signal_jones = np.zeros((self.num_modes_s,))
        signal_jones[0::2] = 1

        # inject power equally among the three spatial modes
        pump_jones = np.zeros((self.num_modes_p,))
        pump_jones[1::2] = 1

        return super().run(signal_jones, pump_jones, thetas)
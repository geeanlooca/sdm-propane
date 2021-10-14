
import sys
import os
from numpy.random.mtrand import rand

# add path containing the code and data files
current_path = os.path.dirname(os.path.abspath(__file__))
experiments_path = os.path.dirname(current_path)
root_path = os.path.dirname(experiments_path)
experiments_path = sys.path.append(root_path)


# %%
import argparse
import datetime
import multiprocessing

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import lambda2nu
import tqdm

from fiber import StepIndexFiber
import polarization
from polarization import hyperstokes_to_jones
import raman_linear_coupling

from experiment import Experiment


class BirefringenceExperiment(Experiment):

    def __init__(self, args) -> None:
        super().__init__()

        self.args = args

        fiber_path = os.path.join(root_path, "fibers")

        self.fiber = StepIndexFiber(clad_index=1.46, delta=0.005,
                                    core_radius=6, clad_radius=60, data_path=fiber_path)

        self.signal_wavelength = 1550
        self.pump_wavelength = 1459.45

        self.signal_freq = lambda2nu(self.signal_wavelength * 1e-9)
        self.pump_freq = lambda2nu(self.pump_wavelength * 1e-9)

        n2 = args.n2
        gR = args.gR

        # n2 = 4e-20
        # gR = 1e-13

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

        Ke = self.fiber.core_ellipticity_coupling_matrix(gamma=self.gamma, wavelength=self.signal_wavelength)
        Kb = self.fiber.birefringence_coupling_matrix(delta_n=self.delta_n, wavelength=self.signal_wavelength)
        self.Ktot_signal = Kb + Ke

        Ke = self.fiber.core_ellipticity_coupling_matrix(gamma=self.gamma, wavelength=self.pump_wavelength)
        Kb = self.fiber.birefringence_coupling_matrix(delta_n=self.delta_n, wavelength=self.pump_wavelength)
        self.Ktot_pump = Kb + Ke

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
        self.Ps0 = args.signal_power_per_group * 1e-3

        pump_attenuation = 0.2 * 1e-3 * np.log(10) / 10
        signal_attenuation = 0.2 * 1e-3 * np.log(10) / 10
        self.alpha_s = signal_attenuation
        self.alpha_p = pump_attenuation

        self.beta_s = self.fiber.propagation_constants(
            wavelength=self.signal_wavelength, remove_mean=True)
        self.beta_p = self.fiber.propagation_constants(
            wavelength=self.pump_wavelength, remove_mean=True)

        self.nonlinear_params['aW'] = np.conj(self.nonlinear_params['aW'])
        self.nonlinear_params['bW'] = np.conj(self.nonlinear_params['bW'])

    def propagate(self, As0, Ap0):
        z, theta, Ap, As = raman_linear_coupling.propagate(
            As0,
            Ap0,
            self.fiber_length,
            self.dz,
            self.indices_s,
            self.indices_p,
            self.correlation_length,
            self.alpha_s,
            self.alpha_p,
            self.beta_s,
            self.beta_p,
            K_s=self.Ktot_signal,
            K_p=self.Ktot_pump,
            seed=args.fiber_seed,
            nonlinear_params=self.nonlinear_params,
            undepleted_pump=self.args.undepleted_pump,
            signal_coupling=self.args.signal_coupling,
            pump_coupling=self.args.pump_coupling,
            signal_spm=True,
            pump_spm=True,
        )

        return z, As, Ap

    def run(self, signal_sop, pump_sop):
        """
        Parameters
        ----------
            signal_sop: np.ndarray
                Generalized SOP vector for the signal, unit power
            pump_sop: np.ndarray
                Generalized SOP vector for the pump, unit power
        """

        signal_power_per_spatial_mode = self.Ps0

        # divide power equally among groups
        num_spatial_modes_signal = int(self.num_modes_s / 2)
        num_spatial_modes_pump = int(self.num_modes_p / 2 )

        input_signal_jones = hyperstokes_to_jones(signal_sop)
        input_pump_jones = hyperstokes_to_jones(pump_sop)

        # inject power only in the two LP11 groups of the pump
        pump_power_per_spatial_mode = self.Pp0 / 2
        input_pump = np.sqrt(pump_power_per_spatial_mode) * np.reshape(input_pump_jones, (num_spatial_modes_pump, 2))
        input_pump[0] = 0
        input_pump = input_pump.flatten()

        input_signal = input_signal_jones * np.sqrt(signal_power_per_spatial_mode)



        z, As, Ap = self.propagate(input_signal, input_pump)

        # downsample data 
        df = self.fiber_length // self.args.sampling
        z = z[::df]
        As = As[::df]
        Ap = Ap[::df]


        return z, As, Ap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--total-pump-power", default=400.0, type=float)
    parser.add_argument("-p", "--signal-power-per_group", default=1e-3, type=float)
    parser.add_argument("-L", "--fiber-length", default=50, type=float)
    parser.add_argument("-d", "--dz", default=1, type=float)
    parser.add_argument("-U", "--undepleted-pump", action="store_true", type=bool)
    parser.add_argument("-sc", "--signal-coupling", action="store_true", type=bool)
    parser.add_argument("-pc", "--pump-coupling", action="store_true", type=bool)
    parser.add_argument("-v", "--verbose", action="store_true", type=bool)
    parser.add_argument("-N", "--runs", default=4, type=int)
    parser.add_argument("-Lc", "--correlation-length", default=10, type=float)
    parser.add_argument("-s", "--fiber-seed", default=0, type=int)
    parser.add_argument("--n2", default=4e-20, type=float)
    parser.add_argument("--gR", default=1e-13, type=float)
    parser.add_argument("--perturbation-beat-length", default=100, type=float)
    parser.add_argument("--numpy-seed", default=None, type=int)
    parser.add_argument("--sampling", default=100, type=int)

    args = parser.parse_args()

    if args.numpy_seed:
        from numpy.random import MT19937, RandomState, SeedSequence
        rs = RandomState(MT19937(SeedSequence(int(args.numpy_seed))))

    exp = BirefringenceExperiment(args)

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    params = [(polarization.random_hypersop(3), polarization.random_hypersop(3)) for _ in range(args.runs)]
    results = pool.starmap(exp.run, tqdm.tqdm(params))
    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()

    with h5py.File(f"results-{timestamp}.h5", "a") as f:
        signal_sops = np.zeros((args.runs, 3, 3), dtype=np.complex128)
        pump_sops = np.zeros_like(signal_sops)

        for i in range(args.runs):
            signal_sops[i] = params[i][0]
            pump_sops[i] = params[i][1]

        z = results[0][0]
        As = np.stack([ s for (_, s, _) in results])
        Ap = np.stack([ p for (_, _, p) in results])

        signal_sop_dset = f.create_dataset("signal_sops", dtype=np.complex128, shape=signal_sops.shape, compression="gzip")
        signal_sop_dset[:] = signal_sops
        pump_sop_dset = f.create_dataset("pump_sops", dtype=np.complex128, shape=pump_sops.shape, compression="gzip")
        pump_sop_dset[:] = pump_sops

        z_dset = f.create_dataset("z", dtype=np.float64, shape=z.shape, compression="gzip")
        z_dset[:] = z

        signal_dset = f.create_dataset("signal", dtype=np.complex128, shape=As.shape, compression="gzip")
        pump_dset = f.create_dataset("pump", dtype=np.complex128, shape=Ap.shape, compression="gzip")

        signal_dset[:] = As
        pump_dset[:] = Ap

        for (k, v) in vars(args).items():
            f[f"params/{k}"] = v


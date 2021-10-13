
# %%
from tqdm.contrib.concurrent import process_map
import argparse

import matplotlib.pyplot as plt
import numpy as np

from scipy.constants import Planck as hp, lambda2nu
from scipy.optimize import nonlin

from fiber import StepIndexFiber

from polarization import stokes_to_jones, random_sop, plot_sphere, compute_stokes, plot_stokes
import raman_linear_coupling

from experiment import Experiment, ParallelExperiment

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sigma", action="store_false")
parser.add_argument("-P", "--power", default=400.0, type=float)
parser.add_argument("-L", "--fiber-length", default=60, type=float)
parser.add_argument("-d", "--dz", default=5, type=float)
parser.add_argument("-U", "--undepleted-pump", action="store_true")
parser.add_argument("-sc", "--signal-coupling", action="store_true")
parser.add_argument("-pc", "--pump-coupling", action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-N", "--number", default=4, type=int)

args = parser.parse_args()


class BirefringenceExperiment(Experiment):

    def __init__(self, args) -> None:
        super().__init__()

        self.args = args

        self.fiber = StepIndexFiber(clad_index=1.46, delta=0.005,
                                    core_radius=6, clad_radius=60, data_path="fibers")

        self.signal_wavelength = 1550
        self.pump_wavelength = 1459.45

        self.signal_freq = lambda2nu(self.signal_wavelength * 1e-9)
        self.pump_freq = lambda2nu(self.pump_wavelength * 1e-9)

        n2 = 4e-20
        gR = 1e-13

        self.fiber.load_data(wavelength=self.signal_wavelength)
        self.fiber.load_data(wavelength=self.pump_wavelength)
        self.fiber.load_nonlinear_coefficients(
            self.signal_wavelength, self.pump_wavelength)

        self.Lbeta = self.fiber.modal_beat_length(
            wavelength=self.signal_wavelength)
        self.Lpert = 1e3 * self.Lbeta
        self.delta_n = self.fiber.birefringence(
            2 * np.pi / self.Lpert, wavelength=self.signal_wavelength)

        self.K_signal = self.fiber.birefringence_coupling_matrix(
            delta_n=self.delta_n, wavelength=self.signal_wavelength)
        self.K_pump = self.fiber.birefringence_coupling_matrix(
            delta_n=self.delta_n, wavelength=self.pump_wavelength)

        self.nonlinear_params = self.fiber.get_raman_coefficients(
            n2, gR, self.signal_wavelength, self.pump_wavelength, as_dict=True)

        self.fiber_length = args.fiber_length * 1e3
        self.correlation_length = 10
        self.dz = args.dz

        self.indices_s = self.fiber.group_azimuthal_orders(
            wavelength=self.signal_wavelength)
        self.indices_p = self.fiber.group_azimuthal_orders(
            wavelength=self.pump_wavelength)

        self.num_modes_s = self.fiber.num_modes(self.signal_wavelength)
        self.num_modes_p = self.fiber.num_modes(self.pump_wavelength)

        self.Pp0 = args.power * 1e-3
        self.Ps0 = 1e-3
        self.Ap0 = np.zeros((self.num_modes_p,)).astype("complex128")
        self.As0 = np.zeros((self.num_modes_s,)).astype("complex128")

        self.pump_sop = random_sop()
        self.Ap0[0:2] = np.sqrt(self.Pp0) * \
            stokes_to_jones(self.pump_sop).squeeze()

        pump_attenuation = 0.2 * 1e-3 * np.log(10) / 10
        signal_attenuation = 0.2 * 1e-3 * np.log(10) / 10
        self.alpha_s = signal_attenuation
        self.alpha_p = pump_attenuation
        self.Kb_s = self.K_signal
        self.Kb_p = self.K_pump

        self.beta_s = self.fiber.propagation_constants(
            wavelength=self.signal_wavelength, remove_mean=True)
        self.beta_p = self.fiber.propagation_constants(
            wavelength=self.pump_wavelength, remove_mean=True)

        if not args.sigma:
            self.nonlinear_params['sigma'] = 0

        self.nonlinear_params['aW'] = np.conj(self.nonlinear_params['aW'])
        self.nonlinear_params['bW'] = np.conj(self.nonlinear_params['bW'])

    def propagate(self, sop):
        jones = stokes_to_jones(sop) * np.random.uniform(low=0, high=2 * np.pi)

        self.As0[0] = jones[0] * np.sqrt(self.Ps0)
        self.As0[1] = jones[1] * np.sqrt(self.Ps0)

        z, theta, Ap, As = raman_linear_coupling.propagate(
            self.As0,
            self.Ap0,
            self.fiber_length,
            self.dz,
            self.indices_s,
            self.indices_p,
            self.correlation_length,
            self.alpha_s,
            self.alpha_p,
            self.beta_s,
            self.beta_p,
            K_s=self.Kb_s,
            K_p=self.Kb_p,
            seed=0,
            nonlinear_params=self.nonlinear_params,
            undepleted_pump=self.args.undepleted_pump,
            signal_coupling=self.args.signal_coupling,
            pump_coupling=self.args.pump_coupling,
            signal_spm=True,
            pump_spm=True,
        )

        As = As[:, 0:2]
        Ap = Ap[:, 0:2]

        signal_sop = compute_stokes(As[1])
        pump_sop = compute_stokes(Ap[-1])

        return signal_sop, pump_sop

    def run(self, sop):
        output_sop, sop_pump = self.propagate(sop)

        results = {"input_sop": sop, "output_sop": output_sop,
                   "output_pump_sop": sop_pump}
        return results


class ParallelBirefringenceExperiment(ParallelExperiment):
    def __init__(self, experiment, args) -> None:
        super().__init__()
        self.args = args
        self.experiment = experiment
        
        # prepare arrays to store inputs and outputs
        self.output_sops = np.zeros((args.number, 3))
        self.output_pump_sops = np.zeros_like(self.output_sops)
        self.input_sops = np.zeros_like(self.output_sops)

        for x in range(args.number):
            sop = random_sop()
            self.input_sops[x] = sop.squeeze()

    def inputs(self):
        return [self.input_sops]

    def post(self):
        print("Post-processing")
        for x, result in enumerate(self.results):
            self.output_sops[x] = result['output_sop']
            self.output_pump_sops[x] = result['output_pump_sop']
            self.input_sops[x] = result['input_sop']

        
        fig = plt.figure(figsize=(7, 5))
        ax = plt.axes(projection='3d')

        plot_sphere(pts=10)
        plot_stokes(self.output_sops.T, label="Signal SOP, output", s=80)
        plot_stokes(self.output_pump_sops.T, label="Pump SOP, output", marker="s", s=80)

        ax.legend()
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":
    exp = BirefringenceExperiment(args)
    parallel_exp = ParallelBirefringenceExperiment(exp, args)
    result = parallel_exp.run()
    parallel_exp.post()


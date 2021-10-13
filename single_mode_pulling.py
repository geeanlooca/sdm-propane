
# %%
from tqdm.contrib.concurrent import process_map
import argparse

import matplotlib.pyplot as plt
import numpy as np

from scipy.constants import Planck as hp, lambda2nu
from scipy.optimize import nonlin

from fiber import StepIndexFiber

from polarization import plot_stokes, stokes_to_jones, random_sop, plot_sphere, compute_stokes, plot_stokes_trajectory
import raman_linear_coupling


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sigma", action="store_false")
parser.add_argument("-P", "--pump-power", default=400.0, type=float)
parser.add_argument("-p", "--signal-power", default=1e-3, type=float)
parser.add_argument("-L", "--fiber-length", default=60, type=float)
parser.add_argument("-d", "--dz", default=5, type=float)
parser.add_argument("-U", "--undepleted-pump", action="store_true")
parser.add_argument("-sc", "--signal-coupling", action="store_true")
parser.add_argument("-pc", "--pump-coupling", action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-N", "--number", default=10, type=int)

args = parser.parse_args()


fiber = StepIndexFiber(clad_index=1.46, delta=0.005,
                       core_radius=6, clad_radius=60, data_path="fibers")
signal_wavelength = 1550
pump_wavelength = 1459.45

signal_freq = lambda2nu(signal_wavelength * 1e-9)
pump_freq = lambda2nu(pump_wavelength * 1e-9)

n2 = 4e-20
gR = 1e-13

fiber.load_data(wavelength=signal_wavelength)
fiber.load_data(wavelength=pump_wavelength)
fiber.load_nonlinear_coefficients(signal_wavelength, pump_wavelength)

Lbeta = fiber.modal_beat_length(wavelength=signal_wavelength)
Lpert = 1e5 * Lbeta
delta_n = fiber.birefringence(2 * np.pi / Lpert, wavelength=signal_wavelength)

K_signal = fiber.birefringence_coupling_matrix(
    delta_n=delta_n, wavelength=signal_wavelength)
K_pump = fiber.birefringence_coupling_matrix(
    delta_n=delta_n, wavelength=pump_wavelength)

nonlinear_params = fiber.get_raman_coefficients(
    n2, gR, signal_wavelength, pump_wavelength, as_dict=True)

fiber_length = args.fiber_length * 1e3
correlation_length = 10
dz = args.dz

indices_s = fiber.group_azimuthal_orders(wavelength=signal_wavelength)
indices_p = fiber.group_azimuthal_orders(wavelength=pump_wavelength)

num_modes_s = fiber.num_modes(signal_wavelength)
num_modes_p = fiber.num_modes(pump_wavelength)

Pp0 = args.pump_power * 1e-3
Ps0 = args.signal_power
Ap0 = np.zeros((num_modes_p,)).astype("complex128")
As0 = np.zeros((num_modes_s,)).astype("complex128")

pump_sop = random_sop()
Ap0[0:2] = np.sqrt(Pp0) * stokes_to_jones(pump_sop).squeeze()

pump_attenuation = 0.2 * 1e-3 * np.log(10) / 10
signal_attenuation = 0.2 * 1e-3 * np.log(10) / 10
alpha_s = signal_attenuation
alpha_p = pump_attenuation
Kb_s = K_signal
Kb_p = K_pump

beta_s = fiber.propagation_constants(
    wavelength=signal_wavelength, remove_mean=True)
beta_p = fiber.propagation_constants(
    wavelength=pump_wavelength, remove_mean=True)

if not args.sigma:
    nonlinear_params['sigma'] = 0

nonlinear_params['aW'] = np.conj(nonlinear_params['aW'])
nonlinear_params['bW'] = np.conj(nonlinear_params['bW'])

if args.verbose:
    print("sigma", nonlinear_params['sigma'])
    print("a0", nonlinear_params['a0'])
    print("b0", nonlinear_params['b0'])
    print("aW", nonlinear_params['aW'])
    print("bW", nonlinear_params['bW'])

propagation_function = raman_linear_coupling.propagate


def propagate(sop):
    jones = stokes_to_jones(sop)# * np.random.uniform(low=0, high=2 * np.pi)
    As0[0] = jones[0] * np.sqrt(Ps0)
    As0[1] = jones[1] * np.sqrt(Ps0)

    z, theta, Ap, As = propagation_function(
        As0,
        Ap0,
        fiber_length,
        dz,
        indices_s,
        indices_p,
        correlation_length,
        alpha_s,
        alpha_p,
        beta_s,
        beta_p,
        K_s=Kb_s,
        K_p=Kb_p,
        seed=0,
        nonlinear_params=nonlinear_params,
        undepleted_pump=args.undepleted_pump,
        signal_coupling=args.signal_coupling,
        pump_coupling=args.pump_coupling,
        signal_spm=True,
        pump_spm=True,
    )

    output_signal_sop = compute_stokes(As[-1, 0:2])
    output_pump_sop = compute_stokes(Ap[-1, 0:2])

    As = As[:, 0:2]
    Ap = Ap[:, 0:2]

    return output_signal_sop, output_pump_sop, As


output_sops = np.zeros((3, args.number))
output_pump_sops = np.zeros_like(output_sops)
input_sops = np.zeros_like(output_sops)

for x in range(args.number):
    s1, s2, s3 = random_sop()
    sop = np.array([s1, s2, s3])
    input_sops[:, x] = sop.squeeze()


# Double check that the convertion from stokes to jones space is correct
input_sops_jones = stokes_to_jones(input_sops)
input_sops_jones = compute_stokes(input_sops_jones)

fig = plt.figure(figsize=(7, 5))
ax = plt.axes(projection='3d')
plot_sphere(pts=10)
plot_stokes(input_sops, label="Stokes", s=80)
plot_stokes(input_sops_jones, label="Jones", s=80)
ax.legend()
plt.tight_layout()

def worker(procnum):
    input_sop = input_sops[:, procnum]
    output_sop, sop_pump, As = propagate(input_sop)
    return output_sop, sop_pump, sop, As


results = process_map(worker, range(args.number), max_workers=4)

plt.figure()
ax = plt.axes(projection='3d')
plot_sphere()
for x, (sop, sop_pump, input_sop, As) in enumerate(results):
    output_sops[:, x] = sop
    output_pump_sops[:, x] = sop_pump
    input_sops[:, x] = input_sop.squeeze()

    sop = compute_stokes(As.T)
    plot_stokes_trajectory(sop)
    plot_stokes(output_sops[:, x], s=40)
    

input_pump_sop = compute_stokes(Ap0)

# %%

fig = plt.figure(figsize=(7, 5))
ax = plt.axes(projection='3d')

plot_sphere(pts=10)
plot_stokes(input_pump_sop, label="Pump SOP", marker="^", s=80)
plot_stokes(output_sops, label="Signal SOP, output", s=80)
plot_stokes(output_pump_sops, label="Pump SOP, output", marker="s", s=80)

ax.legend()
plt.tight_layout()
plt.show()

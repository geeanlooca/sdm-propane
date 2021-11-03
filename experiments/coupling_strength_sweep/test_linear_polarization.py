
import sys
import os

# add path containing the code and data files
current_path = os.path.dirname(os.path.abspath(__file__))
experiments_path = os.path.dirname(current_path)
root_path = os.path.dirname(experiments_path)
experiments_path = sys.path.append(root_path)

from stats_manager import OnlineMeanManager

# %%
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np

import polarization

from uniform_pumping_experiments import ParallelLinearPolarizationsUniformPumpingExperiment, OrthogonalLinearPolarizationsUniformPumpingExperiment
from utils import process_results, write_metadata, cmd_parser, build_params_string
from perturbation_angles import generate_perturbation_angles

def dBm(x):
    return 10 * np.log10(x * 1e3)

if __name__ == "__main__":

    parser = cmd_parser()
    parser.add_argument("--polarization", choices=["parallel", "orthogonal"])
    args = parser.parse_args()


    args.perturbation_beat_length = 1e2

    selected_params = ["fiber_length", "correlation_length", "perturbation_beat_length", "dz"]
    params_string = build_params_string(args, selected_params)


    exp = OrthogonalLinearPolarizationsUniformPumpingExperiment(args)


    signal_manager = OnlineMeanManager("Signal power")
    pump_manager = OnlineMeanManager("Pump power")
    output_signal_manager = OnlineMeanManager("Output signal power")

    angles = generate_perturbation_angles(args.correlation_length, args.dz, args.fiber_length * 1e3)

    # propagate 
    z, As, Ap = exp.run(angles)


    Ps = np.abs(As) ** 2
    Ps_pol = (Ps[:, ::2] + Ps[:, 1::2])

    Pp = np.abs(Ap) ** 2
    Pp_pol = (Pp[:, ::2] + Pp[:, 1::2])



    plt.figure(1)
    plt.cla()

    plt.plot(z * 1e-3, dBm(Ps))
    plt.xlabel("Position [km]")
    plt.ylabel("Power [dBm]")
    plt.tight_layout()
    
    plt.figure(2)
    plt.cla()

    plt.plot(z * 1e-3, dBm(Pp))
    plt.xlabel("Position [km]")
    plt.ylabel("Power [dBm]")
    plt.tight_layout()
    plt.show()


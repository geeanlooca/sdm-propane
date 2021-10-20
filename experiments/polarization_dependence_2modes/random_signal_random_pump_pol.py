
import sys
import os

# add path containing the code and data files
current_path = os.path.dirname(os.path.abspath(__file__))
experiments_path = os.path.dirname(current_path)
root_path = os.path.dirname(experiments_path)
experiments_path = sys.path.append(root_path)

from stats_manager import OnlineMeanManager

# %%
import argparse
import datetime
import multiprocessing

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tqdm

import polarization

from base_experiment import VaryPolarizationExperiment
from utils import process_results, write_metadata, cmd_parser, build_params_string

def dBm(x):
    return 10 * np.log10(x * 1e3)

if __name__ == "__main__":

    parser = cmd_parser()
    args = parser.parse_args()

    selected_params = ["fiber_length", "correlation_length", "perturbation_beat_length", "dz"]
    params_string = build_params_string(args, selected_params)

    exp_name = args.experiment_name
    filename = f"{exp_name}-{params_string}.h5"


    exp = VaryPolarizationExperiment(args)


    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    batch_idx = 0

    signal_manager = OnlineMeanManager("Signal power")
    pump_manager = OnlineMeanManager("Pump power")
    output_signal_manager = OnlineMeanManager("Output signal power")

    # generate parallel input polarizations between signal and pump
    pol_angle = 0



    write_metadata(filename, exp)

    def condition(i, args):
        if args.forever:
            print(f"Batch {i}...")
            return True
        else:
            print(f"Batch {i}/{args.batches}...")
            return i < args.batches


    while condition(batch_idx, args):
        params = [(polarization.random_hypersop(3), polarization.random_hypersop(3)) for _ in range(args.runs_per_batch)]

        # propagate 
        results = pool.starmap(exp.run, params)

        # process results and save data to file
        z, As, Ap, theta = process_results(results, params, filename)

        Ps = np.abs(As) ** 2
        Ps_pol = (Ps[:, :, ::2] + Ps[:,:, 1::2])

        Pp = np.abs(Ap) ** 2
        Pp_pol = (Pp[:, :, ::2] + Pp[:,:, 1::2])


        output_signal_manager.update(dBm(Ps_pol[:,-1,:]), accumulate=True)
        signal_manager.update(Ps_pol, accumulate=False)
        pump_manager.update(Pp_pol, accumulate=False)

        plt.figure(1)
        output_signal_manager.plot(f"{exp_name}-output_power_convergence-{params_string}.png")
        plt.pause(0.05)

        plt.figure(2)
        plt.cla()

        above = dBm(signal_manager.mean + signal_manager.std)
        below = dBm(signal_manager.mean - signal_manager.std)
        plt.plot(z * 1e-3, dBm(signal_manager.mean))
        for x in range(signal_manager.mean.shape[-1]):
            plt.fill_between(z * 1e-3, below[:, x], above[:, x], color=f"C{x}", alpha=0.3)
        plt.xlabel("Position [km]")
        plt.ylabel("Power [dBm]")
        plt.title("Average signal power and standard dev. in each spatial mode")
        plt.tight_layout()
        plt.savefig(f"{exp_name}-mean_signal_power-{params_string}.png")
        plt.pause(0.05)

        plt.figure(3)
        plt.cla()

        above = dBm(pump_manager.mean + pump_manager.std)
        below = dBm(pump_manager.mean - pump_manager.std)
        plt.plot(z * 1e-3, dBm(pump_manager.mean))
        for x in range(pump_manager.mean.shape[-1]):
            plt.fill_between(z * 1e-3, below[:, x], above[:, x], color=f"C{x}", alpha=0.3)
        plt.xlabel("Position [km]")
        plt.ylabel("Power [dBm]")
        plt.title("Average pump power and standard dev. in each spatial mode")
        plt.tight_layout()
        plt.savefig(f"{exp_name}-mean_pump_power-{params_string}.png")
        plt.pause(0.05)

        batch_idx += 1

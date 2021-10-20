
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

    Lk_min = args.min_beat_length
    Lk_max = args.max_beat_length

    id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    num_beat_lengths = int(os.getenv("SLURM_ARRAY_TASK_COUNT"))

    Lk = np.geomspace(Lk_min, Lk_max, num_beat_lengths)

    args.perturbation_beat_length = Lk[id]

    selected_params = ["fiber_length", "correlation_length", "perturbation_beat_length", "dz"]
    params_string = build_params_string(args, selected_params)

    exp_name  = "random_polarizations_Lk_sweep"
    filename = f"{exp_name}-{params_string}.h5"


    exp = VaryPolarizationExperiment(args)


    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    batch_idx = 0

    signal_manager = OnlineMeanManager("Signal power")
    signal_manager_dBm = OnlineMeanManager("Signal power dBm")
    output_signal_manager = OnlineMeanManager("Output signal power")


    write_metadata(filename, exp)

    def condition(i, args):

        simulated_fibers = args.runs_per_batch * i
        if args.forever:
            string = f"Batch {i}...\t{simulated_fibers} fibers..."
            return True
        elif args.max_fibers:
            batches = np.ceil(args.max_fibers / args.runs_per_batch)
            string = f"Batch {i}/{batches}...\t{simulated_fibers} fibers..."
            print(string)
            return i < batches
        else:
            string = f"Batch {i}/{args.batches}...\t{simulated_fibers} fibers..."
            print(string)
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
        signal_manager_dBm.update(dBm(Ps_pol), accumulate=False)

        plt.figure(1)
        output_signal_manager.plot(f"{exp_name}-output_power_convergence-{params_string}.png")
        plt.pause(0.05)

        plt.figure(2)
        plt.clf()
        plt.subplot(121)

        plt.plot(z * 1e-3, dBm(signal_manager.mean))
        plt.xlabel("Position [km]")
        plt.ylabel("Power [dBm]")
        plt.tight_layout()


        plt.subplot(122)
        plt.plot(z * 1e-3, signal_manager_dBm.std)
        plt.xlabel("Position [km]")
        plt.ylabel("Power [dBm]")

        plt.suptitle("Average signal power and standard dev. in each spatial mode")

        plt.savefig(f"{exp_name}-mean_signal_power-{params_string}.png")
        plt.tight_layout()

        batch_idx += 1

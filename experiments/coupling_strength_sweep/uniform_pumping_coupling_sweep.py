import argparse
import sys
import os
import pathlib
from typing import Tuple

# add path containing the code and data files
current_path = os.path.dirname(os.path.abspath(__file__))
experiments_path = os.path.dirname(current_path)
root_path = os.path.dirname(experiments_path)
experiments_path = sys.path.append(root_path)

from perturbation_angles import generate_perturbation_angles
from utils import (
    process_results_fixed_polarizations,
    write_metadata,
    cmd_parser,
    build_params_string,
)
from experiment import Experiment
from uniform_pumping_experiments import (
    OrthogonalEllipticalPolarizationsUniformPumpingExperiment as OrthogonalEllip,
    ParallelEllipticalPolarizationsUniformPumpingExperiment as ParallelEllip,
    OrthogonalLinearPolarizationsUniformPumpingExperiment as OrthogonalLinear,
    ParallelLinearPolarizationsUniformPumpingExperiment as ParallelLinear,
)
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from stats_manager import OnlineMeanManager


# %%


def build_experiment(args: argparse.Namespace) -> Experiment:
    """Build an Experiment given its parameters"""
    if args.polarization == "linear":
        if args.polarization_orientation == "orthogonal":
            exp = OrthogonalLinear(args)
        else:
            exp = ParallelLinear(args)
    else:
        if args.polarization_orientation == "orthogonal":
            exp = OrthogonalEllip(args)
        else:
            exp = ParallelEllip(args)

    return exp


def condition(i: int, args: argparse.Namespace) -> bool:
    """Check when to stop the loop"""
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


def dBm(x: np.ndarray) -> np.ndarray:
    """Convert to dBm"""
    return 10 * np.log10(x * 1e3)


def make_output_directory(args: argparse.Namespace) -> Tuple[str, str]:
    """Names and creates the output directories given the input arguments"""
    filename = (
        f"fiber_length_{args.fiber_length}km"
        + f"-dz_{args.dz}m"
        + f"-birefringence_{args.birefringence_weight}"
        f"-filtering_{args.percent}"
        f"-pump_power_{args.total_pump_power}mW"
    )

    if args.no_kerr:
        filename += "-no_kerr"

    output_dir = os.path.join(
        args.output_dir,
        "uniform_pumping",
        f"{args.modes}modes",
        f"{args.polarization}_{args.polarization_orientation}_polarizations",
        f"Lc_{args.correlation_length}m",
        filename,
    )

    img_dir = os.path.join(output_dir, "convergence_imgs")

    if not args.dry_run:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(img_dir).mkdir(parents=True, exist_ok=True)

    return output_dir, img_dir


if __name__ == "__main__":

    parser = cmd_parser()
    args = parser.parse_args()

    # read slurm parameters from the environment
    try:
        id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        num_beat_lengths = int(os.getenv("SLURM_ARRAY_TASK_COUNT"))
    except:
        id = 0
        num_beat_lengths = 1

    # sets the perturbation beat length
    Lk_min = args.min_beat_length
    Lk_max = args.max_beat_length
    Lk = np.logspace(Lk_min, Lk_max, num_beat_lengths)
    args.perturbation_beat_length = Lk[id]
    perturbation_beat_length = Lk[id]

    # build the experiment and the output directory structure
    exp = build_experiment(args)
    output_dir, img_dir = make_output_directory(args)
    output_filename = os.path.join(output_dir, f"{perturbation_beat_length=}m.h5")

    pump_power_filename = os.path.join(
        img_dir, f"mean_pump_power-{perturbation_beat_length=}m.png"
    )
    signal_power_filename = os.path.join(
        img_dir, f"mean_signal_power-{perturbation_beat_length=}m.png"
    )

    output_power_filename = os.path.join(
        img_dir,
        f"output_power_convergence-{perturbation_beat_length=}m.png",
    )

    signal_manager = OnlineMeanManager("Signal power")
    pump_manager = OnlineMeanManager("Pump power")
    output_signal_manager = OnlineMeanManager("Output signal power")

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    batch_idx = 0

    if args.dry_run:
        print("Dry run")
        print(output_dir)
        print(img_dir)
        print(output_filename)
        print(output_power_filename)
        print(signal_power_filename)
        print(pump_power_filename)
        raise SystemExit

    write_metadata(output_filename, exp)

    while condition(batch_idx, args):
        fibers = [
            generate_perturbation_angles(
                args.correlation_length, args.dz, args.fiber_length * 1e3
            )
            for _ in range(args.runs_per_batch)
        ]
        params = [f for f in fibers]

        # propagate
        results = pool.map(exp.run, params)

        # process results and save data to file
        z, As, Ap = process_results_fixed_polarizations(
            results, params, output_filename
        )

        Ps = np.abs(As) ** 2
        Ps_pol = Ps[:, :, ::2] + Ps[:, :, 1::2]

        Pp = np.abs(Ap) ** 2
        Pp_pol = Pp[:, :, ::2] + Pp[:, :, 1::2]

        output_signal_manager.update(dBm(Ps_pol[:, -1, :]), accumulate=True)
        signal_manager.update(dBm(Ps_pol), accumulate=False)
        pump_manager.update(dBm(Pp_pol), accumulate=False)

        plt.figure(1)
        output_signal_manager.plot(output_power_filename)
        plt.pause(0.05)

        plt.figure(2)
        plt.cla()

        above = signal_manager.mean + signal_manager.std
        below = signal_manager.mean - signal_manager.std
        plt.plot(z * 1e-3, (signal_manager.mean))
        for x in range(signal_manager.mean.shape[-1]):
            plt.fill_between(
                z * 1e-3, below[:, x], above[:, x], color=f"C{x}", alpha=0.3
            )
        plt.xlabel("Position [km]")
        plt.ylabel("Power [dBm]")
        plt.title("Average signal power and standard dev. in each spatial mode")
        plt.tight_layout()
        plt.savefig(signal_power_filename)
        plt.pause(0.05)

        plt.figure(3)
        plt.cla()

        above = pump_manager.mean + pump_manager.std
        below = pump_manager.mean - pump_manager.std
        plt.plot(z * 1e-3, (pump_manager.mean))
        for x in range(pump_manager.mean.shape[-1]):
            plt.fill_between(
                z * 1e-3, below[:, x], above[:, x], color=f"C{x}", alpha=0.3
            )
        plt.xlabel("Position [km]")
        plt.ylabel("Power [dBm]")
        plt.title("Average pump power and standard dev. in each spatial mode")
        plt.tight_layout()
        plt.savefig(pump_power_filename)
        plt.pause(0.05)

        batch_idx += 1

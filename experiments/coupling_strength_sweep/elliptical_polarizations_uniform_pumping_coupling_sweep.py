import sys
import os

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


def condition(i, args):
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


def dBm(x):
    return 10 * np.log10(x * 1e3)


if __name__ == "__main__":

    parser = cmd_parser()
    args = parser.parse_args()

    Lk_min = args.min_beat_length
    Lk_max = args.max_beat_length

    try:
        id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        num_beat_lengths = int(os.getenv("SLURM_ARRAY_TASK_COUNT"))
    except:
        id = 0
        num_beat_lengths = 1

    Lk = np.logspace(Lk_min, Lk_max, num_beat_lengths)

    args.perturbation_beat_length = Lk[id]
    perturbation_beat_length = Lk[id]

    selected_params = [
        "modes",
        "fiber_length",
        "correlation_length",
        "perturbation_beat_length",
        "dz",
        "polarization",
        "percent",
    ]
    params_string = build_params_string(args, selected_params)

    exp_name = f"{args.polarization}_{args.polarization_orientation}_uniform_pumping"

    output_dir = os.path.join(
        "uniform_pumping",
        f"{args.modes}modes-"
        + f"{args.fiber_length}km-"
        + f"{args.polarization}-{args.polarization_orientation}-"
        + f"birefringence_{args.birefringence_weight}-"
        f"filtering_{args.percent}",
    )

    img_dir = os.path.join(output_dir, "convergence_imgs")

    filename = os.path.join(output_dir, f"{perturbation_beat_length=}m.h5")

    print(output_dir)

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    batch_idx = 0

    signal_manager = OnlineMeanManager("Signal power")
    pump_manager = OnlineMeanManager("Pump power")
    output_signal_manager = OnlineMeanManager("Output signal power")

    # generate parallel input polarizations between signal and pump

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

    write_metadata(filename, exp)

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
        z, As, Ap = process_results_fixed_polarizations(results, params, filename)

        Ps = np.abs(As) ** 2
        Ps_pol = Ps[:, :, ::2] + Ps[:, :, 1::2]

        Pp = np.abs(Ap) ** 2
        Pp_pol = Pp[:, :, ::2] + Pp[:, :, 1::2]

        output_signal_manager.update(dBm(Ps_pol[:, -1, :]), accumulate=True)
        signal_manager.update(dBm(Ps_pol), accumulate=False)
        pump_manager.update(dBm(Pp_pol), accumulate=False)

        plt.figure(1)
        output_signal_manager.plot(
            os.path.join(
                img_dir,
                f"output_power_convergence-{perturbation_beat_length=}m.png",
            )
        )
        plt.savefig()
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

        plt.savefig(
            os.path.join(img_dir, f"mean_signal_power-{perturbation_beat_length=}m.png")
        )
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
        plt.savefig(
            os.path.join(img_dir, f"mean_pump_power-{perturbation_beat_length=}m.png")
        )
        plt.pause(0.05)

        batch_idx += 1

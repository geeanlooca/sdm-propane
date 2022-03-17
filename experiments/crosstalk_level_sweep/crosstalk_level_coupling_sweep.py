import argparse
import json
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
from crosstalk_experiments import CrossTalkLevelExperiment
import numpy as np
import multiprocessing
import tqdm


# %%


def dBm(x: np.ndarray) -> np.ndarray:
    """Convert to dBm"""
    return 10 * np.log10(x * 1e3)


def dB(x: np.ndarray) -> np.ndarray:
    """Convert to dB"""
    return 10 * np.log10(x)


def generate_fiber_batch(batch_size, fiber_length, dz, correlation_length):
    """Generate a list of fiber realizations."""
    return [
        generate_perturbation_angles(correlation_length, dz, fiber_length)
        for _ in range(batch_size)
    ]


def cmd_parser():
    """Build the command line parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--signal-power-per-mode", default=1e-3, type=float)
    parser.add_argument("-L", "--fiber-length", default=50, type=float)
    parser.add_argument("-d", "--dz", default=1, type=float)
    parser.add_argument("-Lc", "--correlation-length", default=10, type=float)
    parser.add_argument("--fiber-seed", default=0, type=int)
    parser.add_argument("--numpy-seed", default=0, type=int)
    parser.add_argument("--min-beat-length", default=1e-3, type=float)
    parser.add_argument("--max-beat-length", default=1e5, type=float)
    parser.add_argument("--beat-length-points", default=50, type=int)
    parser.add_argument("--fibers", default=5000, type=int)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--alpha", default=0.2, type=float)

    return parser


def compute_crosstalk(results):
    """Compute the crosstalk given the results of the experiment"""
    z = results[0][0]
    results = [b for (_, b) in results]

    # average over the realizations
    results = np.stack(results).mean(axis=0)
    groups = list(range(results.shape[0]))

    XT = np.zeros((results.shape[0], results.shape[1]))
    for group_idx in groups:
        indeces = [i for i in groups if i != group_idx]

        power_other_modes = np.zeros((XT.shape[1],))
        for i in indeces:
            power_other_modes = power_other_modes + results[group_idx, :, i]
        power_same_mode = results[group_idx].sum(axis=-1)
        XT[group_idx, :] = power_other_modes / power_same_mode

    return XT, z


if __name__ == "__main__":
    parser = cmd_parser()
    args = parser.parse_args()

    # save params to file
    with open(os.path.join(args.output_dir, "crosstalk_params.json"), "w") as f:
        json.dump(vars(args), f)

    num_beat_lengths = args.beat_length_points
    Lk_min = args.min_beat_length
    Lk_max = args.max_beat_length
    Lk = np.logspace(Lk_min, Lk_max, num_beat_lengths)

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    batch_idx = 0

    progress_bar = tqdm.tqdm(Lk)

    XT = []

    for iteration, beat_length in enumerate(progress_bar):

        # build the experiment
        args.perturbation_beat_length = beat_length
        exp = CrossTalkLevelExperiment(args)

        # generate a bunch of fiber realizations
        fibers = generate_fiber_batch(
            args.fibers,
            args.fiber_length * 1e3,
            args.dz,
            args.correlation_length,
        )

        # propagate
        results = pool.map(exp.run, fibers)

        crosstalk, z = compute_crosstalk(results)
        XT.append(crosstalk)

XT = np.stack(XT)

np.savez(os.path.join(args.output_dir, "crosstalk_4modes.npz"), Lk=Lk, XT=XT, z=z)


#     return z, XT[:, -1], exp.Lbeta

# Lk = np.logspace(-2, 6, 30)
# XT = np.zeros((Lk.shape[0], 4))
# for i, l in tqdm(enumerate(Lk)):
#     z, xt, Lbeta = get_XT(l)
#     XT[i] = xt

# plt.figure()
# plt.semilogx(Lk, dB(XT), label=f"Lk={l}")
# # plt.plot(z, dB(XT.T))

# # plt.plot(z, dB(results[0]))
# plt.show()

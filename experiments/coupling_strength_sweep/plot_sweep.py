import sys

sys.path.append("../../")

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from read_data import read_data
import tqdm


def find_files(args):
    filenames = os.listdir(args.directory)
    data_files = [
        os.path.join(args.directory, f) for f in filenames if f.endswith(".h5")
    ]
    return data_files


def read_from_files(args, index=None):
    data_files = find_files(args)
    As = []
    Ap = []
    Lk = []
    Ps0 = []

    for f in tqdm.tqdm(data_files):
        try:
            data = read_data(f, idx=index)
            As.append(data["signal"])
            Ap.append(data["pump"])
            Ps0.append(data["Ps0"] * 1e-3)
            z = data["z"]
            Lk.append(data["Lk"])
        except:
            print(f"Error reading {f}")

    xy = sorted(zip(Lk, As))
    Lk = [x for x, y in xy]
    As = [y for x, y in xy]

    return As, Ap, z, np.array(Lk), Ps0


def dBm(x):
    return 10 * np.log10(x * 1e3)


def dB(x):
    return 10 * np.log10(x)


parser = argparse.ArgumentParser()
parser.add_argument("directory", nargs="?")
parser.add_argument("-s", "--save", action="store_true")
parser.add_argument("-S", "--save-data", action="store_true")
parser.add_argument("-L", "--lengths", nargs="+", type=float)
parser.add_argument("-t", "--tex", action="store_true")

args = parser.parse_args()

if args.tex:
    plt.rcParams.update(
        {
            "text.latex.preamble": r"\usepackage{mathpazo}",
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["palatino"],
        }
    )


def compute_gain_statistics(args, index=None):
    As, Ap, z, Lk, Ps0 = read_from_files(args, index=index)
    num_files = len(As)
    std = []
    average_gain = []

    gain_family = []
    std_family = []
    for i in range(num_files):
        nmodes = As[i].shape[-1] // 2
        Ps = np.abs(As[i]) ** 2
        Ps_pol = Ps[:, :, ::2] + Ps[:, :, 1::2]
        gain = Ps_pol / Ps0[i]
        average_gain.append(dB(gain.mean(axis=0)))

        std_ = np.sqrt(np.mean(Ps_pol ** 2, axis=0) / Ps0[i] ** 2 - 1)
        std.append(std)

        if nmodes == 3:
            # get the total power on the LP01 and LP11 groups
            Ps_LP01 = Ps_pol[:, :, 0]
            Ps_LP11 = Ps_pol[:, :, 1] + Ps_pol[:, :, 2]
            gain_LP01 = Ps_LP01 / Ps0[i]
            gain_LP11 = Ps_LP11 / (Ps0[i] + Ps0[i])
            gain_groups = np.stack([(gain_LP01), (gain_LP11)], axis=-1)
            gain_family.append(dB(gain_groups.mean(axis=0)))

            std_01 = np.sqrt(np.mean(Ps_LP01 ** 2, axis=0) / Ps0[i] ** 2 - 1)
            std_11 = np.sqrt(np.mean(Ps_LP11 ** 2, axis=0) / (2 * Ps0[i]) ** 2 - 1)

            std_family.append(np.stack([std_01, std_11], axis=-1))

        elif nmodes == 6:
            Ps_LP11 = Ps_pol[:, :, 1] + Ps_pol[:, :, 2]
            gain_LP11 = Ps_LP11 / (Ps0[i] + Ps0[i])

            Ps_01_02_21 = (
                Ps_pol[:, :, 0] + Ps_pol[:, :, 3] + Ps_pol[:, :, 4] + Ps_pol[:, :, 5]
            )
            gain_01_02_21 = Ps_01_02_21 / (Ps0[i] + Ps0[i] + Ps0[i] + Ps0[i])
            gain_groups = np.stack([gain_LP11, gain_01_02_21], axis=-1)
            gain_family.append(dB(gain_groups.mean(axis=0)))

            std_11 = np.sqrt(np.mean(Ps_LP11 ** 2, axis=0) / (2 * Ps0[i]) ** 2 - 1)
            std_01_02_21 = np.sqrt(
                np.mean(Ps_01_02_21 ** 2, axis=0)
                / (Ps0[i] + Ps0[i] + Ps0[i] + Ps0[i]) ** 2
                - 1
            )

            std_family.append(np.stack([std_11, std_01_02_21], axis=-1))

    average_gain = np.stack(average_gain)
    std = np.stack(std)

    if nmodes == 3 or nmodes == 6:
        average_gain_family = np.stack(gain_family)
        std_family = np.stack(std_family)

        return z, Lk, average_gain, std, average_gain_family, std_family
    else:
        return z, Lk, average_gain, std


# check if passed parameter is a directory
if os.path.isdir(args.directory):

    try:
        z, Lk, mean, std, mean_family, std_family = compute_gain_statistics(
            args, index=None
        )
        # save arrays in numpy file
        if args.save_data:
            data_file = os.path.join(args.directory, "data.npz")
            np.savez(
                data_file,
                z=z,
                Lk=Lk,
                average_gain=mean,
                std=std,
                average_gain_family=mean_family,
                std_family=std_family,
            )
    except ValueError:
        z, Lk, mean, std = compute_gain_statistics(args, index=None)

        # save arrays in numpy file
        if args.save_data:
            data_file = os.path.join(args.directory, "data.npz")
            np.savez(
                data_file,
                z=z,
                Lk=Lk,
                average_gain=mean,
                std=std,
            )

else:
    # read from npz data file
    data = np.load(args.directory)
    args.directory = os.path.dirname(args.directory)

    z = data["z"]
    Lk = data["Lk"]
    mean = data["average_gain"]
    std = data["std"]

    try:
        mean_family = data["average_gain_family"]
        std_family = data["std_family"]
    except KeyError:
        mean_family = None
        std_family = None


def get_mode_names(num_modes):
    """From the number of spatial modes, generate the appropriate mode names."""

    if num_modes == 3:
        return ["LP01", "LP11a", "LP11b"]
    elif num_modes == 6:
        return ["LP01", "LP11a", "LP11b", "LP21a", "LP21b", "LP02"]
    else:
        raise ValueError("Invalid number of modes.")


def get_markers(num_modes):
    markers = ["o", "s", "x", "^", ".", "D", "*", ">", "<"]
    return markers[:num_modes]


nmodes = mean.shape[-1]
mode_labels = get_mode_names(nmodes)
markers = get_markers(nmodes)
colors = [f"C{m}" for m in range(nmodes)]

mode_handles = [
    lines.Line2D(
        [],
        [],
        color=colors[x],
        marker=markers[x],
        label=mode_labels[x],
        fillstyle="none",
    )
    for x in range(nmodes)
]


if args.lengths is None:
    args.lengths = [z[-1] / 1000]

args.lengths = [x if x > 0 else z[-1] / 1000 for x in args.lengths]
args.lengths = np.array(args.lengths)

dz = z[1] - z[0]
idx = np.round(args.lengths * 1000 / dz).astype(int)
actual_lengths = idx * dz / 1000

if mean_family is not None:

    mode_family_labels = ["LP01", "LP11"]
    if nmodes == 6:
        mode_family_labels = ["LP11", "LP01 + LP21 + LP02"]

    mode_handles_family = [
        lines.Line2D(
            [],
            [],
            color=colors[x],
            marker=markers[x],
            label=y,
            fillstyle="none",
        )
        for x, y in enumerate(mode_family_labels)
    ]

FIGSIZE = (8, 7)
for i, length in zip(idx, actual_lengths):

    plt.minorticks_on()
    fig, axs = plt.subplots(
        nrows=2, sharex=True, num=f"{length:.2f}km", figsize=FIGSIZE
    )
    for m in range(nmodes):
        color = colors[m]
        marker = markers[m]
        axs[0].minorticks_on()
        axs[0].semilogx(
            Lk,
            mean[:, i, m].squeeze(),
            color=color,
            marker=marker,
            fillstyle="none",
            markersize=6,
        )
        plt.minorticks_on()
        axs[0].set_ylabel(r"$\langle G \rangle$ [dB]")

        axs[1].semilogx(
            Lk,
            100 * std[:, i, m].squeeze(),
            std[:, i, m].squeeze(),
            color=color,
            marker=marker,
            fillstyle="none",
            markersize=6,
        )
        # axs[1].set_ylabel(r"$\sigma_G$ [\%]")
        axs[1].set_ylabel(r"$\sigma_G$ [dB]")
        axs[1].set_xlabel(r"$L_{\kappa}$ [m]")
    axs[0].legend(
        handles=mode_handles,
        ncol=int(nmodes // 2),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
    )

    if args.save:
        ext = "pdf" if args.tex else "png"
        output_file = os.path.join(args.directory, f"plot_{length}km.{ext}")
        plt.savefig(output_file, bbox_inches="tight")

    if mean_family is not None:
        fig, axs = plt.subplots(
            nrows=2, sharex=True, num=f"family, {length:.2f}km", figsize=FIGSIZE
        )
        for m in range(mean_family.shape[-1]):
            color = colors[m]
            marker = markers[m]
            axs[0].minorticks_on()
            axs[0].semilogx(
                Lk,
                mean_family[:, i, m].squeeze(),
                color=color,
                marker=marker,
                fillstyle="none",
                markersize=6,
            )
            plt.minorticks_on()
            axs[0].set_ylabel(r"$\langle G \rangle$ [dB]")

            axs[1].semilogx(
                Lk,
                100 * std_family[:, i, m].squeeze(),
                std_family[:, i, m].squeeze(),
                color=color,
                marker=marker,
                fillstyle="none",
                markersize=6,
            )
            # axs[1].set_ylabel(r"$\sigma_G$ [\%]")
            axs[1].set_ylabel(r"$\sigma_G$ [dB]")
            axs[1].set_xlabel(r"$L_{\kappa}$ [m]")

        axs[0].legend(
            handles=mode_handles_family,
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
        )

        plt.tight_layout()

        if args.save:
            ext = "pdf" if args.tex else "png"
            output_file = os.path.join(args.directory, f"plot_family_{length}km.{ext}")
            plt.savefig(output_file, bbox_inches="tight")

if not args.save:
    plt.show()

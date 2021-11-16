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

import polarization


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

args = parser.parse_args()


def compute_gain_statistics(args, index=None):
    As, Ap, z, Lk, Ps0 = read_from_files(args, index=index)
    num_files = len(As)
    std = []
    average_gain = []
    for i in range(num_files):
        Ps = np.abs(As[i]) ** 2
        Ps_pol = Ps[:, ::2] + Ps[:, 1::2]
        gain = dB(Ps_pol / Ps0[i])
        average_gain.append(gain.mean(axis=0))
        std.append(gain.std(axis=0))

    average_gain = np.stack(average_gain)
    std = np.stack(std)
    return Lk, average_gain, std


# check if passed parameter is a directory
if os.path.isdir(args.directory):
    Lk, mean, std = compute_gain_statistics(args, index=-1)

    # save arrays in numpy file
    if args.save_data:
        data_file = os.path.join(args.directory, "data.npz")
        np.savez(data_file, Lk=Lk, average_gain=mean, std=std)
else:
    # read from npz data file
    data = np.load(args.directory)
    args.directory = os.path.dirname(args.directory)

    Lk = data["Lk"]
    mean = data["average_gain"]
    std = data["std"]


def get_mode_names(num_modes):
    """From the number of spatial modes, generate the appropriate mode names."""

    if num_modes == 3:
        return ["LP01", "LP11a", "LP11b"]
    elif num_modes == 6:
        return ["LP01", "LP11a", "LP11b", "LP21a", "LP21b", "LP02"]
    else:
        raise ValueError("Invalid number of modes.")


def get_markers(num_modes):
    markers = ["o", "s", "x", "^", "D", "*", "v", ">", "<"]
    return markers[:num_modes]


nmodes = mean.shape[-1]
mode_labels = get_mode_names(nmodes)
markers = get_markers(nmodes)
colors = [f"C{m}" for m in range(nmodes)]

mode_handles = [
    lines.Line2D([], [], color=colors[x], marker=markers[x], label=mode_labels[x])
    for x in range(nmodes)
]

fig, axs = plt.subplots(nrows=2, sharex=True)
for m in range(nmodes):
    color = colors[m]
    marker = markers[m]
    axs[0].semilogx(
        Lk, mean[:, m].squeeze(), color=color, marker=marker, fillstyle="none"
    )
    axs[0].set_ylabel(r"$\langle G \rangle$ [dB]")

    axs[1].semilogx(
        Lk, std[:, m].squeeze(), color=color, marker=marker, fillstyle="none"
    )
    axs[1].set_ylabel(r"$\sigma_G$ [dB]")
    axs[1].set_xlabel(r"$L_{\kappa}$ [m]")
axs[0].legend(handles=mode_handles)
plt.tight_layout()


if args.save:
    output_file = os.path.join(args.directory, "plot.png")
    plt.savefig(output_file, dpi=500)
else:
    plt.show()

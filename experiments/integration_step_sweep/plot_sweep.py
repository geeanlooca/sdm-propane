
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
    data_files = [os.path.join(args.directory, f) for f in filenames if f.endswith(".h5")]
    return data_files

def get_data(args):
    data_files = find_files(args)
    As = []
    Lk = []
    dz = []
    z : np.array

    if len(data_files) == 0:
        raise ValueError("Empty directory")

    for f in tqdm.tqdm(data_files):
        try:
            data = read_data(f)
            As.append(data['signal'])
            z = data['z']
            Lk.append( data['Lk'] )
            dz.append(data['dz'])
        except:
            print(f"Error while reading {f}")

    xy = sorted(zip(dz, As))
    dz = [x for x, y in xy]
    As = [y for x, y in xy]

    return As, z, np.array(dz), Lk

def dBm(x):
    return 10 * np.log10(x * 1e3)

def dB(x):
    return 10 * np.log10(x)


parser = argparse.ArgumentParser()
parser.add_argument("directory")
args = parser.parse_args()
As, z, dz, Lk = get_data(args)

num_files = len(As)

std = []
average_power = []

for x in range(num_files):
    Ps = np.abs(As[x] ** 2)
    Ps_pol = (Ps[:, -1, ::2] + Ps[:,-1, 1::2])
    average_power.append(dBm(Ps_pol.mean(axis=0)))
    std.append(dBm(Ps_pol).std(axis=0))


average_power = np.stack(average_power)
std = np.stack(std)




nmodes = average_power.shape[-1]
mode_labels = ["LP01", "LP11a", "LP11b"]
markers = ["o", 's', 'x', '^']
colors = [f"C{x}" for x in range(nmodes)]
mode_handles = [lines.Line2D([], [], color='k', marker=markers[x], label=mode_labels[x]) for x in range(nmodes)]


fig, axs = plt.subplots(nrows=2, sharex=True)

fiber_length = z[-1] * 1e-3

for m in range(nmodes):
    color = colors[m]
    marker = markers[m]
    axs[0].semilogx(dz, average_power[:, m].squeeze(), color=color, marker=marker, fillstyle='none')

    axs[1].semilogx(dz, std[:, m].squeeze(), color=color, marker=marker, fillstyle='none')

axs[0].set_ylabel(r"$\langle P \rangle$ [dBm]")
axs[1].set_ylabel(r"$\sigma$ [dBm]")
axs[1].set_xlabel(r"$dz$ [m]")
axs[1].legend(mode_labels)
plt.suptitle(rf"$L_{{\kappa}} = {Lk[0]}$ m, $L = {fiber_length}$ km")
plt.tight_layout()
plt.show()

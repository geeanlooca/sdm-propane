
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from read_data import read_data


def find_files(args):
    filenames = os.listdir(args.directory)
    data_files = [os.path.join(args.directory, f) for f in filenames if f.endswith(".h5")]
    return data_files

def get_data(args):
    data_files = find_files(args)
    As = []
    Lk = []
    

    for f in data_files:
        data = read_data(f)
        As.append(data['signal'])
        z = data['z']
        Lk.append( data['Lk'] )

    xy = sorted(zip(Lk, As))
    Lk = [x for x, y in xy]
    As = [y for x, y in xy]

    return As, z, np.array(Lk)

def dBm(x):
    return 10 * np.log10(x * 1e3)

def dB(x):
    return 10 * np.log10(x)


parser = argparse.ArgumentParser()
parser.add_argument("directory")
args = parser.parse_args()
As, z, Lk = get_data(args)

num_files = len(As)



std = []
average_power = []

for x in range(num_files):
    Ps = np.abs(As[x] ** 2)
    Ps_pol = (Ps[:, :, ::2] + Ps[:,:, 1::2])
    average_power.append(dBm(Ps_pol.mean(axis=0)))
    std.append(dBm(Ps_pol).std(axis=0))


average_power = np.stack(average_power)
std = np.stack(std)


lengths = np.array([10, 25, 50])
dz = z[1] - z[0]

idx = lengths * 1e3 / dz


nlenghts = len(lengths)
nmodes = average_power.shape[-1]
mode_labels = ["LP01", "LP11a", "LP11b"]
length_labels = [f"{length} km" for length in lengths]
markers = ["o", 's', 'x', '^']
colors= [ f"C{m}" for m in range(nlenghts) ] 
length_handles = [ patches.Patch(color=colors[l], label=length_labels[l]) for l in range(nlenghts)]
mode_handles = [lines.Line2D([], [], color='k', marker=markers[x], label=mode_labels[x]) for x in range(nmodes)]


fig, axs = plt.subplots(nrows=2, sharex=True)

for x, id in enumerate(idx):
    color = colors[x]
    for m in range(nmodes):
        marker = markers[m]
        axs[0].semilogx(Lk, average_power[:, int(id), m].squeeze(), color=color, marker=marker, fillstyle='none')
        axs[0].set_ylabel("Power [dBm]")

        axs[1].semilogx(Lk, std[:, int(id), m].squeeze(), color=color, marker=marker, fillstyle='none')
        axs[1].set_ylabel("Power [dBm]")
        axs[1].set_xlabel(r"$L_{\kappa}$ [m]")
axs[0].legend(handles=mode_handles + length_handles, ncol=3, loc="upper left")
plt.tight_layout()
plt.show()

import sys

sys.path.append("/home/gianluca/sdm-propane")

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
    data_files = [os.path.join(args.directory, f) for f in filenames if f.endswith(".h5")]
    return data_files

def get_data(args):
    data_files = find_files(args)
    As = []
    Ap = []
    Lk = []
    

    for f in tqdm.tqdm(data_files):
        try:
            data = read_data(f)
            As.append(data['signal'])
            Ap.append(data['pump'])
            z = data['z']
            Lk.append( data['Lk'] )
        except:
            print(f"Error reading {f}")

    xy = sorted(zip(Lk, As))
    Lk = [x for x, y in xy]
    As = [y for x, y in xy]

    return As, Ap, z, np.array(Lk)

def dBm(x):
    return 10 * np.log10(x * 1e3)

def dB(x):
    return 10 * np.log10(x)


default_filename="/home/gianluca/sdm-propane/experiments/coupling_strength_sweep/random_polarizations/results/dz_1m_L_50km/"
parser = argparse.ArgumentParser()
parser.add_argument("directory", nargs='?', default=default_filename)
args = parser.parse_args()
As, Ap, z, Lk = get_data(args)

num_files = len(As)


def compute_angle(a, b, axis=0):
    norm_a = np.linalg.norm(a, axis=axis)
    norm_b = np.linalg.norm(b, axis=axis)
    ab = np.sum(a * b, axis=axis)
    return ab / (norm_a * norm_b)


std = []
average_power = []

for x in range(num_files):
    Ps = np.abs(As[x] ** 2)
    Ps_pol = (Ps[:, :, ::2] + Ps[:,:, 1::2])
    average_power.append(dBm(Ps_pol.mean(axis=0)))
    std.append(dBm(Ps_pol).std(axis=0))


average_power = np.stack(average_power)
std = np.stack(std)

As = np.stack(As)
Ap = np.stack(Ap)

S_s = polarization.hyperjones_to_hyperstokes(As, axis=-1)
S_p = polarization.hyperjones_to_hyperstokes(Ap, axis=-1)

angle = compute_angle(S_s, S_p, axis=-1)
angle_avg = angle.mean(axis=1)

plt.figure()
plt.plot(z * 1e-3, angle_avg[0], label=rf"$L_{{\kappa}} = {Lk[0]}$ m")
plt.plot(z * 1e-3, angle_avg[-1], label=rf"$L_{{\kappa}} = {Lk[-1]}$ m")
plt.xlabel(r"$z$ [km]")
plt.ylabel(r"$\langle \cos\theta \rangle$") 
plt.ylim((-1, 1))
plt.legend()
plt.tight_layout()


lengths = np.array([10])
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
        axs[0].set_ylabel(r"$\langle P \rangle$ [dBm]")

        axs[1].semilogx(Lk, std[:, int(id), m].squeeze(), color=color, marker=marker, fillstyle='none')
        axs[1].set_ylabel(r"$\sigma$ [dBm]")
        axs[1].set_xlabel(r"$L_{\kappa}$ [m]")
axs[0].legend(handles=mode_handles + length_handles, ncol=3, loc="upper left")
plt.tight_layout()
plt.show()

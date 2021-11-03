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

def get_data(args, index=None):
    data_files = find_files(args)
    As = []
    Ap = []
    Lk = []
    Ps0 = []

    for f in tqdm.tqdm(data_files):
        try:
            data = read_data(f, idx=index)
            As.append(data['signal'])
            Ap.append(data['pump'])
            Ps0.append(data['Ps0'] * 1e-3)
            z = data['z']
            Lk.append( data['Lk'] )
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


default_filename="/home/gianluca/sdm-propane/experiments/coupling_strength_sweep/random_polarizations/results/dz_1m_L_50km/"
parser = argparse.ArgumentParser()
parser.add_argument("directory", nargs='?', default=default_filename)
args = parser.parse_args()
As, Ap, z, Lk, Ps0 = get_data(args, index=-1)

num_files = len(As)


def compute_angle(a, b, axis=0):
    norm_a = np.linalg.norm(a, axis=axis)
    norm_b = np.linalg.norm(b, axis=axis)
    ab = np.sum(a * b, axis=axis)
    return ab


std = []
average_gain = []

for x in range(num_files):
    Ps = np.abs(As[x] ** 2)
    Ps_pol = (Ps[:, ::2] + Ps[:, 1::2])
    gain = dB(Ps_pol / Ps0[x])
    average_gain.append(gain.mean(axis=0))
    std.append(gain.std(axis=0))


average_gain = np.stack(average_gain)
std = np.stack(std)

# S_s = polarization.hyperjones_to_hyperstokes(As, axis=-1)
# S_p = polarization.hyperjones_to_hyperstokes(Ap, axis=-1)

# S_s_01 = S_s[:, :, :, (0,1,2)]
# S_p_01 = S_p[:, :, :, (0,1,2)]
# S_s_11a = S_s[:, :, :, (3,4,5)]
# S_p_11a = S_p[:, :, :, (3,4,5)]
# S_s_11b = S_s[:, :, :, (6,7,8)]
# S_p_11b = S_p[:, :, :, (6,7,8)]

# angle = compute_angle(S_s_11b, S_p_11b, axis=-1)
# angle_avg = angle.mean(axis=1)

# plt.figure()

# for x in range(len(Lk)):
#     plt.plot(z * 1e-3, angle_avg[x], label=rf"$L_{{\kappa}} = {Lk[x]}$ m")
# plt.xlabel(r"$z$ [km]")
# plt.ylabel(r"$\langle \cos\theta \rangle$") 
# plt.ylim((-1, 1))
# plt.legend()
# plt.tight_layout()


# lengths = np.array([10])
# dz = z[1] - z[0]

# idx = lengths * 1e3 / dz


nmodes = average_gain.shape[-1]
mode_labels = ["LP01", "LP11a", "LP11b"]
markers = ["o", 's', 'x', '^']
colors= [ f"C{m}" for m in range(nmodes) ] 
mode_handles = [lines.Line2D([], [], color=colors[x], marker=markers[x], label=mode_labels[x]) for x in range(nmodes)]


fig, axs = plt.subplots(nrows=2, sharex=True)

for m in range(nmodes):
    color = colors[m]
    marker = markers[m]
    axs[0].semilogx(Lk, average_gain[:, m].squeeze(), color=color, marker=marker, fillstyle='none')
    axs[0].set_ylabel(r"$\langle G \rangle$ [dB]")

    axs[1].semilogx(Lk, std[:, m].squeeze(), color=color, marker=marker, fillstyle='none')
    axs[1].set_ylabel(r"$\sigma_G$ [dB]")
    axs[1].set_xlabel(r"$L_{\kappa}$ [m]")
axs[0].legend(handles=mode_handles)
plt.tight_layout()

plt.show()
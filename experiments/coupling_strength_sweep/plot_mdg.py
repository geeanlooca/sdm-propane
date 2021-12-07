import argparse
import os

import itertools

import numpy as np
import matplotlib.pyplot as plt

image_dir = os.path.join(
    os.path.expanduser("~"), "thesis", "thesis", "Chapters", "linear_coupling", "images"
)


parser = argparse.ArgumentParser()
parser.add_argument("directory")
parser.add_argument("-s", "--save", action="store_true")
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
fiber_path = "/home/gianluca/sdm-propane/fibers"

args = parser.parse_args()
directory = args.directory

if not os.path.isdir(directory):
    raise ValueError("Directory does not exist")


data = np.load(os.path.join(directory, "data.npz"))


z = data["z"]
Lk = data["Lk"]
mean = data["average_gain"]
std = data["std"]


print(Lk.shape)
print(z.shape)
print(mean.shape)

# modes for which to compute the gain difference
modes_idx = [(1, 2)]

mdg = np.zeros((mean.shape[0], mean.shape[1], len(modes_idx)))

for x, (m1, m2) in enumerate(modes_idx):
    mdg[:, :, x] = np.abs(mean[:, :, m1] - mean[:, :, m2])


markers = [".", "o", "x"]
Lk_idx = [-1]

plt.figure()

for x in range(mdg.shape[-1]):
    for y in Lk_idx:

        exp = np.log10(Lk[y])
        plt.plot(
            z * 1e-3,
            mdg[y, :, x].squeeze(),
            marker=markers[x],
            label=fr"$L_{{\kappa}} = 10^{{{exp:.2f}}}$ m",
        )
plt.xlabel(r"$z$ [km]")
plt.ylabel(r"$\Delta$G [dB]")
plt.legend()
plt.tight_layout()

if args.save:
    plt.savefig(
        os.path.join(image_dir, f"mdg_2modes_ellip_para.pdf"), bbox_inches="tight"
    )


plt.figure()
plt.plot(z * 1e-3, std[-10, :, :].squeeze())
plt.plot(z * 1e-3, std[-20, :, :].squeeze())
plt.plot(z * 1e-3, std[-30, :, :].squeeze())
plt.xlabel(r"$z$ [km]")
plt.ylabel(r"$\sigma_G$ [dB]")


plt.show()
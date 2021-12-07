import argparse
import sys

sys.path.append("/home/gianluca/sdm-propane")

import os
from scipy.constants import epsilon_0 as e0
from fiber import StepIndexFiber
from scipy.constants import Planck as hp, lambda2nu
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1)

FIG_SIZE = (9, 4)

image_dir = os.path.join(
    os.path.expanduser("~"), "thesis", "thesis", "Chapters", "linear_coupling", "images"
)

parser = argparse.ArgumentParser()
parser.add_argument("-a", default=0.5, type=float)
parser.add_argument("-t", "--tex", action="store_true")
parser.add_argument("-m", "--modes", default=2, type=int)
parser.add_argument("-s", "--save", action="store_true")

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


print(image_dir)
clad_index = 1.46 if args.modes == 2 else 1.4545
fiber = StepIndexFiber(
    clad_index=clad_index,
    delta=0.005,
    core_radius=6,
    clad_radius=60,
    data_path=fiber_path,
)
signal_wavelength = 1550

signal_freq = lambda2nu(signal_wavelength * 1e-9)

fiber.load_data(wavelength=signal_wavelength)


Ke = np.real(fiber.core_ellipticity_coupling_matrix())

Ke = Ke / np.max(np.abs(Ke))
Kb = np.real(fiber.birefringence_coupling_matrix())
Kb = Kb / np.max(np.abs(Kb))


nmodes = Ke.shape[0]

print(fiber.group_orders())
degen = fiber.group_degeneracies()

degen_ = np.cumsum(degen)
degen_ = np.insert(degen_, 0, 0)
print(degen_)

plt.set_cmap("inferno")

plt.figure(figsize=FIG_SIZE)
plt.subplot(121)
plt.imshow(np.log10(np.abs(Kb)))
ax = plt.gca()

ax.set_xticks(degen_ - 0.5)
ax.set_yticks(degen_ - 0.5)
ax.set_xticklabels(["" for _ in range(len(degen_))])
ax.set_yticklabels(["" for _ in range(len(degen_))])

minors = degen_[:-1] + np.array(degen) / 2
print(minors)

ax.set_xticks(minors - 0.5, minor=True)
ax.set_yticks(minors - 0.5, minor=True)
ax.xaxis.set_ticks_position("none")
ax.yaxis.set_ticks_position("none")
ax.xaxis.set_minor_formatter(plt.FixedFormatter(fiber.group_names()))
ax.yaxis.set_minor_formatter(plt.FixedFormatter(fiber.group_names()))
ax.xaxis.set_tick_params(which="minor", pad=15)
ax.yaxis.set_tick_params(which="minor", pad=15)

plt.grid(True, color="#000000", which="major")
plt.grid(False, which="minor")
plt.colorbar()
plt.title("(a)", pad=20)


###############################################

plt.subplot(122)
plt.imshow(np.log10(np.abs(Ke)))
ax = plt.gca()

ax.set_xticks(degen_ - 0.5)
ax.set_yticks(degen_ - 0.5)
ax.set_xticklabels(["" for _ in range(len(degen_))])
ax.set_yticklabels(["" for _ in range(len(degen_))])

minors = degen_[:-1] + np.array(degen) / 2
print(minors)

ax.set_xticks(minors - 0.5, minor=True)
ax.set_yticks(minors - 0.5, minor=True)
ax.xaxis.set_ticks_position("none")
ax.yaxis.set_ticks_position("none")
ax.xaxis.set_minor_formatter(plt.FixedFormatter(fiber.group_names()))
ax.yaxis.set_minor_formatter(plt.FixedFormatter(fiber.group_names()))
ax.xaxis.set_tick_params(which="minor", pad=15)
ax.yaxis.set_tick_params(which="minor", pad=15)

plt.grid(True, color="#000000", which="major")
plt.grid(False, which="minor")
plt.colorbar()
plt.title("(b)", pad=20)
plt.tight_layout()

if args.save:
    plt.savefig(
        os.path.join(image_dir, f"{args.modes}modes_matrix.pdf"), bbox_inches="tight"
    )

plt.show()
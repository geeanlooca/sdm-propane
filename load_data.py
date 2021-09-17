import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams as rc
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.constants import speed_of_light as c0, epsilon_0 as e0, lambda2nu, nu2lambda

rc["font.family"] = "Arial"
plt.style.use("seaborn-dark")
plt.style.use("ggplot")
# rc["axes.prop_cycle"] = plt.cycler(color=plt.cm.Dark2.colors)
colors = rc["axes.prop_cycle"].by_key()["color"]

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--save", action="store_true")
args = parser.parse_args()

IMG_DIR = os.path.join("..", "images")
data = np.load("raman_data.npz")
frequency_shift = data["frequency"]
a_response = data["a"]
b_response = data["b"]


n2 = 2.18e-20
n = 1.46

sigma_weight = 3 / 2 + 2 * 0.05 + 2 * 4.3 / 28
sigma = n2 * 4 * c0 * e0 * n ** 2 / sigma_weight
b0 = 0.05 * sigma
a0 = 4.3 / 28 * sigma

print("sigma=", sigma, "m2/V2")
print("a0=", a0, "m2/V2")
print("b0=", b0, "m2/V2")

a_response[0] = np.real(a_response[0]) + 1j * 0
b_response[0] = np.real(b_response[0]) + 1j * 0

# set values of response in Omega=0
a_response = a_response / np.real(a_response[0]) * a0
b_response = b_response / np.real(b_response[0]) * b0

wavelength = 795.5e-9
freqs = lambda2nu(wavelength) + frequency_shift * 1e12

gain_coefficient = (
    2 * np.pi * (freqs) / (n * e0 * c0 ** 2) * np.imag(a_response + b_response)
)

fix, axs = plt.subplots(ncols=2, figsize=(9, 4))
axs[0].plot(frequency_shift, np.real(a_response), color="C0", linestyle="-")
axs[0].plot(frequency_shift, np.imag(a_response), color="C0", linestyle="--")
axs[0].plot(frequency_shift, np.real(b_response), color="C1", linestyle="-")
axs[0].plot(frequency_shift, np.imag(b_response), color="C1", linestyle="--")
axs[0].set_xlabel("Frequency shift [THz]")

handles = [
    Patch(facecolor="C0", label="$\\tilde{a}$"),
    Patch(facecolor="C1", label="$\\tilde{b}$"),
    Line2D([0], [0], color="k", linestyle="-", label="Real"),
    Line2D([0], [0], color="k", linestyle="--", label="Imag."),
]
axs[0].legend(handles=handles)  # , loc="lower center", ncol=len(handles))

axs[1].plot(frequency_shift, gain_coefficient * 1e2, color="C0", linestyle="-")
axs[1].set_xlabel("Frequency shift [THz]")
axs[1].set_ylabel("Gain coefficient [cm/W]")

text_str = "\n".join(
    (
        f"$\sigma = {sigma * 1e22:.2f} \\times 10^{{-22}} m^2/V^2$",
        f"$\\tilde{{a}}_0 = {a0 * 1e22:.2f} \\times 10^{{-22}} m^2/V^2$",
        f"$\\tilde{{b}}_0 = {b0 * 1e22:.2f} \\times 10^{{-22}} m^2/V^2$",
        f"$n_2 = {n2 * 1e20:.2f} \\times 10^{{-20}} m^2/W$",
    )
)

axs[1].text(
    0.43,
    0.95,
    text_str,
    fontsize=11,
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox={"edgecolor": "none", "facecolor": "none"},
)

args.save and plt.savefig(
    os.path.join(IMG_DIR, "gain_coefficient_scaled_response_set_n2.pdf")
)

plt.show()

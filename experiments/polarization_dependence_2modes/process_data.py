#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%%

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()
filename = args.filename

mode_names = ["LP01", "LP11a", "LP11b"]
pol_names = ["LP01x", "LP01y", "LP11ax", "LP011ay", "LP11bx", "LP11by"]

f = h5py.File(filename, "r")
z = f["z"][:]

keys = f.keys()
batch_keys = list(filter(lambda x : x.startswith("batch-"), keys))

As = [ f[k]['signal'][:] for k in batch_keys]

As = np.vstack(As)

#%%

def dBm(x):
    return 10 * np.log10(x * 1e3)

def dB(x):
    return 10 * np.log10(x)

#%%
As_avg = np.mean(As, axis=0)
Ps = np.abs(As) ** 2 
Ps_avg = Ps.mean(axis=0)

# average over the polarizations
Ps_avg_x = Ps_avg[:, ::2]
Ps_avg_y = Ps_avg[:, 1::2]
Ps_avg_pol = (Ps_avg_x + Ps_avg_y)
Ps_pol = (Ps[:, :, ::2] + Ps[:,:, 1::2])
Ps_pol_std = dBm(Ps_pol).std(axis=0)



#%%

# Statistics on total power of the spatial mode (addin the two polarizations)

plt.figure()

above = dBm(Ps_pol.mean(axis=0) + Ps_pol.std(axis=0))
below = dBm(Ps_pol.mean(axis=0) - Ps_pol.std(axis=0))
plt.plot(z * 1e-3, dBm(Ps_pol.mean(axis=0)))
for x in range(Ps_avg_pol.shape[-1]):
    plt.fill_between(z * 1e-3, below[:, x], above[:, x], color=f"C{x}", alpha=0.3, label=mode_names[x])
plt.xlabel("Position [km]")
plt.ylabel("Power [dBm]")
plt.title("Average power and standard dev. in each spatial mode")
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(z * 1e-3, Ps_pol_std)
plt.xlabel("Position [km]")
plt.ylabel("Standard deviation [dBm]")
plt.title("Power standard dev. in each spatial mode")
plt.legend(mode_names)
plt.tight_layout()

#%%

# Statistics on each polarization
plt.figure()

above = dBm(Ps.mean(axis=0) + Ps.std(axis=0))
below = dBm(Ps.mean(axis=0) - Ps.std(axis=0))
plt.plot(z * 1e-3, dBm(Ps.mean(axis=0)))

for x in range(Ps.shape[-1]):
    plt.fill_between(z * 1e-3, below[:, x], above[:, x], color=f"C{x}", alpha=0.3)
plt.xlabel("Position [km]")
plt.ylabel("Power [dBm]")
plt.title("Average power and standard dev. in each polarization")
plt.legend(pol_names)
plt.tight_layout()

plt.figure()
plt.plot(z * 1e-3, dBm(Ps).std(axis=0))
plt.xlabel("Position [km]")
plt.ylabel("Standard deviation [dBm]")
plt.title("Power standard dev. in each polarization")
plt.legend(pol_names)
plt.tight_layout()

# %% Difference in power between LP11a and LP11b

power_difference = np.abs((Ps_pol[:, :, 1] - Ps_pol[:,:,2]))
above = dB(power_difference.mean(axis=0) + power_difference.std(axis=0))
below = dB(power_difference.mean(axis=0) - power_difference.std(axis=0))

plt.figure()
plt.plot(z * 1e-3, dB(power_difference.mean(axis=0)))
plt.fill_between(z * 1e-3, below, above, color=f"C0", alpha=0.3)
plt.xlabel("Position [km]")
plt.ylabel("Power difference [dB]")
plt.title("Difference in power between LP11a and LP11b")
plt.tight_layout()

# %% Histogram on the total power of each spatial mode at the end of the fiber

plt.figure()

for x in range(Ps_pol.shape[-1]):
    n,bins, patchs = plt.hist(dBm(Ps_pol[:, -1, x]), 30, density=True, alpha=0.5, label=mode_names[x])
plt.xlabel("Power [dBm]")
plt.ylabel("Probability")
plt.title("Power in each spatial mode, at fiber end")
plt.legend()
plt.tight_layout()
# %%

# %%

plt.show()


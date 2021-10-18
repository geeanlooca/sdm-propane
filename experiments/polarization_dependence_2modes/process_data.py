#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%%

filename = "results-2021-10-15T22:02:32.h5"
filename = "results-total_pump_power=1000.0;signal_power_per_group=0.001;fiber_length=5.0;dz=0.1;correlation_length=50.0;perturbation_beat_length=10.0;n2=4e-20;gR=1e-13;fiber_seed=None;numpy_seed=0;sampling=50;batches=500;runs=4-2021-10-16T04:01:02.h5"

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
    plt.fill_between(z * 1e-3, below[:, x], above[:, x], color=f"C{x}", alpha=0.3)
plt.xlabel("Position [km]")
plt.ylabel("Power [dBm]")
plt.title("Average power and standard dev. in each spatial mode")
plt.tight_layout()

plt.figure()
plt.plot(z * 1e-3, Ps_pol_std)
plt.xlabel("Position [km]")
plt.ylabel("Standard deviation [dBm]")
plt.title("Power standard dev. in each spatial mode")
plt.tight_layout()

#%%

# Statistics on each polarization
plt.figure()

above = dBm(Ps.mean(axis=0) + Ps.std(axis=0))
below = dBm(Ps.mean(axis=0) - Ps.std(axis=0))
plt.plot(z * 1e-3, dBm(Ps.mean(axis=0)))

for x in range(Ps_avg_pol.shape[-1]):
    plt.fill_between(z * 1e-3, below[:, x], above[:, x], color=f"C{x}", alpha=0.3)
plt.xlabel("Position [km]")
plt.ylabel("Power [dBm]")
plt.title("Average power and standard dev. in each polarization")
plt.tight_layout()

plt.figure()
plt.plot(z * 1e-3, dBm(Ps).std(axis=0))
plt.xlabel("Position [km]")
plt.ylabel("Standard deviation [dBm]")
plt.title("Power standard dev. in each polarization")
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
    n,bins, patchs = plt.hist(dBm(Ps_pol[:, -1, x]), 50, density=True, alpha=0.5)
plt.xlabel("Power [dBm]")
plt.ylabel("Probability")
plt.title("Power in each spatial mode, at fiber end")
plt.tight_layout()
# %%

# positions = [20, 40, 80, 100]

# nbins = 50
# for p in positions:
#     plt.figure()
#     for x in range(Ps_pol.shape[-1]):
#         n,bins, patchs = plt.hist(dBm(Ps_pol[:, p, x]), 50, density=False, alpha=0.5)
#     plt.xlabel("Power [dBm]")
#     plt.ylabel("Probability")
#     plt.title(f"{p*1e-3} km")
# %%

plt.show()
#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%%


def read_data(filename, idx=None):

    f = h5py.File(filename, "r")
    z = f["z"][:]

    keys = f.keys()
    batch_keys = list(filter(lambda x : x.startswith("batch-"), keys))

    if idx:
        As = [f[k]['signal'][:, idx, :] for k in batch_keys]
        Ap = [f[k]['pump'][:, idx, :] for k in batch_keys]
    else:
        As = [f[k]['signal'][:] for k in batch_keys]
        Ap = [f[k]['pump'][:] for k in batch_keys]

    As = np.vstack(As)
    Ap = np.vstack(Ap)

    Lk = f['perturbation_beat_length'][()]
    Ps0 = f['signal_power_per_mode'][()]

    return {
        "signal" : As,
        "pump" : Ap,
        "z" : z,
        "Lk" : Lk,
        "Ps0" : Ps0
    }





#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%%


def read_data(filename):

    f = h5py.File(filename, "r")
    z = f["z"][:]

    keys = f.keys()
    batch_keys = list(filter(lambda x : x.startswith("batch-"), keys))

    As = [f[k]['signal'][:] for k in batch_keys]
    As = np.vstack(As)

    Ap = [f[k]['pump'][:] for k in batch_keys]
    Ap = np.vstack(Ap)

    Lk = f['perturbation_beat_length'][()]

    return {
        "signal" : As,
        "pump" : Ap,
        "z" : z,
        "Lk" : Lk
    }





import h5py

import numpy as np
import matplotlib.pyplot as plt



class OnlineMeanManager:
    def __init__(self, filename=None):
        self.mean = 0
        self.means = []
        self.std = 0
        self.stds = []
        self.num_obs = 0
        self.m2 = 0

        self.filename = filename
        if self.filename:
            with h5py.File(self.filename, 'a') as _file:
                try:
                    dset = _file.create_dataset("results", shape=(0,), maxshape=(None,), dtype="f")
                    dset = _file.create_dataset("mean", shape=(1,), maxshape=(None,), dtype="f")
                    dset = _file.create_dataset("num_observations", shape=(1,), maxshape=(None,), dtype="f")
                except ValueError:
                    pass

    
    def _append_data(self, dataset, new_data):
        dataset_size = dataset.shape[0]
        data_size = new_data.shape[0]
        dataset.resize(dataset_size + data_size, axis=0)
        dataset[-data_size:] = new_data
        print(dataset.shape)


    def update(self, data):
        b = len(data)
        m = self.num_obs * 1.0

        # update number of observations
        self.num_obs += b

        # new values minus old mean
        delta = data - self.mean

        # update the mean
        self.mean += np.sum(delta) / self.num_obs

        # new values minus new mean
        delta2 = data - self.mean
        
        # update residuals
        self.m2 += np.sum(delta * delta2)

        self.means.append(self.mean)


        variance = self.m2 / (self.num_obs - 1)
        self.std = np.sqrt(variance)
        self.stds.append(self.std)

        if self.filename:
            with h5py.File(self.filename, 'a') as _file:
                _file["mean"][:] = self.mean
                _file["num_observations"][:] = self.num_obs
                self._append_data(_file["results"], data)

        plt.figure()
        plt.plot(self.means, label="mean", marker="x")
        plt.plot(self.stds, label="std", marker="^")
        plt.legend()
        plt.ylabel("Mean")
        plt.xlabel("Batches")
        plt.savefig("mean-convergence.png")
        plt.close()

from copy import copy
import h5py

import numpy as np
import matplotlib.pyplot as plt



class OnlineMeanManager:
    def __init__(self, name=None):
        self.mean = 0
        self.means = []
        self.std = 0
        self.stds = []
        self.num_obs = 0
        self.m2 = 0
        self.name = name


    def update(self, data, axis=0, accumulate=None):
        b = data.shape[axis]

        m = self.num_obs * 1.0

        # update number of observations
        self.num_obs += b

        # new values minus old mean
        delta = data - self.mean

        # update the mean
        self.mean += np.sum(delta, axis=axis) / self.num_obs

        # new values minus new mean
        delta2 = data - self.mean
        
        # update residuals
        self.m2 += np.sum(delta * delta2, axis=0)


        variance = self.m2 / (self.num_obs - 1)
        self.std = np.sqrt(variance)

        if accumulate:
            self.means.append(copy(self.mean))
            self.stds.append(self.std)



    def plot(self, filename=None, **kwargs):

        data = np.array(self.means)

        plt.subplot(121)
        plt.cla()
        plt.plot(data, label="mean", marker="x")
        plt.ylabel("Mean")
        plt.xlabel("Batches")

        plt.subplot(122)
        plt.cla()
        plt.plot(self.stds, label="std", marker="^")
        plt.ylabel("Std")
        plt.xlabel("Batches")
        
        if self.name:
            plt.suptitle(self.name)

        plt.tight_layout()

        if filename:
            plt.savefig(filename, **kwargs)


import argparse
import multiprocessing
import random
import signal
import time

import matplotlib.pyplot as plt
import numpy as np
import h5py
import tqdm

from stats_manager import StatsManager

parser = argparse.ArgumentParser()
parser.add_argument("-B", "--batch", default=20, type=int)
parser.add_argument("-P", "--pump-power", default=500, type=float)
parser.add_argument("-p", "--signal-power", default=1, type=float)
parser.add_argument("-g", "--raman-gain", default=1e-17, type=float)
parser.add_argument("-n2", "--n2", default=4e-20, type=float)

args = parser.parse_args()

def build_filename(args):
    filename = ",".join([f"{k}={v}" for (k,v) in vars(args).items()])
    return filename + ".h5"

def simulate(x):
    time.sleep(0.1)
    return random.random() * 10


if __name__ == "__main__":
    manager = StatsManager("saving_results_test.h5")
    while True:
        # keep looping until external force intervenes
        pool = multiprocessing.Pool()
        inputs = range(args.batch)
        results = np.array(pool.map(simulate, inputs))

        manager.update(results)

        print("Estimate mean: ", manager.mean, "Std.", manager.std)

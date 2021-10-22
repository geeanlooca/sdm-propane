import argparse
import numpy as np
import h5py

def process_results(results, inputs, filename):

    runs = len(results)

    signal_sops = np.zeros((runs, 3*3), dtype=np.complex128)
    pump_sops = np.zeros_like(signal_sops)

    for i in range(runs):
        signal_sops[i] = inputs[i][0]
        pump_sops[i] = inputs[i][1]

    z = results[0][0]
    As = np.stack([ s for (_, s, _) in results])
    Ap = np.stack([ p for (_, _, p) in results])

    # save to hdf5 file
    with h5py.File(filename, "a") as f:

        if "total_batches" in f:
            batch_idx = f["total_batches"][()] + 1
        else:
            f["total_batches"] = 0
            batch_idx = 1



        signal_sop_dset = f.create_dataset(f"batch-{batch_idx}/signal_sops", dtype=np.complex128, shape=signal_sops.shape, compression="gzip")
        signal_sop_dset[:] = signal_sops
        pump_sop_dset = f.create_dataset(f"batch-{batch_idx}/pump_sops", dtype=np.complex128, shape=pump_sops.shape, compression="gzip")
        pump_sop_dset[:] = pump_sops

        # only save the positions once
        if batch_idx == 1:
            z_dset = f.create_dataset("z", dtype=np.float64, shape=z.shape, compression="gzip")
            z_dset[:] = z

        signal_dset = f.create_dataset(f"batch-{batch_idx}/signal", dtype=np.complex128, shape=As.shape, compression="gzip")
        pump_dset = f.create_dataset(f"batch-{batch_idx}/pump", dtype=np.complex128, shape=Ap.shape, compression="gzip")

        signal_dset[:] = As
        pump_dset[:] = Ap

        batches = f["total_batches"]
        batches[...] = batch_idx

    return z, As, Ap

def write_metadata(filename, experiment):
    # save to hdf5 file
    with h5py.File(filename, "a") as f:
        args = experiment.metadata()
        for (k,v) in args.items():
            key = str(k)
            if v and not key in f:
                f[key] = v



def cmd_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="", type=str)
    parser.add_argument("-P", "--total-pump-power", default=1000.0, type=float)
    parser.add_argument("-p", "--signal-power-per-mode", default=1e-3, type=float)
    parser.add_argument("-L", "--fiber-length", default=50, type=float)
    parser.add_argument("-d", "--dz", default=1, type=float)
    parser.add_argument("-Lc", "--correlation-length", default=10, type=float)
    parser.add_argument("-Lk", "--perturbation-beat-length", default=100, type=float)
    parser.add_argument("--n2", default=4e-20, type=float)
    parser.add_argument("--gR", default=1e-13, type=float)
    parser.add_argument("--fiber-seed", default=None, type=int)
    parser.add_argument("--numpy-seed", default=None, type=int)
    parser.add_argument("--sampling", default=100, type=int)
    parser.add_argument("-B", "--batches", default=1, type=int)
    parser.add_argument("-N", "--runs-per-batch", default=4, type=int)
    parser.add_argument("-f", "--forever", action="store_true", help="Run an infinite while loop")
    parser.add_argument("--sigma", default=None, help="Kerr parameter")
    parser.add_argument("--max-fibers", type=int)

    return parser


def build_params_string(args, fields, sep=';'):
    filtered_params = { k : v for k, v in vars(args).items() if k in fields}
    params_string = sep.join([f"{k}={v}" for (k,v) in filtered_params.items()])
    return params_string
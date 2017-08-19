import sys
import h5py
#import numpy as np


def info(filename):
    with h5py.File(filename, "r") as f:
        first = ("version", "model")
        for k in first:
            print("{} = {}".format(k, f["metadata"][k][...]))
        for k, v in f["metadata"].items():
            if k not in first:
                print("{} = {}".format(k, v[...]))
        for k in ("N", "L", "dt", "n_matmul", "n_delay", "n_sweep_warm",
                  "n_sweep_meas", "period_eqlt", "period_uneqlt"):
            print("{} = {}".format(k, f["params"][k][...]))


def main(argv):
    for path in argv[1:]:
        print(path)
        info(path)

if __name__ == "__main__":
    main(sys.argv)

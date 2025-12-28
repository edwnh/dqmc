import sys
import h5py
from glob import glob


def info(path):
    with h5py.File(path, "r") as f:
        first = ("version", "model")
        for k in first:
            print("{} = {}".format(k, f["metadata"][k][...]))
        for k, v in f["metadata"].items():
            if k not in first:
                print("{} = {}".format(k, v[...]))
        for k in ("N", "L", "dt", "n_matmul", "n_delay", "n_sweep_warm",
                  "n_sweep_meas", "period_eqlt", "period_uneqlt"):
            print("{} = {}".format(k, f["params"][k][...]))


def main(argv=None):
    if argv is None:
        argv = sys.argv
    for path_spec in argv[1:]:
        files = sorted(glob(path_spec))
        if len(files) == 0:
            print("No files matching:"+path_spec)
        else:
            for f in files:
                print(f)
                info(f)

if __name__ == "__main__":
    main()

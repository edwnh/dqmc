import sys
from glob import glob

from . import core


def info(path):
    n_sample, sign, density, double_occ, sweep, n_sweep = \
        core.load_file(path, "meas_eqlt/n_sample", "meas_eqlt/sign",
                             "meas_eqlt/density", "meas_eqlt/double_occ",
                             "state/sweep", "params/n_sweep")
    print(f"n_sample={n_sample or 0}, sweep={sweep}/{n_sweep}")
    if n_sample is not None and n_sample > 0:
        print(f"<sign>={(sign/n_sample)}")
        print(f"<n>={(density/sign)}")
        print(f"<m_z^2>={((density-2*double_occ)/sign)}")


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

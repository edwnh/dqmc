import sys
import util


def info(path):
    n_sample, sign, density, double_occ, sweep, n_sweep = \
        util.load_file(path, "meas_eqlt/n_sample", "meas_eqlt/sign",
                             "meas_eqlt/density", "meas_eqlt/double_occ",
                             "state/sweep", "params/n_sweep")
    print(f"n_sample={n_sample}, sweep={sweep}/{n_sweep}")
    if n_sample > 0:
        print(f"<sign>={float(sign/n_sample):.20f}")
        print(f"<n>={float(density/sign):.20f}")
        print(f"<m_z^2>={float((density-2*double_occ)/sign):.20f}")


def main(argv):
    for path in argv[1:]:
        print(path)
        info(path)

if __name__ == "__main__":
    main(sys.argv)

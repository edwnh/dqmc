import sys
import h5py
import util


def info(path):
    n_sample = util.load(path, "meas_eqlt/n_sample")
    sign = util.load(path, "meas_eqlt/sign")
    density = util.load(path, "meas_eqlt/density")

    print("<sign>=", util.jackknife(n_sample, sign)[:, 0])
    print("<n>=", util.jackknife(sign, density)[:, 0])


def main(argv):
    for path in argv[1:]:
        print(path)
        info(path)

if __name__ == "__main__":
    main(sys.argv)

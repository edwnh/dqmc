import sys
import h5py
import util


def info(path):
    n_sample = util.load(path, "meas_eqlt/n_sample")
    if n_sample.max() == 0:
        print("no data")
        return
    mask = (n_sample == n_sample.max())
    sign = util.load(path, "meas_eqlt/sign")[mask]
    density = util.load(path, "meas_eqlt/density")[mask]

    print(mask.sum(), "/", len(n_sample), " complete bins", sep="")
    print("<sign>=", util.jackknife(n_sample[mask], sign)[:, 0])
    print("<n>=", util.jackknife(sign, density)[:, 0])


def main(argv):
    for path in argv[1:]:
        print(path)
        info(path)

if __name__ == "__main__":
    main(sys.argv)

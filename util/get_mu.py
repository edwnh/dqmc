from glob import glob
import sys
import h5py
import util


def get_mu(path):
    return util.load_1(glob(path + "/*.h5")[0], "metadata/mu")


def get_n(path):
    sign = util.load(path, "meas_eqlt/sign")
    n = util.load(path, "meas_eqlt/density")
    return util.jackknife(sign, n)[:, 0]


def main(argv):
    target = float(argv[3])
    path1, path2 = argv[1], argv[2]
    mu1, mu2 = get_mu(path1), get_mu(path2)
    n1, n2 = get_n(path1), get_n(path2)
    print(path1, "mu=", mu1, "<n>=", n1)
    print(path2, "mu=", mu2, "<n>=", n2)
    print("target mu=", (target-n1[0])/(n2[0]-n1[0]) * (mu2-mu1) + mu1)

if __name__ == "__main__":
    main(sys.argv)

import sys
import h5py
#import numpy as np


def info(filename):
    with h5py.File(filename, "r") as f:
        n_sample = f["meas_eqlt"]["n_sample"][...]
        sign = f["meas_eqlt"]["sign"][...]
        density = f["meas_eqlt"]["density"][...]
        double_occ = f["meas_eqlt"]["double_occ"][...]
        print("n_sample = {}, sweep = {}/{}".format(
              n_sample,
              int(f["state"]["sweep"][...]),
              int(f["params"]["n_sweep"][...])))
        if n_sample > 0:
            print("<sign>={:.20f}".format(float(sign/n_sample)))
            print("<n>={:.20f}".format(float(density/sign)))
            print("<m_z^2>={:.20f}".format(float((density-2*double_occ)/sign)))


def main(argv):
    for path in argv[1:]:
        print(path)
        info(path)

if __name__ == "__main__":
    main(sys.argv)

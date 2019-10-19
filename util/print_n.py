import sys
import util
from glob import glob


def info(path):
    n_sample, sign, density = \
        util.load(path, "meas_eqlt/n_sample", "meas_eqlt/sign",
                        "meas_eqlt/density")
    if n_sample.max() == 0:
        print("no data")
        return
    mask = (n_sample == n_sample.max())
    sign, density = sign[mask], density[mask]
    print(f"complete: {mask.sum()}/{len(n_sample)}")
    print(f"<sign>={util.jackknife(n_sample[mask], sign)}")
    print(f"<n>={util.jackknife(sign, density.sum(1))}")


def main(argv):
    #wildcard path expansion on Windows
    for path in argv[1:]:
        paths = sorted(glob(path))
        if len(paths) == 0:
            print("No paths matching:"+ path)
        else:
            for p in paths:
                print(p)
                info(p)

if __name__ == "__main__":
    main(sys.argv)

import sys
import util
from glob import glob

def info(path):
    period_eqlt, L, n_sweep_meas = util.load_firstfile(path,
        "params/period_eqlt", "params/L", "params/n_sweep_meas")
    full_n_sample = n_sweep_meas*(L//period_eqlt)

    n_sample, sign, density = util.load(path,
        "meas_eqlt/n_sample", "meas_eqlt/sign", "meas_eqlt/density")

    frac = n_sample.sum()/(full_n_sample * len(n_sample))
    print(f"samples: {n_sample.sum()}/{full_n_sample * len(n_sample)}={frac*100:.2f}%")
    print(f"complete bins: {(n_sample == full_n_sample).sum()}/{len(n_sample)}")

    if frac == 0:
        print("no data")
        return
    mask = (n_sample > 0)
    n_sample, sign, density = n_sample[mask], sign[mask], density[mask]
    print(f"<sign>={util.jackknife_noniid(n_sample, n_sample, sign)}")
    print(f"<n>={util.jackknife_noniid(n_sample, sign, density.sum(1))}")


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

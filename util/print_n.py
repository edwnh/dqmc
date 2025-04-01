import sys
import util
from glob import glob

def get_sign_n(path):
    period_eqlt, L, n_sweep_meas = util.load_firstfile(path,
        "params/period_eqlt", "params/L", "params/n_sweep_meas")
    n_sample, sign, density = util.load(path,
        "meas_eqlt/n_sample", "meas_eqlt/sign", "meas_eqlt/density")

    full_n_sample = n_sweep_meas*(L//period_eqlt)
    frac = n_sample.mean()/full_n_sample
    print(f"{path}: complete bins {(n_sample == full_n_sample).sum()}/{len(n_sample)}, samples {frac*100:.3f}%")

    if frac == 0:
        print("no data")
        return

    signj = util.jackknife_noniid(n_sample, n_sample, sign)
    nj = util.jackknife_noniid(n_sample, sign, density.sum(1))
    return signj, nj


def main(argv):
    # wildcard path expansion on Windows
    for path in argv[1:]:
        paths = sorted(glob(path))
        if len(paths) == 0:
            print("No paths matching:"+ path)
        else:
            for p in paths:
                signj, nj = get_sign_n(p)
                print(f"<sign> = {signj}")
                print(f"<n> = {nj}")

if __name__ == "__main__":
    main(sys.argv)

import sys
import util


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
    for path in argv[1:]:
        print(path)
        info(path)

if __name__ == "__main__":
    main(sys.argv)

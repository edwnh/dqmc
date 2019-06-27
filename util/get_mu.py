from glob import glob
import sys
import numpy as np
import util


def get_mu_n(path):
    n_sample, sign, density = \
        util.load(path, "meas_eqlt/n_sample", "meas_eqlt/sign",
                        "meas_eqlt/density")
    mask = (n_sample == n_sample.max())
    if not mask.all():
        print(f"{path} incomplete: {mask.sum()}/{len(n_sample)}")
    sign, density = sign[mask], density[mask]
    nj = util.jackknife(sign, density.sum(1))
    return util.load_firstfile(path, "metadata/mu")[0], nj[0], nj[1]


def get_mu(targets, paths):
    data = np.array([get_mu_n(path) for path in paths])
    data = data[np.argsort(data[:, 0])]
    
    mus = np.zeros(len(targets))
    for i in range(len(targets)):
        y = data[:, 1] - targets[i]

        closest = np.abs(y).argmin()
        j = np.clip(closest, 1, data.shape[0] - 2)
        if j != closest:
            print(f"warning: target {targets[i]} out of range of data")

        p = np.polyfit(data[j-1:j+2, 0], y[j-1:j+2], 2)
        r = np.roots(p)
        mus[i] = r[np.abs(r - data[closest, 0]).argmin()]
    
    return data, mus


def main(argv):
    target = float(argv[1])
    paths = argv[2:]

    data, mus = get_mu([target], paths)
    
    diffs = np.zeros(data.shape[0])
    diffs[:-1] = data[1:, 1] - data[:-1, 1]
    print(np.hstack((data, diffs[:, None])))
    
    print(mus)


if __name__ == "__main__":
    main(sys.argv)

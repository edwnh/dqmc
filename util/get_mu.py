import sys
import numpy as np
from scipy.interpolate import CubicSpline
import util


def get_mu_n_chi(path):
    period_eqlt, L, n_sweep_meas, N, mu, beta = util.load_firstfile(path,
        "params/period_eqlt", "params/L", "params/n_sweep_meas", "params/N", "metadata/mu", "metadata/beta")
    n_sample, sign, density, nn = util.load(path,
        "meas_eqlt/n_sample", "meas_eqlt/sign", "meas_eqlt/density", "meas_eqlt/nn")

    full_n_sample = n_sweep_meas*(L//period_eqlt)
    frac = n_sample.mean()/full_n_sample
    print(f"{path}: complete bins {(n_sample == full_n_sample).sum()}/{len(n_sample)}, samples {frac*100:.3f}%")

    nj = util.jackknife_noniid(n_sample, sign, density.sum(1))
    chij = beta*util.jackknife_noniid(n_sample, sign, density.sum(1), nn.sum(1),
                                      f=lambda ns, s, n, nn: ((nn.T/s.T) - N*((n.T/s.T)**2)).T)

    return mu, nj[0], nj[1], chij[0], chij[1]


def main(argv):
    target = float(argv[1])
    paths = argv[2:]
    if len(paths) == 1:
        mu, n, n_err, chi, chi_err = get_mu_n_chi(paths[0])
        print(mu, n, n_err, chi, chi_err)
        target_mu = mu + (target - n)/chi
    else:
        data = np.array([get_mu_n_chi(path) for path in paths])
        data = data[np.argsort(data[:, 0])] # sort rows by mu
        print(data[:, :3])
        cs = CubicSpline(data[:, 0], data[:, 1] - target)
        target_mu = cs.roots(extrapolate=False)[0]
    print(target_mu)


if __name__ == "__main__":
    main(sys.argv)

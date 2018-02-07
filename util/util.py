from glob import glob
from math import fsum

import h5py
import numpy as np


def load_1(filename, x):
    with h5py.File(filename, "r") as f:
        return f[x][...]


def load(path, x):
    files = sorted(glob(path + "*.h5"))
    if not files:
        return np.zeros((0,))

    nbin = len(files)

    a_last = load_1(files.pop(), x)
    shape = a_last.shape
    a = np.zeros((nbin,) + shape)
    a[-1, ...] = a_last

    for i, filename in enumerate(files):
        a[i, ...] = load_1(filename, x)

    return a


def jackknife(signs, sx):
    # sx is <s X>, not <s X>/<s>
    n = signs.shape[0]
    if n == 1:
        return np.stack((sx/signs, np.zeros_like(sx)))

    if signs.ndim == 1:
        signs = signs[:, np.newaxis]
    if sx.ndim == 1:
        sx = sx[:, np.newaxis]

    sum_s = np.apply_along_axis(fsum, 0, signs)
    sum_sx = np.apply_along_axis(fsum, 0, sx)
    jk_vals = ((sum_sx - sx).T / (sum_s - signs).ravel()).T

    jk_sum = np.apply_along_axis(fsum, 0, jk_vals)
    jk_diff = jk_vals - jk_sum / n
    jk_sumdiffsq = np.apply_along_axis(fsum, 0, jk_diff * jk_diff)

    return np.stack((sum_sx * (n/sum_s) - ((n - 1)/n) * jk_sum,
                     (((n - 1)/n) * jk_sumdiffsq) ** 0.5))

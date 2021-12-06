from glob import glob
import h5py
import numpy as np


def load_file(path, *args):
    with h5py.File(path, "r") as f:
        return tuple(f[x][...] for x in args)


def load_firstfile(path, *args):
    return load_file(min(glob(path + "*.h5")), *args)


def load(path, *args):
    files = sorted(glob(path + "*.h5"))
    nbins = len(files)
    if nbins == 0:
        print(f"no files matching: {path}*.h5")
        return
    last = load_file(files.pop(), *args)
    data = tuple(np.zeros((nbins,) + a.shape, dtype=a.dtype) for a in last)
    for a, a_last in zip(data, last):
        a[-1, ...] = a_last
    for i, f in enumerate(files):
        for a, a_i in zip(data, load_file(f, *args)):
            a[i, ...] = a_i
    return data


def jackknife(*args, f=lambda s, sx: (sx.T/s.T).T.real):
    '''
    jackknife resampling assuming i.i.d. bins.
    positional arguments should be arrays, with first index for bin index.
    the keyword argument f is a function that defines the desired statistic in
        terms of the arguments, summed over the bin index.
    '''
    n = args[0].shape[0]
    sums = tuple(a.sum(0) for a in args)
    res_all = f(*sums)
    if n == 1:
        return np.stack((res_all, np.zeros_like(res_all)))
    res_jk = f(*(s - a for s, a in zip(sums, args)))
    res_jk_mean = res_jk.mean(0)
    res_jk_var = ((res_jk - res_jk_mean)**2).mean(0)
    return np.stack(((n*res_all - (n-1)*res_jk_mean), ((n-1)*res_jk_var)**0.5))


def jackknife_noniid(*args, f=lambda n, ns, nsx: (nsx.T/ns.T).T.real):
    '''
    jackknife resampling assuming independent and non-identical bins.
    positional arguments should be arrays, with first index for bin index.
    the first argument must be equal (or proportional) to the number of samples
    in each bin.
    the keyword argument f is a function that defines the desired statistic in
    terms of the arguments, summed over the bin index.
    '''
    nbin, = args[0].shape
    sums = tuple(a.sum(0) for a in args)
    res_all = f(*sums)
    if nbin == 1:
        return np.stack((res_all, np.zeros_like(res_all)))
    res_jk = f(*(s - a for s, a in zip(sums, args)))
    m = sums[0] - args[0]
    res_jk_mean = (m * res_jk.T).T.sum(0)/m.sum()
    res_jk_var = (m*(m/args[0]) * ((res_jk - res_jk_mean)**2).T).T.sum(0)/m.sum()
    return np.stack((res_all, res_jk_var**0.5))

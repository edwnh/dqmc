from glob import glob
import h5py
import numpy as np


def load_file(filename, *args):
    with h5py.File(filename, "r") as f:
        return tuple((f[x][...] if x in f else None) for x in args)


def load_firstfile(prefix, *args):
    return load_file(min(glob(prefix + "*.h5")), *args)


def load(prefix, *args):
    files = sorted(glob(prefix + "*.h5"))
    nbins = len(files)
    if nbins == 0:
        print(f"no files matching: {prefix}*.h5")
        if not prefix.endswith("/"):
            print("probably forgot trailing slash.")
        return
    data_raw = [load_file(f, *args) for f in files]
    data = []
    for i in range(len(data_raw[0])):
        a = [d[i] for d in data_raw]
        t = next((x for x in a if x is not None), None)
        if t is None:
            data.append(None)
        else:
            data.append(np.stack([x if x is not None else np.zeros_like(t) for x in a]))
    return data


def load_complete(path, *args):
    n_sweep_max, = load_firstfile(path, "params/n_sweep")
    data = load(path, "state/sweep", *args)
    mask = (data[0] == n_sweep_max)
    return tuple(a[mask] for a in data[1:])


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
    nbin = args[0].shape[0]
    sums = tuple(a.sum(0) for a in args)
    res_all = f(*sums)
    if nbin == 1:
        return np.stack((res_all, np.zeros_like(res_all)))
    res_jk = f(*(s - a for s, a in zip(sums, args)))
    m = sums[0] - args[0]
    m[args[0] == 0] = 0
    res_jk_mean = (m * res_jk.T).T.sum(0)/m.sum()
    r = np.divide(m, args[0], where=(args[0] != 0))
    res_jk_var = (m*r * ((res_jk - res_jk_mean)**2).T).T.sum(0)/m.sum()
    return np.stack((res_all, res_jk_var**0.5))

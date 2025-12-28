import numpy as np
from scipy.interpolate import CubicSpline

from . import core


def _kernel(n):
    return CubicSpline(np.arange(n), np.identity(n)).integrate(0, n - 1)


def spline_sum(f, axis, include_beta=False):
    if not include_beta:
        k = _kernel(f.shape[axis] + 1)
        k[0] += k[-1]
        k = k[:-1]
    else:
        k = _kernel(f.shape[axis])
    return np.tensordot(f, k, axes=(axis, 0))


def sym_square(corr, xy_sym=False):
    Ny, Nx = corr.shape[-2:]
    corr = 0.5 * (corr + corr[..., (-np.arange(Ny)) % Ny, :])
    corr = 0.5 * (corr + corr[..., :, (-np.arange(Nx)) % Nx])
    if xy_sym and Nx == Ny:
        corr = 0.5 * (corr + np.swapaxes(corr, -1, -2))
    return corr


def sym_tau(corr):
    L = corr.shape[1]
    return 0.5 * (corr + corr[:, (-np.arange(L)) % L, ...])


def fnnq(s, nnq, d):
    N = nnq.shape[-2] * nnq.shape[-1]
    ret = nnq.T / s.T
    sub = (d.T / s.T) ** 2
    ret[0, 0, ...] -= N * sub
    return ret.T


class _Ctx:
    def __init__(self, path, needed):
        self.path = path
        param_keys = (
            "params/period_eqlt",
            "params/L",
            "params/n_sweep_meas",
            "metadata/mu",
            "metadata/Ny",
            "metadata/Nx",
            "metadata/beta",
            "metadata/bps",
            "params/bonds",
        )
        stripped_keys = tuple(k.split("/")[-1] for k in param_keys)
        self.params = dict(zip(stripped_keys, core.load_firstfile(path, *param_keys)))
        self.Ny = int(self.params["Ny"])
        self.Nx = int(self.params["Nx"])
        self.L = int(self.params["L"])
        self.mu = float(self.params["mu"])
        self.beta = float(self.params["beta"])
        self.bps = int(self.params["bps"])
        self.bonds = self.params["bonds"].reshape(2, self.bps, self.Ny, self.Nx)

        self.data = dict(zip(needed, core.load(path, *needed)))
        self.print_status()
        self.reshape_and_symmetrize()

    def print_status(self):
        full_n_sample = self.params["n_sweep_meas"] * (
            self.L // self.params["period_eqlt"]
        )
        n_sample = self.data["meas_eqlt/n_sample"]
        frac = n_sample.mean() / full_n_sample
        print(
            f"{self.path}: complete bins {(n_sample == full_n_sample).sum()}/{len(n_sample)}, samples {frac * 100:.3f}%"
        )

    def reshape_and_symmetrize(self):
        Ny, Nx, L = self.Ny, self.Nx, self.L

        def _site_observable(k):
            if k in self.data:
                self.data[k].shape = -1

        def _eq_corr_ij(k):
            if k in self.data:
                self.data[k].shape = -1, Ny, Nx
                self.data[k] = sym_square(self.data[k], xy_sym=(self.Nx == self.Ny))

        def _ue_corr_ij(k, tau_sym=False):
            if k in self.data:
                self.data[k].shape = -1, L, Ny, Nx
                self.data[k] = sym_square(self.data[k])
                if tau_sym:
                    self.data[k] = sym_tau(self.data[k])

        def _ue_corr_bb(k, tau_sym=False):
            if k in self.data:
                self.data[k].shape = -1, L, self.bps, self.bps, Ny, Nx
                if tau_sym:
                    self.data[k] = sym_tau(self.data[k])

        _site_observable("meas_eqlt/density")
        _site_observable("meas_eqlt/double_occ")

        _eq_corr_ij("meas_eqlt/nn")
        _eq_corr_ij("meas_eqlt/zz")
        _eq_corr_ij("meas_eqlt/xx")
        _eq_corr_ij("meas_eqlt/g00")
        _eq_corr_ij("meas_eqlt/pair_sw")

        _ue_corr_ij("meas_uneqlt/nn", tau_sym=True)
        _ue_corr_ij("meas_uneqlt/zz", tau_sym=True)
        _ue_corr_ij("meas_uneqlt/gt0")
        _ue_corr_ij("meas_uneqlt/pair_sw")

        _ue_corr_bb("meas_uneqlt/pair_bb")


_OBS = {}


def observable(*, description, requires, name=None):
    def deco(fn):
        key = fn.__name__.lstrip("_")
        _OBS[key] = {
            "fn": fn,
            "description": description,
            "requires": requires,
        }
        return fn

    return deco


def list_obs():
    return {k: v["description"] for k, v in _OBS.items()}


@observable(
    description="fermion sign",
    requires=("meas_eqlt/n_sample", "meas_eqlt/sign"),
)
def _sign(ctx):
    return core.jackknife_noniid(
        ctx.data["meas_eqlt/n_sample"],
        ctx.data["meas_eqlt/n_sample"],
        ctx.data["meas_eqlt/sign"],
    )


@observable(
    description="density",
    requires=("meas_eqlt/n_sample", "meas_eqlt/sign", "meas_eqlt/density"),
)
def _den(ctx):
    return core.jackknife_noniid(
        ctx.data["meas_eqlt/n_sample"],
        ctx.data["meas_eqlt/sign"],
        ctx.data["meas_eqlt/density"],
    )


@observable(
    description="double occupancy",
    requires=("meas_eqlt/n_sample", "meas_eqlt/sign", "meas_eqlt/double_occ"),
)
def _docc(ctx):
    return core.jackknife_noniid(
        ctx.data["meas_eqlt/n_sample"],
        ctx.data["meas_eqlt/sign"],
        ctx.data["meas_eqlt/double_occ"],
    )


@observable(
    description="equal-time Green's function (real space)",
    requires=("meas_eqlt/n_sample", "meas_eqlt/sign", "meas_eqlt/g00"),
)
def _gr(ctx):
    return core.jackknife_noniid(
        ctx.data["meas_eqlt/n_sample"],
        ctx.data["meas_eqlt/sign"],
        ctx.data["meas_eqlt/g00"],
    )


@observable(
    description="equal-time Green's function (momentum space)",
    requires=("meas_eqlt/n_sample", "meas_eqlt/sign", "meas_eqlt/g00"),
)
def _gk(ctx):
    return core.jackknife_noniid(
        ctx.data["meas_eqlt/n_sample"],
        ctx.data["meas_eqlt/sign"],
        np.fft.fft2(ctx.data["meas_eqlt/g00"]).real,
    )


@observable(
    description="equal-time density-density (connected) correlator",
    requires=(
        "meas_eqlt/n_sample",
        "meas_eqlt/sign",
        "meas_eqlt/density",
        "meas_eqlt/nn",
    ),
)
def _nnr(ctx):
    return core.jackknife_noniid(
        ctx.data["meas_eqlt/n_sample"],
        ctx.data["meas_eqlt/sign"],
        ctx.data["meas_eqlt/nn"],
        ctx.data["meas_eqlt/density"],
        f=lambda ns, s, nn, d: ((nn.T / s.T) - ((d.T / s.T) ** 2)).T,
    )


@observable(
    description="equal-time density structure factor",
    requires=(
        "meas_eqlt/n_sample",
        "meas_eqlt/sign",
        "meas_eqlt/density",
        "meas_eqlt/nn",
    ),
)
def _nnq(ctx):
    return core.jackknife_noniid(
        ctx.data["meas_eqlt/n_sample"],
        ctx.data["meas_eqlt/sign"],
        np.fft.fft2(ctx.data["meas_eqlt/nn"]).real,
        ctx.data["meas_eqlt/density"],
        f=lambda ns, s, n, d: fnnq(s, n, d),
    )


@observable(
    description="equal-time spin_z-spin_z correlator",
    requires=("meas_eqlt/n_sample", "meas_eqlt/sign", "meas_eqlt/zz"),
)
def _zzr(ctx):
    return core.jackknife_noniid(
        ctx.data["meas_eqlt/n_sample"],
        ctx.data["meas_eqlt/sign"],
        ctx.data["meas_eqlt/zz"],
    )


@observable(
    description="equal-time spin_x-spin_x correlator",
    requires=("meas_eqlt/n_sample", "meas_eqlt/sign", "meas_eqlt/xx"),
)
def _xxr(ctx):
    return core.jackknife_noniid(
        ctx.data["meas_eqlt/n_sample"],
        ctx.data["meas_eqlt/sign"],
        ctx.data["meas_eqlt/xx"],
    )


@observable(
    description="equal-time spin_z structure factor",
    requires=("meas_eqlt/n_sample", "meas_eqlt/sign", "meas_eqlt/zz"),
)
def _zzq(ctx):
    return core.jackknife_noniid(
        ctx.data["meas_eqlt/n_sample"],
        ctx.data["meas_eqlt/sign"],
        np.fft.fft2(ctx.data["meas_eqlt/zz"]).real,
    )


@observable(
    description="equal-time dressed bubble (real space)",
    requires=("meas_eqlt/n_sample", "meas_eqlt/sign", "meas_eqlt/g00"),
)
def _ggr(ctx):
    ns, s = ctx.data["meas_eqlt/n_sample"], ctx.data["meas_eqlt/sign"]
    g00 = ctx.data["meas_eqlt/g00"]
    gb0 = -g00.copy()
    gb0[:, 0, 0] += s
    return core.jackknife_noniid(ns, s * s, 2 * g00 * gb0)


@observable(
    description="unequal-time density-density (connected) correlator",
    requires=("meas_uneqlt/n_sample", "meas_uneqlt/sign", "meas_uneqlt/nn"),
)
def _nnrt(ctx):
    return core.jackknife_noniid(
        ctx.data["meas_uneqlt/n_sample"],
        ctx.data["meas_uneqlt/sign"],
        ctx.data["meas_uneqlt/nn"],  # TODO: subtract disconnected part
    )


@observable(
    description="unequal-time spin_z-spin_z correlator",
    requires=("meas_uneqlt/n_sample", "meas_uneqlt/sign", "meas_uneqlt/zz"),
)
def _zzrt(ctx):
    return core.jackknife_noniid(
        ctx.data["meas_uneqlt/n_sample"],
        ctx.data["meas_uneqlt/sign"],
        ctx.data["meas_uneqlt/zz"],
    )


@observable(
    description="zero Matsubara frequency density-density (connected) correlator",
    requires=(
        "meas_uneqlt/n_sample",
        "meas_uneqlt/sign",
        "meas_uneqlt/nn",
        "meas_uneqlt/gt0",
    ),
)
def _nnrw0(ctx):
    ns = ctx.data["meas_uneqlt/n_sample"]
    s = ctx.data["meas_uneqlt/sign"]
    unnw0 = spline_sum(ctx.data["meas_uneqlt/nn"], 1) / ctx.L
    uden = 2 * (s - ctx.data["meas_uneqlt/gt0"][:, 0, 0, 0])
    return ctx.beta * core.jackknife_noniid(
        ns, s, unnw0, uden, f=lambda ns, s, nn, d: ((nn.T / s.T) - ((d.T / s.T) ** 2)).T
    )


@observable(
    description="zero Matsubara frequency density susceptibility",
    requires=(
        "meas_uneqlt/n_sample",
        "meas_uneqlt/sign",
        "meas_uneqlt/nn",
        "meas_uneqlt/gt0",
    ),
)
def _nnqw0(ctx):
    ns = ctx.data["meas_uneqlt/n_sample"]
    s = ctx.data["meas_uneqlt/sign"]
    uden = 2 * (s - ctx.data["meas_uneqlt/gt0"][:, 0, 0, 0])
    unnqw0 = np.fft.fft2(spline_sum(ctx.data["meas_uneqlt/nn"], 1) / ctx.L).real
    return ctx.beta * core.jackknife_noniid(
        ns, s, unnqw0, uden, f=lambda ns, s, n, d: fnnq(s, n, d)
    )


@observable(
    description="zero Matsubara frequency spin_z-spin_z correlator",
    requires=("meas_uneqlt/n_sample", "meas_uneqlt/sign", "meas_uneqlt/zz"),
)
def _zzrw0(ctx):
    return ctx.beta * core.jackknife_noniid(
        ctx.data["meas_uneqlt/n_sample"],
        ctx.data["meas_uneqlt/sign"],
        spline_sum(ctx.data["meas_uneqlt/zz"], 1) / ctx.L,
    )


@observable(
    description="zero Matsubara frequency spin_z susceptibility",
    requires=("meas_uneqlt/n_sample", "meas_uneqlt/sign", "meas_uneqlt/zz"),
)
def _zzqw0(ctx):
    return ctx.beta * core.jackknife_noniid(
        ctx.data["meas_uneqlt/n_sample"],
        ctx.data["meas_uneqlt/sign"],
        np.fft.fft2(spline_sum(ctx.data["meas_uneqlt/zz"], 1) / ctx.L).real,
    )


@observable(
    description="zero Matsubara frequency dressed bubble (real space)",
    requires=("meas_uneqlt/n_sample", "meas_uneqlt/sign", "meas_uneqlt/gt0"),
)
def _ggrw0(ctx):
    ns = ctx.data["meas_uneqlt/n_sample"]
    s = ctx.data["meas_uneqlt/sign"]
    gt0 = ctx.data["meas_uneqlt/gt0"]
    gt0_ex = np.zeros(gt0.shape[:1] + (ctx.L + 1,) + gt0.shape[2:], dtype=gt0.dtype)
    gt0_ex[:, :-1, ...] = gt0
    gt0_ex[:, -1, ...] = -gt0[:, 0, ...]
    gt0_ex[:, -1, 0, 0] += s
    ugg = 2 * gt0_ex * gt0_ex[:, ::-1]
    uggw0 = spline_sum(ugg, 1, include_beta=True) / ctx.L
    return ctx.beta * core.jackknife_noniid(ns, s * s, uggw0)


@observable(
    description="zero Matsubara frequency dressed bubble (momentum space)",
    requires=("meas_uneqlt/n_sample", "meas_uneqlt/sign", "meas_uneqlt/gt0"),
)
def _ggqw0(ctx):
    ns = ctx.data["meas_uneqlt/n_sample"]
    s = ctx.data["meas_uneqlt/sign"]
    gt0 = ctx.data["meas_uneqlt/gt0"]
    gt0_ex = np.zeros(gt0.shape[:1] + (ctx.L + 1,) + gt0.shape[2:])
    gt0_ex[:, :-1, ...] = gt0
    gt0_ex[:, -1, ...] = -gt0[:, 0, ...]
    gt0_ex[:, -1, 0, 0] += s
    ugg = 2 * gt0_ex * gt0_ex[:, ::-1]
    uggqw0 = np.fft.fft2(spline_sum(ugg, 1, include_beta=True) / ctx.L).real
    return ctx.beta * core.jackknife_noniid(ns, s * s, uggqw0)


@observable(
    description="equal-time s-wave pair-field structure factor (q=0)",
    requires=("meas_eqlt/n_sample", "meas_eqlt/sign", "meas_eqlt/pair_sw"),
)
def _swq0(ctx):
    return core.jackknife_noniid(
        ctx.data["meas_eqlt/n_sample"],
        ctx.data["meas_eqlt/sign"],
        ctx.data["meas_eqlt/pair_sw"].sum(axis=(-2, -1)),
    )


@observable(
    description="unequal-time d-wave pair-field structure factor (q=0)",
    requires=(
        "meas_uneqlt/n_sample",
        "meas_uneqlt/sign",
        "meas_uneqlt/gt0",
        "meas_uneqlt/pair_bb",
    ),
)
def _dwq0t(ctx):
    ns = ctx.data["meas_uneqlt/n_sample"]
    sign = ctx.data["meas_uneqlt/sign"]
    Nx, Ny = ctx.Nx, ctx.Ny
    bps = 2  # only need nearest neighbor bonds
    pair_bb = ctx.data["meas_uneqlt/pair_bb"][:, :, :bps, :bps, :, :]

    # extend to tau=beta point
    # could also use meas_eqlt/g00, but different n_sample and sign complicates things
    g00 = ctx.data["meas_uneqlt/gt0"][:, 0, ...]
    comm = np.zeros_like(pair_bb[:, 0, ...])
    for bj in range(bps):  # bond type for j
        j0x = ctx.bonds[0, bj, 0, 0] % Nx
        j0y = ctx.bonds[0, bj, 0, 0] // Nx
        j1x = ctx.bonds[1, bj, 0, 0] % Nx
        j1y = ctx.bonds[1, bj, 0, 0] // Nx
        for bi in range(bps):  # bond type for i
            i0 = ctx.bonds[0, bi, 0, 0]
            i1 = ctx.bonds[1, bi, 0, 0]
            i0x, i0y = i0 % Nx, i0 // Nx
            i1x, i1y = i1 % Nx, i1 // Nx
            for ry in range(Ny):
                for rx in range(Nx):  # r = i - j
                    j0 = ((-rx + j0x) % Nx) + Nx * ((-ry + j0y) % Ny)
                    j1 = ((-rx + j1x) % Nx) + Nx * ((-ry + j1y) % Ny)
                    i0j0x, i0j0y = (rx - j0x + i0x) % Nx, (ry - j0y + i0y) % Ny
                    i0j1x, i0j1y = (rx - j1x + i0x) % Nx, (ry - j1y + i0y) % Ny
                    i1j0x, i1j0y = (rx - j0x + i1x) % Nx, (ry - j0y + i1y) % Ny
                    i1j1x, i1j1y = (rx - j1x + i1x) % Nx, (ry - j1y + i1y) % Ny
                    comm[:, bj, bi, ry, rx] = (
                        (i0 == j0) * g00[:, i1j1y, i1j1x]
                        + (i1 == j1) * g00[:, i0j0y, i0j0x]
                        - (i1 == j1) * (i0 == j0) * sign
                        + (i1 == j0) * g00[:, i0j1y, i0j1x]
                        + (i0 == j1) * g00[:, i1j0y, i1j0x]
                        - (i0 == j1) * (i1 == j0) * sign
                    )
    pair_bb_ex = np.concatenate((pair_bb, (pair_bb[:, 0, ...] - comm)[:, None, ...]), 1)

    pair_bb_q0 = pair_bb_ex.sum((-1, -2))
    pd = 0.5 * (
        pair_bb_q0[..., 0, 0]
        - pair_bb_q0[..., 0, 1]
        - pair_bb_q0[..., 1, 0]
        + pair_bb_q0[..., 1, 1]
    )
    return core.jackknife_noniid(ns, sign, pd)


def get(path, *names):
    if not names:
        names = _OBS.keys()
    missing = tuple(n for n in names if n not in _OBS)
    if missing:
        raise KeyError(f"unknown observables: {missing}. available: {_OBS.keys()}")
    needed = set(req for n in names for req in _OBS[n]["requires"])
    ctx = _Ctx(path, needed)
    return ctx.params | {n: _OBS[n]["fn"](ctx) for n in names}

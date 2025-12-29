"""Generate HDF5 simulation files for 1-band Hubbard model."""

import os
import shutil
import sys
from dataclasses import dataclass, fields

import h5py
import numpy as np
from scipy.linalg import expm

from .lattices import make_square_2d

np.seterr(over="ignore")


@dataclass
class SimParams:
    """Default simulation parameters for 1-band Hubbard on square lattice."""
    Nx: int = 8
    Ny: int = 8
    mu: float = 0.0
    tp: float = 0.0 # 2nd neighbor hopping
    U: float = 6.0
    dt: float = 0.1 # imaginary time step
    L: int = 40 # number of time slices
    nflux: int = 0 # total magnetic flux through torus (0 for no field)
    n_delay: int = 16 # number of updates to batch in delayed update scheme
    n_matmul: int = 5 # number of direct multiplications before a stabilized multiplication
    n_sweep_warm: int = 200
    n_sweep_meas: int = 2000
    period_eqlt: int = 5 # period of equal-time measurements, in units of time slices
    period_uneqlt: int = 0 # period of unequal-time measurements, in units of space-time sweeps (0 to disable)
    meas_bond_corr: int = 0 # toggle bond correlation measurements
    meas_energy_corr: int = 0 # toggle energy correlation measurements
    meas_nematic_corr: int = 0 # toggle nematic (8 fermion) correlation measurements
    trans_sym: int = 1 # 1 to apply translation symmetry to measurement data


def rand_seed_urandom():
    rng = np.zeros(17, dtype=np.uint64)
    rng[:16] = np.frombuffer(os.urandom(16 * 8), dtype=np.uint64)
    return rng


def rand_seed_splitmix64(x):
    x = np.uint64(x)
    rng = np.zeros(17, dtype=np.uint64)
    for i in range(16):
        x += np.uint64(0x9E3779B97F4A7C15)
        z = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        rng[i] = z ^ (z >> np.uint64(31))
    return rng


def rand_uint(rng):
    s0 = rng[rng[16]]
    p = (int(rng[16]) + 1) & 15
    rng[16] = p
    s1 = rng[p]
    s1 ^= s1 << np.uint64(31)
    rng[p] = s1 ^ s0 ^ (s1 >> np.uint64(11)) ^ (s0 >> np.uint64(30))
    return rng[p] * np.uint64(1181783497276652981)


def rand_jump(rng):
    JMP = np.array([
        0x84242f96eca9c41d, 0xa3c65b8776f96855, 0x5b34a39f070b5837,
        0x4489affce4f31a1e, 0x2ffeeb0a48316f40, 0xdc2d9891fe68c022,
        0x3659132bb12fea70, 0xaac17d8efa43cab8, 0xc4cb815590989b13,
        0x5ee975283d71c93b, 0x691548c86c1bd540, 0x7910c41d10a1e6a5,
        0x0b5fc64563b3e2a8, 0x047f7684e9fc949d, 0xb99181f2d8f685ca,
        0x284600e3f30e38c3], dtype=np.uint64)
    t = np.zeros(16, dtype=np.uint64)
    for i in range(16):
        for b in range(64):
            if JMP[i] & (np.uint64(1) << np.uint64(b)):
                for j in range(16):
                    t[j] ^= rng[(np.uint64(j) + rng[16]) & np.uint64(15)]
            rand_uint(rng)
    for j in range(16):
        rng[(np.uint64(j) + rng[16]) & np.uint64(15)] = t[j]


def _init_hs(rng, L, N):
    """Initialize HS field with random bits from rng."""
    hs = np.zeros((L, N), dtype=np.int32)
    for l in range(L):
        for i in range(N):
            hs[l, i] = rand_uint(rng) >> np.uint64(63)
    return hs


def verify_params(p: SimParams):
    for key in ["Nx", "Ny", "L", "dt", "n_matmul", "n_delay", "period_eqlt"]:
        if getattr(p, key) <= 0:
            raise ValueError(f"{key} must be positive, got {getattr(p, key)}")
    for key in ["n_sweep_warm", "n_sweep_meas"]:
        if getattr(p, key) < 0:
            raise ValueError(f"{key} must be non-negative, got {getattr(p, key)}")
    if p.L % p.n_matmul != 0:
        raise ValueError(f"L must be divisible by n_matmul, got L={p.L}, n_matmul={p.n_matmul}")
    if p.L % p.period_eqlt != 0:
        raise ValueError(f"L must be divisible by period_eqlt, got L={p.L}, period_eqlt={p.period_eqlt}")


def create_1(file_sim, file_params, init_rng, overwrite=False,**kwargs):
    """Create a single simulation HDF5 file."""
    p = SimParams(**{k: v for k, v in kwargs.items() if k in {f.name for f in fields(SimParams)}})
    verify_params(p)

    lat = make_square_2d(p.Nx, p.Ny, p.tp, p.nflux, p.trans_sym)
    N = lat["N"]

    dtype_num = np.complex128 if p.nflux != 0 else np.float64

    rng = init_rng.copy()
    init_hs = _init_hs(rng, p.L, N)

    # kinetic matrices
    tij, peierls = lat["tij"], lat["peierls"]
    if dtype_num == np.complex128:
        Ku = -tij * peierls
        assert np.max(np.abs(Ku - Ku.T.conj())) < 1e-10
    else:
        Ku = -tij.real
        assert np.max(np.abs(peierls.imag)) < 1e-10
        peierls = peierls.real

    for i in range(N):
        Ku[i, i] -= p.mu

    exp_Ku = expm(-p.dt * Ku)
    inv_exp_Ku = expm(p.dt * Ku)
    exp_halfKu = expm(-p.dt / 2 * Ku)
    inv_exp_halfKu = expm(p.dt / 2 * Ku)

    # interaction parameters
    U_i = p.U * np.ones_like(lat["degen_i"], dtype=np.float64)
    exp_lmbd = np.exp(0.5 * U_i * p.dt) + np.sqrt(np.expm1(U_i * p.dt))
    exp_lambda = np.array([exp_lmbd[lat["map_i"]]**-1, exp_lmbd[lat["map_i"]]])
    delll = np.array([exp_lmbd[lat["map_i"]]**2 - 1, exp_lmbd[lat["map_i"]]**-2 - 1])

    with h5py.File(file_params, "w" if overwrite else "x") as f:
        # Metadata (for analysis)
        g = f.create_group("metadata")
        g["version"] = 0.1
        g["model"] = "Hubbard (complex)" if dtype_num == np.complex128 else "Hubbard"
        g["Nx"], g["Ny"] = p.Nx, p.Ny
        g["bps"] = lat["bps"]
        g["U"], g["t'"], g["nflux"], g["mu"] = p.U, p.tp, p.nflux, p.mu
        g["beta"] = p.L * p.dt

        # Parameters (used by dqmc code)
        g = f.create_group("params")
        g["N"] = np.int32(N)
        g["L"] = np.int32(p.L)
        g["map_i"], g["map_ij"] = lat["map_i"], lat["map_ij"]
        g["bonds"], g["map_bs"], g["map_bb"] = lat["bonds"], lat["map_bs"], lat["map_bb"]
        g["peierlsu"] = peierls
        g["peierlsd"] = g["peierlsu"]
        g["Ku"] = Ku
        g["Kd"] = g["Ku"]
        g["U"] = U_i
        g["dt"] = np.float64(p.dt)
        g["n_matmul"] = np.int32(p.n_matmul)
        g["n_delay"] = np.int32(p.n_delay)
        g["n_sweep_warm"] = np.int32(p.n_sweep_warm)
        g["n_sweep_meas"] = np.int32(p.n_sweep_meas)
        g["period_eqlt"] = np.int32(p.period_eqlt)
        g["period_uneqlt"] = np.int32(p.period_uneqlt)
        g["meas_bond_corr"] = p.meas_bond_corr
        g["meas_energy_corr"] = p.meas_energy_corr
        g["meas_nematic_corr"] = p.meas_nematic_corr
        g["num_i"], g["num_ij"] = lat["num_i"], lat["num_ij"]
        g["num_b"], g["num_bs"], g["num_bb"] = lat["num_b"], lat["num_bs"], lat["num_bb"]
        g["degen_i"], g["degen_ij"] = lat["degen_i"], lat["degen_ij"]
        g["degen_bs"], g["degen_bb"] = lat["degen_bs"], lat["degen_bb"]
        g["exp_Ku"], g["inv_exp_Ku"] = exp_Ku, inv_exp_Ku
        g["exp_Kd"], g["inv_exp_Kd"] = g["exp_Ku"], g["inv_exp_Ku"]
        g["exp_halfKu"], g["inv_exp_halfKu"] = exp_halfKu, inv_exp_halfKu
        g["exp_halfKd"], g["inv_exp_halfKd"] = g["exp_halfKu"], g["inv_exp_halfKu"]
        g["exp_lambda"], g["del"] = exp_lambda, delll
        g["F"] = np.int32(p.L // p.n_matmul)
        g["n_sweep"] = np.int32(p.n_sweep_warm + p.n_sweep_meas)

    with h5py.File(file_sim, "w" if overwrite else "x") as f:
        params_relpath = os.path.relpath(file_params, os.path.dirname(file_sim))
        f["params_file"] = params_relpath
        f["metadata"] = h5py.ExternalLink(params_relpath, "metadata")
        f["params"] = h5py.ExternalLink(params_relpath, "params")

        g = f.create_group("state")
        g["sweep"] = np.int32(0)
        g["init_rng"] = init_rng
        g["rng"] = rng
        g["hs"] = init_hs

        f.create_group("meas_eqlt")
        if p.period_uneqlt > 0:
            f.create_group("meas_uneqlt")


def create_batch(Nfiles=1, prefix=None, seed=None, **kwargs):
    """Create multiple simulation files with independent RNG streams."""
    init_rng = rand_seed_urandom() if seed is None else rand_seed_splitmix64(seed)
    prefix = prefix or "sim"

    dirname = os.path.dirname(prefix)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    file_0 = f"{prefix}_0.h5"
    file_p = f"{prefix}.h5.params"

    create_1(file_0, file_p, init_rng, **kwargs)

    with h5py.File(file_p, "r") as f:
        N, L = int(f["params"]["N"][...]), int(f["params"]["L"][...])

    for i in range(1, Nfiles):
        rand_jump(init_rng)
        rng = init_rng.copy()
        init_hs = _init_hs(rng, L, N)

        file_i = f"{prefix}_{i}.h5"
        shutil.copy2(file_0, file_i)
        with h5py.File(file_i, "r+") as f:
            f["state"]["init_rng"][...] = init_rng
            f["state"]["rng"][...] = rng
            f["state"]["hs"][...] = init_hs

    last_file = file_0 if Nfiles == 1 else f"{prefix}_{Nfiles-1}.h5"
    print(f"created: {file_0}" + (f" ... {last_file}" if Nfiles > 1 else ""))
    print(f"params:  {file_p}")


def main(argv=None):
    if argv is None:
        argv = sys.argv
    kwargs = {}
    for arg in argv[1:]:
        if "=" not in arg:
            print(f"couldn't find '=' in argument: {arg}")
            return
        key, val = arg.split("=", 1)
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        kwargs[key] = val
    create_batch(**kwargs)


if __name__ == "__main__":
    main()

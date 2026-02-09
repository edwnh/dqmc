"""Generate HDF5 simulation files for 1-band Hubbard model."""

import os
import shutil
import sys
from dataclasses import dataclass, fields

import h5py
import numpy as np
from scipy.linalg import expm

np.seterr(over="ignore")


@dataclass
class SimParams:
    """Default simulation parameters for 1-band Hubbard on square lattice."""
    Nx: int = 8
    Ny: int = 8
    mu: float = 0.0
    t: float = 1.0 # nearest neighbor hopping
    tp: float = 0.0 # 2nd neighbor hopping
    U: float = 4.0
    V: float = 0.5
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

def make_square_2d(Nx, Ny, t, tp, nflux, trans_sym):
    """
    Build geometry data for a 2D square lattice with periodic boundaries.

    Parameters
    ----------
    Nx, Ny : int
        Lattice dimensions.
    tp : float
        Next-nearest neighbor hopping (adds diagonal bonds if nonzero).
    nflux : int
        Number of magnetic flux quanta (Peierls phases).
    trans_sym : bool
        Use translational symmetry to reduce measurement storage.

    Returns
    -------
    dict with keys:
        N, bps, num_b, num_i, num_ij, num_bs, num_bb,
        bonds, map_i, map_ij, map_bs, map_bb,
        degen_i, degen_ij, degen_bs, degen_bb,
        tij, peierls
    """
    N = Nx * Ny
    bps = 4 if tp != 0.0 else 2  # bonds per site

    bps_V = 2 # number of oriented pairs in V

    # --- 1-site mapping ---
    if trans_sym:
        map_i = np.zeros(N, dtype=np.int32)
        degen_i = np.array([N], dtype=np.int32)
    else:
        map_i = np.arange(N, dtype=np.int32)
        degen_i = np.ones(N, dtype=np.int32)
    num_i = degen_i.size

    # --- 2-site mapping ---
    num_ij = N if trans_sym else N * N
    map_ij = np.zeros((N, N), dtype=np.int32)
    degen_ij = np.zeros(num_ij, dtype=np.int32)
    for jy in range(Ny):
        for jx in range(Nx):
            j = jx + Nx * jy
            for iy in range(Ny):
                for ix in range(Nx):
                    i = ix + Nx * iy
                    if trans_sym:
                        k = (ix - jx) % Nx + Nx * ((iy - jy) % Ny)
                    else:
                        k = i + N * j
                    map_ij[j, i] = k
                    degen_ij[k] += 1

    # --- Bond definitions ---
    num_b = bps * N
    bonds = np.zeros((2, num_b), dtype=np.int32)
    for iy in range(Ny):
        for ix in range(Nx):
            i = ix + Nx * iy
            ix1, iy1 = (ix + 1) % Nx, (iy + 1) % Ny
            bonds[0, i] = i
            bonds[1, i] = ix1 + Nx * iy        # +x
            bonds[0, i + N] = i
            bonds[1, i + N] = ix + Nx * iy1    # +y
            if bps == 4:
                bonds[0, i + 2*N] = i
                bonds[1, i + 2*N] = ix1 + Nx * iy1  # +x+y
                bonds[0, i + 3*N] = ix1 + Nx * iy
                bonds[1, i + 3*N] = ix + Nx * iy1   # +x to +y

    # --- V bond definitions ---
    num_b_V = bps_V * N
    bonds_V = np.zeros((2, num_b_V), dtype=np.int32)
    for iy in range(Ny):
        for ix in range(Nx):
            i = ix + Nx * iy
            ix1, iy1 = (ix + 1) % Nx, (iy + 1) % Ny
            bonds_V[0, i] = i
            bonds_V[1, i] = ix1 + Nx * iy        # +x
            bonds_V[0, i + N] = i
            bonds_V[1, i + N] = ix + Nx * iy1    # +y

    # --- Bond-site mapping ---
    num_bs = bps * N if trans_sym else num_b * N
    map_bs = np.zeros((N, num_b), dtype=np.int32)
    degen_bs = np.zeros(num_bs, dtype=np.int32)
    for j in range(N):
        for i in range(N):
            k = map_ij[j, i]
            for ib in range(bps):
                kk = k + num_ij * ib
                map_bs[j, i + N * ib] = kk
                degen_bs[kk] += 1

    # --- Bond-bond mapping ---
    num_bb = bps * bps * N if trans_sym else num_b * num_b
    map_bb = np.zeros((num_b, num_b), dtype=np.int32)
    degen_bb = np.zeros(num_bb, dtype=np.int32)
    for j in range(N):
        for i in range(N):
            k = map_ij[j, i]
            for jb in range(bps):
                for ib in range(bps):
                    kk = k + num_ij * (ib + bps * jb)
                    map_bb[j + N * jb, i + N * ib] = kk
                    degen_bb[kk] += 1

    # --- Hopping matrix ---
    tij = np.zeros((N, N), dtype=np.complex128)
    for iy in range(Ny):
        for ix in range(Nx):
            i = ix + Nx * iy
            ix1, iy1 = (ix + 1) % Nx, (iy + 1) % Ny
            # nearest neighbor
            tij[ix + Nx * iy1, i] += t
            tij[i, ix + Nx * iy1] += t
            tij[ix1 + Nx * iy, i] += t
            tij[i, ix1 + Nx * iy] += t
            # next-nearest neighbor
            tij[ix1 + Nx * iy1, i] += tp
            tij[i, ix1 + Nx * iy1] += tp
            tij[ix1 + Nx * iy, ix + Nx * iy1] += tp
            tij[ix + Nx * iy1, ix1 + Nx * iy] += tp

    # --- Peierls phases (symmetric gauge) ---
    alpha = 0.5
    phi = np.zeros((N, N))
    for dy in range((1 - Ny) // 2, (1 + Ny) // 2):
        for dx in range((1 - Nx) // 2, (1 + Nx) // 2):
            for iy in range(Ny):
                for ix in range(Nx):
                    jy, jx = iy + dy, ix + dx
                    jjy, jjx = jy % Ny, jx % Nx
                    off_y, off_x = jy - jjy, jx - jjx
                    mx, my = (ix + jx) / 2, (iy + jy) / 2
                    phi[jjx + Nx * jjy, ix + Nx * iy] = (
                        -alpha * my * dx + (1 - alpha) * mx * dy
                        - (1 - alpha) * off_x * jy + alpha * off_y * jx
                        - alpha * off_x * off_y
                    )
    peierls = np.exp(2j * np.pi * (nflux / N) * phi)

    return dict(
        N=N, bps=bps, num_b=num_b, bps_V=bps_V, z=2*bps_V, num_b_V=num_b_V,
        num_i=num_i, num_ij=num_ij, num_bs=num_bs, num_bb=num_bb,
        bonds=bonds, bonds_V=bonds_V, map_i=map_i, map_ij=map_ij, map_bs=map_bs, map_bb=map_bb,
        degen_i=degen_i, degen_ij=degen_ij, degen_bs=degen_bs, degen_bb=degen_bb,
        tij=tij, peierls=peierls,
    )


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
            hs[l, i] = rand_uint(rng) >> np.uint64(62) # 0, 1, 2, 3
    return hs


def verify_params(p: SimParams, lat):
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
    # requirement for HS transformation
    if np.abs(p.U) < lat["z"]*np.abs(p.V):
        raise ValueError(f"require |U| >= 4|V| for HS transformation, got U={p.U}, V={p.V}")


def create_1(file_sim, file_params, init_rng, overwrite=False,**kwargs):
    """Create a single simulation HDF5 file."""
    p = SimParams(**{k: v for k, v in kwargs.items() if k in {f.name for f in fields(SimParams)}})
    lat = make_square_2d(p.Nx, p.Ny, p.t, p.tp, p.nflux, p.trans_sym)
    verify_params(p, lat)

    N = lat["N"]
    num_b_V = lat["num_b_V"]

    dtype_num = np.complex128 if p.nflux != 0 or p.U > 0 else np.float64

    rng = init_rng.copy()
    init_hs = _init_hs(rng, p.L, num_b_V)

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
    x_sq = np.abs(p.U)/(2*lat["z"]) + ((p.U/(2*lat["z"]))**2 - p.V**2 / 4)**0.5
    y_sq = np.abs(p.U)/(2*lat["z"]) - ((p.U/(2*lat["z"]))**2 - p.V**2 / 4)**0.5
    # x_sq, y_sq = y_sq, x_sq # shouldn't affect things unless x_sq = 0
    gg = np.sign(p.U) * 2 * x_sq
    aa = np.sign(p.U) * np.sign(p.V) * (y_sq / x_sq)**0.5

    assert np.abs(p.U - lat["z"]*gg/2*(1+aa*aa)) < 1e-10
    assert np.abs(p.V - gg*aa) < 1e-10

    gamma_1 = 1/4 + 6**0.5/12
    gamma_2 = 1/4 - 6**0.5/12
    gamma = np.array([gamma_2, gamma_1, gamma_1, gamma_2])

    lmbd_1 = (-p.dt * gg * (3 - 6**0.5)).astype(dtype_num)**0.5
    if np.abs(lmbd_1.real) < 1e-12:
        lmbd_1 = lmbd_1.imag * 1j
    lmbd_2 = (-p.dt * gg * (3 + 6**0.5)).astype(dtype_num)**0.5
    if np.abs(lmbd_2.real) < 1e-12:
        lmbd_2 = lmbd_2.imag * 1j
    lmbd = np.array([-lmbd_2, -lmbd_1, lmbd_1, lmbd_2])

    exp_lambda = np.exp(lmbd)
    exp_lambdaa = np.exp(aa*lmbd)

    delll = np.zeros((4, 4), dtype=dtype_num)
    dellla = np.zeros((4, 4), dtype=dtype_num)
    pre_ratio = np.zeros((4, 4), dtype=dtype_num)
    for i in range(4):
        for j in range(4):
            # i is hs field before, j is hs field after
            delll[i, j] = exp_lambda[j] / exp_lambda[i] - 1
            dellla[i, j] = exp_lambdaa[j] / exp_lambdaa[i] - 1
            pre_ratio[i, j] = (gamma[j] * np.exp(-lmbd[j]*(1 + aa))) / (gamma[i] * np.exp(-lmbd[i]*(1 + aa)))
    pre_phase = np.exp(-lmbd*(1 + aa))
    pre_phase /= np.abs(pre_phase)

    with h5py.File(file_params, "w" if overwrite else "x") as f:
        # Metadata (for analysis)
        g = f.create_group("metadata")
        g["version"] = 0.1
        g["model"] = "extended Hubbard (complex)" if dtype_num == np.complex128 else "extended Hubbard"
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
        g["bonds_V"] = lat["bonds_V"]
        g["peierlsu"] = peierls
        g["peierlsd"] = g["peierlsu"]
        # g["Ku"] = Ku
        # g["Kd"] = g["Ku"]
        # g["U"] = U_i
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
        g["num_b_V"] = lat["num_b_V"]
        g["degen_i"], g["degen_ij"] = lat["degen_i"], lat["degen_ij"]
        g["degen_bs"], g["degen_bb"] = lat["degen_bs"], lat["degen_bb"]
        g["exp_Ku"], g["inv_exp_Ku"] = exp_Ku, inv_exp_Ku
        g["exp_Kd"], g["inv_exp_Kd"] = g["exp_Ku"], g["inv_exp_Ku"]
        g["exp_halfKu"], g["inv_exp_halfKu"] = exp_halfKu, inv_exp_halfKu
        g["exp_halfKd"], g["inv_exp_halfKd"] = g["exp_halfKu"], g["inv_exp_halfKu"]
        g["exp_lambda"], g["del"] = exp_lambda, delll
        g["exp_lambdaa"], g["dela"] = exp_lambdaa, dellla
        g["pre_ratio"] = pre_ratio
        g["pre_phase"] = pre_phase
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
        num_b_V = int(f["params"]["num_b_V"][...])

    for i in range(1, Nfiles):
        rand_jump(init_rng)
        rng = init_rng.copy()
        init_hs = _init_hs(rng, L, num_b_V)

        file_i = f"{prefix}_{i}.h5"
        shutil.copy2(file_0, file_i)
        with h5py.File(file_i, "r+") as f:
            f["state"]["init_rng"][...] = init_rng
            f["state"]["rng"][...] = rng
            f["state"]["hs"][...] = init_hs

    last_file = file_0 if Nfiles == 1 else f"{prefix}_{Nfiles-1}.h5"
    print(f"created: {file_0}" + (f" ... {last_file}" if Nfiles > 1 else "") + f", {file_p}")


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

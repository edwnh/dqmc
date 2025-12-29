"""Lattice geometry definitions for DQMC simulations."""

import numpy as np


def make_square_2d(Nx, Ny, tp, nflux, trans_sym):
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
            tij[ix + Nx * iy1, i] += 1
            tij[i, ix + Nx * iy1] += 1
            tij[ix1 + Nx * iy, i] += 1
            tij[i, ix1 + Nx * iy] += 1
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
        N=N, bps=bps, num_b=num_b, num_i=num_i, num_ij=num_ij, num_bs=num_bs, num_bb=num_bb,
        bonds=bonds, map_i=map_i, map_ij=map_ij, map_bs=map_bs, map_bb=map_bb,
        degen_i=degen_i, degen_ij=degen_ij, degen_bs=degen_bs, degen_bb=degen_bb,
        tij=tij, peierls=peierls,
    )

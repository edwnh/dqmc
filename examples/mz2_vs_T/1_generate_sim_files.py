import numpy as np
from dqmc_util import gen_1band_hub

PARAMS = {
    "Nx": 6,
    "Ny": 6,
    "mu": 0.0,
    "tp": 0.0,
    "n_sweep_warm": 200,
    "n_sweep_meas": 1000,
}
NFILES = 4

# temperatures for each U
U_T = {
    0.5: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
    1.0: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
    2.0: [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
    4.0: [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
    6.0: [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
    8.0: [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
    10.0: [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
    12.0: [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
}

for U, Ts in U_T.items():
    for T in Ts:
        # find some reasonable values of L and dt
        # such that L >= 10, U * dt**2 <= 0.05, and L % n_matmul = 0 (default n_matmul=5)
        beta = 1.0 / T
        dt = min((0.05/U)**0.5, beta / 10)
        L = int(np.ceil(beta / dt / 5) * 5)
        dt = beta / L
        print(f"U={U}, T={T}, L={L}, dt={dt}")
        gen_1band_hub.create_batch(
            prefix=f"data/U{U}_T{T}/bin",
            Nfiles=NFILES, U=U, L=L, dt=dt, **PARAMS
        )

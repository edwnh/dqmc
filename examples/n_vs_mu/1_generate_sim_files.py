import itertools
import numpy as np
from dqmc_util import gen_1band_hub

PARAMS = {
    "Nx": 4,
    "Ny": 4,
    "U": 4.0,
    "tp": 0.0,
    "dt": 0.1,
    "n_sweep_warm": 200,
    "n_sweep_meas": 1000,
}
NFILES = 4  # bins/files per parameter set

# scanned parameters
mus = np.linspace(-5, 5, 21)
Ls = [20, 50, 100]  # beta = 2, 5, 10 (T = 0.5, 0.2, 0.1)

for L, mu in itertools.product(Ls, mus):
    gen_1band_hub.create_batch(
        prefix=f"data/L{L}_mu{mu:.1f}/bin",
        Nfiles=NFILES, L=L, mu=mu, **PARAMS
    )

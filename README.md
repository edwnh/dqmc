# Determinantal quantum Monte Carlo for the Hubbard model

High-performance DQMC (C) for the Hubbard model, plus Python utilities (`dqmc_util`) for generating inputs and analyzing outputs.

## Supported platforms

- Linux and macOS
- Windows is not supported natively; use WSL2 or GitHub Codespaces if needed.

## Dependencies

**Build (Linux):**
- Intel compiler `icx`
- Intel MKL headers and libraries
- HDF5 headers and libraries (downloaded via `make deps`)

**Build (macOS):**
- `clang` (Xcode Command Line Tools)
- HDF5 headers and libraries (downloaded via `make deps`)

**Python (`dqmc_util`):**
- Installed via `pip install -e .` (included in `environment-*.yml`)
- Runtime deps: `numpy`, `scipy`, `h5py`

## Install and build

1. Create the conda environment:
   ```bash
   # macOS
   conda env create -f environment-macos.yml

   # Linux
   conda env create -f environment-linux.yml
   ```
2. Activate it:
   ```bash
   conda activate dqmc
   ```
3. Download/extract the bundled HDF5 dependency (first time only):
   ```bash
   make deps
   ```
4. Build:
   ```bash
   make
   ```
   The binary is `build/dqmc`.

## Quickstart (smoke run)

```bash
conda activate dqmc
make deps  # first time only
make

dqmc-util gen Nx=4 Ny=4 U=4 dt=0.1 L=20 n_sweep_warm=50 n_sweep_meas=100
build/dqmc sim_0.h5
dqmc-util summary sim_0.h5
```

## Workflow

1. Generate HDF5 simulation files: `dqmc-util gen` (alias: `dqmc-util gen-1band-hub`)
2. Run Monte Carlo sweeps: `build/dqmc sim_0.h5` (optionally with checkpointing)
3. Analyze results: `dqmc-util summary`, `dqmc-util print-n`, or `dqmc_util.analyze_hub` in Python

### Run one file

```bash
build/dqmc sim_0.h5
dqmc-util summary sim_0.h5
```

### Run many files (sharded queue)

For 2+ files, prefer the queue/worker system (`dqmc-util enqueue` + `dqmc-util worker`).

```bash
dqmc-util gen Nx=6 Ny=6 U=4 mu=0 dt=0.1 L=50 Nfiles=160 prefix=runs/U4_T0.25/bin
dqmc-util enqueue queue 'runs/U4_T0.25/bin_*.h5'
dqmc-util worker queue ./build/dqmc -n 8 -s 300 -t 3600
dqmc-util queue-status queue
dqmc-util print-n runs/U4_T0.25/
```

## Examples

See `examples/README.md` and:
- `examples/mz2_vs_T/` - magnetic moment vs temperature
- `examples/n_vs_mu/` - density vs chemical potential

## HDF5 layout

One Markov chain = one HDF5 file = one “bin”.

- `/metadata`: model parameters for analysis (e.g., `beta`, `mu`, `Nx`, `Ny`)
- `/params`: simulation parameters and matrices (often stored in a shared `*.h5.params`)
- `/state`: sweep counter, RNG state, auxiliary field configuration
- `/meas_eqlt`: equal-time measurements
- `/meas_uneqlt`: unequal-time measurements (optional)

For details and authoritative defaults, see `dqmc_util/gen_1band_hub.py` (`SimParams`).

## Parameter notes

- Trotter error rule of thumb: ensure `U * dt^2 <= 0.05`
- `L` must be divisible by `n_matmul` and `period_eqlt`

## AI agent docs

- `AGENTS.md`: working on this codebase (build, architecture, guardrails)
- `.claude/skills/`: task runbooks (generate, run, analyze, scans, dev, advanced)

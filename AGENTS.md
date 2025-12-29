# AGENTS.md

## Project Overview
Determinantal Quantum Monte Carlo (DQMC) simulation for the Hubbard model. High-performance C code with Python utilities for simulation setup and data analysis.

## Repository Architecture

### Core C Code (`src/`)
- **Entry point**: `src/main_1.c` → `src/wrapper.c` → `src/rc/dqmc.c`
- **Real/Complex dual compilation**: Files in `src/rc/` are compiled twice—once for real matrices and once for complex (via `-DUSE_CPLX`). The `RC(name)` macro in `src/rc/numeric.h` expands to `name_real` or `name_cplx`.
- **Key data flow**: HDF5 simulation file → `data.c` (load) → `dqmc.c` (sweep loop) → `greens.c` + `updates.c` + `meas.c` → `data.c` (save)

### Module Responsibilities
| File | Purpose |
|------|---------|
| `src/rc/greens.c` / `src/rc/greens.h` | Equal/unequal-time Green's function calculation using QdX decomposition |
| `src/rc/updates.c` / `src/rc/updates.h` | Delayed update scheme for auxiliary field changes |
| `src/rc/meas.c` / `src/rc/meas.h` | Equal-time and unequal-time correlation measurements |
| `src/rc/linalg.h` | Inline BLAS/LAPACK wrappers (MKL on Linux, Accelerate on macOS) |
| `src/rc/sim_types.h` | X-macro lists defining simulation parameters and measurement arrays |

### Python Package (`dqmc_util/`)
Installable package (`pip install -e .`) with CLI tool `dqmc-util`.
- `dqmc_util/core.py`: Data loading helpers with jackknife resampling
- `dqmc_util/gen_1band_hub.py`: Generate HDF5 simulation files with configurable parameters
- `dqmc_util/analyze_hub.py`: Analysis routines (symmetrization, structure factors)
- `dqmc_util/cli.py`: Unified CLI entry point for all commands

## Build & Development

### Prerequisites
```bash
# For macOS
conda env create -f environment-macos.yml && conda activate dqmc

# For Linux (includes Intel compilers and MKL)
conda env create -f environment-linux.yml && conda activate dqmc

make deps  # downloads HDF5
```

### Build Commands
```bash
make          # builds build/dqmc
make clean    # removes build directory
```

### Running Simulations
```bash
dqmc-util gen                                   # create sim_0.h5
build/dqmc sim_0.h5                             # run simulation
build/dqmc -s 300 -t 3600 sim_0.h5              # with checkpointing
dqmc-util summary sim_0.h5                      # check results
```

### Sharded Queue (for clusters)
```bash
dqmc-util enqueue /path/to/queue 'sim_*.h5'     # add jobs to queue
dqmc-util queue-status /path/to/queue           # check queue status
dqmc-util worker /path/to/queue ./build/dqmc -s 300 -t 3600  # run worker
dqmc-util dequeue /path/to/queue 'sim_*.h5'     # remove from queue
```

## Common Workflows

1. **Local dev**
   - Build the `dqmc` binary locally.
   - Run `dqmc-util gen` once or a few times to generate small test HDF5 files.
   - Execute `build/dqmc` on those files to validate changes.

2. **Cluster / production runs**
   - Build the `dqmc` binary on a compute cluster login node.
   - Run a batching script (often named something like `create_batch.py`) that imports and calls functions from `dqmc_util.gen_1band_hub`, iterating over arrays/grids of parameters to generate many HDF5 files across directories.
   - Run the jobs on the cluster, then download the resulting HDF5 files locally.
   - Do analysis locally in Jupyter notebooks; notebooks typically `from dqmc_util import analyze_hub` and use its `get(...)` function to load/organize results.

### Tests (`test/`)
```bash
cd test && make && ./test_greens
```

## Code Patterns & Conventions

### X-Macro Pattern for Data Structures
Parameters and measurements are defined via X-macros in `src/rc/sim_types.h`:
```c
#define PARAMS_SCALAR_INT_LIST \
    X(N) \
    X(L) \
    ...
```
These macros generate struct definitions, HDF5 I/O code, and allocation tables automatically. When adding new parameters, add to the appropriate `*_LIST` macro.

### Memory Management
Use `my_calloc()` from `src/mem.h` for 64-byte aligned allocations. Bulk allocations use `my_calloc_table()` with `struct alloc_entry` arrays.

### Leading Dimension Padding
`best_ld(N)` in `src/rc/linalg.h` computes optimal leading dimensions for BLAS performance—avoids powers of 2 that cause cache conflicts.

### Profiling
Wrap code sections with `profile_begin(name)` / `profile_end(name)`. Profile points are defined in `PROFILE_LIST` in `src/prof.h`.

## HDF5 File Structure
- `/metadata/`: Model info (mu, Nx, Ny, beta)
- `/params/`: Simulation parameters and precomputed matrices
- `/state/`: RNG state, sweep number, auxiliary field config
- `/meas_eqlt/`: Equal-time measurements
- `/meas_uneqlt/`: Unequal-time measurements (optional)

## Platform-Specific Notes
- **Linux**: Requires Intel `icx` compiler and MKL
- **macOS**: Uses `clang` with Apple Accelerate framework
- Compiler selection is automatic via `UNAME` detection in Makefile

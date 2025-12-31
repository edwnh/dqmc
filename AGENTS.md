# AGENTS.md

> Operating manual for LLM agents working with this DQMC codebase.

## Mission & Boundaries

**What this repo does:** Determinantal Quantum Monte Carlo (DQMC) simulation for the Hubbard model. High-performance C code with Python utilities for simulation setup and data analysis.

**Agent modes:**
- **Mode A: Code Development** - Editing C/Python code, running tests, refactoring. **Check `.claude/skills/dqmc-dev/` first.**
- **Mode B: Simulation & Analysis** - Generating sim files, running DQMC, analyzing results. **Check `.claude/skills/` for the relevant runbook before starting.**

**CRITICAL: Skills First**
This repository contains optimized runbooks in `.claude/skills/`.
- **NEVER** manually script parallel runs or parameter sweeps without first consulting `dqmc-run` and `dqmc-parameter-scans`.
- Use the built-in `dqmc-util` queue and worker system as documented in the skills.
- **Default rule:** if you are running **2+ HDF5 files**, use `dqmc-util enqueue` + `dqmc-util worker` (even on a local workstation). Avoid ad-hoc `for` loops unless explicitly requested.

**Generally avoid:**
- Deleting or overwriting existing HDF5 data files
- Changing default parameter values in `gen_1band_hub.py`
- Running simulations with `n_sweep_meas > 10000`
- Creating new virtual environments

---

## Quickstart Commands

```bash
# Environment setup
conda activate dqmc
make deps          # first time only: downloads HDF5

# Build
make               # -> build/dqmc
make clean         # remove build/

# Test
cd test && make && ./test_greens   # unit tests

# Smoke run (minimal simulation)
dqmc-util gen Nx=4 Ny=4 U=4 L=20 n_sweep_warm=50 n_sweep_meas=100
build/dqmc sim_0.h5
dqmc-util summary sim_0.h5
```

---

## Repository Map

```
├── src/                    # C source code
│   ├── main_1.c           # Entry point
│   ├── wrapper.c          # Dispatches to real/complex
│   ├── mem.c/h            # Memory allocation (64-byte aligned)
│   ├── prof.c/h           # Profiling infrastructure
│   └── rc/                # Real/Complex dual-compiled code
│       ├── dqmc.c         # Main sweep loop
│       ├── greens.c/h     # Green's function calculation
│       ├── updates.c/h    # Delayed update scheme
│       ├── meas.c/h       # Measurements (equal/unequal time)
│       ├── data.c/h       # HDF5 I/O
│       ├── sim_types.h    # X-macro parameter/measurement definitions
│       ├── linalg.h       # BLAS/LAPACK wrappers
│       └── numeric.h      # RC() macro for real/complex dispatch
│
├── dqmc_util/              # Python package (pip install -e .)
│   ├── cli.py             # CLI entry point (dqmc-util)
│   ├── gen_1band_hub.py   # Simulation file generation
│   ├── analyze_hub.py     # Analysis routines with @observable pattern
│   ├── core.py            # Jackknife resampling, data loading
│   ├── queue.py           # Sharded queue for clusters
│   └── worker.py          # Worker process management
│
├── test/                   # Tests
│   ├── test_greens.c      # Green's function tests
│   └── bench_linalg.c     # BLAS benchmarks
│
├── examples/               # Complete workflow examples
│   ├── mz2_vs_T/          # Magnetic moment vs temperature
│   └── n_vs_mu/           # Density vs chemical potential
│
├── build/dqmc             # Compiled binary (after make)
└── .claude/skills/         # Agent Skills (agentskills format)
    ├── dqmc-generate/      # Create simulation files
    ├── dqmc-run/           # Run simulations (checkpointing, queue)
    ├── dqmc-analyze/       # Analyze results
    ├── dqmc-parameter-scans/ # Parameter sweeps
    ├── dqmc-dev/           # Code development workflow
    └── dqmc-advanced/      # Unequal-time, MaxEnt
```

---

## Mode A: Code Development

### Call graph
```
main_1.c -> wrapper.c -> rc/dqmc.c
                          ├── data.c (load HDF5)
                          ├── greens.c (Green's function)
                          ├── updates.c (aux field updates)
                          ├── meas.c (measurements)
                          └── data.c (save HDF5)
```

### Key patterns

**Dual compilation:** Files in `src/rc/` compile twice (real + complex). The `RC(name)` macro in `numeric.h` expands to `name_real` or `name_cplx`.

**X-macros:** Parameters and measurements defined in `sim_types.h`:
```c
#define PARAMS_SCALAR_INT_LIST \
    X(N) X(L) ...
```
To add a parameter: add to the appropriate `*_LIST` macro.

**Memory:** Use `my_calloc()` for 64-byte aligned allocations.

**Profiling:** Wrap code with `profile_begin(name)` / `profile_end(name)`.

### Definition of Done (code changes)
- [ ] `make` succeeds without warnings
- [ ] `cd test && make && ./test_greens` passes
- [ ] Smoke run completes: `dqmc-util gen Nx=4 Ny=4 && build/dqmc sim_0.h5`
- [ ] If touching measurements: verify via `dqmc-util summary`

---

## Mode B: Simulation & Analysis

### Three-phase workflow
```
1. Generate  ->  dqmc-util gen [params]  ->  *.h5 files
2. Run       ->  build/dqmc sim.h5       ->  measurements written to HDF5
3. Analyze   ->  analyze_hub.get(...)    ->  (mean, stderr) tuples
```

See the Agent Skills runbooks in `.claude/skills/` for detailed procedures.

### HDF5 file structure
```
/metadata/     # Model info (mu, Nx, Ny, beta)
/params/       # Simulation parameters, precomputed matrices
/state/        # RNG state, sweep number, aux field config
/meas_eqlt/    # Equal-time measurements (n_sample, sign, den, ...)
/meas_uneqlt/  # Unequal-time measurements (optional)
```

### Definition of Done (simulations)
- [ ] `dqmc-util print-n` shows 100% measurements done
- [ ] Error bars reasonable
- [ ] No NaN/Inf in measurements

---

## Guardrails & Safety Rules

1. **Never overwrite raw HDF5 data** - simulations may take hours/days
2. **Never modify `sim_*.h5` files directly** - always use `dqmc-util` or the binary
3. **Checkpoint long runs** - use `build/dqmc -s 300 -t 3600 file.h5` or similar
4. **Trotter error** - ensure `U * dt^2 <= 0.05`

---

## Platform Notes

| Platform | Compiler | BLAS/LAPACK | Notes |
|----------|----------|-------------|-------|
| Linux | `icx` (Intel) | MKL | Auto-detected via Makefile |
| macOS | `clang` | Accelerate | Auto-detected via Makefile |

---

## Links

- `.claude/skills/dqmc-generate/` - Create simulation files
- `.claude/skills/dqmc-run/` - Run simulations (including queue/checkpointing)
- `.claude/skills/dqmc-analyze/` - Analyze results
- `.claude/skills/dqmc-parameter-scans/` - Parameter sweeps
- `.claude/skills/dqmc-dev/` - Code development workflow
- `.claude/skills/dqmc-advanced/` - Unequal-time, MaxEnt
- [README.md](README.md) - Human-oriented documentation
- [examples/](examples/) - Complete workflow examples
- [dqmc_util/gen_1band_hub.py](dqmc_util/gen_1band_hub.py) - All parameter definitions
- [dqmc_util/analyze_hub.py](dqmc_util/analyze_hub.py) - Observable definitions with `@observable` pattern

# Determinantal quantum Monte Carlo for the Hubbard model

## Environment and dependencies

Linux and macOS are the only supported OS currently.

### Build dependencies (Linux)
- Intel compiler `icx`
- Intel MKL headers and libraries
- HDF5 headers and libraries

### Build dependencies (macOS)
- `clang` compiler from Xcode Command Line Tools
- HDF5 headers and libraries

### Runtime dependencies
- None. Libraries are statically linked by default.

### Python packages (for scripts in `util/`)
- `numpy`
- `scipy`
- `h5py`

## Compilation instructions

These instructions should work on most Linux and macOS systems. The easiest way to try the code is to create a GitHub Codespaces with this repository.

1.  If conda/mamba is not installed yet, install conda/mamba (recommend miniforge or micromamba) and initialize (`conda init`).
2.  Create a new conda environment according to environment.yml.
    ```bash
    conda env create -f environment.yml
    ```
    Wait for all the packages to download and install. This might take a while. Then activate the new environment named `dqmc`.
    ```bash
    conda activate dqmc
    ```
3.  Download and extract the HDF5 dependency.
    ```bash
    make deps
    ```
4.  Now the environment is ready for compiling and running the code.
    ```bash
    make
    ```
    The binary executable is located at `build/dqmc`.

## Usage

1. Generate simulation files using `gen_1band_hub.py` or a similar script. Parameters can be passed through the command line. A list of parameters and their default values can be found in the function definitions of `create_1` and `create_batch` in `util/gen_1band_hub.py`.
2. Perform the Monte Carlo sweeps using the `build/dqmc` binary
    1. Usage: `./build/dqmc [-b] [-l log_file.log] [-s interval] [-t max_time] sim_file.h5`.
        * `-b`: Benchmark mode, data is not saved.
        * `-l log_file.log`: Write output to log_file.log instead of stdout.
        * `-s interval`: Saves a checkpoint every interval seconds.
        * `-t max_time`: Run for a maximum of max_time seconds. Saves a checkpoint if simulation does not complete.
4. Analyze the data using the scripts in `util/`. Typically, this is done inside Jupyter notebooks.

### Example

```
(dqmc) @edwnh ➜ /workspaces/dqmc (master) $ python util/gen_1band_hub.py
created simulation files: sim_0.h5
parameter file: sim.h5.params
(dqmc) @edwnh ➜ /workspaces/dqmc (master) $ build/dqmc sim_0.h5
commit id 03f858a
compiled on Jan 21 2025 05:35:45
opening sim_0.h5
0/2200 sweeps completed
starting dqmc
2200/2200 sweeps completed
saving data
cpu model name  : AMD EPYC 7763 64-Core Processor
wall time: 21.158
thread_1/1_______|_% of all_|___total (s)_|___us per call_|___# calls
            wrap |   30.276 |       6.406 |        38.824 |    165000
          recalc |   29.423 |       6.225 |       282.970 |     22000
         updates |   19.414 |       4.108 |        46.679 |     88000
           multb |   14.079 |       2.979 |       135.401 |     22000
       half_wrap |    3.677 |       0.778 |        38.905 |     20000
           calcb |    1.992 |       0.421 |         2.394 |    176000
         meas_eq |    0.970 |       0.205 |        20.531 |     10000
---------------------------------------------------------------------
(dqmc) @edwnh ➜ /workspaces/dqmc (master) $ python util/summary.py sim_0.h5
sim_0.h5
n_sample=10000, sweep=2200/2200
<sign>=1.0
<n>=[1.]
<m_z^2>=[0.82869916]
```

## Details

### Simulation files

One Markov chain = one HDF5 file = one bin. Each HDF5 file contains the following groups:
* metadata: Miscellaneous information and parameters not used during the simulation, but possibly useful in data analysis. Examples: name of the model, Hamiltonian parameters, temperature.
* params: Simulation parameters and pre-calculated matrices used in the simulation. Examples: kinetic energy matrix and its exponential, number of warmup and measurement sweeps. The data in this group may be stored in a separate HDF5 file, with extension .h5.params. This saves storage space since a batch of simulation files typically has identical params.
* state: Simulation state: sweep number, RNG state, auxiliary field configuration
* meas_eqlt: equal-time measurements
* meas_uneqlt: unequal-time measurements i.e. <O(tau) P>. This group exists only if unequal-time measurements are enabled.

`util/gen_1band_hub.py` is the best reference to see the contents of each group.

### Simulation file generation

List of parameters
* `Nfiles`: Number of simulation files to generate. Each file is identical except for the RNG seed. Default: `Nfiles=1`.
* `prefix`: Prefix for the name of each simulation file. Default: `prefix="sim"`.
* `seed`: RNG seed. Default: RNG is initialized by `os.urandom()`.
* `overwrite`: Whether to overwrite existing files. Default: `False`.
* `Nx`, `Ny`: Rectangular cluster geometry. Default: `Nx=16 Ny=4`.
* `mu`: Chemical potential. Default: `mu=0`.
* `tp`: 2nd neighbor hopping. Default: `tp=0`.
* `U`: Hubbard interaction. Default: `U=6`.
* `dt`: Imaginary time discretization. Default: `dt=0.115`.
* `L`: Number of imaginary time steps. Default: `L=40`.
* `nflux`: Number of magnetic flux quanta perpendicular to the plane. Default: `nflux=0`.
* `n_delay`: Number of updates to group together in the delayed update scheme. Default: `n_delay=16`.
* `n_matmul`: Half the maximum number of direct matrix multiplications before applying a QR decomposition for stability. Default: `n_matmul=8`.
* `n_sweep_warm`: Number of warmup sweeps. Default: `n_sweep_warm=200`.
* `n_sweep_meas`: Number of measurement sweeps. Default: `n_sweep_meas=800`.
* `period_eqlt`: Period of equal-time measurements. 1 means equal-time measurements are performed `L` times per spacetime sweep. Default: `period_eqlt=8`.
* `period_uneqlt`: Period of unequal-time measurements. 1 means unequal-time measurements are performed once per spacetime sweep. 0 means disabled. Default: `period_uneqlt=0`.
* `meas_bond_corr`: Whether to measure bond-bond correlations (current, kinetic energy, bond singlets). Default: `meas_bond_corr=1`.
* `meas_energy_corr`: Whether to measure energy-energy correlations. Default: `meas_energy_corr=0`.
* `meas_nematic_corr`: Whether to measure spin and charge nematic correlations. Default: `meas_nematic_corr=0`.
* `trans_sym`: Whether to apply translational symmetry to compress measurement data. Default: `trans_sym=1`.

### Code description

* `build`: build directory
    * `Makefile`: makefile
* `src`: C code
    * `data.c/data.h`: loading and saving simulation data to and from the simulation file
    * `dqmc.c/dqmc.h`: core DQMC algorithm
    * `greens.c/greens.h`: functions for calculating equal and unequal-time Green's functions
    * `linalg.h`: inline wrappers for BLAS/LAPACK functions, to minimize code changes when complex numbers are used
    * `main_1.c`: main function for a binary that runs only a single simulation file
    * `main_stack.c`: main function for a binary that runs the simulation files listed in the stack file
    * `meas.c/meas.h`: code for performing measurements
    * `mem.c/mem.h`: minimal implementation of an aligned memory pool
    * `prof.c/prof.h`: code related to profiling
    * `rand.h`: random number generator
    * `sig.c/sig.h`: signal handling
    * `time_.h`: high resolution timer
    * `updates.c/updates.h`: propose changes to the auxiliary field and update the Green's function accordingly
* `src_py`
    * `dqmc.py`: python implementation of dqmc code.
* `util`: various python scripts for simulation file generation and data analysis
    * `gen_1band_hub.py`: generate HDF5 simulation file for the Hubbard model
    * `get_mu.py`: estimate desired chemical potential for a target filling based on fitting to a sweep of simulations at different chemical potentials
    * `info.py`: print out some parameters of a single simulation file
    * `maxent.py`: code for Maximum Entropy Method analytic continuation
    * `print_n.py`: print out average sign and density, including errorbars, for a batch of simulation files
    * `push.py`: push simulation filenames onto a stack file
    * `summary.py`: print out average sign, density, and local moment for a single simulation file
    * `util.py`: utility functions for data analysis

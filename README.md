# Determinantal quantum Monte Carlo for the Hubbard model

## Prerequisites

- Linux is the only supported OS at this moment.
- `make`
- Intel compiler `icx`
- Intel MKL headers and libraries
- HDF5 headers and libraries

### For python scripts in `util/`
- Python 3
- `numpy`
- `scipy`
- `h5py`

On a local computer: you can get these via Intel oneAPI and miniconda/anaconda.
On a cluster: these are likely installed already as modules.

## Compilation

Go to build/

Optionally, replace `-xHost` in Makefile or Makefile.icx with appropriate instruction set flag for optimization.

Mandatory: pick whether to compile with `-DUSE_CPLX`. Real DQMC uses 8 byte`double`, and can only be used with hdf5 files generated with `nflux=0` option, while Complex DQMC uses 16 byte `complex double`, and can only be used with hdf5 files generated with `nflux!=0` option. 

Run `make`.

## Usage

To (batch-)generate simulation files, run
`python3 gen_1band_hub.py <parameter arguments>`

To push some .h5 files to a stack, run
`python3 push.py <stackfile_name> <some .h5 files>`

Run dqmc in single file mode:
`./dqmc_1 [options] file.h5`

Run dqmc in stack mode:
`./dqmc_stack [options] stackfile`

## Usage details

### Outline

1. Generate simulation files using `gen_1band_hub.py` or a similar script. Parameters can be passed through the command line. A list of parameters and their default values can be found in the function definitions of `create_1` and `create_batch` in `util/gen_1band_hub.py`.
2. Perform the Monte Carlo sweeps using `dqmc_1` (single file mode) or `dqmc_stack` (stack mode).
    1. Single file mode usage: `./build/dqmc_1 [-b] [-l log_file.log] [-t max_time] sim_file.h5`.
        * `-b`: Benchmark mode, data is not saved.
        * `-l log_file.log`: Write output to log_file.log instead of stdout.
        * `-t max_time`: Run for a maximum of max_time seconds. Saves a checkpoint if simulation does not complete.
    2. Stack mode usage:
        1. Append simulation filenames to a stack file: `python util/push.py stack_file file_0.h5 file_1.h5 file_2.h5 ...`.
        2. Run multiple instances of dqmc_stack. For instance, `mpirun -n 4 ./build/dqmc_stack -t 100 stack_file`. The `-t` flag is as in `dqmc_1`. Each instance will read a simulation filename from the last line of the stack file. It then erases that line and starts the simulation. If the simulation does not complete, the filename will be appended back onto the stack file. If the simulation does complete, `dqmc_stack` will try to read another filename from the last line of the stack file. `dqmc_stack` finishes once the stack file is empty or the time limit set by `-t` is reached, in which case all ongoing simulations will be checkpointed and their names appended back onto the stack file.
        3. Logs will be saved to, for instance, `file_0.h5.log`.
3. Analyze the data using the scripts in `util/`. Typically this is done inside Jupyter notebooks.

### Examples

#### Single file mode

```
ewh@7980xe ~/dqmc (master)> python util/gen_1band_hub.py U=8 mu=0 n_sweep_warm=100 n_sweep_meas=500 prefix=abcd Nfiles=1
created simulation files: abcd_0.h5
parameter file: abcd.h5.params
ewh@7980xe ~/dqmc (master)> build/dqmc_1 abcd_0.h5 
commit id 4a55817
compiled on Sep  8 2022 17:03:38
opening abcd_0.h5
0/600 sweeps completed
starting dqmc
600/600 sweeps completed
saving data
wall time: 2.937
thread_1/1_______|_% of all_|___total (s)_|___us per call_|___# calls
          recalc |   47.154 |       1.385 |       230.784 |      6000
         updates |   15.001 |       0.441 |        18.355 |     24000
            wrap |   13.493 |       0.396 |         9.434 |     42000
           calcb |   12.510 |       0.367 |         7.653 |     48000
           multb |    7.456 |       0.219 |        36.493 |      6000
       half_wrap |    1.799 |       0.053 |        10.568 |      5000
         meas_eq |    1.485 |       0.044 |        17.447 |      2500
---------------------------------------------------------------------
ewh@7980xe ~/dqmc (master)> python util/summary.py abcd_0.h5 
abcd_0.h5
n_sample=2500, sweep=600/600
<sign>=1.0
<n>=[1.]
<m_z^2>=[0.88583784]
```

#### Stack mode: 10 Markov chains, 4 instances of dqmc_stack

```
ewh@7980xe ~/dqmc (master)> python util/gen_1band_hub.py U=8 mu=0 n_sweep_warm=100 n_sweep_meas=500 prefix=abcd Nfiles=10
created simulation files: abcd_0.h5 ... abcd_9.h5
parameter file: abcd.h5.params
ewh@7980xe ~/dqmc (master)> python util/push.py stack_file ./abcd_*.h5
ewh@7980xe ~/dqmc (master)> mpirun -n 4 build/dqmc_stack stack_file
          7980xe   6467: starting: /home/ewh/dqmc/abcd_9.h5
          7980xe   6465: starting: /home/ewh/dqmc/abcd_8.h5
          7980xe   6466: starting: /home/ewh/dqmc/abcd_7.h5
          7980xe   6467: completed: /home/ewh/dqmc/abcd_9.h5
          7980xe   6467: starting: /home/ewh/dqmc/abcd_6.h5
          7980xe   6468: starting: /home/ewh/dqmc/abcd_5.h5
          7980xe   6465: completed: /home/ewh/dqmc/abcd_8.h5
          7980xe   6465: starting: /home/ewh/dqmc/abcd_4.h5
          7980xe   6466: completed: /home/ewh/dqmc/abcd_7.h5
          7980xe   6466: starting: /home/ewh/dqmc/abcd_3.h5
          7980xe   6467: completed: /home/ewh/dqmc/abcd_6.h5
          7980xe   6467: starting: /home/ewh/dqmc/abcd_2.h5
          7980xe   6468: completed: /home/ewh/dqmc/abcd_5.h5
          7980xe   6468: starting: /home/ewh/dqmc/abcd_1.h5
          7980xe   6465: completed: /home/ewh/dqmc/abcd_4.h5
          7980xe   6465: starting: /home/ewh/dqmc/abcd_0.h5
          7980xe   6466: completed: /home/ewh/dqmc/abcd_3.h5
          7980xe   6466: pop_stack() returned 1; idling
          7980xe   6467: completed: /home/ewh/dqmc/abcd_2.h5
          7980xe   6467: pop_stack() returned 1; idling
          7980xe   6468: completed: /home/ewh/dqmc/abcd_1.h5
          7980xe   6468: pop_stack() returned 1; idling
          7980xe   6465: completed: /home/ewh/dqmc/abcd_0.h5
          7980xe   6465: pop_stack() returned 1; idling
ewh@7980xe ~/dqmc (master)> python util/print_n.py ./
./
complete: 10/10
<sign>=[1. 0.]
<n>=[1.00000000e+00 1.53378912e-12]
```
### More details

#### Simulation files

One Markov chain = one HDF5 file. A file contains the following groups:
* metadata: Miscellaneous information and parameters not used during the simulation, but possibly useful in data analysis. Examples: name of the model, Hamiltonian parameters, temperature.
* params: Simulation parameters and pre-calculated matrices used in the simulation. Examples: kinetic energy matrix and its exponential, number of warmup and measurement sweeps. The data in this group may be stored in a separate HDF5 file, with extension .h5.params. This saves storage space since a batch of simulation files typically has identical params.
* state: Simulation state: sweep number, RNG state, auxiliary field configuration
* meas_eqlt: equal-time measurements
* meas_uneqlt: unequal-time measurements i.e. <O(tau) P>. This group exists only if unequal-time measurements are enabled.

#### Simulation file generation

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
* `period_eqlt`: Period of equal-time measurements. 1 means equal-time measurements are performed `L` times per sweep. Default: `period_eqlt=8`.
* `period_uneqlt`: Period of unequal-time measurements. 1 means unequal-time measurements are performed once per sweep. 0 means disabled. Default: `period_uneqlt=0`.
* `meas_bond_corr`: Whether to measure bond-bond correlations (current, kinetic energy, bond singlets). Default: `meas_bond_corr=1`.
* `meas_energy_corr`: Whether to measure energy-energy correlations. Default: `meas_energy_corr=0`.
* `meas_nematic_corr`: Whether to measure spin and charge nematic correlations. Default: `meas_nematic_corr=0`.
* `trans_sym`: Whether to apply translational symmetry to compress measurement data. Default: `trans_sym=1`.

#### Code description

* `build`: build directory
    * `Makefile`: makefile for use with icc compiler
    * `Makefile.icx`: makefile for use with icx compiler
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


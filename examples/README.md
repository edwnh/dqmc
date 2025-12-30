This directory contains a few examples demonstrating typical workflows using the DQMC code.

1. Make sure the dqmc binary has been compiled and the dqmc environment is activated (see the main README.md for details). `cd` into an example directory and run the following commands in order:
2. `python 1_generate_sim_files.py`: generate simulation files with parameter tables defined inside.
3. `bash 2_run_local.sh`: run the generated simulations with the binary at `../../build/dqmc`. Since this is an example, the simulations are run locally. For actual production runs, you'll want to use a cluster with its job scheduler, so make a job submission script.
4. `python 3_analyze_and_plot.py`: analyze the raw data in the simulation files and plot the results.

Examples:
- `mz2_vs_T/`: magnetic moment vs temperature
- `n_vs_mu/`: density vs chemical potential

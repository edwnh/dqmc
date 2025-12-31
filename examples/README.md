This directory contains end-to-end workflow examples (generate -> run -> analyze).

From an example directory:

1. `python 1_generate_sim_files.py`: generate HDF5 simulation files (parameters are defined in the script).
2. `bash 2_run_local.sh`: run the simulations locally using `../../build/dqmc`.
3. `python 3_analyze_and_plot.py`: analyze results and generate plots.

For real multi-file production runs, prefer the sharded queue system described in `README.md` (or `.claude/skills/dqmc-run/`).

Examples:
- `mz2_vs_T/`: magnetic moment vs temperature
- `n_vs_mu/`: density vs chemical potential

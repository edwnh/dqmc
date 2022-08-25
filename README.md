# Complex-capable DQMC

## Prerequisites

- `git`
- `make`

### For compilation

- Intel compiler `icc`/`icx`
- `imkl` headers and libraries
- `hdf5` headers and libraries

### For python scripts in `util/`
- Python 3
- `numpy`
- `h5py`
- `scipy`

You can get these via miniconda/anaconda.

## Compilation

Go to build/

Optionally, replace `-xHost` in Makefile or Makefile.icx with appropriate instruction set flag for optimization.

Mandatory: pick whether to compile with `-DUSE_CPLX`. Real DQMC uses 8 byte`double`, and can only be used with hdf5 files generated with `nflux=0` option, while Complex DQMC uses 16 byte `complex double`, and can only be used with hdf5 files generated with `nflux!=0` option. 

Run `make` if using `icc` or `make -f Makefile.icx` when using `icx`.

## Usage

To (batch-)generate simulation files, run
`python3 gen_1band_hub.py <parameter arguments>`

To push some .h5 files to a stack, run
`python3 push.py <stackfile_name> <some .h5 files>`

Run dqmc in single file mode:
`./dqmc_1 [options] file.h5`

Run dqmc in stack mode:
`./dqmc_stack [options] stackfile`

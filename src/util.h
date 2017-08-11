#ifndef _UTIL_H
#define _UTIL_H

#include <stdint.h>
#include <string.h>
#include <mkl.h>

#define MEM_ALIGN 64
#define DBL_ALIGN (MEM_ALIGN/sizeof(double))

#define _aa(a) __assume_aligned((a), MEM_ALIGN);

#define my_malloc(size) _mm_malloc((size), MEM_ALIGN)

#define my_calloc(size) memset(my_malloc((size)), 0, (size))

#define my_free(ptr) _mm_free((ptr))

#define my_copy(dest, src, N) memcpy((dest), (src), (N)*sizeof((src)[0]))

// use these since fortran blas/lapack function take pointers for arguments
#define cint(x) &(const int){(x)}
#define cdbl(x) &(const double){(x)}

struct params {
	int N, L;
	int *map_i, *map_ij;
//	double *K, *U;
//	double dt;

	int n_matmul, n_delay;
	int n_sweep_warm, n_sweep_meas;
	int period_eqlt, period_uneqlt;

	int num_i, num_ij;
	int *degen_i, *degen_ij;
	double *exp_K, *inv_exp_K;
	double *exp_lambda, *del;
	int F, n_sweep;
};

struct state {
	uint64_t rng[17];
	int sweep;
	int *hs;
};

struct meas_eqlt {
	int n_sample;
	double sign;

	double *density;
	double *double_occ;

	double *g00;
	double *nn;
	double *xx;
	double *zz;
	double *pair_sw;
};

struct meas_uneqlt {
	int n_sample;
	double sign;

	double *g0t;
	double *gt0;
	double *nn;
	double *xx;
	double *zz;
	double *pair_sw;
};

#endif

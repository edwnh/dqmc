#ifndef _UPDATES_H
#define _UPDATES_H

#include <stdint.h>

int update_delayed(const int N, const int n_delay, const double *const restrict del,
		uint64_t *const restrict rng, int *const restrict hs,
		double *const restrict Gu, double *const restrict Gd,
		int *const restrict site_order, // for order of sites
		// work arrays (sizes: N*N, N*N, N)
		double *const restrict au, double *const restrict bu, double *const restrict du,
		double *const restrict ad, double *const restrict bd, double *const restrict dd);

/*
// regular sherman morrison
int update_shermor(const int N, const double *const restrict del,
		uint64_t *const restrict rng, int *const restrict hs,
		double *const restrict Gu, double *const restrict Gd,
		int *const restrict site_order, // for order of sites
		// work arrays (sizes: N, N)
		double *const restrict cu, double *const restrict du,
		double *const restrict cd, double *const restrict dd);

// submatrix updates. generally worse performance except for very large N (>1000)
int update_submat(const int N, const int q, const double *const restrict del,
		uint64_t *const restrict rng, int *const restrict hs,
		double *const restrict Gu, double *const restrict Gd,
		int *const restrict site_order, // for order of sites
		// work arrays (sizes: N*N, N*N, N, N, N)
		double *const restrict Gr_u, double *const restrict G_ru,
		double *const restrict DDu, double *const restrict yu, double *const restrict xu,
		double *const restrict Gr_d, double *const restrict G_rd,
		double *const restrict DDd, double *const restrict yd, double *const restrict xd);
*/

#endif

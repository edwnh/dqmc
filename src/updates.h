#pragma once

#include <stdint.h>
#include "linalg.h"

void update_delayed(const int N, const int ld, const int n_delay, const double *const del,
		const int *const site_order,
		uint64_t *const rng, int *const hs,
		num *const Gu, num *const Gd, num *const phase,
		// work arrays (sizes: N*N, N*N, N)
		num *const au, num *const bu, num *const du,
		num *const ad, num *const bd, num *const dd);

/*
// regular sherman morrison
void update_shermor(const int N, const double *const del,
		const int *const site_order,
		uint64_t *const rng, int *const hs,
		double *const Gu, double *const Gd, int *const sign,
		// work arrays (sizes: N, N)
		double *const cu, double *const du,
		double *const cd, double *const dd);

// submatrix updates. generally worse performance except for very large N (>1000)
void update_submat(const int N, const int q, const double *const del,
		const int *const site_order,
		uint64_t *const rng, int *const hs,
		double *const Gu, double *const Gd, int *const sign,
		// work arrays (sizes: N*N, N*N, N, N, N)
		double *const Gr_u, double *const G_ru,
		double *const DDu, double *const yu, double *const xu,
		double *const Gr_d, double *const G_rd,
		double *const DDd, double *const yd, double *const xd);
*/

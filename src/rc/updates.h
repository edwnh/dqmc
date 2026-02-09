#pragma once

#include <stdint.h>
#include "linalg.h"

void RC(update_delayed)(const int N, const int ld, const int n_delay, const double *const del,
		const int *const site_order,
		uint64_t *const rng, int *const hs,
		num *const Gu, num *const Gd, num *const phase,
		// work arrays (sizes: N*N, N*N, N)
		num *const au, num *const bu, num *const du,
		num *const ad, num *const bd, num *const dd);

void RC(update_woodbury)(const int N, const int ld, const int num_b_V,
		const num *const del, const num *const dela, const num *const pre_ratio,
		const int *const bond_order, const int *const bonds_V,
		uint64_t *const rng, int *const hs,
		num *const gu, num *const gd, num *const phase,
		// work arrays (size: N*N, N*N, N*2)
		num *const au, num *const bu, num *const cu,
		num *const ad, num *const bd, num *const cd);
/*
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

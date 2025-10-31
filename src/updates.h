#pragma once

#include <stdint.h>
#include "linalg.h"

void update_shermor(const int N, const int ld, const int N_inter, const int *const bonds_inter,
	const int *const site_order, uint64_t *const rng, int *const hs,
	num *const gu, num *const gd, num *const phase,
	double *const Delta_p, double *const Delta_q, double *const weight_comparison_matrice,
	num *const cu_p_col, num *const cu_q_col, num *const cu_p_row, 
	num *const cu_q_row, num *const cu_col, num *const cu_row, 
	num *const cd_p_col, num *const cd_q_col, num *const cd_p_row, 
	num *const cd_q_row, num *const cd_col, num *const cd_row, 
	num *const tmpuld2, num *const tmpdld2
);

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

#pragma once

int get_lwork_eq_g(const int N);

int calc_eq_g(const int l, const int N, const int stride, const int L, const int n_mul,
		const double *const restrict B, double *const restrict g,
		// work arrays
		double *const restrict Q, double *const restrict T,
		double *const restrict tau, double *const restrict d,
		double *const restrict v, int *const restrict pvt,
		double *const restrict work, const int lwork);

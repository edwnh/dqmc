#pragma once

int get_lwork_ue_G(const int N, const int L);

void calc_ue_g(const int N, const int stride, const int L, const int F,
		const double *const restrict B, const double *const restrict iB,
		const double *const restrict C,
		double *const restrict G,
		double *const restrict Gred, // NF * NF
		double *const restrict tau, // NF
		double *const restrict Q, // 2*N * 2*N
		double *const restrict work, const int lwork);

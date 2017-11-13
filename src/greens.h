#pragma once

void mul_seq(const int N, const int L,
		const int min, const int maxp1,
		const double alpha, const double *const restrict B,
		double *const restrict A, const int ldA,
		double *const restrict work);

int get_lwork_eq_g(const int N);

int calc_eq_g(const int l, const int N, const int L, const int n_mul,
		const double *const restrict B, double *const restrict g,
		// work arrays
		double *const restrict Q, double *const restrict T,
		double *const restrict tau, double *const restrict d,
		double *const restrict v, int *const restrict pvt,
		double *const restrict work, const int lwork);

int get_lwork_ue_g(const int N, const int L);

void calc_ue_g(const int N, const int L, const int F, const int n_mul,
		const double *const restrict B, const double *const restrict iB,
		const double *const restrict C,
		double *const restrict G0t, double *const restrict Gtt,
		double *const restrict Gt0,
		double *const restrict Gred,
		double *const restrict tau,
		double *const restrict Q,
		double *const restrict work, const int lwork);

/*
void calc_ue_g_full(const int N, const int L, const int F, const int n_mul,
		const double *const restrict B, const double *const restrict iB,
		const double *const restrict C,
		double *const restrict G, // NL * NL
		double *const restrict Gred, // NF * NF
		double *const restrict tau, // NF
		double *const restrict Q, // 2*N * 2*N
		double *const restrict work, const int lwork);
*/

#pragma once

#include <complex.h>

void mul_seq(const int N, const int L,
		const int min, const int maxp1,
		const complex double alpha, const complex double *const restrict B,
		complex double *const restrict A, const int ldA,
		complex double *const restrict work);

int get_lwork_eq_g(const int N);

complex double calc_eq_g(const int l, const int N, const int L, const int n_mul,
		const complex double *const restrict B, complex double *const restrict g,
		// work arrays
		complex double *const restrict Q, complex double *const restrict T,
		complex double *const restrict tau, complex double *const restrict d,
		complex double *const restrict v, int *const restrict pvt,
		complex double *const restrict work, const int lwork);

int get_lwork_ue_g(const int N, const int L);

void calc_ue_g(const int N, const int L, const int F, const int n_mul,
		const complex double *const restrict B, const complex double *const restrict iB,
		const complex double *const restrict C,
		complex double *const restrict G0t, complex double *const restrict Gtt,
		complex double *const restrict Gt0,
		complex double *const restrict Gred,
		complex double *const restrict tau,
		complex double *const restrict Q,
		complex double *const restrict work, const int lwork);

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

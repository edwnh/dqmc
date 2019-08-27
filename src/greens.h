#pragma once

#include "util.h"

void mul_seq(const int N, const int L,
		const int min, const int maxp1,
		const num alpha, const num *const restrict B,
		num *const restrict A, const int ldA,
		num *const restrict work);

int get_lwork_eq_g(const int N);

num calc_eq_g(const int l, const int N, const int L, const int n_mul,
		const num *const restrict B, num *const restrict g,
		// work arrays
		num *const restrict Q, num *const restrict T,
		num *const restrict tau, num *const restrict d,
		num *const restrict v, int *const restrict pvt,
		num *const restrict work, const int lwork);

int get_lwork_ue_g(const int N, const int L);

void calc_ue_g(const int N, const int L, const int F, const int n_mul,
		const num *const restrict B, const num *const restrict iB,
		const num *const restrict C,
		num *const restrict G0t, num *const restrict Gtt,
		num *const restrict Gt0,
		num *const restrict Gred,
		num *const restrict tau,
		num *const restrict Q,
		num *const restrict work, const int lwork);

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

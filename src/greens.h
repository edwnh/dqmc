#pragma once

#include "linalg.h"

struct QdX {
    num *Q;
    num *d;
    num *X;
	num *iL;
	num *R;
	num phase_iL;
};

void mul_seq(const int N,
		const int min, const int maxp1,
		const num alpha, num *const *const B, const int ldB,
		num *const A, const int ldA,
		num *const tmpNN);

int get_lwork(const int N, const int ld);

void calc_QdX_first(
		const int trans, // if 1, calculate QdX of B^T (conjugate transpose for complex)
		const int N, const int ld,
		const num *const B, // input
		struct QdX *const QdX, // output
		num *const tmpN, // work arrays
		int *const pvt,
		num *const work, const int lwork);

void calc_QdX(
		const int trans, // if 1, calculate QdX of B^T (conjugate transpose for complex)
		const int N, const int ld,
		const num *const B, // input
		const struct QdX *const QdX_prev,  // input, previous QdX
		struct QdX *const QdX,  // output
		num *const tmpN, // work arrays
		int *const pvt,
		num *const work, const int lwork);

num calc_Gtt_last(
		const int trans, // if 0 calculate, calculate G = (1 + L R)^-1. if 1, calculate G = (1 + R.T L.T)^-1
		const int N, const int ld,
		const struct QdX *const QdX, // input
		num *const G, // output
		num *const tmpNN, // work arrays
		int *const pvt);

num calc_Gtt(
		const int N, const int ld,
		const struct QdX *const QdX0, // input
		const struct QdX *const QdX1, // input
		num *const G, // output
		num *const tmpNN, // work arrays
		int *const pvt);

void calc_G0t_Gtt_Gt0(
		const int N, const int ld,
		const struct QdX *const QdX0, // input
		const struct QdX *const QdX1, // input
		num *const G0t, // output
		num *const Gtt, // output
		num *const Gt0, // output
		num *const tmpNN, // work arrays
		int *const pvt);

void calc_ue_g(const int N, const int ld, const int L, const int F, const int n_matmul,
		num *const *const B,
		num *const *const iB,
		const struct QdX *const QdX0, // input
		const struct QdX *const QdXL, // input
		num *const *const G0t, // output
		num *const *const Gtt, // output
		num *const *const Gt0, // output
		num *const tmpNN, // work arrays
		int *const pvt);

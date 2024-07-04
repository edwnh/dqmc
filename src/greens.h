#pragma once

#include "linalg.h"

struct QdX {
    num *Q;
    num *d;
    num *X;
};

struct LR {
	num *iL;
	num *R;
	num *phase_iL;
};

void mul_seq(const int N, const int ld,
		const int min, const int maxp1,
		const num alpha, const num *const B,
		num *const A,
		num *const tmpNN);

void wrap(const int N, const int ld,
		num *const G,
		const num *const L, const num *const R,
		num *const tmpNN);

int get_lwork(const int N, const int ld);

void calc_QdX(
		const int trans, // if 1, calculate QdX of B^T (conjugate transpose for complex)
		const int N, const int ld,
		const num *const B, // input
		const struct QdX QdX_prev,  // input, previous QdX or NULL if none
		struct QdX QdX,  // output
		struct LR LR, // output
		num *const tmpN, // work arrays
		int *const pvt,
		num *const work, const int lwork);

num calc_Gtt(
		const int N, const int ld,
		const struct LR LR0, // input or NULL if none
		const struct LR LR1, // input or NULL if none
		num *const G, // output
		num *const tmpNN, // work arrays
		int *const pvt);

void calc_G0t_Gtt_Gt0(
		const int N, const int ld,
		const struct LR LR0, // input
		const struct LR LR1, // input
		num *const G0t, // output
		num *const Gtt, // output
		num *const Gt0, // output
		num *const tmpNN, // work arrays
		int *const pvt);

void calc_ue_g(const int N, const int ld, const int L, const int F, const int n_matmul,
		const num *const B, // input
		const num *const iB, // input
		num *const iL_0, // input
		num *const R_0, // input
		num *const iL_L, // input
		num *const R_L, // input
		num *const G0t, // output
		num *const Gtt, // output
		num *const Gt0, // output
		num *const tmpNN, // work arrays
		int *const pvt);

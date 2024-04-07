#pragma once

#include "linalg.h"

struct QdX {
    num *Q;
    num *tau;
    num *d;
    num *X;
};

void mul_seq(const int N,
		const int min, const int maxp1,
		const num alpha, num *const *const B, const int ldB,
		num *const A, const int ldA,
		num *const tmpNN);

int get_lwork_eq_g(const int N, const int ld);

void calc_QdX_first(
		const int trans, // if 1, calculate QdX of B^T (conjugate transpose for complex)
		const int N, const int ld,
		const num *const B, // input
		const struct QdX *const QdX, // output
		num *const tmpN, // work arrays
		int *const pvt,
		num *const work, const int lwork);

void calc_QdX(
		const int trans, // if 1, calculate QdX of B^T (conjugate transpose for complex)
		const int N, const int ld,
		const num *const B, // input
		const struct QdX *const QdX_prev,  // input, previous QdX
		const struct QdX *const QdX,  // output
		num *const tmpN, // work arrays
		int *const pvt,
		num *const work, const int lwork);

num calc_Gtt_last(
		const int trans, // if 0 calculate, calculate G = (1 + Q d X)^-1. if 1, calculate G = (1 + X.T d Q.T)^-1
		const int N, const int ld,
		const struct QdX *const QdX, // input
		num *const G, // output
		num *const tmpNN, // work arrays
		num *const tmpN,
		int *const pvt,
		num *const work, const int lwork);

num calc_Gtt(
		const int N, const int ld,
		const struct QdX *const QdX0, // input
		const struct QdX *const QdX1, // input
		num *const G, // output
		num *const tmpNN0, // work arrays
		num *const tmpNN1,
		num *const tmpN0,
		num *const tmpN1,
		int *const pvt,
		num *const work, const int lwork);


int get_lwork_ue_g(const int N, const int L);

void calc_ue_g(const int N, const int ld, const int L, const int F, const int n_mul,
		num *const *const B,
		num *const *const iB,
		num *const *const C,
		num *const G0t, num *const Gtt,
		num *const Gt0,
		num *const Gred,
		num *const tau,
		num *const Q,
		num *const work, const int lwork);

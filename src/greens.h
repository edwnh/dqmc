#pragma once

#include "linalg.h"

struct QdX {
    num *restrict Q;
    num *restrict tau;
    num *restrict d;
    num *restrict X;
};

void mul_seq(const int N, const int L,
		const int min, const int maxp1,
		const num alpha, num *const restrict *const restrict B,
		num *const restrict A, const int ldA,
		num *const restrict tmpNN);

int get_lwork_eq_g(const int N);

void calc_QdX_first(
		const int trans, // if 1, calculate QdX of B^T (conjugate transpose for complex)
		const int N,
		const num *const restrict B, // input
		const struct QdX *const restrict QdX, // output
		num *const restrict tmpN, // work arrays
		int *const restrict pvt,
		num *const restrict work, const int lwork);

void calc_QdX(
		const int trans, // if 1, calculate QdX of B^T (conjugate transpose for complex)
		const int N,
		const num *const restrict B, // input
		const struct QdX *const restrict QdX_prev,  // input, previous QdX
		const struct QdX *const restrict QdX,  // output
		num *const restrict tmpN, // work arrays
		int *const restrict pvt,
		num *const restrict work, const int lwork);

num calc_Gtt_last(
		const int trans, // if 0 calculate, calculate G = (1 + Q d X)^-1. if 1, calculate G = (1 + X.T d Q.T)^-1
		const int N,
		const struct QdX *const restrict QdX, // input
		num *const restrict G, // output
		num *const restrict tmpNN, // work arrays
		num *const restrict tmpN,
		int *const restrict pvt,
		num *const restrict work, const int lwork);

num calc_Gtt(
		const int N,
		const struct QdX *const restrict QdX0, // input
		const struct QdX *const restrict QdX1, // input
		num *const restrict G, // output
		num *const restrict tmpNN0, // work arrays
		num *const restrict tmpNN1,
		num *const restrict tmpN0,
		num *const restrict tmpN1,
		int *const restrict pvt,
		num *const restrict work, const int lwork);


int get_lwork_ue_g(const int N, const int L);

void calc_ue_g(const int N, const int L, const int F, const int n_mul,
		num *const restrict *const restrict B,
		num *const restrict *const restrict iB,
		num *const restrict *const restrict C,
		num *const restrict G0t, num *const restrict Gtt,
		num *const restrict Gt0,
		num *const restrict Gred,
		num *const restrict tau,
		num *const restrict Q,
		num *const restrict work, const int lwork);

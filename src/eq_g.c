#include "eq_g.h"
#include <math.h>
#include <mkl.h>
#include "util.h"

int get_lwork_eq_g(const int N)
{
	double lwork;
	int info = 0;
	int max_lwork = N*N; // can be smaller if mul_seq doesn't use work

	dgeqp3(&N, &N, NULL, &N, NULL, NULL, &lwork, cint(-1), &info);
	if (lwork > max_lwork) max_lwork = (int)lwork;

	dgeqrf(&N, &N, NULL, &N, NULL, &lwork, cint(-1), &info);
	if (lwork > max_lwork) max_lwork = (int)lwork;

	dormqr("R", "N", &N, &N, &N, NULL, &N, NULL, NULL, &N, &lwork, cint(-1), &info);
	if (lwork > max_lwork) max_lwork = (int)lwork;

	dormqr("R", "T", &N, &N, &N, NULL, &N, NULL, NULL, &N, &lwork, cint(-1), &info);
	if (lwork > max_lwork) max_lwork = (int)lwork;

	return max_lwork;
}

static inline void mul_seq(const int N, const int stride, const int L,
		const int min, const int maxp1, const double *const restrict B,
		double *const restrict A, double *const restrict work)
{
	__assume(stride % DBL_ALIGN == 0);
	_aa(B); _aa(A); _aa(work);

	const int n_mul = (L + maxp1 - min) % L;
	if (n_mul == 1) {
		my_copy(A, B + stride*min, N*N);
		return;
	}

	int l = min;
	if (n_mul % 2 == 0) {
		dgemm("N", "N", &N, &N, &N, cdbl(1.0), B + stride*((l + 1)%L),
		      &N, B + stride*l, &N, cdbl(0.0), A, &N);
		l = (l + 2) % L;
	} else {
		dgemm("N", "N", &N, &N, &N, cdbl(1.0), B + stride*((l + 1)%L),
		      &N, B + stride*l, &N, cdbl(0.0), work, &N);
		dgemm("N", "N", &N, &N, &N, cdbl(1.0), B + stride*((l + 2)%L),
		      &N, work, &N, cdbl(0.0), A, &N);
		l = (l + 3) % L;
	}

	for (; l != maxp1; l = (l + 2) % L) {
		dgemm("N", "N", &N, &N, &N, cdbl(1.0), B + stride*l,
		      &N, A, &N, cdbl(0.0), work, &N);
		dgemm("N", "N", &N, &N, &N, cdbl(1.0), B + stride*((l + 1)%L),
		      &N, work, &N, cdbl(0.0), A, &N);
	}
}

int calc_eq_g(const int l, const int N, const int stride, const int L, const int n_mul,
		const double *const restrict B, double *const restrict g,
		double *const restrict Q, double *const restrict T,
		double *const restrict tau, double *const restrict d,
		double *const restrict v, int *const restrict pvt,
		double *const restrict work, const int lwork)
{
	__assume(stride % DBL_ALIGN == 0);
	_aa(B); _aa(g); _aa(Q); _aa(T); _aa(tau); _aa(d); _aa(v); _aa(pvt);

	int info = 0;

	// algorithm 3 of 10.1109/IPDPS.2012.37
	// slightly modified; groups of n_mul matrices are multiplied with dgemm
	// if n_mul == 2 && l == 0:
	// like (B5 B4)(B3 B2)(B1 B0) if L is even
	// or (B6 B5)(B4 B3)(B2 B1)(B0) if L is odd
	// (1)

	int m = (l + 1 + (L - 1) % n_mul) % L;
	mul_seq(N, stride, L, l, m, B, Q, work);

	for (int i = 0; i < N; i++) pvt[i] = 0;
	dgeqp3(&N, &N, Q, &N, pvt, tau, work, &lwork, &info);

	// (2)
	for (int i = 0; i < N; i++) {
		d[i] = Q[i + i*N];
		if (d[i] == 0.0) d[i] = 1.0;
		v[i] = 1.0/d[i];
	}

	for (int i = 0; i < N*N; i++) T[i] = 0.0;
	for (int j = 0; j < N; j++)
		for (int i = 0; i <= j; i++)
			T[i + (pvt[j]-1)*N] = v[i] * Q[i + j*N];


	while (m != l) {
		const int next = (m + n_mul) % L;
		mul_seq(N, stride, L, m, next, B, g, work);
		m = next;

		// (3a)
		dormqr("R", "N", &N, &N, &N, Q, &N, tau, g, &N, work, &lwork, &info);

		for (int j = 0; j < N; j++)
			for (int i = 0; i < N; i++)
				g[i + j*N] *= d[j];

		// (3b)
		for (int j = 0; j < N; j++) { // use d for norms
			d[j] = 0.0;
			for (int i = 0; i < N; i++)
				d[j] += g[i + j*N] * g[i + j*N];
		}

		pvt[0] = 0;
		for (int i = 1; i < N; i++) { // insertion sort
			int j;
			for (j = i; j > 0 && d[pvt[j-1]] < d[i]; j--)
				pvt[j] = pvt[j-1];
			pvt[j] = i;
		}

		for (int j = 0; j < N; j++) // pre-pivot
			my_copy(Q + j*N, g + pvt[j]*N, N);

		// (3c)
		dgeqrf(&N, &N, Q, &N, tau, work, &lwork, &info);

		// (3d)
		for (int i = 0; i < N; i++) {
			d[i] = Q[i + i*N];
			if (d[i] == 0.0) d[i] = 1.0;
			v[i] = 1.0/d[i];
		}

		for (int j = 0; j < N; j++)
			for (int i = 0; i <= j; i++)
				Q[i + j*N] *= v[i];

		for (int j = 0; j < N; j++) {
			for (int i = 0; i < N; i++)
				v[i] = T[pvt[i] + j*N];
			my_copy(T + j*N, v, N);
		}

		dtrmm("L", "U", "N", "N", &N, &N, cdbl(1.0), Q, &N, T, &N);
	}

	// construct g from Eq 2.12 of 10.1016/j.laa.2010.06.023
	for (int i = 0; i < N*N; i++) g[i] = 0.0;
	for (int i = 0; i < N; i++) {
		if (fabs(d[i]) > 1.0) { // v = 1/Db; d = Ds
			v[i] = 1.0/d[i];
			d[i] = 1.0;
		} else {
			v[i] = 1.0;
		}
		g[i+i*N] = v[i];
	}

	dormqr("R", "T", &N, &N, &N, Q, &N, tau, g, &N, work, &lwork, &info);

	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			T[i + j*N] *= d[i];

	for (int i = 0; i < N*N; i++) T[i] += g[i];

	dgetrf(&N, &N, T, &N, pvt, &info);
	dgetrs("N", &N, &N, T, &N, pvt, g, &N, &info);

	// calculate sign of det(g)
	int sign = 1;
	for (int i = 0; i < N; i++)
		if ((T[i + N*i] < 0) ^ (pvt[i] != i + 1) ^ (v[i] < 0) ^ (tau[i] > 0))
			sign *= -1;

	return sign;
}

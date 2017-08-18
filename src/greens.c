#include "greens.h"
#include <math.h>
#include <mkl.h>
#include "prof.h"
#include "util.h"

void mul_seq(const int N, const int stride, const int L,
		const int min, const int maxp1,
		const double alpha, const double *const restrict B,
		double *const restrict A, const int ldA,
		double *const restrict work)
{
	__assume(stride % DBL_ALIGN == 0);
	_aa(B); _aa(work);

	const int n_mul = (min == maxp1) ? L : (L + maxp1 - min) % L;
	if (n_mul == 1) {
		for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			A[i + ldA*j] = alpha*B[i + N*j + stride*min];
		return;
	}

	int l = min;
	if (n_mul % 2 == 0) {
		dgemm("N", "N", &N, &N, &N, &alpha, B + stride*((l + 1)%L),
		      &N, B + stride*l, &N, cdbl(0.0), A, &ldA);
		l = (l + 2) % L;
	} else {
		dgemm("N", "N", &N, &N, &N, &alpha, B + stride*((l + 1)%L),
		      &N, B + stride*l, &N, cdbl(0.0), work, &N);
		dgemm("N", "N", &N, &N, &N, cdbl(1.0), B + stride*((l + 2)%L),
		      &N, work, &N, cdbl(0.0), A, &ldA);
		l = (l + 3) % L;
	}

	for (; l != maxp1; l = (l + 2) % L) {
		dgemm("N", "N", &N, &N, &N, cdbl(1.0), B + stride*l,
		      &N, A, &ldA, cdbl(0.0), work, &N);
		dgemm("N", "N", &N, &N, &N, cdbl(1.0), B + stride*((l + 1)%L),
		      &N, work, &N, cdbl(0.0), A, &ldA);
	}
}

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
	mul_seq(N, stride, L, l, m, 1.0, B, Q, N, work);

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
		mul_seq(N, stride, L, m, next, 1.0, B, g, N, work);
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

#define G_BLK(i, j) (G + N*(i) + NL*N*(j))

int get_lwork_ue_g(const int N, const int L)
{
	double lwork;
	int info = 0;
	int max_lwork = N*N; // can start smaller if mul_seq doesn't use work

	const int NL = N*L;
	const int N2 = 2*N;

	dgeqrf(&N2, &N, NULL, &NL, NULL, &lwork, cint(-1), &info);
	if (lwork > max_lwork) max_lwork = (int)lwork;

	dormqr("L", "T", &N2, &N, &N, NULL, &NL, NULL, NULL, &NL, &lwork, cint(-1), &info);
	if (lwork > max_lwork) max_lwork = (int)lwork;

	dgeqrf(&N2, &N2, NULL, &NL, NULL, &lwork, cint(-1), &info);
	if (lwork > max_lwork) max_lwork = (int)lwork;

	dormqr("R", "T", &NL, &N2, &N2, NULL, &N2, NULL, NULL, &NL, &lwork, cint(-1), &info);
	if (lwork > max_lwork) max_lwork = (int)lwork;

	dormqr("R", "T", &NL, &N2, &N, NULL, &N2, NULL, NULL, &NL, &lwork, cint(-1), &info);
	if (lwork > max_lwork) max_lwork = (int)lwork;

	return max_lwork;
}

static void calc_o(const int N, const int stride, const int L, const int n_mul,
		const double *const restrict B, double *const restrict G,
		double *const restrict work)
{
	__assume(stride % DBL_ALIGN == 0);
	_aa(B); _aa(G);

	const int E = 1 + (L - 1) / n_mul;
	const int NE = N*E;

	for (int i = 0; i < NE * NE; i++) G[i] = 0.0;

	for (int e = 0; e < E - 1; e++) // subdiagonal blocks
		mul_seq(N, stride, L, e*n_mul, (e + 1)*n_mul, -1.0, B,
		        G + N*(e + 1) + NE*N*e, NE, work);

	mul_seq(N, stride, L, (E - 1)*n_mul, 0, 1.0, B, // top right corner
		G + NE*N*(E - 1), NE, work);

	for (int i = 0; i < NE; i++) G[i + NE*i] += 1.0; // 1 on diagonal
}

static void bsofi(const int N, const int L,
		double *const restrict G, // input: O matrix, output: G = O^-1
		double *const restrict tau, // NL
		double *const restrict Q, // 2*N * 2*N
		double *const restrict work, const int lwork)
{
	_aa(G); _aa(tau); _aa(Q); _aa(work);

	int info;

	if (L == 1) {
		dgetrf(&N, &N, G, &N, (int *)tau, &info);
		dgetri(&N, G, &N, (int *)tau, work, &lwork, &info);
		return;
	}

	const int NL = N*L;
	const int N2 = 2*N;

	// block qr
	for (int l = 0; l < L - 2; l++) {
		dgeqrf(&N2, &N, G_BLK(l, l), &NL, tau + N*l, work, &lwork, &info);
		dormqr("L", "T", &N2, &N, &N, G_BLK(l, l), &NL, tau + N*l,
		       G_BLK(l, l + 1), &NL, work, &lwork, &info);
		dormqr("L", "T", &N2, &N, &N, G_BLK(l, l), &NL, tau + N*l,
		       G_BLK(l, L - 1), &NL, work, &lwork, &info);
	}
	dgeqrf(&N2, &N2, G_BLK(L - 2, L - 2), &NL, tau + N*(L - 2), work, &lwork, &info);

	// invert r
	if (L <= 2) {
		dtrtri("U", "N", &NL, G, &NL, &info);
	} else {
		dtrtri("U", "N", cint(3*N), G_BLK(L - 3, L - 3), &NL, &info);
		if (L > 3) {
			dtrmm("R", "U", "N", "N", cint(N*(L - 3)), &N, cdbl(1.0),
			      G_BLK(L - 1, L - 1), &NL, G_BLK(0, L - 1), &NL);
			for (int l = L - 4; l >= 0; l--) {
				dtrtri("U", "N", &N, G_BLK(l, l), &NL, &info);
				dtrmm("L", "U", "N", "N", &N, &N, cdbl(-1.0),
				      G_BLK(l, l), &NL, G_BLK(l, L - 1), &NL);
				dtrmm("L", "U", "N", "N", &N, &N, cdbl(-1.0),
				      G_BLK(l, l), &NL, G_BLK(l, l + 1), &NL);
				dgemm("N", "N", &N, cint(N*(L - l - 2)), &N, cdbl(1.0),
				      G_BLK(l, l + 1), &NL, G_BLK(l + 1, l + 2), &NL, cdbl(1.0),
				      G_BLK(l, l + 2), &NL);
				dtrmm("R", "U", "N", "N", &N, &N, cdbl(1.0),
				      G_BLK(l + 1, l + 1), &NL, G_BLK(l, l + 1), &NL);
			}
		}
	}

	// multiply by q inverse
	for (int i = 0; i < 4*N*N; i++) Q[i] = 0.0;

	for (int j = 0; j < N2; j++)
	for (int i = j + 1; i < N2; i++) {
		Q[i + N2*j] = G_BLK(L - 2, L - 2)[i + NL*j];
		G_BLK(L - 2, L - 2)[i + NL*j] = 0.0;
	}
	dormqr("R", "T", &NL, &N2, &N2, Q, &N2, tau + N*(L - 2),
	       G_BLK(0, L - 2), &NL, work, &lwork, &info);
	for (int l = L - 3; l >= 0; l--) {
		for (int j = 0; j < N; j++)
		for (int i = j + 1; i < N2; i++) {
			Q[i + N2*j] = G_BLK(l, l)[i + NL*j];
			G_BLK(l, l)[i + NL*j] = 0.0;
		}
		dormqr("R", "T", &NL, &N2, &N, Q, &N2, tau + N*l,
		       G_BLK(0, l), &NL, work, &lwork, &info);
	}
}

static void expand_g(const int N, const int stride, const int L, const int E, const int n_matmul,
		const double *const restrict B,
		const double *const restrict iB,
		const double *const restrict Gred,
		double *const restrict G)
{
	__assume(stride % DBL_ALIGN == 0);
	_aa(B); _aa(iB); _aa(Gred); _aa(G);

	const int NL = N*L;

	// copy Gred to G
	for (int f = 0; f < E; f++)
	for (int e = 0; e < E; e++) {
		const int l = f*n_matmul;
		const int k = e*n_matmul;
		for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			G_BLK(k, l)[i + NL*j] = Gred[(i + N*e) + N*E*(j + N*f)];
	}

	// number of steps to move in each direction
	// except for boundaries, when L % n_matmul != 0
	const int n_left = (n_matmul - 1)/2;
	const int n_right = n_matmul/2;
	const int n_up = n_left;
	const int n_down = n_right;

	const int rstop_last = ((E - 1)*n_matmul + L)/2;
	const int lstop_first = (rstop_last + 1) % L;
	const int dstop_last = rstop_last;
	const int ustop_first = lstop_first;

	// left and right
	for (int f = 0; f < E; f++)
	for (int e = 0; e < E; e++) {
		const int l = f*n_matmul;
		const int k = e*n_matmul;
		const int lstop = (f == 0) ? lstop_first : l - n_left;
		const int rstop = (f == E - 1) ? rstop_last : l + n_right;
		for (int m = l; m != lstop;) {
			const int next = (m - 1 + L) % L;
			const double alpha = (m == 0) ? -1.0 : 1.0;
			dgemm("N", "N", &N, &N, &N, &alpha,
			      G_BLK(k, m), &NL, B + stride*next, &N, cdbl(0.0),
			      G_BLK(k, next), &NL);
			m = next;
		}
		for (int m = l; m != rstop;) {
			const int next = (m + 1) % L;
			const double alpha = (next == 0) ? -1.0 : 1.0;
			const double beta = (k == m) ? -alpha : 0.0;
			if (k == m) {
				for (int j = 0; j < N; j++)
				for (int i = 0; i < N; i++)
					G_BLK(k, next)[i + NL*j] = iB[i + N*j + stride*m];
			}
			dgemm("N", "N", &N, &N, &N, &alpha,
			      G_BLK(k, m), &NL, iB + stride*m, &N, &beta,
			      G_BLK(k, next), &NL);
			m = next;
		}
	}

	// up and down
	for (int e = 0; e < E; e++)
	for (int l = 0; l < L; l++) {
		const int k = e*n_matmul;
		const int ustop = (e == 0) ? ustop_first : k - n_up;
		const int dstop = (e == E - 1) ? dstop_last : k + n_down;
		for (int m = k; m != ustop;) {
			const int next = (m - 1 + L) % L;
			const double alpha = (m == 0) ? -1.0 : 1.0;
			const double beta = (m == l) ? -alpha : 0.0;
			if (m == l)
				for (int j = 0; j < N; j++)
				for (int i = 0; i < N; i++)
					G_BLK(next, l)[i + NL*j] = iB[i + N*j + stride*next];
			dgemm("N", "N", &N, &N, &N, &alpha,
			      iB + stride*next, &N, G_BLK(m, l), &NL, &beta,
			      G_BLK(next, l), &NL);
			m = next;
		}
		for (int m = k; m != dstop;) {
			const int next = (m + 1) % L;
			const double alpha = (next == 0) ? -1.0 : 1.0;
			dgemm("N", "N", &N, &N, &N, &alpha,
			      B + stride*m, &N, G_BLK(m, l), &NL, cdbl(0.0),
			      G_BLK(next, l), &NL);
			if (next == l)
				for (int i = 0; i < N; i++)
					G_BLK(next, l)[i + NL*i] += 1.0;
			m = next;
		}
	}
}

void calc_ue_g(const int N, const int stride, const int L, const int F, const int n_mul,
		const double *const restrict B, const double *const restrict iB,
		const double *const restrict C,
		double *const restrict G,
		double *const restrict Gred,
		double *const restrict tau,
		double *const restrict Q,
		double *const restrict work, const int lwork)
{
	const int E = 1 + (F - 1) / n_mul;

	profile_begin(calc_o);
	calc_o(N, stride, F, n_mul, C, Gred, work);
	profile_end(calc_o);

	profile_begin(bsofi);
	bsofi(N, E, Gred, tau, Q, work, lwork);
	profile_end(bsofi);

	profile_begin(expand_g);
	expand_g(N, stride, L, E, (L/F) * n_mul, B, iB, Gred, G);
	profile_end(expand_g);
}

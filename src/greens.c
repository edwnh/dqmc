#include "greens.h"
#include <complex.h>
#include <tgmath.h>
#include "linalg.h"
#include "prof.h"
#include "util.h"
#include "mem.h"

void mul_seq(const int N, const int L,
		const int min, const int maxp1,
		const num alpha, const num *const restrict B,
		num *const restrict A, const int ldA,
		num *const restrict work)
{
	const int n_mul = (min == maxp1) ? L : (L + maxp1 - min) % L;
	if (n_mul == 1) {
		for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			A[i + ldA*j] = alpha*B[i + N*j + N*N*min];
		return;
	}

	int l = min;
	if (n_mul % 2 == 0) {
		xgemm("N", "N", N, N, N, alpha, B + N*N*((l + 1)%L),
		      N, B + N*N*l, N, 0.0, A, ldA);
		l = (l + 2) % L;
	} else {
		xgemm("N", "N", N, N, N, alpha, B + N*N*((l + 1)%L),
		      N, B + N*N*l, N, 0.0, work, N);
		xgemm("N", "N", N, N, N, 1.0, B + N*N*((l + 2)%L),
		      N, work, N, 0.0, A, ldA);
		l = (l + 3) % L;
	}

	for (; l != maxp1; l = (l + 2) % L) {
		xgemm("N", "N", N, N, N, 1.0, B + N*N*l,
		      N, A, ldA, 0.0, work, N);
		xgemm("N", "N", N, N, N, 1.0, B + N*N*((l + 1)%L),
		      N, work, N, 0.0, A, ldA);
	}
}

int get_lwork_eq_g(const int N)
{
	num lwork;
	int info = 0;
	int max_lwork = N*N; // can be smaller if mul_seq doesn't use work

	xgeqp3(N, N, NULL, N, NULL, NULL, &lwork, -1, NULL, &info);
	if (creal(lwork) > max_lwork) max_lwork = (int)lwork;

	xgeqrf(N, N, NULL, N, NULL, &lwork, -1, &info);
	if (creal(lwork) > max_lwork) max_lwork = (int)lwork;

	xunmqr("R", "N", N, N, N, NULL, N, NULL, NULL, N, &lwork, -1, &info);
	if (creal(lwork) > max_lwork) max_lwork = (int)lwork;

	xunmqr("R", "C", N, N, N, NULL, N, NULL, NULL, N, &lwork, -1, &info);
	if (creal(lwork) > max_lwork) max_lwork = (int)lwork;

	return max_lwork;
}

num calc_eq_g(const int l, const int N, const int L, const int n_mul,
		const num *const restrict B, num *const restrict g,
		num *const restrict Q, num *const restrict T,
		num *const restrict tau, num *const restrict d,
		num *const restrict v, int *const restrict pvt,
		num *const restrict work, const int lwork)
{
	int info = 0;

	// algorithm 3 of 10.1109/IPDPS.2012.37
	// slightly modified; groups of n_mul matrices are multiplied with gemm
	// if n_mul == 2 && l == 0:
	// like (B5 B4)(B3 B2)(B1 B0) if L is even
	// or (B6 B5)(B4 B3)(B2 B1)(B0) if L is odd

	// (1)
	int m = (l + 1 + (L - 1) % n_mul) % L;
	mul_seq(N, L, l, m, 1.0, B, Q, N, work);

	for (int i = 0; i < N; i++) pvt[i] = 0;
	xgeqp3(N, N, Q, N, pvt, tau, work, lwork, (double *)d, &info); // use d as RWORK

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
		mul_seq(N, L, m, next, 1.0, B, g, N, work);
		m = next;

		// (3a)
		xunmqr("R", "N", N, N, N, Q, N, tau, g, N, work, lwork, &info);

		for (int j = 0; j < N; j++)
			for (int i = 0; i < N; i++)
				g[i + j*N] *= d[j];

		// (3b)
		for (int j = 0; j < N; j++) { // use d for norms
			d[j] = 0.0;
			for (int i = 0; i < N; i++)
				d[j] += g[i + j*N] * conj(g[i + j*N]);
		}

		pvt[0] = 0;
		for (int i = 1; i < N; i++) { // insertion sort
			int j;
			for (j = i; j > 0 && creal(d[pvt[j-1]]) < creal(d[i]); j--)
				pvt[j] = pvt[j-1];
			pvt[j] = i;
		}

		for (int j = 0; j < N; j++) // pre-pivot
			my_copy(Q + j*N, g + pvt[j]*N, N);

		// (3c)
		xgeqrf(N, N, Q, N, tau, work, lwork, &info);

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

		xtrmm("L", "U", "N", "N", N, N, 1.0, Q, N, T, N);
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
		g[i + i*N] = v[i];
	}

	xunmqr("R", "C", N, N, N, Q, N, tau, g, N, work, lwork, &info);

	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			T[i + j*N] *= d[i];

	for (int i = 0; i < N*N; i++) T[i] += g[i];

	xgetrf(N, N, T, N, pvt, &info);
	xgetrs("N", N, N, T, N, pvt, g, N, &info);

	// determinant
	// num det = 1.0;
	// for (int i = 0; i < N; i++) {
	// 	det *= v[i]/T[i + N*i];
	// 	double vv = 1.0;
	// 	for (int j = i + 1; j < N; j++)
	// 		vv += creal(Q[j + i*N])*creal(Q[j + i*N])
	// 		    + cimag(Q[j + i*N])*cimag(Q[j + i*N]);
	// 	det /= 1 - tau[i]*vv;
	// 	if (pvt[i] != i+1)
	// 		det *= -1.0;
	// }

	// probably can be done more efficiently but it's O(N) so whatev
	num phase = 1.0;
	for (int i = 0; i < N; i++) {
		const num c = v[i]/T[i + N*i];
		phase *= c/fabs(c);
		double vv = 1.0;
		for (int j = i + 1; j < N; j++)
			vv += creal(Q[j + i*N])*creal(Q[j + i*N])
			    + cimag(Q[j + i*N])*cimag(Q[j + i*N]);
		const num ref = 1.0 - tau[i]*vv;
		phase *= fabs(ref)/ref;
		if (pvt[i] != i+1) phase *= -1.0;
	}

	return 1.0/phase;
}

int get_lwork_ue_g(const int N, const int L)
{
	num lwork;
	int info = 0;
	int max_lwork = N*N; // can start smaller if mul_seq doesn't use work

	if (L == 1) {  // then bsofi doesn't use QR
		xgetri(N, NULL, N, NULL, &lwork, -1, &info);
		if (creal(lwork) > max_lwork) max_lwork = (int)lwork;
		return max_lwork;
	}

	const int NL = N*L;
	const int N2 = 2*N;

	xgeqrf(N2, N, NULL, NL, NULL, &lwork, -1, &info);
	if (creal(lwork) > max_lwork) max_lwork = (int)lwork;

	xunmqr("L", "C", N2, N, N, NULL, NL, NULL, NULL, NL, &lwork, -1, &info);
	if (creal(lwork) > max_lwork) max_lwork = (int)lwork;

	xgeqrf(N2, N2, NULL, NL, NULL, &lwork, -1, &info);
	if (creal(lwork) > max_lwork) max_lwork = (int)lwork;

	xunmqr("R", "C", NL, N2, N2, NULL, N2, NULL, NULL, NL, &lwork, -1, &info);
	if (creal(lwork) > max_lwork) max_lwork = (int)lwork;

	xunmqr("R", "C", NL, N2, N, NULL, N2, NULL, NULL, NL, &lwork, -1, &info);
	if (creal(lwork) > max_lwork) max_lwork = (int)lwork;

	return max_lwork;
}

static void calc_o(const int N, const int L, const int n_mul,
		const num *const restrict B, num *const restrict G,
		num *const restrict work)
{
	const int E = 1 + (L - 1) / n_mul;
	const int NE = N*E;

	for (int i = 0; i < NE * NE; i++) G[i] = 0.0;

	for (int e = 0; e < E - 1; e++) // subdiagonal blocks
		mul_seq(N, L, e*n_mul, (e + 1)*n_mul, -1.0, B,
		        G + N*(e + 1) + NE*N*e, NE, work);

	mul_seq(N, L, (E - 1)*n_mul, 0, 1.0, B, // top right corner
		G + NE*N*(E - 1), NE, work);

	for (int i = 0; i < NE; i++) G[i + NE*i] += 1.0; // 1 on diagonal
}

static void bsofi(const int N, const int L,
		num *const restrict G, // input: O matrix, output: G = O^-1
		num *const restrict tau, // NL
		num *const restrict Q, // 2*N * 2*N
		num *const restrict work, const int lwork)
{
	int info;

	if (L == 1) {
		xgetrf(N, N, G, N, (int *)tau, &info);
		xgetri(N, G, N, (int *)tau, work, lwork, &info);
		return;
	}

	const int NL = N*L;
	const int N2 = 2*N;

	#define G_BLK(i, j) (G + N*(i) + NL*N*(j))
	// block qr
	for (int l = 0; l < L - 2; l++) {
		xgeqrf(N2, N, G_BLK(l, l), NL, tau + N*l, work, lwork, &info);
		xunmqr("L", "C", N2, N, N, G_BLK(l, l), NL, tau + N*l,
		       G_BLK(l, l + 1), NL, work, lwork, &info);
		xunmqr("L", "C", N2, N, N, G_BLK(l, l), NL, tau + N*l,
		       G_BLK(l, L - 1), NL, work, lwork, &info);
	}
	xgeqrf(N2, N2, G_BLK(L - 2, L - 2), NL, tau + N*(L - 2), work, lwork, &info);

	// invert r
	if (L <= 2) {
		xtrtri("U", "N", NL, G, NL, &info);
	} else {
		xtrtri("U", "N", 3*N, G_BLK(L - 3, L - 3), NL, &info);
		if (L > 3) {
			xtrmm("R", "U", "N", "N", N*(L - 3), N, 1.0,
			      G_BLK(L - 1, L - 1), NL, G_BLK(0, L - 1), NL);
			for (int l = L - 4; l >= 0; l--) {
				xtrtri("U", "N", N, G_BLK(l, l), NL, &info);
				xtrmm("L", "U", "N", "N", N, N, -1.0,
				      G_BLK(l, l), NL, G_BLK(l, L - 1), NL);
				xtrmm("L", "U", "N", "N", N, N, -1.0,
				      G_BLK(l, l), NL, G_BLK(l, l + 1), NL);
				xgemm("N", "N", N, N*(L - l - 2), N, 1.0,
				      G_BLK(l, l + 1), NL, G_BLK(l + 1, l + 2), NL, 1.0,
				      G_BLK(l, l + 2), NL);
				xtrmm("R", "U", "N", "N", N, N, 1.0,
				      G_BLK(l + 1, l + 1), NL, G_BLK(l, l + 1), NL);
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
	xunmqr("R", "C", NL, N2, N2, Q, N2, tau + N*(L - 2),
	       G_BLK(0, L - 2), NL, work, lwork, &info);
	for (int l = L - 3; l >= 0; l--) {
		for (int j = 0; j < N; j++)
		for (int i = j + 1; i < N2; i++) {
			Q[i + N2*j] = G_BLK(l, l)[i + NL*j];
			G_BLK(l, l)[i + NL*j] = 0.0;
		}
		xunmqr("R", "C", NL, N2, N, Q, N2, tau + N*l,
		       G_BLK(0, l), NL, work, lwork, &info);
	}
	#undef G_BLK
}

static void expand_g(const int N, const int L, const int E, const int n_matmul,
		const num *const restrict B,
		const num *const restrict iB,
		const num *const restrict Gred,
		num *const restrict G0t, num *const restrict Gtt,
		num *const restrict Gt0)
{
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

	// copy Gred to G0t
	for (int f = 0; f < E; f++) {
		const int t = f*n_matmul;
		for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			G0t[i + N*j + N*N*t] = Gred[i + N*E*(j + N*f)];
	}

	// expand G0t
	for (int f = 0; f < E; f++) {
		const int l = f*n_matmul;
		const int lstop = (f == 0) ? lstop_first : l - n_left;
		const int rstop = (f == E - 1) ? rstop_last : l + n_right;
		for (int m = l; m != lstop;) {
			const int next = (m - 1 + L) % L;
			const num alpha = (m == 0) ? -1.0 : 1.0;
			xgemm("N", "N", N, N, N, alpha,
			      G0t + N*N*m, N, B + N*N*next, N, 0.0,
			      G0t + N*N*next, N);
			m = next;
		}
		for (int m = l; m != rstop;) {
			const int next = (m + 1) % L;
			const num alpha = (next == 0) ? -1.0 : 1.0;
			const num beta = (m == 0) ? -alpha : 0.0;
			if (m == 0)
				for (int j = 0; j < N; j++)
				for (int i = 0; i < N; i++)
					G0t[i + N*j + N*N*next] = iB[i + N*j + N*N*m];
			xgemm("N", "N", N, N, N, alpha,
			      G0t + N*N*m, N, iB + N*N*m, N, beta,
			      G0t + N*N*next, N);
			m = next;
		}
	}


	// copy Gred to Gtt
	for (int e = 0; e < E; e++) {
		const int k = e*n_matmul;
		for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			Gtt[i + N*j + N*N*k] = Gred[(i + N*e) + N*E*(j + N*e)];
	}

	// expand Gtt
	// possible to save 2 gemm's here by using Gt0 and G0t but whatever
	for (int e = 0; e < E; e++) {
		const int k = e*n_matmul;
		const int ustop = (e == 0) ? ustop_first : k - n_up;
		const int dstop = (e == E - 1) ? dstop_last : k + n_down;
		for (int m = k; m != ustop;) {
			const int next = (m - 1 + L) % L;
			xgemm("N", "N", N, N, N, 1.0,
			      Gtt + N*N*m, N, B + N*N*next, N, 0.0,
			      Gt0, N); // use Gt0 as temporary
			xgemm("N", "N", N, N, N, 1.0,
			      iB + N*N*next, N, Gt0, N, 0.0,
			      Gtt + N*N*next, N);
			m = next;
		}
		for (int m = k; m != dstop;) {
			const int next = (m + 1) % L;
			xgemm("N", "N", N, N, N, 1.0,
			      Gtt + N*N*m, N, iB + N*N*m, N, 0.0,
			      Gt0, N);
			xgemm("N", "N", N, N, N, 1.0,
			      B + N*N*m, N, Gt0, N, 0.0,
			      Gtt + N*N*next, N);
			m = next;
		}
	}

	// copy Gred to Gt0
	for (int e = 0; e < E; e++) {
		const int t = e*n_matmul;
		for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			Gt0[i + N*j + N*N*t] = Gred[(i + N*e) + N*E*j];
	}

	// expand Gt0
	for (int e = 0; e < E; e++) {
		const int k = e*n_matmul;
		const int ustop = (e == 0) ? ustop_first : k - n_up;
		const int dstop = (e == E - 1) ? dstop_last : k + n_down;
		for (int m = k; m != ustop;) {
			const int next = (m - 1 + L) % L;
			const num alpha = (m == 0) ? -1.0 : 1.0;
			const num beta = (m == 0) ? -alpha : 0.0;
			if (m == 0)
				for (int j = 0; j < N; j++)
				for (int i = 0; i < N; i++)
					Gt0[i + N*j + N*N*next] = iB[i + N*j + N*N*next];
			xgemm("N", "N", N, N, N, alpha,
			      iB + N*N*next, N, Gt0 + N*N*m, N, beta,
			      Gt0 + N*N*next, N);
			m = next;
		}
		for (int m = k; m != dstop;) {
			const int next = (m + 1) % L;
			const num alpha = (next == 0) ? -1.0 : 1.0;
			xgemm("N", "N", N, N, N, alpha,
			      B + N*N*m, N, Gt0 + N*N*m, N, 0.0,
			      Gt0 + N*N*next, N);
			if (next == 0) // should never happen
				for (int i = 0; i < N; i++)
					Gt0[i + N*i + N*N*next] += 1.0;
			m = next;
		}
	}
}

void calc_ue_g(const int N, const int L, const int F, const int n_mul,
		const num *const restrict B, const num *const restrict iB,
		const num *const restrict C,
		num *const restrict G0t, num *const restrict Gtt,
		num *const restrict Gt0,
		num *const restrict Gred,
		num *const restrict tau,
		num *const restrict Q,
		num *const restrict work, const int lwork)
{
	const int E = 1 + (F - 1) / n_mul;

	profile_begin(calc_o);
	calc_o(N, F, n_mul, C, Gred, work);
	profile_end(calc_o);

	profile_begin(bsofi);
	bsofi(N, E, Gred, tau, Q, work, lwork);
	profile_end(bsofi);

	profile_begin(expand_g);
	expand_g(N, L, E, (L/F) * n_mul, B, iB, Gred, G0t, Gtt, Gt0);
	profile_end(expand_g);
}

/*
static void expand_g_full(const int N, const int L, const int E, const int n_matmul,
		const double *const restrict B,
		const double *const restrict iB,
		const double *const restrict Gred,
		double *const restrict G)
{
	const int ldG = N;

	#define G_BLK(i, j) (G + N*N*((i) +  L*(j)))
	// copy Gred to G
	for (int f = 0; f < E; f++)
	for (int e = 0; e < E; e++) {
		const int l = f*n_matmul;
		const int k = e*n_matmul;
		for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			G_BLK(k, l)[i + ldG*j] = Gred[(i + N*e) + N*E*(j + N*f)];
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
			      G_BLK(k, m), &ldG, B + N*N*next, &N, cdbl(0.0),
			      G_BLK(k, next), &ldG);
			m = next;
		}
		for (int m = l; m != rstop;) {
			const int next = (m + 1) % L;
			const double alpha = (next == 0) ? -1.0 : 1.0;
			const double beta = (k == m) ? -alpha : 0.0;
			if (k == m)
				for (int j = 0; j < N; j++)
				for (int i = 0; i < N; i++)
					G_BLK(k, next)[i + ldG*j] = iB[i + N*j + N*N*m];
			dgemm("N", "N", &N, &N, &N, &alpha,
			      G_BLK(k, m), &ldG, iB + N*N*m, &N, &beta,
			      G_BLK(k, next), &ldG);
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
					G_BLK(next, l)[i + ldG*j] = iB[i + N*j + N*N*next];
			dgemm("N", "N", &N, &N, &N, &alpha,
			      iB + N*N*next, &N, G_BLK(m, l), &ldG, &beta,
			      G_BLK(next, l), &ldG);
			m = next;
		}
		for (int m = k; m != dstop;) {
			const int next = (m + 1) % L;
			const double alpha = (next == 0) ? -1.0 : 1.0;
			dgemm("N", "N", &N, &N, &N, &alpha,
			      B + N*N*m, &N, G_BLK(m, l), &ldG, cdbl(0.0),
			      G_BLK(next, l), &ldG);
			if (next == l)
				for (int i = 0; i < N; i++)
					G_BLK(next, l)[i + ldG*i] += 1.0;
			m = next;
		}
	}
	#undef G_BLK
}

void calc_ue_g_full(const int N, const int L, const int F, const int n_mul,
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
	calc_o(N, F, n_mul, C, Gred, work);
	profile_end(calc_o);

	profile_begin(bsofi);
	bsofi(N, E, Gred, tau, Q, work, lwork);
	profile_end(bsofi);

	profile_begin(expand_g);
	expand_g_full(N, L, E, (L/F) * n_mul, B, iB, Gred, G);
	profile_end(expand_g);
}
*/

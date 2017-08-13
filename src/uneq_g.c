#include "uneq_g.h"
#include <mkl.h>
#include "util.h"

#define G_BLK(i, j) (G + N*(i) + NL*N*(j))

int get_lwork_ue_G(const int N, const int L)
{
	double lwork;
	int info = 0;
	int max_lwork = N;

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

static void calc_o(const int N, const int stride, const int L,
		const double *const restrict B, double *const restrict G)
{
	__assume(stride % DBL_ALIGN == 0);
	_aa(B); _aa(G);

	const int NL = N*L;
	for (int i = 0; i < NL * NL; i++) G[i] = 0.0;
	for (int i = 0; i < NL; i++) G[i + NL*i] = 1.0; // 1 on diagonal
	for (int l = 0; l < L - 1; l++) // subdiagonal blocks
		for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			G_BLK(l + 1, l)[i + NL*j] = -B[i + N*j + stride*l];
	for (int j = 0; j < N; j++) // top right corner
	for (int i = 0; i < N; i++)
		G_BLK(0, L - 1)[i + NL*j] += B[i + N*j + stride*(L - 1)];
}

static void bsofi(const int N, const int L,
		double *const restrict G, // input: O matrix, output: G = O^-1
		double *const restrict tau, // NL
		double *const restrict Q, // 2*N * 2*N
		double *const restrict work, const int lwork)
{
	_aa(G); _aa(tau); _aa(Q); _aa(work);

	const int NL = N*L;
	const int N2 = 2*N;
	int info;

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

static void expand_g(const int N, const int stride, const int L, const int F,
		const double *const restrict B,
		const double *const restrict iB,
		const double *const restrict Gred,
		double *const restrict G)
{
	__assume(stride % DBL_ALIGN == 0);
	_aa(B); _aa(iB); _aa(Gred); _aa(G);
	const int NL = N*L;
	const int n_matmul = L/F;

	// copy Gred to G
	for (int f = 0; f < F; f++)
	for (int e = 0; e < F; e++) {
		const int l = f*n_matmul;
		const int k = e*n_matmul;
		for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			G_BLK(k, l)[i + NL*j] = Gred[(i + N*e) + N*F*(j + N*f)];
	}

	// number of steps to move in each direction
	const int n_left = (n_matmul - 1)/2;
	const int n_right = n_matmul/2;
	const int n_up = n_left;
	const int n_down = n_right;

	// left and right
	for (int f = 0; f < F; f++)
	for (int e = 0; e < F; e++) {
		const int l = f*n_matmul;
		const int k = e*n_matmul;
		const int lstop = (l - n_left + L) % L;
		const int rstop = (l + n_right) % L;
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
	for (int e = 0; e < F; e++)
	for (int l = 0; l < L; l++) {
		const int k = e*n_matmul;
		const int ustop = (k - n_up + L) % L;
		const int dstop = (k + n_down) % L;
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

void calc_ue_g(const int N, const int stride, const int L, const int F,
		const double *const restrict B, const double *const restrict iB,
		const double *const restrict C,
		double *const restrict G,
		double *const restrict Gred,
		double *const restrict tau,
		double *const restrict Q,
		double *const restrict work, const int lwork)
{
	calc_o(N, stride, F, C, Gred);
	bsofi(N, F, Gred, tau, Q, work, lwork);
	expand_g(N, stride, L, F, B, iB, Gred, G);
}


// tests
// icc -D_TEST_ -std=gnu11 -Wall -Wextra -Ofast -xHost -DMKL_DIRECT_CALL_SEQ -mkl=sequential uneq_g.c -o test
#ifdef _TEST_

#include <math.h>
#include <stdio.h>
#include "rand.h"
#include "util.h"
static void test1(const int N, const int F)
{
	const int stride = N*N;
	const int NF = N*F;

	double *C = my_calloc(stride*F * sizeof(double));
	double *O = my_calloc(NF*NF * sizeof(double));
	double *G = my_calloc(NF*NF * sizeof(double));
	double *I = my_calloc(NF*NF * sizeof(double));
	double *tau = my_calloc(NF * sizeof(double));
	double *Q = my_calloc(2*N * 2*N * sizeof(double));
	const int lwork = get_lwork_ue_G(N, F);
	double *work = my_calloc(lwork * sizeof(double));

	uint64_t rng[17] = {0};
	rng[7] = 117;
	for (int i = 0; i < 1117; i++) rand_uint(rng);

	for (int i = 0; i < stride*F; i++) C[i] = 2.0*rand_doub(rng) - 1.0;
	calc_o(N, stride, F, C, O);
	for (int i = 0; i < NF*NF; i++) G[i] = O[i];
	bsofi(N, F, G, tau, Q, work, lwork);

	double avg;

	dgemm("N", "N", &NF, &NF, &NF, cdbl(1.0), O, &NF, G, &NF, cdbl(0.0), I, &NF);
	avg = 0.0;
	for (int i = 0; i < NF*NF; i++) avg += fabs(I[i] - (i % (NF + 1) == 0));
	avg /= NF*NF;
	printf("%g\n", avg);

	dgemm("N", "N", &NF, &NF, &NF, cdbl(1.0), G, &NF, O, &NF, cdbl(0.0), I, &NF);
	avg = 0.0;
	for (int i = 0; i < NF*NF; i++) avg += fabs(I[i] - (i % (NF + 1) == 0));
	avg /= NF*NF;
	printf("%g\n", avg);
}

static void test2(const int N, const int L, const int F)
{
	if (L % F != 0) {
		printf("%d %% %d != 0\n", L, F);
		return;
	}
	const int n_matmul = L/F;
	const int stride = N*N;
	const int NF = N*F;
	const int NL = N*L;

	double *B = my_calloc(stride*L * sizeof(double));
	double *iB = my_calloc(stride*L * sizeof(double));
	double *C = my_calloc(stride*F * sizeof(double));
	double *Gred = my_calloc(NF*NF * sizeof(double));
	double *O = my_calloc(NL*NL * sizeof(double));
	double *G = my_calloc(NL*NL * sizeof(double));
	double *temp = my_calloc(NL*NL * sizeof(double));

	double *tau = my_calloc(NL * sizeof(double));
	double *Q = my_calloc(2*N * 2*N * sizeof(double));
	const int lwork = get_lwork_ue_G(N, L);
	double *work = my_calloc(lwork * sizeof(double));

	double avg;

	uint64_t rng[17] = {0};
	rng[7] = 117;
	for (int i = 0; i < 1117; i++) rand_uint(rng);

	for (int i = 0; i < stride*L; i++) B[i] = 2.0*rand_doub(rng) - 1.0;

	int info;
	int *piv = my_calloc(N * sizeof(int));
	for (int l = 0; l < L; l++) {
		for (int i = 0; i < N*N; i++) iB[i + stride*l] = B[i + stride*l];
		dgetrf(&N, &N, iB + stride*l, &N, piv, &info);
		dgetri(&N, iB + stride*l, &N, piv, work, &lwork, &info);
	}

	for (int f = 0; f < F; f++) {
		int l = f*n_matmul;
		for (int i = 0; i < N*N; i++) C[i + stride*f] = B[i + stride*l];
		for (l++; l < (f+1)*n_matmul; l++) {
			dgemm("N", "N", &N, &N, &N, cdbl(1.0),
			      B + stride*l, &N, C + stride*f, &N, cdbl(0.0), temp, &N);
			for (int i = 0; i < N*N; i++) C[i + stride*f] = temp[i];
		}
	}

	calc_ue_g(N, stride, L, F, B, iB, C, G, Gred, tau, Q, work, lwork);

	calc_o(N, stride, L, B, O);
	for (int i = 0; i < NL*NL; i++) temp[i] = O[i];
	bsofi(N, L, temp, tau, Q, work, lwork);

	avg = 0.0;
	for (int j = 0; j < NL; j++)
	for (int i = 0; i < NL; i++)
		avg += fabs(G[i + NL*j] - temp[i + NL*j]);
	avg /= NL * NL;
	printf("%g\n", avg);

	// for (int m = 0; m < L; m++)
	// for (int k = 0; k < L; k++) {
		// avg = 0.0;
		// for (int j = 0; j < N; j++)
		// for (int i = 0; i < N; i++)
			// avg += fabs(G[(i + k*N) + NL*(j + m*N)] - temp[(i + k*N) + NL*(j + m*N)]);
		// avg /= N*N;
		// printf("%d\t%d\t%g\n", k, m, avg);
	// }

	// dgemm("N", "N", &NL, &NL, &NL, cdbl(1.0), O, &NL, G, &NL, cdbl(0.0), temp, &NL);
	// avg = 0.0;
	// for (int i = 0; i < NL*NL; i++) avg += fabs(temp[i] - (i % (NL + 1) == 0));
	// avg /= NL*NL;
	// printf("%g\n", avg);

	// dgemm("N", "N", &NL, &NL, &NL, cdbl(1.0), G, &NL, O, &NL, cdbl(0.0), temp, &NL);
	// avg = 0.0;
	// for (int i = 0; i < NL*NL; i++) avg += fabs(temp[i] - (i % (NL + 1) == 0));
	// avg /= NL*NL;
	// printf("%g\n", avg);
}

int main(void)
{
	test1(1, 2);
	test1(1, 13);
	test1(8, 2);
	test1(8, 8);
	test1(17, 23);
	test1(32, 24);
	test2(4, 4, 2);
	test2(17, 16, 2);
	test2(17, 16, 8);
	test2(67, 35, 7);
	test2(7, 63, 7);
	test2(1, 63, 7);
	test2(100, 45, 9);
	return 0;
}

#endif // _TEST_

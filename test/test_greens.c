#include "greens.h"
#include "linalg.h"
#include "mem.h"
#include "rand.h"

#include <stdio.h>

// old code
void mul_seq_old(const int N, const int L,
		const int min, const int maxp1,
		const num alpha, const num *const restrict B, const int ldB,
		num *const restrict A, const int ldA,
		num *const restrict work)
{
	const int n_mul = (min == maxp1) ? L : (L + maxp1 - min) % L;
	if (n_mul == 1) {
		for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			A[i + ldA*j] = alpha*B[i + ldB*j + ldB*N*min];
		return;
	}

	int l = min;
	if (n_mul % 2 == 0) {
		xgemm("N", "N", N, N, N, alpha, B + ldB*N*((l + 1)%L),
		      ldB, B + ldB*N*l, ldB, 0.0, A, ldA);
		l = (l + 2) % L;
	} else {
		xgemm("N", "N", N, N, N, alpha, B + ldB*N*((l + 1)%L),
		      ldB, B + ldB*N*l, ldB, 0.0, work, ldB);
		xgemm("N", "N", N, N, N, 1.0, B + ldB*N*((l + 2)%L),
		      ldB, work, ldB, 0.0, A, ldA);
		l = (l + 3) % L;
	}

	for (; l != maxp1; l = (l + 2) % L) {
		xgemm("N", "N", N, N, N, 1.0, B + ldB*N*l,
		      ldB, A, ldA, 0.0, work, ldB);
		xgemm("N", "N", N, N, N, 1.0, B + ldB*N*((l + 1)%L),
		      ldB, work, ldB, 0.0, A, ldA);
	}
}

num calc_eq_g(const int l, const int N, const int ld, const int L, const int n_mul,
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
	mul_seq_old(N, L, l, m, 1.0, B, ld, Q, ld, work);

	for (int i = 0; i < N; i++) pvt[i] = 0;
	xgeqp3(N, N, Q, ld, pvt, tau, work, lwork, (double *)d, &info); // use d as RWORK

	// (2)
	for (int i = 0; i < N; i++) {
		d[i] = Q[i + i*ld];
		if (d[i] == 0.0) d[i] = 1.0;
		v[i] = 1.0/d[i];
	}

	for (int i = 0; i < N*ld; i++) T[i] = 0.0;
	for (int j = 0; j < N; j++)
		for (int i = 0; i <= j; i++)
			T[i + (pvt[j]-1)*ld] = v[i] * Q[i + j*ld];


	while (m != l) {
		const int next = (m + n_mul) % L;
		mul_seq_old(N, L, m, next, 1.0, B, ld, g, ld, work);
		m = next;

		// (3a)
		xunmqr("R", "N", N, N, N, Q, ld, tau, g, ld, work, lwork, &info);

		for (int j = 0; j < N; j++)
			for (int i = 0; i < N; i++)
				g[i + j*ld] *= d[j];

		// (3b)
		for (int j = 0; j < N; j++) { // use d for norms
			d[j] = 0.0;
			for (int i = 0; i < N; i++)
				d[j] += g[i + j*ld] * conj(g[i + j*ld]);
		}

		pvt[0] = 0;
		for (int i = 1; i < N; i++) { // insertion sort
			int j;
			for (j = i; j > 0 && creal(d[pvt[j-1]]) < creal(d[i]); j--)
				pvt[j] = pvt[j-1];
			pvt[j] = i;
		}

		for (int j = 0; j < N; j++) // pre-pivot
			my_copy(Q + j*ld, g + pvt[j]*ld, N);

		// (3c)
		xgeqrf(N, N, Q, ld, tau, work, lwork, &info);

		// (3d)
		for (int i = 0; i < N; i++) {
			d[i] = Q[i + i*ld];
			if (d[i] == 0.0) d[i] = 1.0;
			v[i] = 1.0/d[i];
		}

		for (int j = 0; j < N; j++)
			for (int i = 0; i <= j; i++)
				Q[i + j*ld] *= v[i];

		for (int j = 0; j < N; j++) {
			for (int i = 0; i < N; i++)
				v[i] = T[pvt[i] + j*ld];
			my_copy(T + j*ld, v, N);
		}

		xtrmm("L", "U", "N", "N", N, N, 1.0, Q, ld, T, ld);
	}

	// construct g from Eq 2.12 of 10.1016/j.laa.2010.06.023
	for (int i = 0; i < N*ld; i++) g[i] = 0.0;
	for (int i = 0; i < N; i++) {
		if (fabs(d[i]) > 1.0) { // v = 1/Db; d = Ds
			v[i] = 1.0/d[i];
			d[i] = 1.0;
		} else {
			v[i] = 1.0;
		}
		g[i + i*ld] = v[i];
	}

	xunmqr("R", "C", N, N, N, Q, ld, tau, g, ld, work, lwork, &info);

	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			T[i + j*ld] *= d[i];

	for (int i = 0; i < N*ld; i++) T[i] += g[i];

	xgetrf(N, N, T, ld, pvt, &info);
	xgetrs("N", N, N, T, ld, pvt, g, ld, &info);

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
		const num c = v[i]/T[i + ld*i];
		phase *= c/fabs(c);
		double vv = 1.0;
		for (int j = i + 1; j < N; j++)
			vv += creal(Q[j + i*ld])*creal(Q[j + i*ld])
			    + cimag(Q[j + i*ld])*cimag(Q[j + i*ld]);
		const num ref = 1.0 - tau[i]*vv;
		phase *= fabs(ref)/ref;
		if (pvt[i] != i+1) phase *= -1.0;
	}

	return 1.0/phase;
}

static void calc_o(const int N, const int ld, const int L, const int n_mul,
		const num *const B, num *const G,
		num *const tmpNN)
{
	const int E = 1 + (L - 1) / n_mul;
	const int NE = N*E;

	for (int i = 0; i < NE * NE; i++) G[i] = 0.0;

	for (int e = 0; e < E - 1; e++) // subdiagonal blocks
		mul_seq_old(N, L, e*n_mul, (e + 1)*n_mul, -1.0, B, ld,
		        G + N*(e + 1) + NE*N*e, NE, tmpNN);

	mul_seq_old(N, L, (E - 1)*n_mul, 0, 1.0, B, ld, // top right corner
		G + NE*N*(E - 1), NE, tmpNN);

	for (int i = 0; i < NE; i++) G[i + NE*i] += 1.0; // 1 on diagonal
}

static void bsofi(const int N, const int L,
		num *const G, // input: O matrix, output: G = O^-1
		num *const tau, // NL
		num *const Q, // 2*N * 2*N
		num *const work, const int lwork)
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


int main(void)
{
	const int F = 40;
	const int N = 72;
	const int ld = best_ld(N);
	printf("%d\n", ld);

	const int lwork = N*N;//RC(get_lwork)(N, N);

	printf("lwork = %d\n", lwork);

	num *const Cu = my_calloc(ld*N*F * sizeof(num));
	struct RC(QdX) *const QdXLu = my_calloc(F * sizeof(struct RC(QdX)));
	struct RC(QdX) *const QdX0u = my_calloc(F * sizeof(struct RC(QdX)));
	struct RC(LR) *const LRLu = my_calloc(F * sizeof(struct RC(LR)));
	struct RC(LR) *const LR0u = my_calloc(F * sizeof(struct RC(LR)));
	for (int f = 0; f < F; f++) {
		QdXLu[f].Q = my_calloc(ld*N * sizeof(num));
		QdXLu[f].d = my_calloc(N * sizeof(num));
		QdXLu[f].X = my_calloc(ld*N * sizeof(num));
		LRLu[f].iL = my_calloc(ld*N * sizeof(num));
		LRLu[f].R = my_calloc(ld*N * sizeof(num));
		LRLu[f].phase_iL = my_calloc(1 * sizeof(num));
		QdX0u[f].Q = my_calloc(ld*N * sizeof(num));
		QdX0u[f].d = my_calloc(N * sizeof(num));
		QdX0u[f].X = my_calloc(ld*N * sizeof(num));
		LR0u[f].iL = my_calloc(ld*N * sizeof(num));
		LR0u[f].R = my_calloc(ld*N * sizeof(num));
		LR0u[f].phase_iL = my_calloc(1 * sizeof(num));
	}
	num *const restrict GuA = my_calloc(ld*N * sizeof(num));
	num *const restrict GuB = my_calloc(ld*N * sizeof(num));
	num *const restrict Gu0t = my_calloc(ld*N * sizeof(num));
	num *const restrict Gutt = my_calloc(ld*N * sizeof(num));
	num *const restrict Gut0 = my_calloc(ld*N * sizeof(num));
	num *const restrict tmpNN0u = my_calloc(ld*N * sizeof(num));
	num *const restrict tmpNN1u = my_calloc(ld*N * sizeof(num));
	num *const restrict tmpN0u = my_calloc(N * sizeof(num));
	num *const restrict tmpN1u = my_calloc(N * sizeof(num));
	num *const restrict tmpN2u = my_calloc(N * sizeof(num));
	num *const restrict worku = my_calloc(lwork * sizeof(num));
	int *const restrict pvtu = my_calloc(N * sizeof(int));
	num *const restrict Gredu = my_calloc(N*F*N*F * sizeof(num));
	num *const restrict tauu = my_calloc(N*F * sizeof(num));
	num *const restrict Qu = my_calloc(4*N*N * sizeof(num));

	struct RC(QdX) QdX_NULL = {NULL, NULL, NULL};
	struct RC(LR) LR_NULL = {NULL, NULL, NULL};

	num phaseuA, phaseuB;

	uint64_t rng[17] = {0};
	for (int i = 0; i < 16; i++) rng[i] = 1234567*i*i + 654321;
	for (int f = 0; f < F; f++)
		for (int j = 0; j < N; j++)
			for (int i = 0; i < N; i++) {
				#ifndef USE_CPLX
				Cu[i + j*ld + f*ld*N] = 2.0*rand_doub(rng) - 1.0;
				#else
				Cu[i + j*ld + f*ld*N] = CMPLX(2.0*rand_doub(rng) - 1.0, 2.0*rand_doub(rng) - 1.0);
				#endif
			}
printf("rng: %f\t%f\n", creal(Cu[0]), creal(Cu[5 + 6*ld + 1*ld*N]));

	for (int f = 0; f < F; f++)
		RC(calc_QdX)(0, N, ld, Cu + f*ld*N, (f == 0) ? QdX_NULL : QdX0u[f - 1], QdX0u[f], LR0u[f], tmpN1u, pvtu, worku, lwork);
for (int i = 0; i < N; i++) printf("d[%d] = %e\n", i, creal(QdX0u[F-1].d[i]));
	phaseuA = RC(calc_Gtt)(N, ld, LR0u[F - 1], LR_NULL, GuA, tmpNN1u, pvtu);

printf("GuA: %f\t%f\t%f\n", cimag(phaseuA), creal(GuA[0]), creal(GuA[2 + 3*ld]));

	phaseuB = calc_eq_g(0, N, ld, F, 1, Cu, GuB, tmpNN0u, tmpNN1u, tmpN0u, tmpN1u, tmpN2u, pvtu, worku, lwork);
printf("old: %f\t%f\t%f\n", cimag(phaseuB), creal(GuB[0]), creal(GuB[2 + 3*ld]));

	for (int f = F - 1; f >= 0; f--)
		RC(calc_QdX)(1, N, ld, Cu + f*ld*N, (f == F - 1) ? QdX_NULL : QdXLu[f + 1], QdXLu[f], LRLu[f], tmpN1u, pvtu, worku, lwork);

	phaseuB = RC(calc_Gtt)(N, ld, LR_NULL, LRLu[0], GuB, tmpNN1u, pvtu);
printf("GuB: %f\t%f\t%f\n", cimag(phaseuB), creal(GuB[0]), creal(GuB[2 + 3*ld]));

	matdiff(N, N, GuA, ld, GuB, ld);

	calc_o(N, ld, F, 1, Cu, Gredu, Qu); // use Q as tmpNN
	bsofi(N, F, Gredu, tauu, Qu, worku, lwork);

	// sweep:
	for (int f = 1; f < F; f++) {
		printf("f=%d\n", f);
		phaseuA = calc_eq_g(f, N, ld, F, 1, Cu, GuA, tmpNN0u, tmpNN1u, tmpN0u, tmpN1u, tmpN2u, pvtu, worku, lwork);

		phaseuB = RC(calc_Gtt)(N, ld, LR0u[f - 1], LRLu[f], GuB, tmpNN0u, pvtu);
		printf("phase %f %f\n", creal(phaseuA), creal(phaseuB));
		matdiff(N, N, GuA, ld, GuB, ld);

		RC(calc_G0t_Gtt_Gt0)(N, ld, LR0u[f - 1], LRLu[f], Gu0t, Gutt, Gut0, tmpNN0u, pvtu);
		matdiff(N, N, GuA, ld, Gutt, ld);
		matdiff(N, N, Gutt, ld, Gredu + f*N + f*N*N*F, N*F);
		matdiff(N, N, Gu0t, ld, Gredu + 0*N + f*N*N*F, N*F);
		matdiff(N, N, Gut0, ld, Gredu + f*N + 0*N*N*F, N*F);
	}

	return 0;
}

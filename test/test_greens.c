#include "greens.h"
#include "linalg.h"
#include "mem.h"
#include "rand.h"

#include <stdio.h>

#define matdiff(m, n, A, ldA, B, ldB) do { \
	double max = 0.0, avg = 0.0; \
	for (int j = 0; j < (n); j++) \
	for (int i = 0; i < (m); i++) { \
		const double diff = fabs((A)[i + (ldA)*j] - (B)[i + (ldB)*j]); \
		if (diff > max) max = diff; \
		avg += diff; \
	} \
	avg /= N*N; \
	printf(#A " - " #B ":\tmax %.3e\tavg %.3e\n", max, avg); \
} while (0);

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


int main(void)
{
	const int F = 40;
	const int N = 72;
	const int ld = mem_best_ld(N);
	printf("%d\n", ld);

	const int lwork = get_lwork_eq_g(N, N);

	printf("lwork = %d\n", lwork);

	#define ALLOC_TABLE(XX, FOR, ENDFOR) \
		XX(num *const Cu, mp, ld*N*F * sizeof(num)) \
		XX(struct QdX *QdXLu, mp, F * sizeof(struct QdX)) \
		XX(struct QdX *QdX0u, mp, F * sizeof(struct QdX)) \
		FOR(f, F) \
			XX(QdXLu[f].Q, mp, ld*N * sizeof(num)) \
			XX(QdXLu[f].d, mp, N * sizeof(num)) \
			XX(QdXLu[f].X, mp, ld*N * sizeof(num)) \
			XX(QdXLu[f].iL, mp, ld*N * sizeof(num)) \
			XX(QdXLu[f].R, mp, ld*N * sizeof(num)) \
			XX(QdX0u[f].Q, mp, ld*N * sizeof(num)) \
			XX(QdX0u[f].d, mp, N * sizeof(num)) \
			XX(QdX0u[f].X, mp, ld*N * sizeof(num)) \
			XX(QdX0u[f].iL, mp, ld*N * sizeof(num)) \
			XX(QdX0u[f].R, mp, ld*N * sizeof(num)) \
		ENDFOR \
		XX(num *const restrict GuA, mp, ld*N * sizeof(num)) \
		XX(num *const restrict GuB, mp, ld*N * sizeof(num)) \
		XX(num *const restrict tmpNN0u, mp, ld*N * sizeof(num)) \
		XX(num *const restrict tmpNN1u, mp, ld*N * sizeof(num)) \
		XX(num *const restrict tmpN0u, mp, N * sizeof(num)) \
		XX(num *const restrict tmpN1u, mp, N * sizeof(num)) \
		XX(num *const restrict tmpN2u, mp, N * sizeof(num)) \
		XX(num *const restrict worku, mp, lwork * sizeof(num)) \
		XX(int *const restrict pvtu, mp, N * sizeof(int))

	struct mem_pool *mp = pool_new(POOL_GET_SIZE(ALLOC_TABLE));
	POOL_DO_ALLOC(ALLOC_TABLE);

	num phaseuA, phaseuB;

	uint64_t rng[17] = {0};
	for (int i = 0; i < 16; i++) rng[i] = 1234567*i*i + 654321;
	for (int f = 0; f < F; f++)
		for (int j = 0; j < N; j++)
			for (int i = 0; i < N; i++)
				Cu[i + j*ld + f*ld*N] =
					2.0*rand_doub(rng) - 1.0;
					// CMPLX(2.0*rand_doub(rng) - 1.0, 2.0*rand_doub(rng) - 1.0);
printf("rng: %f\t%f\n", creal(Cu[0]), creal(Cu[5 + 6*ld + 1*ld*N]));

	calc_QdX_first(0, N, ld, Cu + 0*ld*N, &QdX0u[0], tmpN1u, pvtu, worku, lwork);
	for (int f = 1; f < F; f++)
		calc_QdX(0, N, ld, Cu + f*ld*N, &QdX0u[f - 1], &QdX0u[f], tmpN1u, pvtu, worku, lwork);
for (int i = 0; i < N; i++) printf("d[%d] = %e\n", i, creal(QdX0u[F-1].d[i]));
	phaseuA = calc_Gtt_last(0, N, ld, &QdX0u[F - 1], GuA, tmpNN1u, pvtu);

printf("GuA: %f\t%f\t%f\n", creal(phaseuA), creal(GuA[0]), creal(GuA[2 + 3*ld]));

	phaseuB = calc_eq_g(0, N, ld, F, 1, Cu, GuB, tmpNN0u, tmpNN1u, tmpN0u, tmpN1u, tmpN2u, pvtu, worku, lwork);
printf("old: %f\t%f\t%f\n", creal(phaseuB), creal(GuB[0]), creal(GuB[2 + 3*ld]));


	calc_QdX_first(1, N, ld, Cu + (F - 1)*ld*N, &QdXLu[F - 1], tmpN1u, pvtu, worku, lwork);
	for (int f = F - 2; f >= 0; f--)
		calc_QdX(1, N, ld, Cu + f*ld*N, &QdXLu[f + 1], &QdXLu[f], tmpN1u, pvtu, worku, lwork);

	phaseuB = calc_Gtt_last(1, N, ld, &QdXLu[0], GuB, tmpNN1u, pvtu);
printf("GuB: %f\t%f\t%f\n", creal(phaseuB), creal(GuB[0]), creal(GuB[2 + 3*ld]));

	matdiff(N, N, GuA, ld, GuB, ld);


	// sweep:
	for (int f = 1; f < F; f++) {
		printf("f=%d\n", f);
		phaseuA = calc_eq_g(f, N, ld, F, 1, Cu, GuA, tmpNN0u, tmpNN1u, tmpN0u, tmpN1u, tmpN2u, pvtu, worku, lwork);

		phaseuB = calc_Gtt(N, ld, &QdX0u[f - 1], &QdXLu[f], GuB, tmpNN0u, tmpNN1u, pvtu);
		printf("phase %f %f\n", creal(phaseuA), creal(phaseuB));
		matdiff(N, N, GuA, ld, GuB, ld);
	}

	pool_free(mp);
	return 0;
}

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
	mul_seq_old(N, L, l, m, 1.0, B, Q, N, work);

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
		mul_seq_old(N, L, m, next, 1.0, B, g, N, work);
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


int main(void)
{
	const int F = 40;
	const int N = 72;

	const int lwork = get_lwork_eq_g(N);

	printf("lwork = %d\n", lwork);
	
	#define ALLOC_TABLE(XX, FOR, ENDFOR) \
		XX(num *const Cu, mp, N*N*F * sizeof(num)) \
		XX(struct QdX *QdXLu, mp, F * sizeof(struct QdX)) \
		XX(struct QdX *QdX0u, mp, F * sizeof(struct QdX)) \
		FOR(f, F) \
	        XX(QdXLu[f].Q, mp, N*N * sizeof(num)) \
	        XX(QdXLu[f].tau, mp, N * sizeof(num)) \
	        XX(QdXLu[f].d, mp, N * sizeof(num)) \
	        XX(QdXLu[f].X, mp, N*N * sizeof(num)) \
	        XX(QdX0u[f].Q, mp, N*N * sizeof(num)) \
	        XX(QdX0u[f].tau, mp, N * sizeof(num)) \
	        XX(QdX0u[f].d, mp, N * sizeof(num)) \
	        XX(QdX0u[f].X, mp, N*N * sizeof(num)) \
		ENDFOR \
		XX(num *const restrict GuA, mp, N*N * sizeof(num)) \
		XX(num *const restrict GuB, mp, N*N * sizeof(num)) \
		XX(num *const restrict tmpNN0u, mp, N*N * sizeof(num)) \
		XX(num *const restrict tmpNN1u, mp, N*N * sizeof(num)) \
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
	for (int i = 0; i < F*N*N; i++) Cu[i] = 2.0*rand_doub(rng) - 1.0;
	// for (int i = 0; i < F*N*N; i++) Cu[i] = CMPLX(2.0*rand_doub(rng) - 1.0, 2.0*rand_doub(rng) - 1.0);
printf("rng: %f\t%f\n", creal(Cu[0]), creal(Cu[5 + 6*N + 1*N*N]));

	calc_QdX_first(0, N, Cu + 0*N*N, &QdX0u[0], tmpN1u, pvtu, worku, lwork);
	for (int f = 1; f < F; f++)
		calc_QdX(0, N, Cu + f*N*N, &QdX0u[f - 1], &QdX0u[f], tmpN1u, pvtu, worku, lwork);
for (int i = 0; i < N; i++) printf("d[%d] = %e\n", i, creal(QdX0u[F-1].d[i]));
	phaseuA = calc_Gtt_last(0, N, &QdX0u[F - 1], GuA, tmpNN1u, tmpN1u, pvtu, worku, lwork);
	
printf("GuA: %f\t%f\t%f\n", creal(phaseuA), creal(GuA[0]), creal(GuA[2  + 3*N]));

	phaseuB = calc_eq_g(0, N, F, 1, Cu, GuB, tmpNN0u, tmpNN1u, tmpN0u, tmpN1u, tmpN2u, pvtu, worku, lwork);
printf("old: %f\t%f\t%f\n", creal(phaseuB), creal(GuB[0]), creal(GuB[2  + 3*N]));
	

	calc_QdX_first(1, N, Cu + (F - 1)*N*N, &QdXLu[F - 1], tmpN1u, pvtu, worku, lwork);
	for (int f = F - 2; f >= 0; f--)
		calc_QdX(1, N, Cu + f*N*N, &QdXLu[f + 1], &QdXLu[f], tmpN1u, pvtu, worku, lwork);

	phaseuB = calc_Gtt_last(1, N, &QdXLu[0], GuB, tmpNN1u, tmpN1u, pvtu, worku, lwork);
printf("GuB: %f\t%f\t%f\n", creal(phaseuB), creal(GuB[0]), creal(GuB[2 + 3*N]));

	matdiff(N, N, GuA, N, GuB, N);


	// sweep:
	for (int f = 1; f < F; f++) {
		printf("f=%d\n", f);
		phaseuA = calc_eq_g(f, N, F, 1, Cu, GuA, tmpNN0u, tmpNN1u, tmpN0u, tmpN1u, tmpN2u, pvtu, worku, lwork);
		
		phaseuB = calc_Gtt(N, &QdX0u[f - 1], &QdXLu[f], GuB, tmpNN0u, tmpNN1u, tmpN0u, tmpN1u, pvtu, worku, lwork);
		printf("phase %f %f\n", creal(phaseuA), creal(phaseuB));
		matdiff(N, N, GuA, N, GuB, N);
	}

	pool_free(mp);
	return 0;
}
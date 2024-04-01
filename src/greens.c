#include "greens.h"
#include "linalg.h"
#include "prof.h"
#include "mem.h"

void mul_seq(const int N,
		const int min, const int maxp1,
		const num alpha, num *const restrict *const restrict B,
		num *const restrict A, const int ldA,
		num *const restrict tmpNN)
{
	const int n_mul = maxp1 - min;
	if (n_mul <= 0)
		return;
	if (n_mul == 1) {
		for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			A[i + ldA*j] = alpha*B[min][i + N*j];
		return;
	}

	int l = min;
	if (n_mul % 2 == 0) {
		xgemm("N", "N", N, N, N, alpha, B[l + 1],
		      N, B[l], N, 0.0, A, ldA);
		l += 2;
	} else {
		xgemm("N", "N", N, N, N, alpha, B[l + 1],
		      N, B[l], N, 0.0, tmpNN, N);
		xgemm("N", "N", N, N, N, 1.0, B[l + 2],
		      N, tmpNN, N, 0.0, A, ldA);
		l += 3;
	}

	for (; l != maxp1; l += 2) {
		xgemm("N", "N", N, N, N, 1.0, B[l],
		      N, A, ldA, 0.0, tmpNN, N);
		xgemm("N", "N", N, N, N, 1.0, B[l + 1],
		      N, tmpNN, N, 0.0, A, ldA);
	}
}

int get_lwork_eq_g(const int N)
{
	num lwork;
	int info = 0;
	int max_lwork = 0;

	xgeqp3(N, N, NULL, N, NULL, NULL, &lwork, -1, NULL, &info);
	if (creal(lwork) > max_lwork) max_lwork = (int)lwork;

	xgeqrf(N, N, NULL, N, NULL, &lwork, -1, &info);
	if (creal(lwork) > max_lwork) max_lwork = (int)lwork;

	xunmqr("R", "N", N, N, N, NULL, N, NULL, NULL, N, &lwork, -1, &info);
	if (creal(lwork) > max_lwork) max_lwork = (int)lwork;

	xunmqr("R", "C", N, N, N, NULL, N, NULL, NULL, N, &lwork, -1, &info);
	if (creal(lwork) > max_lwork) max_lwork = (int)lwork;

	xunmqr("L", "N", N, N, N, NULL, N, NULL, NULL, N, &lwork, -1, &info);
	if (creal(lwork) > max_lwork) max_lwork = (int)lwork;

	return max_lwork;
}

void calc_QdX_first(
		const int trans, // if 1, calculate QdX of B^T (conjugate transpose for complex)
		const int N,
		const num *const restrict B, // input
		const struct QdX *const restrict QdX, // output
		num *const restrict tmpN, // work arrays
		int *const restrict pvt,
		num *const restrict work, const int lwork)
{
	int info = 0;
	for (int i = 0; i < N; i++) pvt[i] = 0;
	xomatcopy('C', trans ? 'C' : 'N', N, N, 1.0, B, N, QdX->Q, N);

	// use d as RWORK for zgeqp3
	xgeqp3(N, N, QdX->Q, N, pvt, QdX->tau, work, lwork, (double *)QdX->d, &info);

	for (int i = 0; i < N; i++) {
		QdX->d[i] = QdX->Q[i + i*N];
		if (QdX->d[i] == 0.0) QdX->d[i] = 1.0;
		tmpN[i] = 1.0/QdX->d[i];
	}

	for (int i = 0; i < N*N; i++) QdX->X[i] = 0.0;

	for (int j = 0; j < N; j++)
		for (int i = 0; i <= j; i++)
			QdX->X[i + (pvt[j]-1)*N] = tmpN[i] * QdX->Q[i + j*N];
}

void calc_QdX(
		const int trans, // if 1, calculate QdX of B^T (conjugate transpose for complex)
		const int N,
		const num *const restrict B, // input
		const struct QdX *const restrict QdX_prev,  // input, previous QdX
		const struct QdX *const restrict QdX,  // output
		num *const restrict tmpN, // work arrays
		int *const restrict pvt,
		num *const restrict work, const int lwork)
{
	// use X as temp N*N storage
	int info = 0;
	xomatcopy('C', trans ? 'C' : 'N', N, N, 1.0, B, N, QdX->X, N);

	xunmqr("R", "N", N, N, N, QdX_prev->Q, N, QdX_prev->tau, QdX->X, N, work, lwork, &info);
	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			QdX->X[i + j*N] *= QdX_prev->d[j];

	for (int j = 0; j < N; j++) { // use tmpN for norms
		tmpN[j] = 0.0;
		for (int i = 0; i < N; i++)
			tmpN[j] += QdX->X[i + j*N] * conj(QdX->X[i + j*N]);
	}

	pvt[0] = 0;
	for (int i = 1; i < N; i++) { // insertion sort
		int j;
		for (j = i; j > 0 && creal(tmpN[pvt[j-1]]) < creal(tmpN[i]); j--)
			pvt[j] = pvt[j-1];
		pvt[j] = i;
	}

	for (int j = 0; j < N; j++) // pre-pivot
		my_copy(QdX->Q + j*N, QdX->X + pvt[j]*N, N);

	xgeqrf(N, N, QdX->Q, N, QdX->tau, work, lwork, &info);

	for (int i = 0; i < N; i++) {
		QdX->d[i] = QdX->Q[i + i*N];
		if (QdX->d[i] == 0.0) QdX->d[i] = 1.0;
		tmpN[i] = 1.0/QdX->d[i];
	}

	for (int j = 0; j < N; j++)
		for (int i = 0; i <= j; i++)
			QdX->Q[i + j*N] *= tmpN[i];

	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			QdX->X[i + j*N] = QdX_prev->X[pvt[i] + j*N];

	xtrmm("L", "U", "N", "N", N, N, 1.0, QdX->Q, N, QdX->X, N);
}

num calc_Gtt_last(
		const int trans, // if 0 calculate, calculate G = (1 + Q d X)^-1. if 1, calculate G = (1 + X.T d Q.T)^-1
		const int N,
		const struct QdX *const restrict QdX, // input
		num *const restrict G, // output
		num *const restrict tmpNN, // work arrays
		num *const restrict tmpN,
		int *const restrict pvt,
		num *const restrict work, const int lwork)
{
	int info = 0;

	// construct g from Eq 2.12 of 10.1016/j.laa.2010.06.023
//todo try double d = 1.0/QdX->d[i];
	for (int i = 0; i < N*N; i++) G[i] = 0.0;
	for (int i = 0; i < N; i++) {
		if (fabs(QdX->d[i]) > 1.0) { // tmpN = 1/Db; tmpNN = Ds X
			tmpN[i] = 1.0/QdX->d[i];
			for (int j = 0; j < N; j++)
				tmpNN[i + j*N] = QdX->X[i + j*N];
		} else {
			tmpN[i] = 1.0;
			for (int j = 0; j < N; j++)
				tmpNN[i + j*N] = QdX->d[i] * QdX->X[i + j*N];
		}
		G[i + i*N] = tmpN[i];
	}

	xunmqr("R", "C", N, N, N, QdX->Q, N, QdX->tau, G, N, work, lwork, &info);

	for (int i = 0; i < N*N; i++) tmpNN[i] += G[i];

	xgetrf(N, N, tmpNN, N, pvt, &info);
	xgetrs("N", N, N, tmpNN, N, pvt, G, N, &info);

	if (trans)
		ximatcopy('C', 'C', N, N, 1.0, G, N, N);

	// determinant
	// num det = 1.0;
	// for (int i = 0; i < N; i++) {
	// 	det *= tmpN[i]/T[i + N*i];
	// 	double vv = 1.0;
	// 	for (int j = i + 1; j < N; j++)
	// 		vv += creal(Q[j + i*N])*creal(Q[j + i*N])
	// 		    + cimag(Q[j + i*N])*cimag(Q[j + i*N]);
	// 	det /= 1 - tau[i]*vv;
	// 	if (pvt[i] != i+1)
	// 		det *= -1.0;
	// }

	// probably can be done more efficiently but it's O(N) so whatev
#ifdef USE_CPLX
	num phase = 1.0;
	for (int i = 0; i < N; i++) {
		const num c = tmpNN[i + N*i]/tmpN[i];
		phase *= c/cabs(c);
		double vv = 1.0;
		for (int j = i + 1; j < N; j++)
			vv += creal(QdX->Q[j + i*N])*creal(QdX->Q[j + i*N])
			    + cimag(QdX->Q[j + i*N])*cimag(QdX->Q[j + i*N]);
		const num ref = 1.0 - QdX->tau[i]*vv;
		phase *= ref/cabs(ref);
		if (pvt[i] != i+1) phase *= -1.0;
	}
	return phase;
#else
	int sign = 1.0;
	for (int i = 0; i < N; i++) 
		if ((tmpN[i] < 0) ^ (QdX->tau[i] > 0) ^ (pvt[i] != i+1) ^ (tmpNN[i + i*N] < 0))
			sign *= -1;
	return (double)sign;
#endif
}

// G = (1 + Q0 d0 X0 X1.T d1 Q1.T)^-1
//   = Q1 id1b (id0b Q0.T Q1 id1b + d0s X0 X1.T d1s)^-1 id0b Q0.T
// step by step:
// 1. tmpNN0 = d0s X0 X1.T d1s
// 2. G = id0b Q0.T
// 3. tmpNN1 = G Q1 id1b
// 4. tmpNN1 += tmpNN0
// 5. G = tmpNN1^-1 G
// 6. G = Q1 id1b G
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
		num *const restrict work, const int lwork)
{
	int info = 0;

	// 1. tmpNN0 = d0s X0 X1.T d1s
	// tmpNN0 = X0 X1.T
	xgemm("N", "C", N, N, N, 1.0, QdX0->X, N, QdX1->X, N, 0.0, tmpNN0, N);
	// tmpN1 = d0s, tmpN0 = id0b
	for (int i = 0; i < N; i++) {
		if (fabs(QdX0->d[i]) > 1.0) {
			tmpN0[i] = 1.0/QdX0->d[i];
			tmpN1[i] = 1.0;
		} else {
			tmpN0[i] = 1.0;
			tmpN1[i] = QdX0->d[i];
		}
	}
	// tmpNN0 = d0s tmpNN0 d1s
	for (int j = 0; j < N; j++) {
		const num d1s = fabs(QdX1->d[j]) > 1.0 ? 1.0 : QdX1->d[j];
		for (int i = 0; i < N; i++)
			tmpNN0[i + j*N] *= tmpN1[i] * d1s;
	}

	// 2. G = id0b Q0.T
	// G = id0b
	for (int i = 0; i < N*N; i++) G[i] = 0.0;
	for (int i = 0; i < N; i++) G[i + i*N] = tmpN0[i];
	// G = G Q0.T
	xunmqr("R", "C", N, N, N, QdX0->Q, N, QdX0->tau, G, N, work, lwork, &info);

	// 3. tmpNN1 = G Q1 id1b
	// tmpNN1 = G Q1
	my_copy(tmpNN1, G, N*N);
	xunmqr("R", "N", N, N, N, QdX1->Q, N, QdX1->tau, tmpNN1, N, work, lwork, &info);
	// tmpN1 = id1b
	for (int i = 0; i < N; i++) {
		if (fabs(QdX1->d[i]) > 1.0)
			tmpN1[i] = 1.0/QdX1->d[i];
		else
			tmpN1[i] = 1.0;
	}

	// combine last part of 3 with 4. tmpNN1 += tmpNN0
	// tmpNN1 = tmpNN1 tmpN1 + tmpNN0
	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			tmpNN1[i + j*N] = tmpNN1[i + j*N]*tmpN1[j] + tmpNN0[i + j*N];

	// 5. G = tmpNN1^-1 G
	xgetrf(N, N, tmpNN1, N, pvt, &info);
	xgetrs("N", N, N, tmpNN1, N, pvt, G, N, &info);

	// 6. G = Q1 id1b G
	// G = id1b G
	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			G[i + j*N] *= tmpN1[i];
	// G = Q1 G
	xunmqr("L", "N", N, N, N, QdX1->Q, N, QdX1->tau, G, N, work, lwork, &info);

#ifdef USE_CPLX
	num phase = 1.0;
	for (int i = 0; i < N; i++) {
		const num c = tmpNN1[i + N*i]/(tmpN0[i] * tmpN1[i]);
		phase *= c/cabs(c);

		double x0 = 1.0, x1 = 1.0;
		for (int j = i + 1; j < N; j++) {
			x0 += creal(QdX0->Q[j + i*N])*creal(QdX0->Q[j + i*N])
			    + cimag(QdX0->Q[j + i*N])*cimag(QdX0->Q[j + i*N]);
			x1 += creal(QdX1->Q[j + i*N])*creal(QdX1->Q[j + i*N])
			    + cimag(QdX1->Q[j + i*N])*cimag(QdX1->Q[j + i*N]);
		}
		const num refs = (1.0 - QdX0->tau[i]*x0)/(1.0 - QdX1->tau[i]*x1);
		phase *= refs/cabs(refs);

		if (pvt[i] != i+1) phase *= -1.0;
	}
	return phase;
#else
	int sign = 1.0;
	for (int i = 0; i < N; i++) 
		if ((tmpNN1[i + i*N] < 0) ^ (tmpN0[i] < 0) ^ (tmpN1[i] < 0) ^
				(QdX0->tau[i] > 0) ^ (QdX1->tau[i] > 0) ^ (pvt[i] != i+1))
			sign *= -1;
	return (double)sign;
#endif
}

int get_lwork_ue_g(const int N, const int L)
{
	num lwork;
	int info = 0;
	int max_lwork = 0;

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
		num *const restrict *const restrict B, num *const restrict G,
		num *const restrict tmpNN)
{
	const int E = 1 + (L - 1) / n_mul;
	const int NE = N*E;

	for (int i = 0; i < NE * NE; i++) G[i] = 0.0;

	for (int e = 0; e < E - 1; e++) // subdiagonal blocks
		mul_seq(N, e*n_mul, (e + 1)*n_mul, -1.0, B,
		        G + N*(e + 1) + NE*N*e, NE, tmpNN);

	mul_seq(N, (E - 1)*n_mul, L, 1.0, B, // top right corner
		G + NE*N*(E - 1), NE, tmpNN);

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
		num *const restrict *const restrict B,
		num *const restrict *const restrict iB,
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
			      G0t + N*N*m, N, B[next], N, 0.0,
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
					G0t[i + N*j + N*N*next] = iB[m][i + N*j];
			xgemm("N", "N", N, N, N, alpha,
			      G0t + N*N*m, N, iB[m], N, beta,
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
			      Gtt + N*N*m, N, B[next], N, 0.0,
			      Gt0, N); // use Gt0 as temporary
			xgemm("N", "N", N, N, N, 1.0,
			      iB[next], N, Gt0, N, 0.0,
			      Gtt + N*N*next, N);
			m = next;
		}
		for (int m = k; m != dstop;) {
			const int next = (m + 1) % L;
			xgemm("N", "N", N, N, N, 1.0,
			      Gtt + N*N*m, N, iB[m], N, 0.0,
			      Gt0, N);
			xgemm("N", "N", N, N, N, 1.0,
			      B[m], N, Gt0, N, 0.0,
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
					Gt0[i + N*j + N*N*next] = iB[next][i + N*j];
			xgemm("N", "N", N, N, N, alpha,
			      iB[next], N, Gt0 + N*N*m, N, beta,
			      Gt0 + N*N*next, N);
			m = next;
		}
		for (int m = k; m != dstop;) {
			const int next = (m + 1) % L;
			const num alpha = (next == 0) ? -1.0 : 1.0;
			xgemm("N", "N", N, N, N, alpha,
			      B[m], N, Gt0 + N*N*m, N, 0.0,
			      Gt0 + N*N*next, N);
			if (next == 0) // should never happen
				for (int i = 0; i < N; i++)
					Gt0[i + N*i + N*N*next] += 1.0;
			m = next;
		}
	}
}

void calc_ue_g(const int N, const int L, const int F, const int n_mul,
		num *const restrict *const restrict B,
		num *const restrict *const restrict iB,
		num *const restrict *const restrict C,
		num *const restrict G0t, num *const restrict Gtt,
		num *const restrict Gt0,
		num *const restrict Gred,
		num *const restrict tau,
		num *const restrict Q,
		num *const restrict work, const int lwork)
{
	const int E = 1 + (F - 1) / n_mul;

	profile_begin(calc_o);
	calc_o(N, F, n_mul, C, Gred, Q); // use Q as tmpNN
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

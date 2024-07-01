#include "greens.h"
#include "linalg.h"

void mul_seq(const int N,
		const int min, const int maxp1,
		const num alpha, num *const *const B, const int ldB,
		num *const A, const int ldA,
		num *const tmpNN) // assume tmpNN has ldB leading dim
{
	const int n_mul = maxp1 - min;
	if (n_mul <= 0)
		return;
	if (n_mul == 1) {
		for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			A[i + ldA*j] = alpha*B[min][i + ldB*j];
		return;
	}

	int l = min;
	if (n_mul % 2 == 0) {
		xgemm("N", "N", N, N, N, alpha, B[l + 1],
		      ldB, B[l], ldB, 0.0, A, ldA);
		l += 2;
	} else {
		xgemm("N", "N", N, N, N, alpha, B[l + 1],
		      ldB, B[l], ldB, 0.0, tmpNN, ldB);
		xgemm("N", "N", N, N, N, 1.0, B[l + 2],
		      ldB, tmpNN, ldB, 0.0, A, ldA);
		l += 3;
	}

	for (; l != maxp1; l += 2) {
		xgemm("N", "N", N, N, N, 1.0, B[l],
		      ldB, A, ldA, 0.0, tmpNN, ldB);
		xgemm("N", "N", N, N, N, 1.0, B[l + 1],
		      ldB, tmpNN, ldB, 0.0, A, ldA);
	}
}

// G = L G R
void wrap(const int N, const int ld,
		num *const G,
		const num *const L, const num *const R,
		num *const tmpNN)
{
	xgemm("N", "N", N, N, N, 1.0, G, ld, R, ld, 0.0, tmpNN, ld);
	xgemm("N", "N", N, N, N, 1.0, L, ld, tmpNN, ld, 0.0, G, ld);
}

int get_lwork(const int N, const int ld)
{
	num lwork;
	int info = 0;
	int max_lwork = 0;

	xgeqp3(N, N, NULL, ld, NULL, NULL, &lwork, -1, NULL, &info);
	if (creal(lwork) > max_lwork) max_lwork = (int)lwork;

	xgeqrf(N, N, NULL, ld, NULL, &lwork, -1, &info);
	if (creal(lwork) > max_lwork) max_lwork = (int)lwork;

	xunmqr("R", "N", N, N, N, NULL, ld, NULL, NULL, ld, &lwork, -1, &info);
	if (creal(lwork) > max_lwork) max_lwork = (int)lwork;

	xunmqr("R", "C", N, N, N, NULL, ld, NULL, NULL, ld, &lwork, -1, &info);
	if (creal(lwork) > max_lwork) max_lwork = (int)lwork;

	xunmqr("L", "N", N, N, N, NULL, ld, NULL, NULL, ld, &lwork, -1, &info);
	if (creal(lwork) > max_lwork) max_lwork = (int)lwork;

	return max_lwork;
}

static void calc_QdX_first(
		const int trans, // if 1, calculate QdX of B^T (conjugate transpose for complex)
		const int N, const int ld,
		const num *const B, // input
		struct QdX *const QdX, // output
		num *const tmpN, // work arrays
		int *const pvt,
		num *const work, const int lwork)
{
	__builtin_assume(ld % MEM_ALIGN_NUM == 0);
	(void)__builtin_assume_aligned(B, MEM_ALIGN);
	(void)__builtin_assume_aligned(tmpN, MEM_ALIGN);
	(void)__builtin_assume_aligned(work, MEM_ALIGN);
	num *const Q = __builtin_assume_aligned(QdX->Q, MEM_ALIGN);
	num *const d = __builtin_assume_aligned(QdX->d, MEM_ALIGN);
	num *const X = __builtin_assume_aligned(QdX->X, MEM_ALIGN);
	num *const iL = __builtin_assume_aligned(QdX->iL, MEM_ALIGN);
	num *const R = __builtin_assume_aligned(QdX->R, MEM_ALIGN);

	int info = 0;
	for (int i = 0; i < N; i++) pvt[i] = 0;
	xomatcopy(trans ? 'C' : 'N', N, N, 1.0, B, ld, Q, ld);

	// use R as tau, and use d as RWORK for zgeqp3
	xgeqp3(N, N, Q, ld, pvt, R, work, lwork, (double *)d, &info);

	for (int i = 0; i < N; i++) {
		d[i] = Q[i + i*ld];
		if (d[i] == 0.0) d[i] = 1.0;
		tmpN[i] = 1.0/d[i];
	}

	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			X[i + j*ld] = 0.0;

	for (int j = 0; j < N; j++)
		for (int i = 0; i <= j; i++)
			X[i + (pvt[j]-1)*ld] = tmpN[i] * Q[i + j*ld];

	// calculate phase of det(Q)
	num phase = 1.0;
#ifdef USE_CPLX
	for (int i = 0; i < N; i++) {
		double vv = 1.0;
		for (int j = i + 1; j < N; j++)
			vv += creal(Q[j + i*ld])*creal(Q[j + i*ld])
			    + cimag(Q[j + i*ld])*cimag(Q[j + i*ld]);
		const num ref = 1.0 - R[i]*vv;
		phase *= ref/cabs(ref);
	}
#else
	for (int i = 0; i < N; i++)
		if (R[i] > 0)
			phase *= -1;
#endif

	xungqr(N, N, N, Q, ld, R, work, lwork, &info); // form Q

	// form iL = invdb Q.T, R = ds X
	for (int i = 0; i < N; i++) {
		if (fabs(d[i]) > 1.0) {
#ifdef USE_CPLX
			phase *= cabs(tmpN[i])/tmpN[i];
#else
			if (tmpN[i] < 0) phase *= -1;
#endif
			for (int j = 0; j < N; j++)
				iL[i + j*ld] = tmpN[i] * conj(Q[j + i*ld]);
			for (int j = 0; j < N; j++)
				R[i + j*ld] = X[i + j*ld];
		} else {
			for (int j = 0; j < N; j++)
				iL[i + j*ld] = conj(Q[j + i*ld]);
			for (int j = 0; j < N; j++)
				R[i + j*ld] = d[i] * X[i + j*ld];
		}
	}
	QdX->phase_iL = phase;
}

void calc_QdX(
		const int trans, // if 1, calculate QdX of B^T (conjugate transpose for complex)
		const int N, const int ld,
		const num *const B, // input
		const struct QdX *const QdX_prev,  // input, previous QdX or NULL if none
		struct QdX *const QdX,  // output
		num *const tmpN, // work arrays
		int *const pvt,
		num *const work, const int lwork)
{
	if (QdX_prev == NULL)
		return calc_QdX_first(trans, N, ld, B, QdX, tmpN, pvt, work, lwork);

	__builtin_assume(ld % MEM_ALIGN_NUM == 0);
	(void)__builtin_assume_aligned(B, MEM_ALIGN);
	(void)__builtin_assume_aligned(tmpN, MEM_ALIGN);
	(void)__builtin_assume_aligned(work, MEM_ALIGN);
	const num *const prevQ = __builtin_assume_aligned(QdX_prev->Q, MEM_ALIGN);
	const num *const prevd = __builtin_assume_aligned(QdX_prev->d, MEM_ALIGN);
	const num *const prevX = __builtin_assume_aligned(QdX_prev->X, MEM_ALIGN);
	const num *const previL = __builtin_assume_aligned(QdX_prev->iL, MEM_ALIGN);
	const num *const prevR = __builtin_assume_aligned(QdX_prev->R, MEM_ALIGN);
	num *const Q = __builtin_assume_aligned(QdX->Q, MEM_ALIGN);
	num *const d = __builtin_assume_aligned(QdX->d, MEM_ALIGN);
	num *const X = __builtin_assume_aligned(QdX->X, MEM_ALIGN);
	num *const iL = __builtin_assume_aligned(QdX->iL, MEM_ALIGN);
	num *const R = __builtin_assume_aligned(QdX->R, MEM_ALIGN);

	int info = 0;

	// use X as temp N*N storage
	xgemm(trans ? "C" : "N", "N", N, N, N, 1.0, B, ld, prevQ, ld, 0.0, X, ld);

	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			X[i + j*ld] *= prevd[j];

	for (int j = 0; j < N; j++) { // use tmpN for norms
		num norm = 0;
		for (int i = 0; i < N; i++)
			norm += X[i + j*ld] * conj(X[i + j*ld]);
		tmpN[j] = norm;
	}

	pvt[0] = 0;
	for (int i = 1; i < N; i++) { // insertion sort
		int j;
		for (j = i; j > 0 && creal(tmpN[pvt[j-1]]) < creal(tmpN[i]); j--)
			pvt[j] = pvt[j-1];
		pvt[j] = i;
	}

	for (int j = 0; j < N; j++) // pre-pivot
		my_copy(Q + j*ld, X + pvt[j]*ld, N);

	// use R as tau
	xgeqrf(N, N, Q, ld, R, work, lwork, &info);

	for (int i = 0; i < N; i++) {
		d[i] = Q[i + i*ld];
		if (d[i] == 0.0) d[i] = 1.0;
		tmpN[i] = 1.0/d[i];
	}

	for (int j = 0; j < N; j++)
		for (int i = 0; i <= j; i++)
			Q[i + j*ld] *= tmpN[i];

	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			X[i + j*ld] = prevX[pvt[i] + j*ld];

	xtrmm("L", "U", "N", "N", N, N, 1.0, Q, ld, X, ld);

	// calculate phase of det(Q)
	num phase = 1.0;
#ifdef USE_CPLX
	for (int i = 0; i < N; i++) {
		double vv = 1.0;
		for (int j = i + 1; j < N; j++)
			vv += creal(Q[j + i*ld])*creal(Q[j + i*ld])
			    + cimag(Q[j + i*ld])*cimag(Q[j + i*ld]);
		const num ref = 1.0 - R[i]*vv;
		phase *= ref/cabs(ref);
	}
#else
	for (int i = 0; i < N; i++)
		if (R[i] > 0)
			phase *= -1;
#endif

	xungqr(N, N, N, Q, ld, R, work, lwork, &info); // form Q

	// form iL = invdb Q.T, R = ds X
	for (int i = 0; i < N; i++) {
		if (fabs(d[i]) > 1.0) {
#ifdef USE_CPLX
			phase *= cabs(tmpN[i])/tmpN[i];
#else
			if (tmpN[i] < 0) phase *= -1;
#endif
			for (int j = 0; j < N; j++)
				iL[i + j*ld] = tmpN[i] * conj(Q[j + i*ld]);
			for (int j = 0; j < N; j++)
				R[i + j*ld] = X[i + j*ld];
		} else {
			for (int j = 0; j < N; j++)
				iL[i + j*ld] = conj(Q[j + i*ld]);
			for (int j = 0; j < N; j++)
				R[i + j*ld] = d[i] * X[i + j*ld];
		}
	}
	QdX->phase_iL = phase;
}

static inline num calc_phase_LU(const int N, const int ld, const num *A, const int *pvt)
{
#ifdef USE_CPLX
	num phase = 1.0;
	for (int i = 0; i < N; i++) {
		const num c = A[i + i*ld];
		phase *= c/cabs(c);
		if (pvt[i] != i+1)
			phase *= -1.0;
	}
	return phase;
#else
	int sign = 1.0;
	for (int i = 0; i < N; i++)
		if ((pvt[i] != i+1) ^ (A[i + i*ld] < 0))
			sign *= -1;
	return (num)sign;
#endif
}

// G = (1 + L R)^-1 = (iL + R)^-1 iL
// 1. G = iL
// 2. tmpNN = G + R
// 3. G = tmpNN^-1 G
static num calc_Gtt_last(
		const int trans, // if 0 calculate, calculate G = (1 + L R)^-1. if 1, calculate G = (1 + R.T L.T)^-1
		const int N, const int ld,
		const struct QdX *const QdX, // input
		num *const G, // output
		num *const tmpNN, // work arrays
		int *const pvt)
{
	__builtin_assume(ld % MEM_ALIGN_NUM == 0);
	(void)__builtin_assume_aligned(G, MEM_ALIGN);
	(void)__builtin_assume_aligned(tmpNN, MEM_ALIGN);
	const num *const iL = __builtin_assume_aligned(QdX->iL, MEM_ALIGN);
	const num *const R = __builtin_assume_aligned(QdX->R, MEM_ALIGN);

	int info = 0;

	xomatcopy('N', N, N, 1.0, iL, ld, G, ld); // 1
	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			tmpNN[i + j*ld] = G[i + j*ld] + R[i + j*ld]; // 2
	xgetrf(N, N, tmpNN, ld, pvt, &info);
	xgetrs("N", N, N, tmpNN, ld, pvt, G, ld, &info); // 3
	num phase = calc_phase_LU(N, ld, tmpNN, pvt)*QdX->phase_iL;
	if (trans) {
		ximatcopy('C', N, N, 1.0, G, ld, ld);
		return conj(phase);
	} else {
		return phase;
	}
}

// G = (1 + L0 R0 R1.T L1.T)^-1
//   = iL1.T (iL0 iL1.T + R0 R1.T)^-1 iL0
// 1. G = R0 R1.T
// 2. G += iL0 iL1.T
// 3. tmpNN = iL0
// 4. tmpNN = G^-1 tmpNN
// 5. G = iL1.T tmpNN
num calc_Gtt(
		const int N, const int ld,
		const struct QdX *const QdX0, // input
		const struct QdX *const QdX1, // input
		num *const G, // output
		num *const tmpNN, // work arrays
		int *const pvt)
{
	if (QdX1 == NULL)
		return calc_Gtt_last(0, N, ld, QdX0, G, tmpNN, pvt);
	else if (QdX0 == NULL)
		return calc_Gtt_last(1, N, ld, QdX1, G, tmpNN, pvt);

	__builtin_assume(ld % MEM_ALIGN_NUM == 0);
	(void)__builtin_assume_aligned(G, MEM_ALIGN);
	(void)__builtin_assume_aligned(tmpNN, MEM_ALIGN);
	const num *const iL0 = __builtin_assume_aligned(QdX0->iL, MEM_ALIGN);
	const num *const R0 = __builtin_assume_aligned(QdX0->R, MEM_ALIGN);
	const num *const iL1 = __builtin_assume_aligned(QdX1->iL, MEM_ALIGN);
	const num *const R1 = __builtin_assume_aligned(QdX1->R, MEM_ALIGN);

	int info = 0;

	xgemm("N", "C", N, N, N, 1.0, R0, ld, R1, ld, 0.0, G, ld); // 1
	xgemm("N", "C", N, N, N, 1.0, iL0, ld, iL1, ld, 1.0, G, ld); // 2
	xomatcopy('N', N, N, 1.0, iL0, ld, tmpNN, ld); // 3
	xgetrf(N, N, G, ld, pvt, &info); // 4
	const num phase_LU = calc_phase_LU(N, ld, G, pvt);
	xgetrs("N", N, N, G, ld, pvt, tmpNN, ld, &info);
	xgemm("C", "N", N, N, N, 1.0, iL1, ld, tmpNN, ld, 0.0, G, ld); //5
	return phase_LU*QdX0->phase_iL*conj(QdX1->phase_iL);
}

// G00 = (1 + R1.T L1.T L0 R0)^-1
// Gtt = (1 + L0 R0 R1.T L1.T)^-1
//     = iL1.T (iL0 iL1.T + R0 R1.T)^-1 iL0
// G0t = -(L1.T^-1 R1.T^-1 + L0 R0)^-1
//     = -R1.T (iL0 iL1.T + R0 R1.T)^-1 iL0
// Gt0 = (X0^-1 d0^-1 Q0.T + X1.T d1 Q1.T)^-1
//     = iL1.T (iL0 iL1.T + R0 R1.T)^-1 R0
// 1. Gt0 = R0 R1.T
// 2. Gt0 += iL0 iL1.T
// 3. tmpNN = iL0
// 4. tmpNN = Gt0^-1 tmpNN
// 5. G0t = -R1.T tmpNN
// 6. Gtt = iL1.T tmpNN
// 7. tmpNN = R0
// 8. tmpNN = Gt0^-1 tmpNN
// 9. Gt0 = iL1.T tmpNN
void calc_G0t_Gtt_Gt0(
		const int N, const int ld,
		const struct QdX *const QdX0, // input
		const struct QdX *const QdX1, // input
		num *const G0t, // output
		num *const Gtt, // output
		num *const Gt0, // output
		num *const tmpNN, // work arrays
		int *const pvt)
{
	__builtin_assume(ld % MEM_ALIGN_NUM == 0);
	(void)__builtin_assume_aligned(Gtt, MEM_ALIGN);
	(void)__builtin_assume_aligned(G0t, MEM_ALIGN);
	(void)__builtin_assume_aligned(Gt0, MEM_ALIGN);
	(void)__builtin_assume_aligned(tmpNN, MEM_ALIGN);
	const num *const iL0 = __builtin_assume_aligned(QdX0->iL, MEM_ALIGN);
	const num *const R0 = __builtin_assume_aligned(QdX0->R, MEM_ALIGN);
	const num *const iL1 = __builtin_assume_aligned(QdX1->iL, MEM_ALIGN);
	const num *const R1 = __builtin_assume_aligned(QdX1->R, MEM_ALIGN);

	int info = 0;

	xgemm("N", "C", N, N, N, 1.0, R0, ld, R1, ld, 0.0, Gt0, ld); // 1
	xgemm("N", "C", N, N, N, 1.0, iL0, ld, iL1, ld, 1.0, Gt0, ld); // 2
	xomatcopy('N', N, N, 1.0, iL0, ld, tmpNN, ld); // 3
	xgetrf(N, N, Gt0, ld, pvt, &info); // 4
	xgetrs("N", N, N, Gt0, ld, pvt, tmpNN, ld, &info);
	xgemm("C", "N", N, N, N, -1.0, R1, ld, tmpNN, ld, 0.0, G0t, ld); // 5
	xgemm("C", "N", N, N, N, 1.0, iL1, ld, tmpNN, ld, 0.0, Gtt, ld); // 6
	xomatcopy('N', N, N, 1.0, R0, ld, tmpNN, ld); // 7
	xgetrs("N", N, N, Gt0, ld, pvt, tmpNN, ld, &info); // 8
	xgemm("C", "N", N, N, N, 1.0, iL1, ld, tmpNN, ld, 0.0, Gt0, ld); // 9
}

static void expand_g(const int N, const int ld, const int L, const int E, const int n_matmul,
		num *const *const B,
		num *const *const iB,
		num *const *const G0t,
		num *const *const Gtt,
		num *const *const Gt0,
		num *const tmpNN)
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

	// expand G0t
	for (int f = 0; f < E; f++) {
		const int l = f*n_matmul;
		const int lstop = (f == 0) ? lstop_first : l - n_left;
		const int rstop = (f == E - 1) ? rstop_last : l + n_right;
		for (int m = l; m != lstop;) {
			const int next = (m - 1 + L) % L;
			const num alpha = (m == 0) ? -1.0 : 1.0;
			xgemm("N", "N", N, N, N, alpha,
			      G0t[m], ld, B[next], ld, 0.0,
			      G0t[next], ld);
			m = next;
		}
		for (int m = l; m != rstop;) {
			const int next = (m + 1) % L;
			const num alpha = (next == 0) ? -1.0 : 1.0;
			const num beta = (m == 0) ? -alpha : 0.0;
			if (m == 0)
				for (int j = 0; j < N; j++)
				for (int i = 0; i < N; i++)
					G0t[next][i + ld*j] = iB[m][i + ld*j];
			xgemm("N", "N", N, N, N, alpha,
			      G0t[m], ld, iB[m], ld, beta,
			      G0t[next], ld);
			m = next;
		}
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
			      Gtt[m], ld, B[next], ld, 0.0,
			      tmpNN, ld);
			xgemm("N", "N", N, N, N, 1.0,
			      iB[next], ld, tmpNN, ld, 0.0,
			      Gtt[next], ld);
			m = next;
		}
		for (int m = k; m != dstop;) {
			const int next = (m + 1) % L;
			xgemm("N", "N", N, N, N, 1.0,
			      Gtt[m], ld, iB[m], ld, 0.0,
			      tmpNN, ld);
			xgemm("N", "N", N, N, N, 1.0,
			      B[m], ld, tmpNN, ld, 0.0,
			      Gtt[next], ld);
			m = next;
		}
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
					Gt0[next][i + ld*j] = iB[next][i + ld*j];
			xgemm("N", "N", N, N, N, alpha,
			      iB[next], ld, Gt0[m], ld, beta,
			      Gt0[next], ld);
			m = next;
		}
		for (int m = k; m != dstop;) {
			const int next = (m + 1) % L;
			const num alpha = (next == 0) ? -1.0 : 1.0;
			xgemm("N", "N", N, N, N, alpha,
			      B[m], ld, Gt0[m], ld, 0.0,
			      Gt0[next], ld);
			if (next == 0) // should never happen
				for (int i = 0; i < N; i++)
					Gt0[next][i + ld*i] += 1.0;
			m = next;
		}
	}
}

void calc_ue_g(const int N, const int ld, const int L, const int F, const int n_matmul,
		num *const *const B,
		num *const *const iB,
		const struct QdX *const QdX0, // input
		const struct QdX *const QdXL, // input
		num *const *const G0t, // output
		num *const *const Gtt, // output
		num *const *const Gt0, // output
		num *const tmpNN, // work arrays
		int *const pvt)
{
	// assumes G0t[0], Gtt[0], Gt0[0] already initialized
	// only need to do every other f
	for (int f = 2; f < F; f += 2) {
		const int l = f*n_matmul;
		calc_G0t_Gtt_Gt0(N, ld, &QdX0[f - 1], &QdXL[f], G0t[l], Gtt[l], Gt0[l], tmpNN, pvt);
	}
	expand_g(N, ld, L, 1 + (F - 1)/2, 2*n_matmul, B, iB, G0t, Gtt, Gt0, tmpNN);
}

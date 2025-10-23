#pragma once

#if __APPLE__

#include <Accelerate/Accelerate.h>
#define MKL_Complex16 double complex
#define zgemm zgemm_
#define dgemm dgemm_
#define zgemv zgemv_
#define dgemv dgemv_
#define ztrmm ztrmm_
#define dtrmm dtrmm_
#define ztrsm ztrsm_
#define dtrsm dtrsm_
#define zgetrf zgetrf_
#define dgetrf dgetrf_
#define zgetri zgetri_
#define dgetri dgetri_
#define zgetrs zgetrs_
#define dgetrs dgetrs_
#define zgeqp3 zgeqp3_
#define dgeqp3 dgeqp3_
#define zgeqrf zgeqrf_
#define dgeqrf dgeqrf_
#define zunmqr zunmqr_
#define dormqr dormqr_
#define zungqr zungqr_
#define dorgqr dorgqr_
#define ztrtri ztrtri_
#define dtrtri dtrtri_

#else

#include <mkl.h>

#endif

#include <tgmath.h>
#include "mem.h"

#ifdef USE_CPLX
	#include <complex.h>
	typedef double complex num;
	#define cast(p) (MKL_Complex16 *)(p)
	#define ccast(p) (const MKL_Complex16 *)(p)

	#define RC(x) x##_cplx
#else
	typedef double num;
	#define cast(p) (p)
	#define ccast(p) (p)

	#define RC(x) x##_real
#endif

#define MEM_ALIGN_NUM (MEM_ALIGN/sizeof(num))

static inline size_t best_ld(const size_t N)
{
	// leading dimension should correspond to a multiple of 64 bytes
	const size_t ld_aligned = ((N + MEM_ALIGN_NUM - 1)/MEM_ALIGN_NUM)*MEM_ALIGN_NUM;
	const size_t ld_bytes = ld_aligned*sizeof(num);
	// but not be a large power of two
	if (ld_bytes >= 512 && !(ld_bytes & (ld_bytes - 1)))
		return ld_aligned + MEM_ALIGN_NUM;
	return ld_aligned;
}

static inline void xgemm(const char *transa, const char *transb,
		const int m, const int n, const int k,
		const num alpha, const num *a, const int lda,
		const num *b, const int ldb,
		const num beta, num *c, const int ldc)
{
#ifdef USE_CPLX
	zgemm(
#else
	dgemm(
#endif
	transa, transb, &m, &n, &k,
	ccast(&alpha), ccast(a), &lda, ccast(b), &ldb,
	ccast(&beta), cast(c), &ldc);
}

static inline void xgemv(const char *trans, const int m, const int n,
		const num alpha, const num *a, const int lda,
		const num *x, const int incx,
		const num beta, num *y, const int incy)
{
#ifdef USE_CPLX
	zgemv(
#else
	dgemv(
#endif
	trans, &m, &n,
	ccast(&alpha), ccast(a), &lda, ccast(x), &incx,
	ccast(&beta), cast(y), &incy);
}

static inline void xtrmm(const char *side, const char *uplo, const char *transa, const char *diag,
		const int m, const int n,
		const num alpha, const num *a, const int lda,
		num *b, const int ldb)
{
#ifdef USE_CPLX
	ztrmm(
#else
	dtrmm(
#endif
	side, uplo, transa, diag, &m, &n,
	ccast(&alpha), ccast(a), &lda, cast(b), &ldb);
}

static inline void xtrsm(const char *side, const char *uplo, const char *transa, const char *diag,
		const int m, const int n,
		const num alpha, const num *a, const int lda,
		num *b, const int ldb)
{
#ifdef USE_CPLX
	ztrmm(
#else
	dtrmm(
#endif
	side, uplo, transa, diag, &m, &n,
	ccast(&alpha), ccast(a), &lda, cast(b), &ldb);
}

static inline void xgetrf(const int m, const int n, num* a,
		const int lda, int* ipiv, int* info)
{
#ifdef USE_CPLX
	zgetrf(
#else
	dgetrf(
#endif
	&m, &n, cast(a), &lda, ipiv, info);
}


static inline void xgetri(const int n, num* a, const int lda, const int* ipiv,
		num* work, const int lwork, int* info)
{
#ifdef USE_CPLX
	zgetri(
#else
	dgetri(
#endif
	&n, cast(a), &lda, ipiv, cast(work), &lwork, info);
}

static inline void xgetrs(const char* trans, const int n, const int nrhs,
		const num* a, const int lda, const int* ipiv,
		num* b, const int ldb, int* info)
{
#ifdef USE_CPLX
	zgetrs(
#else
	dgetrs(
#endif
	trans, &n, &nrhs, ccast(a), &lda, ipiv, cast(b), &ldb, info);
}

static inline void xgeqp3(const int m, const int n, num* a, const int lda, int* jpvt, num* tau,
		num* work, const int lwork, double* rwork, int* info)
{
#ifdef USE_CPLX
	zgeqp3(&m, &n, cast(a), &lda, jpvt, cast(tau),
	cast(work), &lwork, rwork, info);
#else
	dgeqp3(&m, &n, cast(a), &lda, jpvt, cast(tau),
	cast(work), &lwork, info);
	(void)rwork; // unused
#endif
}

static inline void xgeqrf(const int m, const int n, num* a, const int lda, num* tau,
		num* work, const int lwork, int* info)
{
#ifdef USE_CPLX
	zgeqrf(
#else
	dgeqrf(
#endif
	&m, &n, cast(a), &lda, cast(tau), cast(work), &lwork, info);
}

static inline void xunmqr(const char* side, const char* trans,
		const int m, const int n, const int k, const num* a,
		const int lda, const num* tau, num* c,
		const int ldc, num* work, const int lwork, int* info)
{
#ifdef USE_CPLX
	zunmqr(side, trans,
#else
	dormqr(side, trans[0] == 'C' ? "T" : trans,
#endif
	&m, &n, &k, ccast(a), &lda, ccast(tau),
	cast(c), &ldc, cast(work), &lwork, info);
}

static inline void xungqr(const int m, const int n, const int k, num* a,
		const int lda, const num* tau, num* work, const int lwork, int* info)
{
#ifdef USE_CPLX
	zungqr(
#else
	dorgqr(
#endif
	&m, &n, &k, cast(a), &lda, ccast(tau), cast(work), &lwork, info);
}

static inline void xtrtri(const char* uplo, const char* diag, const int n,
		num* a, const int lda, int* info)
{
#ifdef USE_CPLX
	ztrtri(
#else
	dtrtri(
#endif
	uplo, diag, &n, cast(a), &lda, info);
}

static inline void ximatcopy(const char trans, const size_t rows, const size_t cols,
const num alpha, num *AB, const size_t lda, const size_t ldb)
{
#ifdef __APPLE__
// assumes rows == cols, lda == ldb, alpha = 1.0
	if (trans == 'C') {
		for (size_t j = 0; j < cols; j++) {
			for (size_t i = 0; i < j; i++) {
				num temp = conj(AB[i + j*lda]);
				AB[i + j*lda] = conj(AB[j + i*lda]);
				AB[j + i*lda] = temp;
			}
			AB[j + j*lda] = conj(AB[j + j*lda]);
		}
	}
	(void)alpha;
	(void)rows;
	(void)ldb;
#else

#ifdef USE_CPLX
	mkl_zimatcopy('C', trans, rows, cols, *cast(&alpha), cast(AB), lda, ldb);
#else
	mkl_dimatcopy('C', trans, rows, cols, alpha, AB, lda, ldb);
#endif

#endif
}

static inline void xomatcopy(const char trans, const size_t rows, const size_t cols,
const num alpha, const num *A, const size_t lda, num *B, const size_t ldb)
{
#ifdef __APPLE__

	if (trans == 'N') {
		for (size_t j = 0; j < cols; j++)
			for (size_t i = 0; i < rows; i++)
				B[i + j*ldb] = alpha*A[i + j*lda];
	} else {
		for (size_t j = 0; j < cols; j++)
			for (size_t i = 0; i < rows; i++)
				B[i + j*ldb] = alpha*conj(A[j + i*lda]);
	}

#else

#ifdef USE_CPLX
	mkl_zomatcopy('C', trans, rows, cols, *cast(&alpha), ccast(A), lda, cast(B), ldb);
#else
	mkl_domatcopy('C', trans, rows, cols, alpha, A, lda, B, ldb);
#endif

#endif
}

// B = A diag(d)
static inline void mul_mat_diag(const int N, const int ld, const num *const A, const num *const d, num *const B)
{
	__builtin_assume(ld % MEM_ALIGN_NUM == 0);
	(void)__builtin_assume_aligned(A, MEM_ALIGN);
	(void)__builtin_assume_aligned(d, MEM_ALIGN);
	(void)__builtin_assume_aligned(B, MEM_ALIGN);

	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			B[i + ld*j] = A[i + ld*j] * d[j];
}


// B = diag(d) A
static inline void mul_diag_mat(const int N, const int ld, const num *const d, const num *const A, num *const B)
{
	__builtin_assume(ld % MEM_ALIGN_NUM == 0);
	(void)__builtin_assume_aligned(d, MEM_ALIGN);
	(void)__builtin_assume_aligned(A, MEM_ALIGN);
	(void)__builtin_assume_aligned(B, MEM_ALIGN);

	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			B[i + ld*j] = d[i] * A[i + ld*j];
}

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
} while (0)

#undef ccast
#undef cast

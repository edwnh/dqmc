#pragma once

#include <mkl.h>
#include <tgmath.h>

#ifdef USE_CPLX
	#include <complex.h>
	typedef double complex num;
	#define cast(p) (MKL_Complex16 *)(p)
	#define ccast(p) (const MKL_Complex16 *)(p)
#else
	typedef double num;
	#define cast(p) (p)
	#define ccast(p) (p)
#endif


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
	cast(work), &lwork, info); // rwork not used
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

static inline void ximatcopy(const char ordering, const char trans, const size_t rows, const size_t cols,
const num alpha, num *AB, const size_t lda, const size_t ldb)
{
#ifdef USE_CPLX
	mkl_zimatcopy(ordering, trans, rows, cols, *cast(&alpha), cast(AB), lda, ldb);
#else
	mkl_dimatcopy(ordering, trans, rows, cols, alpha, AB, lda, ldb);
#endif
}

static inline void xomatcopy(const char ordering, const char trans, const size_t rows, const size_t cols,
const num alpha, const num *A, const size_t lda, num *B, const size_t ldb)
{
#ifdef USE_CPLX
	mkl_zomatcopy(ordering, trans, rows, cols, *cast(&alpha), ccast(A), lda, cast(B), ldb);
#else
	mkl_domatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb);
#endif
}

#undef ccast
#undef cast

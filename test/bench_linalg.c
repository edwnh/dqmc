#include "linalg.h"
#include "time_.h"
#include "mem.h"
#include "rand.h"
#include <stdio.h>
#include <stdlib.h>

#define MIN_TIME 1

static void bench_gemm(int N, int ld, num *A, num *B, num *C)
{
	tick_t elapsed = 0;
	int trials = 0;
	const tick_t time_start = time_wall();
	for (; (elapsed = time_wall() - time_start) < MIN_TIME*TICK_PER_SEC; trials++) {
		xgemm("N", "N", N, N, N, 1.0, A, ld, B, ld, 0.0, C, ld);
	}
	const double ops = 2.0*N*N*N;
	printf("%.3f us\t%.3f GFlops\n", US_PER_TICK*((double)elapsed)/trials, ops*trials/(elapsed));
}

static void bench_trmm(int N, int ld, num *A, num *B, num *C)
{
	tick_t elapsed = 0;
	int trials = 0;
	const tick_t time_start = time_wall();
	for (; (elapsed = time_wall() - time_start) < MIN_TIME*TICK_PER_SEC; trials++) {
		for (int i = 0; i < N*ld; i++) C[i] = B[i];
		xtrmm("R", "L", "N", "U", N, N, 1.0, A, ld, C, ld);
	}
	const double ops = 1.0*N*N*N;
	printf("%.3f us\t%.3f GFlops\n", US_PER_TICK*((double)elapsed)/trials, ops*trials/(elapsed));
}

static void bench_trsm(int N, int ld, num *A, num *B, num *C)
{
	tick_t elapsed = 0;
	int trials = 0;
	const tick_t time_start = time_wall();
	for (; (elapsed = time_wall() - time_start) < MIN_TIME*TICK_PER_SEC; trials++) {
		for (int i = 0; i < N*ld; i++) C[i] = B[i];
		xtrsm("R", "L", "N", "U", N, N, 1.0, A, ld, C, ld);
	}
	const double ops = 1.0*N*N*N;
	printf("%.3f us\t%.3f GFlops\n", US_PER_TICK*((double)elapsed)/trials, ops*trials/(elapsed));
}

static void bench_trtri(int N, int ld, num *A, num *B, num *C)
{
	tick_t elapsed = 0;
	int trials = 0;
	const tick_t time_start = time_wall();
	for (; (elapsed = time_wall() - time_start) < MIN_TIME*TICK_PER_SEC; trials++) {
		for (int i = 0; i < N*ld; i++) C[i] = A[i];
		xtrtri("L", "U", N, C, ld, &(int){0});
	}
	const double ops = (1.0/3)*N*N*N;
	printf("%.3f us\t%.3f GFlops\n", US_PER_TICK*((double)elapsed)/trials, ops*trials/(elapsed));
}

static void bench_getrf(int N, int ld, num *A, num *B, num *C, int *pvt)
{
	tick_t elapsed = 0;
	int trials = 0;
	const tick_t time_start = time_wall();
	for (; (elapsed = time_wall() - time_start) < MIN_TIME*TICK_PER_SEC; trials++) {
		for (int i = 0; i < N*ld; i++) C[i] = A[i];
		xgetrf(N, N, C, ld, pvt, &(int){0});
	}
	const double ops = (2.0/3)*N*N*N - 0.5*N*N + (5.0/6)*N;
	printf("%.3f us\t%.3f GFlops\n", US_PER_TICK*((double)elapsed)/trials, ops*trials/(elapsed));
}

static void bench_geqrf(int N, int ld, num *A, num *B, num *C, num *D)
{
	tick_t elapsed = 0;
	int trials = 0;
	const tick_t time_start = time_wall();
	for (; (elapsed = time_wall() - time_start) < MIN_TIME*TICK_PER_SEC; trials++) {
		for (int i = 0; i < N*ld; i++) C[i] = A[i];
		xgeqrf(N, N, C, ld, B, D, N*N, &(int){0});
	}
	const double ops = (4.0/3)*N*N*N + 2.0*N*N + (14.0/3)*N;
	printf("%.3f us\t%.3f GFlops\n", US_PER_TICK*((double)elapsed)/trials, ops*trials/(elapsed));
}

// inconsistent performance on laptop
static void bench_geqp3(int N, int ld, num *A, num *B, num *C, num *D, int *pvt)
{
	tick_t elapsed = 0;
	int trials = 0;
	const tick_t time_start = time_wall();
	for (; (elapsed = time_wall() - time_start) < MIN_TIME*TICK_PER_SEC; trials++) {
		for (int i = 0; i < N*ld; i++) C[i] = A[i];
		for (int i = 0; i < N; i++) pvt[i] = 0;
		xgeqp3(N, N, C, ld, pvt, B, D, N*N, (double *)(B + N), &(int){0});
	}
	const double ops = (4.0/3)*N*N*N + 2.0*N*N + (14.0/3)*N;
	printf("%.3f us\t%.3f GFlops\n", US_PER_TICK*((double)elapsed)/trials, ops*trials/(elapsed));
}

int main(int argc, char *argv[])
{
	if (argc < 2) {
		printf("usage: %s <N>\n", argv[0]);
		return 1;
	}

	const int N = atoi(argv[1]);
	const int ld = best_ld(N);

	#define ALLOC_TABLE(X, FOR, ENDFOR) \
		X(num *const restrict A, N*ld * sizeof(num)) \
		X(num *const restrict B, N*ld * sizeof(num)) \
		X(num *const restrict C, N*ld * sizeof(num)) \
		X(num *const restrict D, N*ld * sizeof(num)) \
		X(int *const restrict pvt, N * sizeof(int))

	void *pool = my_calloc(POOL_GET_SIZE(ALLOC_TABLE));
	POOL_DO_ALLOC(pool, ALLOC_TABLE);
	#undef ALLOC_TABLE

	uint64_t rng[17] = {0};
	for (int i = 0; i < 16; i++) rng[i] = 1234567*i*i + 654321;
	for (int i = 0; i < N*ld; i++) {
		A[i] = 2.0*rand_doub(rng) - 1.0;
		B[i] = 2.0*rand_doub(rng) - 1.0;
	}

	printf("gemm:\t"); bench_gemm(N, ld, A, B, C);
	printf("trmm:\t"); bench_trmm(N, ld, A, B, C);
	printf("trsm:\t"); bench_trsm(N, ld, A, B, C);
	printf("trtri:\t"); bench_trtri(N, ld, A, B, C);
	printf("getrf:\t"); bench_getrf(N, ld, A, B, C, pvt);
	printf("geqrf:\t"); bench_geqrf(N, ld, A, B, C, D);
	printf("geqp3:\t"); bench_geqp3(N, ld, A, B, C, D, pvt);

	my_free(pool);
}

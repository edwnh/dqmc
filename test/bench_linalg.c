#include "linalg.h"
#include "time_.h"
#include "mem.h"
#include "rand.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    if (argc != 3) {
        printf("Usage: %s <N> <trials>\n", argv[0]);
        return 1;
    }

    const int N = atoi(argv[1]);
	const int ld = mem_best_ld(N);
    const int trials = atoi(argv[2]);

	struct mem_pool *mp = pool_new(4 * MEM_ALIGN_UP(N*ld * sizeof(num)) + MEM_ALIGN_UP(N * sizeof(int)));
	num *const restrict A = pool_alloc(mp, N*ld * sizeof(num));
	num *const restrict B = pool_alloc(mp, N*ld * sizeof(num));
	num *const restrict C = pool_alloc(mp, N*ld * sizeof(num));
	num *const restrict D = pool_alloc(mp, N*ld * sizeof(num));
	int *const restrict pvt = pool_alloc(mp, N * sizeof(int));

	uint64_t rng[17] = {0};
	for (int i = 0; i < 16; i++) rng[i] = 1234567*i*i + 654321;
	for (int i = 0; i < N*ld; i++) {
		A[i] = 2.0*rand_doub(rng) - 1.0;
		B[i] = 2.0*rand_doub(rng) - 1.0;
	}

	// xgeqrf(N, N, A, N, B, D, N*N, &(int){0});

	const tick_t time_start = time_wall();
	for (int t = 0; t < trials; t++) {
		// for (int i = 0; i < N*N; i++) C[i] = A[i];
		// for (int i = 0; i < N; i++) pvt[i] = N - i;

		// xgeqrf(N, N, C, N, B, D, N*N, &(int){0});
		// xgeqp3(N, N, C, N, pvt, B, D, N*N, NULL, &(int){0});

		// for (int i = 0; i < N*N; i++) {C[i] = A[i]; B[i] = A[i];}
		// C[0] += 0.01*t;
		// xgetrf(N, N, C, N, (int *)D, &(int){0});
		// xgetri(N, C, N, (int *)D, B, N*N, &(int){0});
		// xgetrs("N", N, N, C, N, (int *)D, B, N, &(int){0});

		// dorgqr(&(int){N}, &(int){N}, &(int){N}, C, &(int){N}, B, D, &(int){N*N}, &(int){0});

		// for (int i = 0; i < N*N; i++) C[i] = A[i];
		// C[0] += 0.01*t;
		// xungqr(N, N, N, C, N, B, D, N*N, &(int){0});

		// for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) C[i + j*ld] = (i == j);
		// xunmqr("R", "N", N, N, N, A, N, B, C, N, D, N*N, &(int){0});

		// for (int i = 0; i < N*N; i++) B[i] = 0.1*i + 0.01*t;
		A[0] += 0.01;
		xgemm("N", "N", N, N, N, 1.0, A, ld, B, ld, 0.0, C, ld);

		if (A[0]+B[1]+C[2] == 0.1203) return 0;
	}
	const tick_t elapsed = time_wall() - time_start ;
	printf("%.3f s\n", SEC_PER_TICK * (double)elapsed);
	printf("%.3f us\n", US_PER_TICK * ((double)elapsed)/trials);
	printf("%.3f GFlops\n", 2.0*N*N*N*trials/(elapsed));

	pool_free(mp);
}

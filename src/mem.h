#pragma once

#include <string.h>
#include <xmmintrin.h>

#define MEM_ALIGN 64
#define MEM_ALIGN_UP(size) ((((size) + MEM_ALIGN - 1)/MEM_ALIGN) * MEM_ALIGN)

#define my_malloc(size) _mm_malloc((size), MEM_ALIGN)
#define my_free(ptr) _mm_free((ptr))
#define my_copy(dest, src, N) memcpy((dest), (src), (N)*sizeof((src)[0]))

static inline void *my_calloc(size_t size)
{
	void *p = my_malloc(size);
	if (p != NULL) memset(p, 0, size);
	return p;
}

struct mem_pool;

struct mem_pool *pool_new(size_t total_mem);
void *pool_alloc(struct mem_pool *mp, size_t size);
void pool_free(struct mem_pool *mp);

#define POOL_SIZE_X(ptr, pool, size) MEM_ALIGN_UP(size) +
#define POOL_SIZE_FOR(i, num) (num)*(
#define POOL_SIZE_ENDFOR 0) +
#define POOL_ALLOC_X(ptr, pool, size) ptr = pool_alloc(pool, (size));
#define POOL_ALLOC_FOR(i, num) for (int (i) = 0; (i) < (num); (i)++) {
#define POOL_ALLOC_ENDFOR }
#define POOL_GET_SIZE(table) (table(POOL_SIZE_X, POOL_SIZE_FOR, POOL_SIZE_ENDFOR) 0)
#define POOL_DO_ALLOC(table) table(POOL_ALLOC_X, POOL_ALLOC_FOR, POOL_ALLOC_ENDFOR)

#if 0

instead of

	int *a = malloc(12 * sizeof(int));
	double **b = malloc(34 * sizeof(int *));
	for (int i = 0; i < 34; i++)
		b[i] = malloc(56 * sizeof(double));
	double *c = NULL;
	if (flag)
		c = malloc(78 * sizeof(double));
	...use a, b, c here...
	if (flag)
		free(c);
	for (int i = 0; i < 34; i++)
		free(b[i]);
	free(b);
	free(a);

the pool_ functions can be used to allocate a contiguous block of memory (set to zero). allocations within the pool are aligned to MEM_ALIGN. the direct way to do this is

	struct mem_pool *mp = pool_new(
		MEM_ALIGN_UP(12 * sizeof(int)) +
		MEM_ALIGN_UP(34 * sizeof(int *)) +
		34*MEM_ALIGN_UP(56 * sizeof(double))
		flag*MEM_ALIGN_UP(78 * sizeof(double))
	);
	
	int *a = pool_alloc(mp, 12 * sizeof(int));
	double **b = pool_alloc(mp, 34 * sizeof(int *));
	for (int i = 0; i < 34; i++)
		b[i] = pool_alloc(mp, 56 * sizeof(double));
	double *c = NULL;
	if (flag)
		c = pool_alloc(mp, 78 * sizeof(double));
	...use a, b, c here...
	pool_free(mp);

but this requires that the exact right size is inputted into pool_new. the POOL_ macros automate this.

	#define ALLOC_TABLE(XX, FOR, ENDFOR) \
	XX(int *a, mp, 12 * sizeof(int)) \
	XX(double **b, mp, 34 * sizeof(int *)) \
	FOR(i, 34) \
		XX(b[i], mp, 56 * sizeof(double)) \
	ENDFOR \
	XX(double *c, mp, flag*78 * sizeof(double))
	
	struct mem_pool *mp = pool_new(POOL_GET_SIZE(ALLOC_TABLE));
	POOL_DO_ALLOC(ALLOC_TABLE);
	...use a, b, c here...
	pool_free(mp);

#endif

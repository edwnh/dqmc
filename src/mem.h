#pragma once
#include <stdlib.h>
#include <string.h>

#define MEM_ALIGN 64
#define MEM_ALIGN_UP(size) ((((size) + MEM_ALIGN - 1)/MEM_ALIGN) * MEM_ALIGN)

static inline void *my_calloc(size_t size)
{
	const size_t size_aligned = MEM_ALIGN_UP(size);
	void *p = aligned_alloc(MEM_ALIGN, size_aligned);
	if (p != NULL) memset(p, 0, size_aligned);
	return p;
}
#define my_free(ptr) free((ptr))
#define my_copy(dest, src, N) memcpy((dest), (src), (N)*sizeof((src)[0]))

#define POOL_SIZE_X(ptr, size) MEM_ALIGN_UP(size) +
#define POOL_SIZE_FOR(i, n) (n)*(
#define POOL_SIZE_ENDFOR 0) +

#define POOL_ALLOC_X(ptr, size) \
	ptr = ((size) == 0) ? NULL : __builtin_assume_aligned(_next, MEM_ALIGN); \
	_next = ((char *)_next) + MEM_ALIGN_UP((size));
#define POOL_ALLOC_FOR(i, n) for (int (i) = 0; (i) < (n); (i)++) {
#define POOL_ALLOC_ENDFOR }

#define POOL_GET_SIZE(table) (table(POOL_SIZE_X, POOL_SIZE_FOR, POOL_SIZE_ENDFOR) 0)
#define POOL_DO_ALLOC(pool, table) \
	void *_next = pool; \
	table(POOL_ALLOC_X, POOL_ALLOC_FOR, POOL_ALLOC_ENDFOR);

#if 0

instead of

	int *a = malloc(12 * sizeof(int));
	double **b = malloc(34 * sizeof(double *));
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

the POOL_ macros can be used to allocate a contiguous block of memory.
allocations within the pool are aligned to MEM_ALIGN.

	#define ALLOC_TABLE(X, FOR, ENDFOR) \
	X(int *a, 12 * sizeof(int)) \
	X(double **b, 34 * sizeof(double *)) \
	FOR(i, 34) \
		X(b[i], 56 * sizeof(double)) \
	ENDFOR \
	X(double *c, flag*78 * sizeof(double))

	void *mp = my_calloc(POOL_GET_SIZE(ALLOC_TABLE));
	POOL_DO_ALLOC(ALLOC_TABLE);
	...use a, b, c here...
	free(mp);

#endif

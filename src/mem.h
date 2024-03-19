#pragma once

#include <string.h>
#include <xmmintrin.h>

#define MEM_ALIGN 64

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

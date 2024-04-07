#include "mem.h"
#include <stdio.h>
#include "linalg.h"

struct mem_pool {
	void *start;
	void *next;  // what the next call to pool_alloc would return
	size_t remaining;  // number of free bytes
};

struct mem_pool *pool_new(const size_t total_mem)
{
	struct mem_pool *mp = my_calloc(sizeof(struct mem_pool));
	if (mp == NULL) return NULL;
	mp->start = my_calloc(total_mem);
	if (mp->start == NULL) {
		my_free(mp);
		return NULL;
	}
	mp->next = mp->start;
	mp->remaining = total_mem;
	return mp;
}

void *pool_alloc(struct mem_pool *mp, const size_t size)
{
	if (size == 0) return NULL;
	const size_t used = MEM_ALIGN_UP(size);
	if (used > mp->remaining) {
        printf("Out of memory!!!\n");
        return NULL;
    }
	mp->remaining -= used;
	void *ret = mp->next;
	mp->next = ((char*)mp->next) + used;
	return ret;
}

void pool_free(struct mem_pool *mp)
{
	my_free(mp->start);
	my_free(mp);
}

size_t mem_best_ld(const size_t N)
{
	// leading dimension should correspond to a multiple of 64 bytes
	const size_t ld_aligned = ((N + MEM_ALIGN_NUM - 1)/MEM_ALIGN_NUM)*MEM_ALIGN_NUM;
	const size_t ld_bytes = ld_aligned*sizeof(num);
	// but not be a large power of two
	if (ld_bytes >= 512 && !(ld_bytes & (ld_bytes - 1)))
		return ld_aligned + MEM_ALIGN_NUM;
	return ld_aligned;
}

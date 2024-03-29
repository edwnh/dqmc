#include "mem.h"
#include <stdio.h>

struct mem_pool {
	void *start;
	void *next;  // what the next call to pool_alloc would return
	size_t remaining;  // number of free bytes
};

struct mem_pool *pool_new(size_t total_mem)
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

void *pool_alloc(struct mem_pool *mp, size_t size)
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

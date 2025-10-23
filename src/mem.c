#include "mem.h"
#include <stdlib.h>
#include <string.h>

#define MEM_ALIGN_UP(size) ((((size) + MEM_ALIGN - 1)/MEM_ALIGN) * MEM_ALIGN)

void *my_calloc(const size_t size)
{
	const size_t size_aligned = MEM_ALIGN_UP(size);
	void *p = aligned_alloc(MEM_ALIGN, size_aligned);
	if (p != NULL) memset(p, 0, size_aligned);
	return p;
}

void *my_calloc_table(const struct alloc_entry *table, const size_t n_alloc)
{
	size_t total_size = 0;
	for (size_t i = 0; i < n_alloc; i++)
		total_size += MEM_ALIGN_UP(table[i].size);

	void *pool = my_calloc(total_size);
	if (pool == NULL) {
		// printf("failed to allocate %zu bytes\n", total_size);
		return NULL;
	}

	void *next = pool;
	for (size_t i = 0; i < n_alloc; i++) {
		if (table[i].size > 0)
			memcpy(table[i].ptr, &next, sizeof(next));
		// direct assignment can lead to issues in bench_linalg.c
		// related to strict aliasing rules. memcpy is a workaround.
		// *table[i].ptr = next;
		next = (char *)next + MEM_ALIGN_UP(table[i].size);
	}
	// printf("allocated %zu bytes\n", total_size);
	return pool;
}

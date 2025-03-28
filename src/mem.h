#pragma once
#include <stdlib.h>
#include <string.h>

#define MEM_ALIGN 64

struct alloc_entry {
	void **ptr;
	size_t size;
};

void *my_calloc(const size_t size);
void *my_calloc_table(const struct alloc_entry *table, const size_t n_alloc);

#define my_free(ptr) free((ptr))
#define my_copy(dest, src, N) memcpy((dest), (src), (N)*sizeof((src)[0]))

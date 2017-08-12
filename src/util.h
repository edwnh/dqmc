#pragma once

#include <string.h>
#include <mkl.h>

#define MEM_ALIGN 64
#define DBL_ALIGN (MEM_ALIGN/sizeof(double))

#define _aa(a) __assume_aligned((a), MEM_ALIGN);

#define my_malloc(size) _mm_malloc((size), MEM_ALIGN)

#define my_calloc(size) memset(my_malloc((size)), 0, (size))

#define my_free(ptr) _mm_free((ptr))

#define my_copy(dest, src, N) memcpy((dest), (src), (N)*sizeof((src)[0]))

// use these since fortran blas/lapack functions take pointers for arguments
#define cint(x) &(const int){(x)}
#define cdbl(x) &(const double){(x)}

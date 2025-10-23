#pragma once

#ifdef USE_CPLX
	#include <complex.h>
	typedef double complex num;
	#define RC(x) x##_cplx
#else
	typedef double num;
	#define RC(x) x##_real
#endif

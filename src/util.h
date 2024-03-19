#pragma once

#ifdef USE_CPLX
	#include <complex.h>
	typedef double complex num;
#else
	typedef double num;
#endif

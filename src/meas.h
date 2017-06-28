#ifndef _MEAS_H
#define _MEAS_H

#include "util.h"

void measure_eqlt(const struct params *const restrict p, const int sign,
		const double *const restrict Gu,
		const double *const restrict Gd,
		struct meas_eqlt *const restrict m);

void measure_uneqlt(const struct params *const restrict p, const int sign,
		const double *const restrict Gu,
		const double *const restrict Gd,
		struct meas_uneqlt *const restrict m);

#endif

#pragma once

#include "data.h"
#include "util.h"

void measure_eqlt(const struct params *const restrict p, const num phase,
		const num *const restrict gu,
		const num *const restrict gd,
		struct meas_eqlt *const restrict m);

void measure_uneqlt(const struct params *const restrict p, const num phase,
		const num *const Gu0t,
		const num *const Gutt,
		const num *const Gut0,
		const num *const Gd0t,
		const num *const Gdtt,
		const num *const Gdt0,
		struct meas_uneqlt *const restrict m);

/*
void measure_uneqlt_full(const struct params *const restrict p, const int sign,
		const double *const restrict Gu,
		const double *const restrict Gd,
		struct meas_uneqlt *const restrict m);
*/

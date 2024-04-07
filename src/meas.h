#pragma once

#include "data.h"
#include "linalg.h"

void measure_eqlt(const struct params *const p, const num phase,
		const int ld,
		const num *const gu,
		const num *const gd,
		struct meas_eqlt *const m);

void measure_uneqlt(const struct params *const p, const num phase,
		const int ld,
		const num *const Gu0t,
		const num *const Gutt,
		const num *const Gut0,
		const num *const Gd0t,
		const num *const Gdtt,
		const num *const Gdt0,
		struct meas_uneqlt *const m);

/*
void measure_uneqlt_full(const struct params *const p, const int sign,
		const double *const Gu,
		const double *const Gd,
		struct meas_uneqlt *const m);
*/

#pragma once

#include "linalg.h"
#include "structs.h"

void RC(measure_eqlt)(const struct RC(params) *const p, const num phase,
		const int ld,
		const num *const gu,
		const num *const gd,
		struct RC(meas_eqlt) *const m);

void RC(measure_uneqlt)(const struct RC(params) *const p, const num phase,
		const int ld,
		const num *const Gu0t,
		const num *const Gutt,
		const num *const Gut0,
		const num *const Gd0t,
		const num *const Gdtt,
		const num *const Gdt0,
		struct RC(meas_uneqlt) *const m);

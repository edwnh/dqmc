#pragma once

#ifdef PROFILE_ENABLE

#include "time_.h"

#define PROFILE_LIST \
	X(wall) \
	X(updates) \
	X(calcb) \
	X(multb) \
	X(recalc) \
	X(wrap) \
	X(half_wrap) \
	X(half_wrap_ue) \
	X(calc_ue) \
	X(meas_eq) \
	X(meas_uneq)

#define X(a) __profile_##a,
enum {
	PROFILE_LIST
	n_profile
};
#undef X

extern tick_t profile_time[n_profile];
extern int profile_count[n_profile];
#pragma omp threadprivate(profile_time, profile_count)

#define profile_begin(a) \
	const tick_t __##a##_start = time_wall()

#define profile_end(a) \
	do { \
		profile_time[__profile_##a] += time_wall() - __##a##_start; \
		profile_count[__profile_##a]++; \
	} while (0)

void profile_print(tick_t wall_time);

void profile_clear(void);

#else // ifndef PROFILE_ENABLE

#define profile_begin(a) ((void)0)
#define profile_end(a) ((void)0)
#define profile_print(a) ((void)0)
#define profile_clear() ((void)0)

#endif

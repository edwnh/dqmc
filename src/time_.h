#pragma once

#include <stdint.h>
#include <time.h>

typedef int64_t tick_t;

#define TICK_PER_SEC INT64_C(1000000000)
#define SEC_PER_TICK 1e-9
#define US_PER_TICK 1e-3

static inline tick_t time_wall(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * TICK_PER_SEC + t.tv_nsec;
}

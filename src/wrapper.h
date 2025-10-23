#pragma once

#include <stdio.h>
#include "time_.h"

extern FILE *log_f;

// returns -1 for failure, 0 for completion, 1 for partial completion
int dqmc_wrapper(const char *sim_file, const char *log_file,
		const tick_t save_interval, const tick_t max_time, const int bench);

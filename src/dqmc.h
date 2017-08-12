#pragma once

#include "time_.h"

// returns -1 for failure, 0 for completion, 1 for partial completion
int dqmc_wrapper(const char *sim_file, const char *log_file,
		const tick_t max_time);

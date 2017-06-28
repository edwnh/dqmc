#ifndef _DQMC_H
#define _DQMC_H

#include "time_.h"

int dqmc_wrapper(const char *sim_file, const char *log_file,
		const tick_t max_time);

#endif

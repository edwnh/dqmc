#ifndef _SIG_H
#define _SIG_H

#include <stdio.h>
#include "time_.h"

void sig_init(FILE *_log, const tick_t _wall_start, const tick_t _max_time);

int sig_check_state(const int sweep, const int n_sweep_warm, const int n_sweep);

#endif

#pragma once

#include "time_.h"

void sig_init(const tick_t _wall_start, const tick_t _save_interval, const tick_t _max_time);

int sig_check_state(const int sweep, const int n_sweep_warm, const int n_sweep);

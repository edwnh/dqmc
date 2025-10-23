#include "sig.h"
#include <signal.h>
#include <stdio.h>
#include "time_.h"
#include "wrapper.h"

static volatile sig_atomic_t progress_flag = 0;
static void progress(int signum) { progress_flag = signum; }

static volatile sig_atomic_t stop_flag = 0;
static void stop(int signum) { stop_flag = signum; }

static tick_t wall_start = 0;
static tick_t save_interval = 0;
static tick_t t_next_save = 0;
static tick_t max_time = 0;
static int first = 0;
static tick_t t_first = 0;

void sig_init(const tick_t _wall_start, const tick_t _save_interval, const tick_t _max_time)
{
	static int called = 0; // could be called multiple times
	if (called == 0) {
		called = 1;
		sigaction(SIGUSR1, &(const struct sigaction){.sa_handler = progress}, NULL);
		sigaction(SIGINT, &(const struct sigaction){.sa_handler = stop}, NULL);
		sigaction(SIGTERM, &(const struct sigaction){.sa_handler = stop}, NULL);
	}

	wall_start = _wall_start;
	save_interval = _save_interval;
	t_next_save = wall_start + save_interval;
	max_time = _max_time;
	first = 0;
	t_first = 0;
}

int sig_check_state(const int sweep, const int n_sweep_warm, const int n_sweep)
{
	const tick_t t_now = time_wall();

	if (save_interval > 0 && t_now >= t_next_save) {
		progress_flag = -1;
		t_next_save = t_now + save_interval;
	}

	if (max_time > 0 && t_now >= wall_start + max_time)
		stop_flag = -1;

	if (t_first == 0) {
		first = sweep;
		t_first = t_now;
	}

	if (stop_flag < 0)
		fprintf(log_f, "reached time limit, checkpointing\n");
	else if (stop_flag > 0)
		fprintf(log_f, "signal %d received, saving data\n", stop_flag);
	else if (progress_flag < 0)
		fprintf(log_f, "periodic checkpoint, saving data\n");
	else if (progress_flag > 0)
		fprintf(log_f, "signal %d received, saving data\n", progress_flag);

	if (stop_flag != 0 || progress_flag != 0) {
		const int warmed_up = (sweep >= n_sweep_warm);
		const double t_elapsed = (t_now - wall_start) * SEC_PER_TICK;
		const double t_done = (t_now - t_first) * SEC_PER_TICK;
		const int sweep_done = sweep - first;
		const int sweep_left = n_sweep - sweep;
		const double t_left = (t_done / sweep_done) * sweep_left;
		fprintf(log_f, "%d/%d sweeps completed (%s)\n",
			sweep,
			n_sweep,
			warmed_up ? "measuring" : "warming up");
		fprintf(log_f, "\telapsed: %.3f%c\n",
			t_elapsed < 3600 ? t_elapsed : t_elapsed/3600,
			t_elapsed < 3600 ? 's' : 'h');
		fprintf(log_f, "\tremaining%s: %.3f%c\n",
			(first < n_sweep_warm) ? " (ignoring measurement cost)" : "",
			t_left < 3600 ? t_left : t_left/3600,
			t_left < 3600 ? 's' : 'h');
		fflush(log_f);
	}

	if (sweep == n_sweep_warm) {
		first = sweep;
		t_first = t_now;
	}

	const int retval = (stop_flag != 0) ? 1 : (progress_flag != 0) ? 2 : 0;
	progress_flag = 0;
	return retval;
}

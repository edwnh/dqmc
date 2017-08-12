#include "sig.h"
#include <signal.h>
#include <stdio.h>
#include "time_.h"

static volatile sig_atomic_t progress_flag = 0;
static void progress(int signum) { progress_flag = 1; }

static volatile sig_atomic_t stop_flag = 0;
static void stop(int signum) { stop_flag = signum; }

static FILE *log = NULL;
static tick_t wall_start = 0;
static tick_t max_time = 0;
static int first = 0;
static tick_t t_first = 0;

void sig_init(FILE *_log, const tick_t _wall_start, const tick_t _max_time)
{
	static int called = 0; // could be called multiple times
	if (called == 0) {
		called = 1;
		sigaction(SIGUSR1, &(const struct sigaction){.sa_handler = progress}, NULL);
		sigaction(SIGINT, &(const struct sigaction){.sa_handler = stop}, NULL);
		sigaction(SIGTERM, &(const struct sigaction){.sa_handler = stop}, NULL);
	}

	log = _log;
	wall_start = _wall_start;
	max_time = _max_time;
	first = 0;
	t_first = 0;
}

int sig_check_state(const int sweep, const int n_sweep_warm, const int n_sweep)
{
	const tick_t t_now = time_wall();

	if (max_time > 0 && t_now >= wall_start + max_time)
		stop_flag = -1;

	if (t_first == 0) {
		first = sweep;
		t_first = t_now;
	}

	if (stop_flag != 0 || progress_flag != 0) {
		progress_flag = 0;
		const int warmed_up = (sweep >= n_sweep_warm);
		const double t_elapsed = (t_now - wall_start) * SEC_PER_TICK;
		const double t_done = (t_now - t_first) * SEC_PER_TICK;
		const int sweep_done = sweep - first;
		const int sweep_left = n_sweep - sweep;
		const double t_left = (t_done / sweep_done) * sweep_left;
		fprintf(log, "%d/%d sweeps completed (%s)\n",
			sweep,
			n_sweep,
			warmed_up ? "measuring" : "warming up");
		fprintf(log, "\telapsed: %.3f%c\n",
			t_elapsed < 3600 ? t_elapsed : t_elapsed/3600,
			t_elapsed < 3600 ? 's' : 'h');
		fprintf(log, "\tremaining%s: %.3f%c\n",
			(first < n_sweep_warm) ? " (ignoring measurement cost)" : "",
			t_left < 3600 ? t_left : t_left/3600,
			t_left < 3600 ? 's' : 'h');
		fflush(log);
	}

	if (sweep == n_sweep_warm) {
		first = sweep;
		t_first = t_now;
	}

	if (stop_flag < 0)
		fprintf(log, "reached time limit, checkpointing\n");
	else if (stop_flag > 0)
		fprintf(log, "signal %d received, checkpointing\n", stop_flag);

	return (stop_flag != 0);
}

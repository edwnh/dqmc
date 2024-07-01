#ifdef PROFILE_ENABLE

#include "prof.h"
#include <stdio.h>
#include <string.h>
#ifdef _OPENMP
	#include <omp.h>
#else
	static inline int omp_get_num_threads(void)
	{
		return 1;
	}
#endif
#include "dqmc.h"
#include "time_.h"

tick_t profile_time[n_profile] = {0};
int profile_count[n_profile] = {0};

void profile_print(tick_t wall_time)
{
	#define X(a) #a,
	const char *name[] = {
		PROFILE_LIST
		NULL
	};
	#undef X

	int n_threads;
	#pragma omp parallel
	{
		#pragma omp single
		{
			n_threads = omp_get_num_threads();
		}
	}

	// ordered for loop through all threads
	#pragma omp parallel for ordered schedule(static, 1)
	for (int thread = 0; thread < n_threads; thread++) {
		#pragma omp ordered
		{
		// insertion sort for time
		int i_sorted[n_profile] = {0};
		for (int i = 1; i < n_profile; i++) {
			int j;
			for (j = i; j > 0 && profile_time[i_sorted[j-1]] < profile_time[i]; j--)
				i_sorted[j] = i_sorted[j-1];
			i_sorted[j] = i;
		}

		fprintf(log_f, "thread_%d/%d_______|_%% of all_|___total (s)_|___us per call_|___# calls\n",
			thread + 1, n_threads);
		for (int j = 0; j < n_profile; j++) {
			const int i = i_sorted[j];
			if (profile_count[i] == 0) continue;
			fprintf(log_f, "%16s |%9.3f |%12.3f |%14.3f |%10d\n",
				name[i],
				100.0 * profile_time[i] / wall_time,
				profile_time[i] * SEC_PER_TICK,
				US_PER_TICK * profile_time[i] / profile_count[i],
				profile_count[i]);
		}
		fprintf(log_f, "---------------------------------------------------------------------\n");
		}
	}
}

void profile_clear(void)
{
	#pragma omp parallel
	{
	memset(profile_time, 0, n_profile * sizeof(tick_t));
	memset(profile_count, 0, n_profile * sizeof(int));
	}
}

#endif

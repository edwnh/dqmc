#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#ifdef _OPENMP
	#include <omp.h>
#else
	static inline void omp_set_num_threads(int n)
	{
		(void)n;
	}
#endif
#include "dqmc.h"
#include "time_.h"

static void usage(const char *name)
{
	printf("usage: %s [-b] [-l log_file.log] [-t max_time] sim_file.h5\n", name);
}

int main(int argc, char **argv)
{
	omp_set_num_threads(2);
	char *log_file = NULL;
	char *max_time = "0";
	int bench = 0;

	int c;
	while ((c = getopt(argc, argv, "bl:t:")) != -1)
		switch (c) {
		case 'b':
			bench = 1;
			break;
		case 'l':
			log_file = optarg;
			break;
		case 't':
			max_time = optarg;
			break;
		default:
			usage(argv[0]);
			return 0;
		}

	if (optind >= argc) {
		usage(argv[0]);
		return 0;
	}

	int status = dqmc_wrapper(argv[optind], log_file,
	                          atoi(max_time) * TICK_PER_SEC, bench);

	if (status < 0) {
		fprintf(stderr, "dqmc_wrapper() failed: %d", status);
		return 1;
	}
	return 0;
}

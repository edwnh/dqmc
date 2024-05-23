#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
	printf("usage: %s [-b] [-c] [-l file.log] [-s interval] [-t max_time] sim_file.h5\n"
	       "options:\n"
	       "  -b           set benchmarking mode. no data is saved.\n"
	       "  -c           print cpu model from /proc/cpuinfo.\n"
	       "  -l file.log  output to log file instead of stdout.\n"
	       "  -s interval  time (seconds) between automatic checkpoints.\n"
	       "  -t max_time  maximum run time (seconds).\n", name);
}

static void print_cpu_model(void)
{
	FILE *cpuinfo = fopen("/proc/cpuinfo", "r");
	if (!cpuinfo) {
		printf("couldn't open /proc/cpuinfo\n");
		return;
	}

	char *line = NULL;
	size_t len = 0;
	while (getline(&line, &len, cpuinfo) != -1) {
		if (strncmp(line, "model name", 10) == 0) {
			printf("cpu %s", line);
			break;
		}
	}
	free(line);
	fclose(cpuinfo);
}

int main(int argc, char **argv)
{
	omp_set_num_threads(2);
	char *log_file = NULL;
	char *save_interval = "0";
	char *max_time = "0";
	int bench = 0;

	int c;
	while ((c = getopt(argc, argv, "bcl:s:t:")) != -1)
		switch (c) {
		case 'b':
			bench = 1;
			break;
		case 'c':
			print_cpu_model();
			break;
		case 'l':
			log_file = optarg;
			break;
		case 's':
			save_interval = optarg;
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
	                          atoi(save_interval) * TICK_PER_SEC,
	                          atoi(max_time) * TICK_PER_SEC,
	                          bench);

	if (status < 0) {
		fprintf(stderr, "dqmc_wrapper() failed: %d", status);
		return 1;
	}
	return 0;
}

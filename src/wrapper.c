#include "wrapper.h"
#include <stdio.h>
#include <string.h>
#ifdef __APPLE__
	#include <sys/sysctl.h>
#else
	#include <stdlib.h>
#endif
#include <hdf5.h>
#include <hdf5_hl.h>
#include "prof.h"
#include "sig.h"
#include "time_.h"

#define RC(x) x##_real
#include "rc/data_decl.h"
#include "rc/dqmc_decl.h"
#undef RC

#define RC(x) x##_cplx
#include "rc/data_decl.h"
#include "rc/dqmc_decl.h"
#undef RC

FILE *log_f;

#define return_if(cond, val, ...) \
	do {if (cond) {fprintf(stderr, __VA_ARGS__); return (val);}} while (0)

static int is_cplx_file(const char *file)
{
	const hid_t file_id = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file_id < 0) {
		fprintf(stderr, "H5Fopen() failed: %ld\n", file_id);
		return -1;
	}
	int is_cplx = -1;
	herr_t status = H5LTread_dataset_int(file_id, "/params/is_cplx", &is_cplx);
	if (status < 0) {
		fprintf(stderr, "H5LTread_dataset() failed for %s: %d\n", "/params/is_cplx", status);
	}
	status = H5Fclose(file_id);
	if (status < 0) {
		fprintf(stderr, "H5Fclose() failed: %d\n", status);
		return -1;
	}
	return is_cplx;
}

static void print_cpu_model(void)
{
#ifdef __APPLE__
	char str[256];
	size_t size = sizeof(str);

	if (sysctlbyname("machdep.cpu.brand_string", str, &size, NULL, 0) == 0) {
		fprintf(log_f, "cpu: %s\n", str);
	} else {
		fprintf(log_f, "couldn't get CPU information\n");
	}
#else
	FILE *cpuinfo = fopen("/proc/cpuinfo", "r");
	if (!cpuinfo) {
		fprintf(log_f, "couldn't open /proc/cpuinfo\n");
		return;
	}

	char *line = NULL;
	size_t len = 0;
	while (getline(&line, &len, cpuinfo) != -1) {
		if (strncmp(line, "model name", 10) == 0) {
			fprintf(log_f, "cpu %s", line);
			break;
		}
	}
	free(line);
	fclose(cpuinfo);
#endif
}

int dqmc_wrapper(const char *sim_file, const char *log_file,
		const tick_t save_interval, const tick_t max_time, const int bench)
{
	const tick_t wall_start = time_wall();
	profile_clear();

	int status = 0;

	// open log file
	log_f = (log_file != NULL) ? fopen(log_file, "a") : stdout;
	if (log_f == NULL) {
		fprintf(stderr, "fopen() failed to open: %s\n", log_file);
		return -1;
	}

	fprintf(log_f, "commit id %s\n", GIT_ID);
	fprintf(log_f, "compiled on %s %s\n", __DATE__, __TIME__);
	print_cpu_model();

	// initialize signal handling
	sig_init(wall_start, save_interval, max_time);

	// open and read simulation file
	fprintf(log_f, "opening %s\n", sim_file);

	// only one of these will be used
	struct sim_data_real *sim_real = NULL;
	struct sim_data_cplx *sim_cplx = NULL;

	const int is_cplx = is_cplx_file(sim_file);
	if (is_cplx) {
		sim_cplx = sim_data_read_alloc_cplx(sim_file);
	} else {
		sim_real = sim_data_read_alloc_real(sim_file);
	}
	if ((sim_real == NULL) && (sim_cplx == NULL)) {
		fprintf(stderr, "read_file() failed: %d\n", status);
		status = -1;
		goto cleanup;
	}

	// run dqmc
	const int retcode = is_cplx ? dqmc_cplx(sim_cplx) : dqmc_real(sim_real);

	// save to simulation file (if not in benchmarking mode)
	if (!bench) {
		fprintf(log_f, "saving data\n");
		status = is_cplx ? sim_data_save_cplx(sim_cplx) : sim_data_save_real(sim_real);
		if (status < 0) {
			fprintf(stderr, "save_file() failed: %d\n", status);
			status = -1;
			goto cleanup;
		}
	} else {
		fprintf(log_f, "benchmark mode enabled; not saving data\n");
	}

	status = retcode;

cleanup:
	if (is_cplx) {
		sim_data_free_cplx(sim_cplx);
	} else {
		sim_data_free_real(sim_real);
	}

	const tick_t wall_time = time_wall() - wall_start;
	fprintf(log_f, "wall time: %.3f\n", wall_time * SEC_PER_TICK);
	profile_print(wall_time);

	if (log_f != stdout)
		fclose(log_f);

	return status;
}

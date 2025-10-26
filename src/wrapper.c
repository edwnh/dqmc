#include "wrapper.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifdef __APPLE__
	#include <sys/sysctl.h>
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

static int is_cplx_file(const char *file)
{
	const hid_t file_id = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file_id < 0) {
		fprintf(stderr, "H5Fopen() failed: %ld\n", file_id);
		return -1;
	}

	hid_t dset_id = H5Dopen2(file_id, "/params/exp_Ku", H5P_DEFAULT);
	if (dset_id < 0) {
		fprintf(stderr, "H5Dopen2() failed: %ld\n", dset_id);
		H5Fclose(file_id);
		return -1;
	}

	hid_t type_id = H5Dget_type(dset_id);
	if (type_id < 0) {
		fprintf(stderr, "H5Dget_type() failed: %ld\n", type_id);
		H5Dclose(dset_id);
		H5Fclose(file_id);
		return -1;
	}

	const H5T_class_t type_class = H5Tget_class(type_id);
	int is_cplx = (type_class == H5T_COMPOUND) ? 1 : (type_class == H5T_FLOAT) ? 0 : -1;
	H5Tclose(type_id);
	H5Dclose(dset_id);
	H5Fclose(file_id);
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
		fprintf(log_f, "couldn't get cpu information\n");
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
		fprintf(stderr, "fopen() failed to open: %s. using stdout", log_file);
		log_f = stdout;
	}

	fprintf(log_f, "commit id %s compiled on %s %s\n", GIT_ID, __DATE__, __TIME__);
	print_cpu_model();

	// initialize signal handling
	sig_init(wall_start, save_interval, max_time);

	// open and read simulation file
	fprintf(log_f, "opening %s\n", sim_file);

	// only one of these will be used
	struct sim_data_real *sim_real = NULL;
	struct sim_data_cplx *sim_cplx = NULL;

	const int is_cplx = is_cplx_file(sim_file);
	if (is_cplx == 1) {
		sim_cplx = sim_data_read_alloc_cplx(sim_file);
	} else if (is_cplx == 0) {
		sim_real = sim_data_read_alloc_real(sim_file);
	} else {
		fprintf(stderr, "is_cplx_file() failed\n");
		status = -1;
		goto cleanup;
	}
	if ((sim_real == NULL) && (sim_cplx == NULL)) {
		fprintf(stderr, "sim_data_read_alloc() failed: %d\n", status);
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
			fprintf(stderr, "sim_data_save() failed: %d\n", status);
			status = -1;
			goto cleanup;
		} else {
			status = retcode;
		}
	} else {
		fprintf(log_f, "benchmark mode enabled; not saving data\n");
	}

cleanup:
	if (is_cplx == 1) {
		sim_data_free_cplx(sim_cplx);
	} else if (is_cplx == 0) {
		sim_data_free_real(sim_real);
	}

	const tick_t wall_time = time_wall() - wall_start;
	fprintf(log_f, "wall time: %.3f\n", wall_time * SEC_PER_TICK);
	profile_print(wall_time);

	if (log_f != stdout)
		fclose(log_f);

	return status;
}

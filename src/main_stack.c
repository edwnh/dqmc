#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>
#include "dqmc.h"
#include "util.h"

#define USE_HARD_LINK

#define my_printf(...) do { \
	printf("%16s %6d: ", hostname, pid); \
	printf(__VA_ARGS__); \
	fflush(stdout); \
} while (0)

static char hostname[65];
static int pid;

static void usage(const char *name)
{
	my_printf("usage: %s [-t max_time] stack_file\n", name);
}

// sleep a number of seconds between min and max (assumes both > 0)
// so that processes don't repeatedly try to do something simultaneously
static void sleep_rand(double min, double max)
{
	const double t = min + (max - min)*(double)rand()/RAND_MAX;
	const struct timespec ts = {(time_t)t, (long)(1e9*(t - (int)t))};
	nanosleep(&ts, NULL);
}

static int lock_file(const char *file, const int retry)
{
	const size_t len_file = strlen(file);
	char *lfile = my_calloc(len_file + 2);
	memcpy(lfile, file, len_file);
	lfile[len_file] = '~';

	struct timespec lock_mtime = {0};
	int cycles_same_mtime = 0;
	while (1) {
#ifdef USE_HARD_LINK
		if (link(file, lfile) == 0) {
#else
		if (symlink(file, lfile) == 0) { // successfully locked
#endif
			my_free(lfile);
			return 0;
		}

		if (!retry) return 1;
		// this method to automatically release zombie locks allows for
		// a possible race condition due to the gap between stat and
		// releasing the zombie lock

		// s = stat, z = zombie lock released, c = create,
		// r = regular lock release, e = editing file
		// process 1: ...s    s    s  z
		// process 2:   ...s    s    s  z
		// process 3:                  c eeeeeee r
		// process 4:                       c eeeeeee r

		// these incidents can be logged because r would fail, but
		// failure of r doesn't necessarily imply simultaneous editing

		// check mtime
		struct stat statbuf = {0};
		if (stat(lfile, &statbuf) != 0) // file gone
			cycles_same_mtime = 0;
		else if (statbuf.st_mtim.tv_sec == lock_mtime.tv_sec &&
				statbuf.st_mtim.tv_nsec == lock_mtime.tv_nsec)
			cycles_same_mtime++;
		else {
			cycles_same_mtime = 0;
			lock_mtime = statbuf.st_mtim;
		}

		// if mtime stays the same for 10 cycles, assume the locking
		// process died and force unlock the file.
		if (cycles_same_mtime >= 10) {
			remove(lfile);
			my_printf("warning: zombie lock released\n");
			sleep_rand(1.0, 2.0);
		}

		// wait 1-3s before looping again
		sleep_rand(1.0, 3.0);
	}
}

static void unlock_file(const char *file)
{
	const size_t len_file = strlen(file);
	char *lfile = my_calloc(len_file + 2);
	memcpy(lfile, file, len_file);
	lfile[len_file] = '~';

	if (remove(lfile) != 0)
		my_printf("warning: lock release failed (already removed?)\n");

	my_free(lfile);
}

static int pop_stack(const char *file, int max_len, char *line)
{
	int ret = 0;
	int len_line = 0;
	memset(line, 0, max_len);

	lock_file(file, 1);

	const int fd = open(file, O_RDWR);
	if (fd == -1) {
		unlock_file(file);
		my_printf("error: open() failed in pop_stack\n");
		return -1;
	}

	#define BUF_SZ 128

	off_t offset = lseek(fd, 0, SEEK_END);
	if (offset == -1) {
		my_printf("error: lseek() failed in pop_stack()\n");
		ret = -1;
		goto end;
	}

	char buf[BUF_SZ + 1];

	int start = BUF_SZ, end = BUF_SZ;

	while (offset > 0) {
		offset -= BUF_SZ;
		const int ofs = offset < 0 ? 0 : offset;
		const int actual_ofs = lseek(fd, ofs, SEEK_SET);
		if (actual_ofs == -1) {
			my_printf("error: lseek() failed in pop_stack()\n");
			ret = -1;
			goto end;
		} else if (actual_ofs != ofs) {
			my_printf("error: actual_ofs=%d, requested=%d\n",
			           actual_ofs, ofs);
			ret = -1;
			goto end;
		}

		memset(buf, 0, sizeof(buf));
		if (end != BUF_SZ)
			end = offset < 0 ? BUF_SZ + offset - 1 : BUF_SZ - 1;
		if (read(fd, buf, end == BUF_SZ ? BUF_SZ : end + 1) == -1) {
			my_printf("error: read() failed in pop_stack()\n");
			ret = -1;
			goto end;
		}

		if (end == BUF_SZ)
			for (; end >= 0; end--)
				if (buf[end] != 0 && buf[end] != '\n')
					break;

		if (end == -1) {
			end = BUF_SZ;
			continue;
		}

		for (start = end; start >= 0; start--)
			if (buf[start] == '\n') {
				start++;
				break;
			}
		if (start == -1) start = 0;

		const int new_chars = end - start + 1;
		if (len_line + new_chars > max_len) {
			my_printf("error: last line length > %d\n", max_len);
			ret = -1;
			goto end;
		}

		if (new_chars > 0) {
			if (len_line > 0)
				memmove(line + new_chars, line, len_line);
			memcpy(line, buf + start, new_chars);
			len_line += new_chars;
		}

		if (start != 0) // start of last line found
			break;
	}

	if (len_line == 0) {
		ret = 1;
	} else if (ftruncate(fd, (offset < 0 ? 0 : offset) + start) == -1) {
		my_printf("error: ftruncate failed in pop_stack()\n");
		ret = -1;
	}

end:
	if (close(fd) == -1)
		my_printf("error: close() failed in pop_stack()\n");
	unlock_file(file);
	return ret;
}

static void push_stack(const char *file, const char *line)
{
	// add newline here instead of using fputc
	const size_t len_line = strlen(line);
	char *line_nl = my_calloc(len_line + 2);
	memcpy(line_nl, line, len_line);
	line_nl[len_line] = '\n';
	char *backup = NULL;

	int status = 0;

	if (lock_file(file, 0) != 0) { // if locking fails on first try
		// save line to backup in case process gets killed
		const size_t len_file = strlen(file);
		backup = my_calloc(len_file + 32);
		snprintf(backup, len_file + 32, "%s_%.20s_%d", file, hostname, pid);

		const int bfd = open(backup, O_CREAT | O_WRONLY | O_APPEND);
		if (bfd == -1)
			status |= 1;
		else {
			if (write(bfd, line_nl, len_line + 1) != (ssize_t)len_line + 1)
				status |= 2;
			if (close(bfd) == -1)
				status |= 4;
		}

		// now try locking with retrying enabled
		lock_file(file, 1);
	}

	const int fd = open(file, O_WRONLY | O_APPEND);
	if (fd == -1)
		status |= 1;
	else {
		if (write(fd, line_nl, len_line + 1) != (ssize_t)len_line + 1)
			status |= 2;
		if (close(fd) == -1)
			status |= 4;
	}

	unlock_file(file);

	if (backup != NULL) {
		if (status == 0)
			remove(backup);
		my_free(backup);
	}

	my_free(line_nl);

	if (status & 1)
		my_printf("error: open() failed in push_stack()\n");
	if (status & 2)
		my_printf("error: write() failed or incomplete in push_stack()\n");
	if (status & 4)
		my_printf("error: close() failed in push_stack()\n");
}

int main(int argc, char **argv)
{
	const tick_t t_start = time_wall();
	omp_set_num_threads(2);
	gethostname(hostname, 64);
	pid = getpid();

	char *str_max_time = NULL;
	int c;
	while ((c = getopt(argc, argv, "t:")) != -1)
		switch (c) {
		case 't':
			str_max_time = optarg;
			break;
		default:
			usage(argv[0]);
			return 0;
		}

	if (argc - optind <= 0) {
		usage(argv[0]);
		return 0;
	}

	srand((unsigned int)pid);
	sleep_rand(0.0, 4.0);

	const char *stack_file = argv[optind];
	const int max_time = (str_max_time == NULL) ? 0 : atoi(str_max_time);
	const tick_t t_stop = t_start + max_time * TICK_PER_SEC;

	#define MAX_LEN 512
	char sim_file[MAX_LEN + 1] = {0};
	char log_file[MAX_LEN + 5] = {0};

	while (1) {
		int status = pop_stack(stack_file, MAX_LEN, sim_file);
		if (status == 1 || status < 0) { // empty or pop_stack failed
			my_printf("pop_stack() returned %d; idling\n", status);
			break;
		}

		const size_t len_sim_file = strlen(sim_file);
		memcpy(log_file, sim_file, len_sim_file);
		memcpy(log_file + len_sim_file, ".log", 5);

		tick_t t_remain;
		if (max_time > 0) {
			t_remain = t_stop - time_wall();
			if (t_remain <= 0) {
				push_stack(stack_file, sim_file);
				break;
			}
		} else
			t_remain = 0;

		my_printf("starting: %s\n", sim_file);
		status = dqmc_wrapper(sim_file, log_file, t_remain, 0);

		if (status > 0) {
			my_printf("checkpointed: %s\n", sim_file);
			push_stack(stack_file, sim_file);
			// checkpoint would only happen if signal received or
			// time limit reached, so break here
			break;
		} else if (status == 0)
			my_printf("completed: %s\n", sim_file);
		else
			my_printf("dqmc_wrapper() failed: %d, %s\n",
			           status, sim_file);
	}

	return 0;
}

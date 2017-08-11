#include "dqmc.h"
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <mkl.h>
#include "eq_g.h"
#include "io.h"
#include "meas.h"
#include "prof.h"
#include "time_.h"
#include "updates.h"
#include "util.h"

// uncomment below to check recalculated g against wrapped g
// #define CHECK_G_WRP

// uncomment below check recalculated g against using QR for every 2nd multiply
// #define CHECK_G_ACC

static volatile sig_atomic_t progress_flag = 0;
static void progress(int signum) { progress_flag = 1; }

static volatile sig_atomic_t stop_flag = 0;
static void stop(int signum) { stop_flag = signum; }

static void print_progress(FILE *log, const tick_t wall_start,
		const tick_t t_first, const tick_t t_now, const int first,
		const int sweep, const int n_sweep_warm, const int n_sweep)
{
	const int warmed_up = (sweep >= n_sweep_warm);
	const double t_elapsed = (t_now - wall_start)/1e9;
	const double t_done = (t_now - t_first)/1e9;
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

static void dqmc(FILE *log, const tick_t wall_start, const tick_t max_time,
		const struct params *p, struct state *s,
		struct meas_eqlt *m_eq, struct meas_uneqlt *m_ue)
{
	const int N = p->N;
	const int n_matmul = p->n_matmul;
	const int n_delay = p->n_delay;
	const int F = p->F;
	const double *const restrict exp_K = p->exp_K; _aa(exp_K);
	const double *const restrict inv_exp_K = p->inv_exp_K; _aa(inv_exp_K);
	const double *const restrict exp_lambda = p->exp_lambda; _aa(exp_lambda);
	const double *const restrict del = p->del; _aa(del);
	uint64_t *const restrict rng = s->rng; _aa(rng);
	int *const restrict hs = s->hs;

	// stride for time index in arrays of B or C matrices
	const int stride = DBL_ALIGN * ((N*N + DBL_ALIGN - 1) / DBL_ALIGN);
	__assume(stride % DBL_ALIGN == 0);

	double *const restrict Bu = my_calloc(stride*p->L * sizeof(double)); _aa(Bu);
	double *const restrict Bd = my_calloc(stride*p->L * sizeof(double)); _aa(Bd);
	double *const restrict Cu = my_calloc(stride*F * sizeof(double)); _aa(Cu);
	double *const restrict Cd = my_calloc(stride*F * sizeof(double)); _aa(Cd);
	double *const restrict gu = my_calloc(N*N * sizeof(double)); _aa(gu);
	double *const restrict gd = my_calloc(N*N * sizeof(double)); _aa(gd);
	#ifdef CHECK_G_WRP
	double *const restrict guwrp = my_calloc(N*N * sizeof(double)); _aa(guwrp);
	double *const restrict gdwrp = my_calloc(N*N * sizeof(double)); _aa(gdwrp);
	#endif
	#ifdef CHECK_G_ACC
	double *const restrict guacc = my_calloc(N*N * sizeof(double)); _aa(guwrp);
	double *const restrict gdacc = my_calloc(N*N * sizeof(double)); _aa(gdwrp);
	#endif
	int sign = 0;
	int *const site_order = my_calloc(N * sizeof(double)); _aa(site_order);

	// work arrays for calc_eq_g and stuff. two sets for easy 2x parallelization
	const int lwork = get_lwork_eq_g(N);

	double *const restrict worku = my_calloc(lwork * sizeof(double)); _aa(worku);
	double *const restrict tmpNN1u = my_calloc(N*N * sizeof(double)); _aa(tmpNN1u);
	double *const restrict tmpNN2u = my_calloc(N*N * sizeof(double)); _aa(tmpNN2u);
	double *const restrict tmpN1u = my_calloc(N * sizeof(double)); _aa(tmpN1u);
	double *const restrict tmpN2u = my_calloc(N * sizeof(double)); _aa(tmpN2u);
	double *const restrict tmpN3u = my_calloc(N * sizeof(double)); _aa(tmpN3u);
	int *const restrict pvtu = my_calloc(N * sizeof(int)); _aa(pvtu);

	double *const restrict workd = my_calloc(lwork * sizeof(double)); _aa(workd);
	double *const restrict tmpNN1d = my_calloc(N*N * sizeof(double)); _aa(tmpNN1d);
	double *const restrict tmpNN2d = my_calloc(N*N * sizeof(double)); _aa(tmpNN2d);
	double *const restrict tmpN1d = my_calloc(N * sizeof(double)); _aa(tmpN1d);
	double *const restrict tmpN2d = my_calloc(N * sizeof(double)); _aa(tmpN2d);
	double *const restrict tmpN3d = my_calloc(N * sizeof(double)); _aa(tmpN3d);
	int *const restrict pvtd = my_calloc(N * sizeof(int)); _aa(pvtd);

	// who needs function calls
	#define matmul(C, A, B) do { \
		dgemm("N", "N", &N, &N, &N, cdbl(1.0), (A), &N, (B), &N, cdbl(0.0), (C), &N); \
	} while (0);

	#define calcBu(B, l) do { \
		for (int j = 0; j < N; j++) { \
			const double el = exp_lambda[j + N*hs[j + N*(l)]]; \
			for (int i = 0; i < N; i++) \
				(B)[i + N*j] = exp_K[i + N*j] * el; \
		} \
	} while (0);

	#define calcBd(B, l) do { \
		for (int j = 0; j < N; j++) { \
			const double el = exp_lambda[j + N*!hs[j + N*(l)]]; \
			for (int i = 0; i < N; i++) \
				(B)[i + N*j] = exp_K[i + N*j] * el; \
		} \
	} while (0);

	#define calcinvBu(invB, l) do { \
		for (int i = 0; i < N; i++) { \
			const double el = exp_lambda[i + N*!hs[i + N*(l)]]; \
			for (int j = 0; j < N; j++) \
				(invB)[i + N*j] = el * inv_exp_K[i + N*j]; \
		} \
	} while (0);

	#define calcinvBd(invB, l) do { \
		for (int i = 0; i < N; i++) { \
			const double el = exp_lambda[i + N*hs[i + N*(l)]]; \
			for (int j = 0; j < N; j++) \
				(invB)[i + N*j] = el * inv_exp_K[i + N*j]; \
		} \
	} while (0);

	{
	int signu, signd;
	#pragma omp parallel sections
	{
	#pragma omp section
	{
	for (int f = 0; f < F; f++) {
		calcBu(Bu + stride*f*n_matmul, f*n_matmul);
		my_copy(Cu + stride*f, Bu + stride*f*n_matmul, N*N);
		for (int l = f*n_matmul + 1 ; l < (f + 1) * n_matmul; l++) {
			calcBu(Bu + stride*l, l);
			my_copy(tmpNN1u, Cu + stride*f, N*N);
			matmul(Cu + stride*f, Bu + stride*l, tmpNN1u);
		}
	}
	signu = calc_eq_g(0, N, stride, F, Cu, gu, tmpNN1u, tmpNN2u,
		          tmpN1u, tmpN2u, tmpN3u, pvtu, worku, lwork);
	}
	#pragma omp section
	{
	for (int f = 0; f < F; f++) {
		calcBd(Bd + stride*f*n_matmul, f*n_matmul);
		my_copy(Cd + stride*f, Bd + stride*f*n_matmul, N*N);
		for (int l = f*n_matmul + 1 ; l < (f + 1) * n_matmul; l++) {
			calcBd(Bd + stride*l, l);
			my_copy(tmpNN1d, Cd + stride*f, N*N);
			matmul(Cd + stride*f, Bd + stride*l, tmpNN1d);
		}
	}
	signd = calc_eq_g(0, N, stride, F, Cd, gd, tmpNN1d, tmpNN2d,
		          tmpN1d, tmpN2d, tmpN3d, pvtd, workd, lwork);
	}
	}
	sign = signu*signd;
	}

	// these are used to estimate remaining time in print_progress
	int first = s->sweep;
	tick_t t_first = time_wall();

	for (; s->sweep < p->n_sweep; s->sweep++) {
		const int warmed_up = (s->sweep >= p->n_sweep_warm);
		const int enabled_eqlt = warmed_up && (p->period_eqlt > 0);
		const int enabled_uneqlt = warmed_up && (p->period_uneqlt > 0);

		const tick_t t_now = time_wall();
		if (max_time > 0 && t_now >= wall_start + max_time)
			stop_flag = -1;

		// signal handling
		if (stop_flag != 0 || progress_flag != 0) {
			progress_flag = 0;
			print_progress(log, wall_start, t_first, t_now, first,
			               s->sweep, p->n_sweep_warm, p->n_sweep);
		}
		if (s->sweep == p->n_sweep_warm) {
			first = s->sweep;
			t_first = t_now;
		}
		if (stop_flag != 0) {
			if (stop_flag < 0)
				fprintf(log, "reached time limit, checkpointing\n");
			else
				fprintf(log, "signal %d received, checkpointing\n", stop_flag);
			break;
		}

		for (int l = 0; l < p->L; l++) {
			profile_begin(updates);
			sign *= update_delayed(N, n_delay, del, rng, hs + N*l, gu, gd, site_order,
			                       tmpNN1u, tmpNN2u, tmpN1u, tmpNN1d, tmpNN2d, tmpN1d);
			profile_end(updates);

			const int f = l / n_matmul;
			const int recalc = ((l + 1) % n_matmul == 0);
			int signu, signd;
			#pragma omp parallel sections
			{
			#pragma omp section
			{
			profile_begin(multb);
			calcBu(Bu + stride*l, l);
			if (l % n_matmul == 0) {
				my_copy(Cu + stride*f, Bu + stride*l, N*N);
			} else {
				my_copy(tmpNN1u, Cu + stride*f, N*N);
				matmul(Cu + stride*f, Bu + stride*l, tmpNN1u);
			}
			profile_end(multb);
			if (recalc) {
				profile_begin(recalc);
				#ifdef CHECK_G_WRP
				calcinvBu(tmpNN1u, l);
				matmul(tmpNN2u, gu, tmpNN1u);
				matmul(guwrp, Bu + stride*l, tmpNN2u);
				#endif
				#ifdef CHECK_G_ACC
				calc_eq_g((l + 1) % p->L, N, stride, p->L, Bu, guacc,
				          tmpNN1u, tmpNN2u, tmpN1u, tmpN2u,
				          tmpN3u, pvtu, worku, lwork);
				#endif
				signu = calc_eq_g((f + 1) % F, N, stride, F, Cu, gu,
				                  tmpNN1u, tmpNN2u, tmpN1u, tmpN2u,
				                  tmpN3u, pvtu, worku, lwork);
				profile_end(recalc);
			} else {
				profile_begin(wrap);
				calcinvBu(tmpNN1u, l);
				matmul(tmpNN2u, gu, tmpNN1u);
				matmul(gu, Bu + stride*l, tmpNN2u);
				profile_end(wrap);
			}
			}
			#pragma omp section
			{
			profile_begin(multb);
			calcBd(Bd + stride*l, l)
			if (l % n_matmul == 0) {
				my_copy(Cd + stride*f, Bd + stride*l, N*N);
			} else {
				my_copy(tmpNN1d, Cd + stride*f, N*N);
				matmul(Cd + stride*f, Bd + stride*l, tmpNN1d);
			}
			profile_end(multb);
			if (recalc) {
				profile_begin(recalc);
				#ifdef CHECK_G_WRP
				calcinvBd(tmpNN1d, l);
				matmul(tmpNN2d, gd, tmpNN1d);
				matmul(gdwrp, Bd + stride*l, tmpNN2d);
				#endif
				#ifdef CHECK_G_ACC
				calc_eq_g((l + 1) % p->L, N, stride, p->L, Bd, gdacc,
				          tmpNN1d, tmpNN2d, tmpN1d, tmpN2d,
				          tmpN3d, pvtd, workd, lwork);
				#endif
				signd = calc_eq_g((f + 1) % F, N, stride, F, Cd, gd,
				                  tmpNN1d, tmpNN2d, tmpN1d, tmpN2d,
				                  tmpN3d, pvtd, workd, lwork);
				profile_end(recalc);
			} else {
				profile_begin(wrap);
				calcinvBd(tmpNN1d, l);
				matmul(tmpNN2d, gd, tmpNN1d);
				matmul(gd, Bd + stride*l, tmpNN2d);
				profile_end(wrap);
			}
			}
			}

			#define matdiff(A, B) do { \
				double max = 0.0, avg = 0.0; \
				for (int i = 0; i < N*N; i++) { \
					const double diff = fabs((A)[i] - (B)[i]); \
					if (diff > max) max = diff; \
					avg += diff; \
				} \
				avg /= N*N; \
				printf(#A " - " #B ":\tmax %.3e\tavg %.3e\n", max, avg); \
			} while (0);

			#ifdef CHECK_G_WRP
			if (recalc) {
				matdiff(gu, guwrp);
				matdiff(gd, gdwrp);
			}
			#endif
			#ifdef CHECK_G_ACC
			if (recalc) {
				matdiff(gu, guacc);
				matdiff(gd, gdacc);
			}
			#endif
			#if defined(CHECK_G_WRP) && defined(CHECK_G_ACC)
			if (recalc) {
				matdiff(guwrp, guacc);
				matdiff(gdwrp, gdacc);
			}
			#endif

			#undef matdiff

			if (recalc) sign = signu*signd;

			if (enabled_eqlt && (l + 1) % p->period_eqlt == 0) {
				profile_begin(meas_eq);
				measure_eqlt(p, sign, gu, gd, m_eq);
				profile_end(meas_eq);
			}
		}

		if (enabled_uneqlt && s->sweep % p->period_uneqlt == 0) {
			profile_begin(meas_uneq);
			measure_uneqlt(p, sign, gu, gd, m_ue);
			profile_end(meas_uneq);
		}
	}
	#undef calcinvBd
	#undef calcinvBu
	#undef calcBd
	#undef calcBu
	#undef matmul

	if (stop_flag == 0) // if all sweeps done
		fprintf(log, "%d/%d sweeps completed\n", s->sweep, p->n_sweep);

	my_free(pvtd);
	my_free(tmpN3d);
	my_free(tmpN2d);
	my_free(tmpN1d);
	my_free(tmpNN2d);
	my_free(tmpNN1d);
	my_free(workd);
	my_free(pvtu);
	my_free(tmpN3u);
	my_free(tmpN2u);
	my_free(tmpN1u);
	my_free(tmpNN2u);
	my_free(tmpNN1u);
	my_free(worku);
	my_free(site_order);
	#ifdef CHECK_G_ACC
	my_free(gdacc);
	my_free(guacc);
	#endif
	#ifdef CHECK_G_WRP
	my_free(gdwrp);
	my_free(guwrp);
	#endif
	my_free(gd);
	my_free(gu);
	my_free(Cd);
	my_free(Cu);
	my_free(Bd);
	my_free(Bu);
}

// returns -1 for failure, 0 for completion, 1 for partial completion
int dqmc_wrapper(const char *sim_file, const char *log_file,
		const tick_t max_time)
{
	const tick_t wall_start = time_wall();

	profile_clear();

	// initialize signal handlers first time dqmc_wrapper() is called
	static int sigaction_called = 0;
	if (sigaction_called == 0) {
		sigaction_called = 1;
		sigaction(SIGUSR1, &(const struct sigaction){.sa_handler = progress}, NULL);
		sigaction(SIGINT, &(const struct sigaction){.sa_handler = stop}, NULL);
		sigaction(SIGTERM, &(const struct sigaction){.sa_handler = stop}, NULL);
	}

	// open log file
	FILE *log = (log_file != NULL) ? fopen(log_file, "a") : stdout;
	if (log == NULL) {
		fprintf(stderr, "fopen() failed to open: %s\n", log_file);
		return -1;
	}

	fprintf(log, "commit id %s\n", GIT_ID);
	fprintf(log, "compiled on %s %s\n", __DATE__, __TIME__);

	// open and read simulation file into structs
	int status = 0;
	struct params *p = my_calloc(sizeof(struct params));
	struct state *s = my_calloc(sizeof(struct state));
	struct meas_eqlt *m_eq = my_calloc(sizeof(struct meas_eqlt));
	struct meas_uneqlt *m_ue = my_calloc(sizeof(struct meas_uneqlt));

	fprintf(log, "opening %s\n", sim_file);
	profile_begin(read_file);
	status = read_file(sim_file, p, s, m_eq, m_ue);
	profile_end(read_file);
	if (status < 0) {
		fprintf(stderr, "read_file() failed: %d\n", status);
		status = -1;
		goto cleanup;
	}

	// if no uneqlt measurements, free m_ue and set to NULL
	if (p->period_uneqlt == 0) {
		my_free(m_ue);
		m_ue = NULL;
	}

	// check existing progress
	fprintf(log, "%d/%d sweeps completed\n", s->sweep, p->n_sweep);
	if (s->sweep >= p->n_sweep) {
		fprintf(log, "already finished\n");
		status = 0;
		goto cleanup;
	}

	fprintf(log, "starting dqmc\n");
	dqmc(log, wall_start, max_time, p, s, m_eq, m_ue);

	fprintf(log, "saving data\n");
	profile_begin(save_file);
	status = save_file(sim_file, s, m_eq, m_ue);
	profile_end(save_file);
	if (status < 0) {
		fprintf(stderr, "save_file() failed: %d\n", status);
		status = -1;
		goto cleanup;
	}

	status = (s->sweep == p->n_sweep) ? 0 : 1;

cleanup:
	if (m_ue != NULL) {
		my_free(m_ue->pair_sw);
		my_free(m_ue->zz);
		my_free(m_ue->xx);
		my_free(m_ue->nn);
		my_free(m_ue->gt0);
		my_free(m_ue->g0t);
	}
	my_free(m_eq->pair_sw);
	my_free(m_eq->zz);
	my_free(m_eq->xx);
	my_free(m_eq->nn);
	my_free(m_eq->g00);
	my_free(m_eq->double_occ);
	my_free(m_eq->density);
	my_free(s->hs);
	my_free(p->del);
	my_free(p->exp_lambda);
	my_free(p->inv_exp_K);
	my_free(p->exp_K);
	my_free(p->degen_ij);
	my_free(p->degen_i);
//	my_free(p->U);
//	my_free(p->K);
	my_free(p->map_ij);
	my_free(p->map_i);

	my_free(m_ue);
	my_free(m_eq);
	my_free(s);
	my_free(p);

	const tick_t wall_time = time_wall() - wall_start;
	fprintf(log, "wall time: %.3f\n", wall_time * SEC_PER_TICK);

	profile_print(log, wall_time);

	if (log != stdout)
		fclose(log);
	else
		fflush(log);

	return status;
}

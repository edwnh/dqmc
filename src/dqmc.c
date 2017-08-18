#include "dqmc.h"
#include <math.h>
#include <stdio.h>
#include <mkl.h>
#include "data.h"
#include "greens.h"
#include "meas.h"
#include "prof.h"
#include "rand.h"
#include "sig.h"
#include "time_.h"
#include "updates.h"
#include "util.h"

#define N_MUL 2 // input parameter to calc_eq_g() and calc_ue_g()

// uncomment to check recalculated g against wrapped g
// #define CHECK_G_WRP

// uncomment to check recalculated g against g from using QR for every multiply
// #define CHECK_G_ACC

// uncomment to check 0,0 block of unequal-time G against recalculated g
// #define CHECK_G_UE

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

#define calciBu(iB, l) do { \
	for (int i = 0; i < N; i++) { \
		const double el = exp_lambda[i + N*!hs[i + N*(l)]]; \
		for (int j = 0; j < N; j++) \
			(iB)[i + N*j] = el * inv_exp_K[i + N*j]; \
	} \
} while (0);

#define calciBd(iB, l) do { \
	for (int i = 0; i < N; i++) { \
		const double el = exp_lambda[i + N*hs[i + N*(l)]]; \
		for (int j = 0; j < N; j++) \
			(iB)[i + N*j] = el * inv_exp_K[i + N*j]; \
	} \
} while (0);

#define matdiff(m, n, A, ldA, B, ldB) do { \
	double max = 0.0, avg = 0.0; \
	for (int j = 0; j < (n); j++) \
	for (int i = 0; i < (m); i++) { \
		const double diff = fabs((A)[i + (ldA)*j] - (B)[i + (ldB)*j]); \
		if (diff > max) max = diff; \
		avg += diff; \
	} \
	avg /= N*N; \
	printf(#A " - " #B ":\tmax %.3e\tavg %.3e\n", max, avg); \
} while (0);

static void dqmc(struct sim_data *sim)
{
	const int N = sim->p.N;
	const int L = sim->p.L;
	const int n_matmul = sim->p.n_matmul;
	const int n_delay = sim->p.n_delay;
	const int F = sim->p.F;
	const double *const restrict exp_K = sim->p.exp_K; _aa(exp_K);
	const double *const restrict inv_exp_K = sim->p.inv_exp_K; _aa(inv_exp_K);
	const double *const restrict exp_lambda = sim->p.exp_lambda; _aa(exp_lambda);
	const double *const restrict del = sim->p.del; _aa(del);
	uint64_t *const restrict rng = sim->s.rng; _aa(rng);
	int *const restrict hs = sim->s.hs;

	// stride for time index in arrays of B or C matrices
	const int stride = DBL_ALIGN * ((N*N + DBL_ALIGN - 1) / DBL_ALIGN);
	__assume(stride % DBL_ALIGN == 0);

	double *const Bu = my_calloc(stride*L * sizeof(double)); _aa(Bu);
	double *const Bd = my_calloc(stride*L * sizeof(double)); _aa(Bd);
	double *const iBu = my_calloc(stride*L * sizeof(double)); _aa(iBu);
	double *const iBd = my_calloc(stride*L * sizeof(double)); _aa(iBd);
	double *const Cu = my_calloc(stride*F * sizeof(double)); _aa(Cu);
	double *const Cd = my_calloc(stride*F * sizeof(double)); _aa(Cd);
	double *const restrict gu = my_calloc(N*N * sizeof(double)); _aa(gu);
	double *const restrict gd = my_calloc(N*N * sizeof(double)); _aa(gd);
	#ifdef CHECK_G_WRP
	double *const restrict guwrp = my_calloc(N*N * sizeof(double)); _aa(guwrp);
	double *const restrict gdwrp = my_calloc(N*N * sizeof(double)); _aa(gdwrp);
	#endif
	#ifdef CHECK_G_ACC
	double *const restrict guacc = my_calloc(N*N * sizeof(double)); _aa(guacc);
	double *const restrict gdacc = my_calloc(N*N * sizeof(double)); _aa(gdacc);
	#endif
	int sign = 0;
	int *const site_order = my_calloc(N * sizeof(double)); _aa(site_order);

	// work arrays for calc_eq_g and stuff. two sets for easy 2x parallelization
	double *const restrict tmpNN1u = my_calloc(N*N * sizeof(double)); _aa(tmpNN1u);
	double *const restrict tmpNN2u = my_calloc(N*N * sizeof(double)); _aa(tmpNN2u);
	double *const restrict tmpN1u = my_calloc(N * sizeof(double)); _aa(tmpN1u);
	double *const restrict tmpN2u = my_calloc(N * sizeof(double)); _aa(tmpN2u);
	double *const restrict tmpN3u = my_calloc(N * sizeof(double)); _aa(tmpN3u);
	int *const restrict pvtu = my_calloc(N * sizeof(int)); _aa(pvtu);

	double *const restrict tmpNN1d = my_calloc(N*N * sizeof(double)); _aa(tmpNN1d);
	double *const restrict tmpNN2d = my_calloc(N*N * sizeof(double)); _aa(tmpNN2d);
	double *const restrict tmpN1d = my_calloc(N * sizeof(double)); _aa(tmpN1d);
	double *const restrict tmpN2d = my_calloc(N * sizeof(double)); _aa(tmpN2d);
	double *const restrict tmpN3d = my_calloc(N * sizeof(double)); _aa(tmpN3d);
	int *const restrict pvtd = my_calloc(N * sizeof(int)); _aa(pvtd);

	// arrays for calc_ue_g
	double *restrict ueGu = NULL;
	double *restrict Gredu = NULL;
	double *restrict tauu = NULL;
	double *restrict Qu = NULL;

	double *restrict ueGd = NULL;
	double *restrict Gredd = NULL;
	double *restrict taud = NULL;
	double *restrict Qd = NULL;

	if (sim->p.period_uneqlt > 0) {
		ueGu = my_calloc(N*L*N*L * sizeof(double)); _aa(ueGu);
		Gredu = my_calloc(N*F*N*F * sizeof(double)); _aa(Gredu);
		tauu = my_calloc(N*F * sizeof(double)); _aa(tauu);
		Qu = my_calloc(4*N*N * sizeof(double)); _aa(Qu);

		ueGd = my_calloc(N*L*N*L * sizeof(double)); _aa(ueGd);
		Gredd = my_calloc(N*F*N*F * sizeof(double)); _aa(Gredd);
		taud = my_calloc(N*F * sizeof(double)); _aa(taud);
		Qd = my_calloc(4*N*N * sizeof(double)); _aa(Qd);
	}

	// lapack work arrays
	int lwork = get_lwork_eq_g(N);
	if (sim->p.period_uneqlt > 0) {
		const int lwork_ue = get_lwork_ue_g(N, F);
		if (lwork_ue > lwork) lwork = lwork_ue;
	}
	double *const restrict worku = my_calloc(lwork * sizeof(double)); _aa(worku);
	double *const restrict workd = my_calloc(lwork * sizeof(double)); _aa(workd);

	{
	int signu, signd;
	#pragma omp parallel sections
	{
	#pragma omp section
	{
	if (sim->p.period_uneqlt > 0)
		for (int l = 0; l < L; l++)
			calciBu(iBu + stride*l, l);
	for (int f = 0; f < F; f++) {
		calcBu(Bu + stride*f*n_matmul, f*n_matmul);
		my_copy(Cu + stride*f, Bu + stride*f*n_matmul, N*N);
		for (int l = f*n_matmul + 1 ; l < (f + 1) * n_matmul; l++) {
			calcBu(Bu + stride*l, l);
			my_copy(tmpNN1u, Cu + stride*f, N*N);
			matmul(Cu + stride*f, Bu + stride*l, tmpNN1u);
		}
	}
	signu = calc_eq_g(0, N, stride, F, N_MUL, Cu, gu, tmpNN1u, tmpNN2u,
	                  tmpN1u, tmpN2u, tmpN3u, pvtu, worku, lwork);
	}
	#pragma omp section
	{
	if (sim->p.period_uneqlt > 0)
		for (int l = 0; l < L; l++)
			calciBu(iBu + stride*l, l);
	for (int f = 0; f < F; f++) {
		calcBd(Bd + stride*f*n_matmul, f*n_matmul);
		my_copy(Cd + stride*f, Bd + stride*f*n_matmul, N*N);
		for (int l = f*n_matmul + 1 ; l < (f + 1) * n_matmul; l++) {
			calcBd(Bd + stride*l, l);
			my_copy(tmpNN1d, Cd + stride*f, N*N);
			matmul(Cd + stride*f, Bd + stride*l, tmpNN1d);
		}
	}
	signd = calc_eq_g(0, N, stride, F, N_MUL, Cd, gd, tmpNN1d, tmpNN2d,
	                  tmpN1d, tmpN2d, tmpN3d, pvtd, workd, lwork);
	}
	}
	sign = signu*signd;
	}

	for (; sim->s.sweep < sim->p.n_sweep; sim->s.sweep++) {
		if (sig_check_state(sim->s.sweep, sim->p.n_sweep_warm, sim->p.n_sweep) != 0)
			break;

		for (int l = 0; l < L; l++) {
			profile_begin(updates);
			shuffle(rng, N, site_order);
			update_delayed(N, n_delay, del, site_order,
			               rng, hs + N*l, gu, gd, &sign,
			               tmpNN1u, tmpNN2u, tmpN1u,
			               tmpNN1d, tmpNN2d, tmpN1d);
			profile_end(updates);

			const int f = l / n_matmul;
			const int recalc = ((l + 1) % n_matmul == 0);
			int signu, signd;
			#pragma omp parallel sections
			{
			#pragma omp section
			{
			profile_begin(multb);
			double *const restrict Bul = Bu + stride*l; _aa(Bul);
			double *const restrict iBul = iBu + stride*l; _aa(iBul);
			double *const restrict Cuf = Cu + stride*f; _aa(Cuf);
			calcBu(Bul, l);
			if (l % n_matmul == 0) {
				my_copy(Cuf, Bul, N*N);
			} else {
				my_copy(tmpNN1u, Cuf, N*N);
				matmul(Cuf, Bul, tmpNN1u);
			}
			if (!recalc || sim->p.period_uneqlt > 0)
				calciBu(iBul, l);
			profile_end(multb);
			if (recalc) {
				profile_begin(recalc);
				#ifdef CHECK_G_WRP
				if (sim->p.period_uneqlt == 0)
					calciBu(iBu + stride*l, l);
				matmul(tmpNN1u, gu, iBu + stride*l);
				matmul(guwrp, Bu + stride*l, tmpNN1u);
				#endif
				#ifdef CHECK_G_ACC
				calc_eq_g((l + 1) % L, N, stride, L, 1, Bu, guacc,
				          tmpNN1u, tmpNN2u, tmpN1u, tmpN2u,
				          tmpN3u, pvtu, worku, lwork);
				#endif
				signu = calc_eq_g((f + 1) % F, N, stride, F, N_MUL, Cu, gu,
				                  tmpNN1u, tmpNN2u, tmpN1u, tmpN2u,
				                  tmpN3u, pvtu, worku, lwork);
				profile_end(recalc);
			} else {
				profile_begin(wrap);
				matmul(tmpNN1u, gu, iBul);
				matmul(gu, Bul, tmpNN1u);
				profile_end(wrap);
			}
			}
			#pragma omp section
			{
			profile_begin(multb);
			double *const restrict Bdl = Bd + stride*l; _aa(Bdl);
			double *const restrict iBdl = iBd + stride*l; _aa(iBdl);
			double *const restrict Cdf = Cd + stride*f; _aa(Cdf);
			calcBd(Bdl, l)
			if (l % n_matmul == 0) {
				my_copy(Cdf, Bdl, N*N);
			} else {
				my_copy(tmpNN1d, Cdf, N*N);
				matmul(Cdf, Bdl, tmpNN1d);
			}
			if (!recalc || sim->p.period_uneqlt > 0)
				calciBd(iBdl, l);
			profile_end(multb);
			if (recalc) {
				profile_begin(recalc);
				#ifdef CHECK_G_WRP
				if (sim->p.period_uneqlt == 0)
					calciBd(iBd + stride*l, l);
				matmul(tmpNN1d, gd, iBd + stride*l);
				matmul(gdwrp, Bd + stride*l, tmpNN1d);
				#endif
				#ifdef CHECK_G_ACC
				calc_eq_g((l + 1) % L, N, stride, L, 1, Bd, gdacc,
				          tmpNN1d, tmpNN2d, tmpN1d, tmpN2d,
				          tmpN3d, pvtd, workd, lwork);
				#endif
				signd = calc_eq_g((f + 1) % F, N, stride, F, N_MUL, Cd, gd,
				                  tmpNN1d, tmpNN2d, tmpN1d, tmpN2d,
				                  tmpN3d, pvtd, workd, lwork);
				profile_end(recalc);
			} else {
				profile_begin(wrap);
				matmul(tmpNN1d, gd, iBdl);
				matmul(gd, Bdl, tmpNN1d);
				profile_end(wrap);
			}
			}
			}

			#ifdef CHECK_G_WRP
			if (recalc) {
				matdiff(N, N, gu, N, guwrp, N);
				matdiff(N, N, gd, N, gdwrp, N);
			}
			#endif
			#ifdef CHECK_G_ACC
			if (recalc) {
				matdiff(N, N, gu, N, guacc, N);
				matdiff(N, N, gd, N, gdacc, N);
			}
			#endif
			#if defined(CHECK_G_WRP) && defined(CHECK_G_ACC)
			if (recalc) {
				matdiff(N, N, guwrp, N, guacc, N);
				matdiff(N, N, gdwrp, N, gdacc, N);
			}
			#endif

			if (recalc) sign = signu*signd;

			if ((sim->s.sweep >= sim->p.n_sweep_warm) &&
					(sim->p.period_eqlt > 0) &&
					(l + 1) % sim->p.period_eqlt == 0) {
				profile_begin(meas_eq);
				measure_eqlt(&sim->p, sign, gu, gd, &sim->m_eq);
				profile_end(meas_eq);
			}
		}

		if ((sim->s.sweep >= sim->p.n_sweep_warm) && (sim->p.period_uneqlt > 0) &&
				sim->s.sweep % sim->p.period_uneqlt == 0) {
			#pragma omp parallel sections
			{
			#pragma omp section
			calc_ue_g(N, stride, L, F, N_MUL, Bu, iBu, Cu,
			          ueGu, Gredu, tauu, Qu, worku, lwork);
			#pragma omp section
			calc_ue_g(N, stride, L, F, N_MUL, Bd, iBd, Cd,
			          ueGd, Gredd, taud, Qd, workd, lwork);
			}

			#ifdef CHECK_G_UE
			matdiff(N, N, gu, N, ueGu, N*L);
			matdiff(N, N, gd, N, ueGd, N*L);
			#endif
			#if defined(CHECK_G_UE) && defined(CHECK_G_ACC)
			matdiff(N, N, ueGu, N*L, guacc, N);
			matdiff(N, N, ueGd, N*L, gdacc, N);
			#endif

			profile_begin(meas_uneq);
			measure_uneqlt(&sim->p, sign, ueGu, ueGd, &sim->m_ue);
			profile_end(meas_uneq);
		}
	}


	my_free(workd);
	my_free(worku);
	if (sim->p.period_uneqlt > 0) {
		my_free(Qd);
		my_free(taud);
		my_free(Gredd);
		my_free(ueGd);
		my_free(Qu);
		my_free(tauu);
		my_free(Gredu);
		my_free(ueGu);
	}
	my_free(pvtd);
	my_free(tmpN3d);
	my_free(tmpN2d);
	my_free(tmpN1d);
	my_free(tmpNN2d);
	my_free(tmpNN1d);
	my_free(pvtu);
	my_free(tmpN3u);
	my_free(tmpN2u);
	my_free(tmpN1u);
	my_free(tmpNN2u);
	my_free(tmpNN1u);
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
	my_free(iBd);
	my_free(iBu);
	my_free(Bd);
	my_free(Bu);
}

int dqmc_wrapper(const char *sim_file, const char *log_file,
		const tick_t max_time)
{
	const tick_t wall_start = time_wall();
	profile_clear();

	int status = 0;

	// open log file
	FILE *log = (log_file != NULL) ? fopen(log_file, "a") : stdout;
	if (log == NULL) {
		fprintf(stderr, "fopen() failed to open: %s\n", log_file);
		return -1;
	}

	fprintf(log, "commit id %s\n", GIT_ID);
	fprintf(log, "compiled on %s %s\n", __DATE__, __TIME__);

	// initialize signal handling
	sig_init(log, wall_start, max_time);

	// open and read simulation file
	struct sim_data *sim = my_calloc(sizeof(struct sim_data));
	fprintf(log, "opening %s\n", sim_file);
	status = sim_data_read_alloc(sim, sim_file);
	if (status < 0) {
		fprintf(stderr, "read_file() failed: %d\n", status);
		status = -1;
		goto cleanup;
	}

	// check existing progress
	fprintf(log, "%d/%d sweeps completed\n", sim->s.sweep, sim->p.n_sweep);
	if (sim->s.sweep >= sim->p.n_sweep) {
		fprintf(log, "already finished\n");
		goto cleanup;
	}

	// run dqmc
	fprintf(log, "starting dqmc\n");
	dqmc(sim);
	fprintf(log, "%d/%d sweeps completed\n", sim->s.sweep, sim->p.n_sweep);

	// save to simulation file
	fprintf(log, "saving data\n");
	status = sim_data_save(sim, sim_file);
	if (status < 0) {
		fprintf(stderr, "save_file() failed: %d\n", status);
		status = -1;
		goto cleanup;
	}

	status = (sim->s.sweep == sim->p.n_sweep) ? 0 : 1;

cleanup:
	sim_data_free(sim);
	my_free(sim);

	const tick_t wall_time = time_wall() - wall_start;
	fprintf(log, "wall time: %.3f\n", wall_time * SEC_PER_TICK);
	profile_print(log, wall_time);

	if (log != stdout)
		fclose(log);
	else
		fflush(log);

	return status;
}

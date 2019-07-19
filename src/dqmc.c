#include "dqmc.h"
#include <tgmath.h>
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

// who needs function calls :D
#define matmul(C, A, B) do { \
	zgemm("N", "N", &N, &N, &N, ccplx(1.0), (A), &N, (B), &N, ccplx(0.0), (C), &N); \
} while (0);

#define calcBu(B, l) do { \
	for (int j = 0; j < N; j++) { \
		const double el = exp_lambda[j + N*hs[j + N*(l)]]; \
		for (int i = 0; i < N; i++) \
			(B)[i + N*j] = exp_Ku[i + N*j] * el; \
	} \
} while (0);

#define calcBd(B, l) do { \
	for (int j = 0; j < N; j++) { \
		const double el = exp_lambda[j + N*!hs[j + N*(l)]]; \
		for (int i = 0; i < N; i++) \
			(B)[i + N*j] = exp_Kd[i + N*j] * el; \
	} \
} while (0);

#define calciBu(iB, l) do { \
	for (int i = 0; i < N; i++) { \
		const double el = exp_lambda[i + N*!hs[i + N*(l)]]; \
		for (int j = 0; j < N; j++) \
			(iB)[i + N*j] = el * inv_exp_Ku[i + N*j]; \
	} \
} while (0);

#define calciBd(iB, l) do { \
	for (int i = 0; i < N; i++) { \
		const double el = exp_lambda[i + N*hs[i + N*(l)]]; \
		for (int j = 0; j < N; j++) \
			(iB)[i + N*j] = el * inv_exp_Kd[i + N*j]; \
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

static int dqmc(struct sim_data *sim)
{
	const int N = sim->p.N;
	const int L = sim->p.L;
	const int n_matmul = sim->p.n_matmul;
	const int n_delay = sim->p.n_delay;
	const int F = sim->p.F;
	const complex double *const restrict exp_Ku = sim->p.exp_Ku;
	const complex double *const restrict exp_Kd = sim->p.exp_Kd;
	const complex double *const restrict inv_exp_Ku = sim->p.inv_exp_Ku;
	const complex double *const restrict inv_exp_Kd = sim->p.inv_exp_Kd;
	const complex double *const restrict exp_halfKu = sim->p.exp_halfKu;
	const complex double *const restrict exp_halfKd = sim->p.exp_halfKd;
	const complex double *const restrict inv_exp_halfKu = sim->p.inv_exp_halfKu;
	const complex double *const restrict inv_exp_halfKd = sim->p.inv_exp_halfKd;
	const double *const restrict exp_lambda = sim->p.exp_lambda;
	const double *const restrict del = sim->p.del;
	uint64_t *const restrict rng = sim->s.rng;
	int *const restrict hs = sim->s.hs;

	complex double *const Bu = my_calloc(N*N*L * sizeof(complex double));
	complex double *const Bd = my_calloc(N*N*L * sizeof(complex double));
	complex double *const iBu = my_calloc(N*N*L * sizeof(complex double));
	complex double *const iBd = my_calloc(N*N*L * sizeof(complex double));
	complex double *const Cu = my_calloc(N*N*F * sizeof(complex double));
	complex double *const Cd = my_calloc(N*N*F * sizeof(complex double));
	complex double *const restrict gu = my_calloc(N*N * sizeof(complex double));
	complex double *const restrict gd = my_calloc(N*N * sizeof(complex double));
	#ifdef CHECK_G_WRP
	complex double *const restrict guwrp = my_calloc(N*N * sizeof(complex double));
	complex double *const restrict gdwrp = my_calloc(N*N * sizeof(complex double));
	#endif
	#ifdef CHECK_G_ACC
	complex double *const restrict guacc = my_calloc(N*N * sizeof(complex double));
	complex double *const restrict gdacc = my_calloc(N*N * sizeof(complex double));
	#endif
	complex double phase;
	int *const site_order = my_calloc(N * sizeof(double));

	// work arrays for calc_eq_g and stuff. two sets for easy 2x parallelization
	complex double *const restrict tmpNN1u = my_calloc(N*N * sizeof(complex double));
	complex double *const restrict tmpNN2u = my_calloc(N*N * sizeof(complex double));
	complex double *const restrict tmpN1u = my_calloc(N * sizeof(complex double));
	complex double *const restrict tmpN2u = my_calloc(N * sizeof(complex double));
	complex double *const restrict tmpN3u = my_calloc(N * sizeof(complex double));
	int *const restrict pvtu = my_calloc(N * sizeof(int));

	complex double *const restrict tmpNN1d = my_calloc(N*N * sizeof(complex double));
	complex double *const restrict tmpNN2d = my_calloc(N*N * sizeof(complex double));
	complex double *const restrict tmpN1d = my_calloc(N * sizeof(complex double));
	complex double *const restrict tmpN2d = my_calloc(N * sizeof(complex double));
	complex double *const restrict tmpN3d = my_calloc(N * sizeof(complex double));
	int *const restrict pvtd = my_calloc(N * sizeof(int));

	// arrays for calc_ue_g
	complex double *restrict Gu0t = NULL;
	complex double *restrict Gutt = NULL;
	complex double *restrict Gut0 = NULL;
	// complex double *restrict ueGu = NULL;
	complex double *restrict Gredu = NULL;
	complex double *restrict tauu = NULL;
	complex double *restrict Qu = NULL;

	complex double *restrict Gd0t = NULL;
	complex double *restrict Gdtt = NULL;
	complex double *restrict Gdt0 = NULL;
	// complex double *restrict ueGd = NULL;
	complex double *restrict Gredd = NULL;
	complex double *restrict taud = NULL;
	complex double *restrict Qd = NULL;

	if (sim->p.period_uneqlt > 0) {
		const int E = 1 + (F - 1) / N_MUL;

		Gredu = my_calloc(N*E*N*E * sizeof(complex double));
		tauu = my_calloc(N*E * sizeof(complex double));
		Qu = my_calloc(4*N*N * sizeof(complex double));

		Gredd = my_calloc(N*E*N*E * sizeof(complex double));
		taud = my_calloc(N*E * sizeof(complex double));
		Qd = my_calloc(4*N*N * sizeof(complex double));

		Gu0t = my_calloc(N*N*L * sizeof(complex double));
		Gutt = my_calloc(N*N*L * sizeof(complex double));
		Gut0 = my_calloc(N*N*L * sizeof(complex double));
		Gd0t = my_calloc(N*N*L * sizeof(complex double));
		Gdtt = my_calloc(N*N*L * sizeof(complex double));
		Gdt0 = my_calloc(N*N*L * sizeof(complex double));
		// ueGu = my_calloc(N*N*L*L * sizeof(complex double));
		// ueGd = my_calloc(N*N*L*L * sizeof(complex double));
		// if (ueGu == NULL || ueGd == NULL) return -1;
	}

	// lapack work arrays
	int lwork = get_lwork_eq_g(N);
	if (sim->p.period_uneqlt > 0) {
		const int lwork_ue = get_lwork_ue_g(N, F);
		if (lwork_ue > lwork) lwork = lwork_ue;
	}
	complex double *const restrict worku = my_calloc(lwork * sizeof(complex double));
	complex double *const restrict workd = my_calloc(lwork * sizeof(complex double));

	{
	complex double phaseu, phased;
	#pragma omp parallel sections
	{
	#pragma omp section
	{
	if (sim->p.period_uneqlt > 0)
		for (int l = 0; l < L; l++)
			calciBu(iBu + N*N*l, l);
	for (int l = 0; l < L; l++)
		calcBu(Bu + N*N*l, l);
	for (int f = 0; f < F; f++)
		mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L, 1.0,
		        Bu, Cu + N*N*f, N, tmpNN1u);
	phaseu = calc_eq_g(0, N, F, N_MUL, Cu, gu, tmpNN1u, tmpNN2u,
	                  tmpN1u, tmpN2u, tmpN3u, pvtu, worku, lwork);
	}
	#pragma omp section
	{
	if (sim->p.period_uneqlt > 0)
		for (int l = 0; l < L; l++)
			calciBd(iBd + N*N*l, l);
	for (int l = 0; l < L; l++)
		calcBd(Bd + N*N*l, l);
	for (int f = 0; f < F; f++)
		mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L, 1.0,
		        Bd, Cd + N*N*f, N, tmpNN1d);
	phased = calc_eq_g(0, N, F, N_MUL, Cd, gd, tmpNN1d, tmpNN2d,
	                  tmpN1d, tmpN2d, tmpN3d, pvtd, workd, lwork);
	}
	}
	phase = phaseu*phased;
	}

	for (; sim->s.sweep < sim->p.n_sweep; sim->s.sweep++) {
		if (sig_check_state(sim->s.sweep, sim->p.n_sweep_warm, sim->p.n_sweep) != 0)
			break;

		for (int l = 0; l < L; l++) {
			profile_begin(updates);
			shuffle(rng, N, site_order);
			update_delayed(N, n_delay, del, site_order,
			               rng, hs + N*l, gu, gd, &phase,
			               tmpNN1u, tmpNN2u, tmpN1u,
			               tmpNN1d, tmpNN2d, tmpN1d);
			profile_end(updates);

			const int f = l / n_matmul;
			const int recalc = ((l + 1) % n_matmul == 0);
			complex double phaseu, phased;
			#pragma omp parallel sections
			{
			#pragma omp section
			{
			complex double *const restrict Bul = Bu + N*N*l;
			complex double *const restrict iBul = iBu + N*N*l;
			complex double *const restrict Cuf = Cu + N*N*f;
			profile_begin(calcb);
			calcBu(Bul, l);
			if (!recalc || sim->p.period_uneqlt > 0)
				calciBu(iBul, l);
			profile_end(calcb);
			if (recalc) {
				profile_begin(multb);
				mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L,
				        1.0, Bu, Cuf, N, tmpNN1u);
				profile_end(multb);
				profile_begin(recalc);
				#ifdef CHECK_G_WRP
				if (sim->p.period_uneqlt == 0)
					calciBu(iBu + N*N*l, l);
				matmul(tmpNN1u, gu, iBu + N*N*l);
				matmul(guwrp, Bu + N*N*l, tmpNN1u);
				#endif
				#ifdef CHECK_G_ACC
				calc_eq_g((l + 1) % L, N, L, 1, Bu, guacc,
				          tmpNN1u, tmpNN2u, tmpN1u, tmpN2u,
				          tmpN3u, pvtu, worku, lwork);
				#endif
				phaseu = calc_eq_g((f + 1) % F, N, F, N_MUL, Cu, gu,
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
			complex double *const restrict Bdl = Bd + N*N*l;
			complex double *const restrict iBdl = iBd + N*N*l;
			complex double *const restrict Cdf = Cd + N*N*f;
			profile_begin(calcb);
			calcBd(Bdl, l);
			if (!recalc || sim->p.period_uneqlt > 0)
				calciBd(iBdl, l);
			profile_end(calcb);
			if (recalc) {
				profile_begin(multb);
				mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L,
				        1.0, Bd, Cdf, N, tmpNN1d);
				profile_end(multb);
				profile_begin(recalc);
				#ifdef CHECK_G_WRP
				if (sim->p.period_uneqlt == 0)
					calciBd(iBd + N*N*l, l);
				matmul(tmpNN1d, gd, iBd + N*N*l);
				matmul(gdwrp, Bd + N*N*l, tmpNN1d);
				#endif
				#ifdef CHECK_G_ACC
				calc_eq_g((l + 1) % L, N, L, 1, Bd, gdacc,
				          tmpNN1d, tmpNN2d, tmpN1d, tmpN2d,
				          tmpN3d, pvtd, workd, lwork);
				#endif
				phased = calc_eq_g((f + 1) % F, N, F, N_MUL, Cd, gd,
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

			if (recalc) phase = phaseu*phased;

			if ((sim->s.sweep >= sim->p.n_sweep_warm) &&
					(sim->p.period_eqlt > 0) &&
					(l + 1) % sim->p.period_eqlt == 0) {
				#pragma omp parallel sections
				{
				#pragma omp section
				{
				profile_begin(half_wrap);
				matmul(tmpNN1u, gu, exp_halfKu);
				matmul(tmpNN2u, inv_exp_halfKu, tmpNN1u);
				profile_end(half_wrap);
				}
				#pragma omp section
				{
				profile_begin(half_wrap);
				matmul(tmpNN1d, gd, exp_halfKd);
				matmul(tmpNN2d, inv_exp_halfKd, tmpNN1d);
				profile_end(half_wrap);
				}
				}

				profile_begin(meas_eq);
				measure_eqlt(&sim->p, phase, tmpNN2u, tmpNN2d, &sim->m_eq);
				profile_end(meas_eq);
			}
		}

		if ((sim->s.sweep >= sim->p.n_sweep_warm) && (sim->p.period_uneqlt > 0) &&
				sim->s.sweep % sim->p.period_uneqlt == 0) {
			#pragma omp parallel sections
			{
			#pragma omp section
			calc_ue_g(N, L, F, N_MUL, Bu, iBu, Cu, Gu0t, Gutt, Gut0,
			          Gredu, tauu, Qu, worku, lwork);
			#pragma omp section
			calc_ue_g(N, L, F, N_MUL, Bd, iBd, Cd, Gd0t, Gdtt, Gdt0,
			          Gredd, taud, Qd, workd, lwork);
			}

			#ifdef CHECK_G_UE
			matdiff(N, N, gu, N, Gutt, N);
			matdiff(N, N, gd, N, Gdtt, N);
			#endif
			#if defined(CHECK_G_UE) && defined(CHECK_G_ACC)
			matdiff(N, N, Gutt, N, guacc, N);
			matdiff(N, N, Gdtt, N, gdacc, N);
			#endif

			#pragma omp parallel sections
			{
			#pragma omp section
			{
			profile_begin(half_wrap);
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1u, Gu0t + N*N*l, exp_halfKu);
				matmul(Gu0t + N*N*l, inv_exp_halfKu, tmpNN1u);
			}
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1u, Gutt + N*N*l, exp_halfKu);
				matmul(Gutt + N*N*l, inv_exp_halfKu, tmpNN1u);
			}
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1u, Gut0 + N*N*l, exp_halfKu);
				matmul(Gut0 + N*N*l, inv_exp_halfKu, tmpNN1u);
			}
			profile_end(half_wrap);
			}
			#pragma omp section
			{
			profile_begin(half_wrap);
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1d, Gd0t + N*N*l, exp_halfKd);
				matmul(Gd0t + N*N*l, inv_exp_halfKd, tmpNN1d);
			}
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1d, Gdtt + N*N*l, exp_halfKd);
				matmul(Gdtt + N*N*l, inv_exp_halfKd, tmpNN1d);
			}
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1d, Gdt0 + N*N*l, exp_halfKd);
				matmul(Gdt0 + N*N*l, inv_exp_halfKd, tmpNN1d);
			}
			profile_end(half_wrap);
			}
			}

			profile_begin(meas_uneq);
			measure_uneqlt(&sim->p, phase,
			               Gu0t, Gutt, Gut0, Gd0t, Gdtt, Gdt0,
			               &sim->m_ue);
			profile_end(meas_uneq);
			// #pragma omp parallel sections
			// {
			// #pragma omp section
			// calc_ue_g(N, L, F, N_MUL, Bu, iBu, Cu,
			          // ueGu, Gredu, tauu, Qu, worku, lwork);
			// #pragma omp section
			// calc_ue_g(N, L, F, N_MUL, Bd, iBd, Cd,
			          // ueGd, Gredd, taud, Qd, workd, lwork);
			// }

			// #ifdef CHECK_G_UE
			// matdiff(N, N, gu, N, ueGu, N);
			// matdiff(N, N, gd, N, ueGd, N);
			// #endif
			// #if defined(CHECK_G_UE) && defined(CHECK_G_ACC)
			// matdiff(N, N, ueGu, N, guacc, N);
			// matdiff(N, N, ueGd, N, gdacc, N);
			// #endif

			// profile_begin(meas_uneq);
			// measure_uneqlt(&sim->p, sign, ueGu, ueGd, &sim->m_ue);
			// profile_end(meas_uneq);
		}
	}


	my_free(workd);
	my_free(worku);
	if (sim->p.period_uneqlt > 0) {
		my_free(Qd);
		my_free(taud);
		my_free(Gredd);
		// my_free(ueGd);
		my_free(Gdt0);
		my_free(Gdtt);
		my_free(Gd0t);
		my_free(Qu);
		my_free(tauu);
		my_free(Gredu);
		// my_free(ueGu);
		my_free(Gut0);
		my_free(Gutt);
		my_free(Gu0t);
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

	return 0;
}

int dqmc_wrapper(const char *sim_file, const char *log_file,
		const tick_t max_time, const int bench)
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
	status = dqmc(sim);
	if (status < 0) {
		fprintf(stderr, "dqmc() failed to allocate memory\n");
		status = -1;
		goto cleanup;
	}
	fprintf(log, "%d/%d sweeps completed\n", sim->s.sweep, sim->p.n_sweep);

	// save to simulation file (if not in benchmarking mode)
	if (!bench) {
		fprintf(log, "saving data\n");
		status = sim_data_save(sim, sim_file);
		if (status < 0) {
			fprintf(stderr, "save_file() failed: %d\n", status);
			status = -1;
			goto cleanup;
		}
	} else {
		fprintf(log, "benchmark mode enabled; not saving data\n");
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

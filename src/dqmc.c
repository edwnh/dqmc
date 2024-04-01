#include "dqmc.h"
#include <tgmath.h>
#include <stdio.h>
#include "data.h"
#include "greens.h"
#include "linalg.h"
#include "meas.h"
#include "prof.h"
#include "rand.h"
#include "sig.h"
#include "time_.h"
#include "updates.h"

#define N_MUL 2 // input parameter to calc_eq_g() and calc_ue_g()

// uncomment to check recalculated g against wrapped g
// #define CHECK_G_WRP

// uncomment to check recalculated g against g from using QR for every multiply
// #define CHECK_G_ACC

// uncomment to check 0,0 block of unequal-time G against recalculated g
// #define CHECK_G_UE

static inline void calcBu(
		const int l, const int N,
		const double *const restrict exp_lambda,
		const int *const restrict hs,
		const num *const restrict exp_K,
		num *const restrict B)
{
	for (int j = 0; j < N; j++) {
		const double el = exp_lambda[j + N*hs[j + N*l]];
		for (int i = 0; i < N; i++)
			B[i + N*j] = exp_K[i + N*j] * el;
	}
}
static inline void calcBd(
		const int l, const int N,
		const double *const restrict exp_lambda,
		const int *const restrict hs,
		const num *const restrict exp_K,
		num *const restrict B)
{
	for (int j = 0; j < N; j++) {
		const double el = exp_lambda[j + N*!hs[j + N*l]];
		for (int i = 0; i < N; i++)
			B[i + N*j] = exp_K[i + N*j] * el;
	}
}
static inline void calciBu(
		const int l, const int N,
		const double *const restrict exp_lambda,
		const int *const restrict hs,
		const num *const restrict inv_exp_K,
		num *const restrict iB)
{
	for (int i = 0; i < N; i++) {
		const double el = exp_lambda[i + N*!hs[i + N*l]];
		for (int j = 0; j < N; j++)
			iB[i + N*j] = el * inv_exp_K[i + N*j];
	}
}
static inline void calciBd(
		const int l, const int N,
		const double *const restrict exp_lambda,
		const int *const restrict hs,
		const num *const restrict inv_exp_K,
		num *const restrict iB)
{
	for (int i = 0; i < N; i++) {
		const double el = exp_lambda[i + N*hs[i + N*l]];
		for (int j = 0; j < N; j++)
			iB[i + N*j] = el * inv_exp_K[i + N*j];
	}
}

// who needs function calls :D
#define matmul(C, A, B) do { \
	xgemm("N", "N", N, N, N, 1.0, (A), N, (B), N, 0.0, (C), N); \
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
	const int F = sim->p.F;
	const int E = 1 + (F - 1) / N_MUL;
	const int n_matmul = sim->p.n_matmul;
	const int n_delay = sim->p.n_delay;
	const num *const restrict exp_Ku = sim->p.exp_Ku;
	const num *const restrict exp_Kd = sim->p.exp_Kd;
	const num *const restrict inv_exp_Ku = sim->p.inv_exp_Ku;
	const num *const restrict inv_exp_Kd = sim->p.inv_exp_Kd;
	const num *const restrict exp_halfKu = sim->p.exp_halfKu;
	const num *const restrict exp_halfKd = sim->p.exp_halfKd;
	const num *const restrict inv_exp_halfKu = sim->p.inv_exp_halfKu;
	const num *const restrict inv_exp_halfKd = sim->p.inv_exp_halfKd;
	const double *const restrict exp_lambda = sim->p.exp_lambda;
	const double *const restrict del = sim->p.del;

	uint64_t *const restrict rng = sim->s.rng;
	int *const restrict hs = sim->s.hs;

	num phase;

	// lapack work array size
	int lwork = N*N;
	if (sim->p.period_uneqlt > 0) {
		const int lwork_ue = get_lwork_ue_g(N, E);
		if (lwork_ue > lwork) lwork = lwork_ue;
	}

#define ALLOC_TABLE(XX, FOR, ENDFOR) \
	XX(int *const site_order, mp, N * sizeof(int)) \
	XX(num *restrict *const restrict iBu, mp, L * sizeof(num *)) \
	FOR(l, L) \
		XX(iBu[l], mp, N*N * sizeof(num)) \
	ENDFOR \
	XX(num *restrict *const restrict Bu, mp, L * sizeof(num *)) \
	FOR(l, L) \
		XX(Bu[l], mp, N*N * sizeof(num)) \
	ENDFOR \
	XX(num *restrict *const restrict Cu, mp, F * sizeof(num *)) \
	FOR(f, F) \
		XX(Cu[f], mp, N*N * sizeof(num)) \
	ENDFOR \
	XX(struct QdX *QdXLu, mp, F * sizeof(struct QdX)) \
	XX(struct QdX *QdX0u, mp, F * sizeof(struct QdX)) \
	FOR(f, F) \
		XX(QdXLu[f].Q, mp, N*N * sizeof(num)) \
		XX(QdXLu[f].tau, mp, N * sizeof(num)) \
		XX(QdXLu[f].d, mp, N * sizeof(num)) \
		XX(QdXLu[f].X, mp, N*N * sizeof(num)) \
		XX(QdX0u[f].Q, mp, N*N * sizeof(num)) \
		XX(QdX0u[f].tau, mp, N * sizeof(num)) \
		XX(QdX0u[f].d, mp, N * sizeof(num)) \
		XX(QdX0u[f].X, mp, N*N * sizeof(num)) \
	ENDFOR \
	XX(num *const restrict gu, mp, N*N * sizeof(num)) \
	XX(num *restrict *const restrict iBd, mp, L * sizeof(num *)) \
	FOR(l, L) \
		XX(iBd[l], mp, N*N * sizeof(num)) \
	ENDFOR \
	XX(num *restrict *const restrict Bd, mp, L * sizeof(num *)) \
	FOR(l, L) \
		XX(Bd[l], mp, N*N * sizeof(num)) \
	ENDFOR \
	XX(num *restrict *const restrict Cd, mp, F * sizeof(num *)) \
	FOR(f, F) \
		XX(Cd[f], mp, N*N * sizeof(num)) \
	ENDFOR \
	XX(struct QdX *QdXLd, mp, F * sizeof(struct QdX)) \
	XX(struct QdX *QdX0d, mp, F * sizeof(struct QdX)) \
	FOR(f, F) \
		XX(QdXLd[f].Q, mp, N*N * sizeof(num)) \
		XX(QdXLd[f].tau, mp, N * sizeof(num)) \
		XX(QdXLd[f].d, mp, N * sizeof(num)) \
		XX(QdXLd[f].X, mp, N*N * sizeof(num)) \
		XX(QdX0d[f].Q, mp, N*N * sizeof(num)) \
		XX(QdX0d[f].tau, mp, N * sizeof(num)) \
		XX(QdX0d[f].d, mp, N * sizeof(num)) \
		XX(QdX0d[f].X, mp, N*N * sizeof(num)) \
	ENDFOR \
	XX(num *const restrict gd, mp, N*N * sizeof(num)) \
	XX(num *const restrict tmpNN1u, mp, N*N * sizeof(num)) \
	XX(num *const restrict tmpNN2u, mp, N*N * sizeof(num)) \
	XX(num *const restrict tmpN1u, mp, N * sizeof(num)) \
	XX(num *const restrict tmpN2u, mp, N * sizeof(num)) \
	XX(num *const restrict tmpN3u, mp, N * sizeof(num)) \
	XX(int *const restrict pvtu, mp, N * sizeof(int)) \
	XX(num *const restrict tmpNN1d, mp, N*N * sizeof(num)) \
	XX(num *const restrict tmpNN2d, mp, N*N * sizeof(num)) \
	XX(num *const restrict tmpN1d, mp, N * sizeof(num)) \
	XX(num *const restrict tmpN2d, mp, N * sizeof(num)) \
	XX(num *const restrict tmpN3d, mp, N * sizeof(num)) \
	XX(int *const restrict pvtd, mp, N * sizeof(int)) \
	XX(num *const restrict worku, mp, lwork * sizeof(num)) \
	XX(num *const restrict workd, mp, lwork * sizeof(num)) \
	XX(num *const restrict Gredu, mp, (sim->p.period_uneqlt > 0)*N*E*N*E * sizeof(num)) \
	XX(num *const restrict tauu, mp, (sim->p.period_uneqlt > 0)*N*E * sizeof(num)) \
	XX(num *const restrict Qu, mp, (sim->p.period_uneqlt > 0)*4*N*N * sizeof(num)) \
	XX(num *const restrict Gredd, mp, (sim->p.period_uneqlt > 0)*N*E*N*E * sizeof(num)) \
	XX(num *const restrict taud, mp, (sim->p.period_uneqlt > 0)*N*E * sizeof(num)) \
	XX(num *const restrict Qd, mp, (sim->p.period_uneqlt > 0)*4*N*N * sizeof(num)) \
	XX(num *const restrict Gu0t, mp, (sim->p.period_uneqlt > 0)*N*N*L * sizeof(num)) \
	XX(num *const restrict Gutt, mp, (sim->p.period_uneqlt > 0)*N*N*L * sizeof(num)) \
	XX(num *const restrict Gut0, mp, (sim->p.period_uneqlt > 0)*N*N*L * sizeof(num)) \
	XX(num *const restrict Gd0t, mp, (sim->p.period_uneqlt > 0)*N*N*L * sizeof(num)) \
	XX(num *const restrict Gdtt, mp, (sim->p.period_uneqlt > 0)*N*N*L * sizeof(num)) \
	XX(num *const restrict Gdt0, mp, (sim->p.period_uneqlt > 0)*N*N*L * sizeof(num))

	struct mem_pool *mp = pool_new(POOL_GET_SIZE(ALLOC_TABLE));
	POOL_DO_ALLOC(ALLOC_TABLE);
#undef ALLOC_TABLE

	{
	num phaseu, phased;
	#pragma omp parallel sections
	{
	#pragma omp section
	{
	if (sim->p.period_uneqlt > 0)
		for (int l = 0; l < L; l++)
			calciBu(l, N, exp_lambda, hs, inv_exp_Ku, iBu[l]);
	for (int l = 0; l < L; l++)
		calcBu(l, N, exp_lambda, hs, exp_Ku, Bu[l]);
	for (int f = 0; f < F; f++)
		mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L, 1.0,
		        Bu, Cu[f], N, tmpNN1u);
	calc_QdX_first(1, N, Cu[F - 1], &QdXLu[F - 1],
	               tmpN1u, pvtu, worku, lwork);
	for (int f = F - 2; f >= 0; f--)
		calc_QdX(1, N, Cu[f], &QdXLu[f + 1], &QdXLu[f],
		         tmpN1u, pvtu, worku, lwork);
	phaseu = calc_Gtt_last(1, N, &QdXLu[0], gu,
	                       tmpNN1u, tmpN1u, pvtu, worku, lwork);
	}
	#pragma omp section
	{
	if (sim->p.period_uneqlt > 0)
		for (int l = 0; l < L; l++)
			calciBd(l, N, exp_lambda, hs, inv_exp_Kd, iBd[l]);
	for (int l = 0; l < L; l++)
		calcBd(l, N, exp_lambda, hs, exp_Kd, Bd[l]);
	for (int f = 0; f < F; f++)
		mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L, 1.0,
		        Bd, Cd[f], N, tmpNN1d);
	calc_QdX_first(1, N, Cd[F - 1], &QdXLd[F - 1],
	               tmpN1d, pvtd, workd, lwork);
	for (int f = F - 2; f >= 0; f--)
		calc_QdX(1, N, Cd[f], &QdXLd[f + 1], &QdXLd[f],
		         tmpN1d, pvtd, workd, lwork);
	phased = calc_Gtt_last(1, N, &QdXLd[0], gd,
	                       tmpNN1d, tmpN1d, pvtd, workd, lwork);
	}
	}
	phase = phaseu*phased;
	}

	for (; sim->s.sweep < sim->p.n_sweep; sim->s.sweep++) {
		const int sig = sig_check_state(sim->s.sweep, sim->p.n_sweep_warm, sim->p.n_sweep);
		if (sig == 1) // stop flag
			break;
		else if (sig == 2) { // progress flag
			const int status = sim_data_save(sim);
			if (status < 0)
				fprintf(stderr, "save_file() failed: %d\n", status);
		}

		const int warmed_up = (sim->s.sweep >= sim->p.n_sweep_warm);
		const int enabled_eqlt = warmed_up && (sim->p.period_eqlt > 0);
		const int enabled_uneqlt = warmed_up && (sim->p.period_uneqlt > 0);

		for (int _l = 0; _l < L; _l++) {
			// on even sweeps, sweep up from l = 0 to L-1
			// on odd sweeps, sweep down from l = L-1 to 0
			const int sweep_up = (sim->s.sweep % 2 == 0);
			const int l = sweep_up ? _l : (L - 1 - _l);
			const int f = (l / n_matmul);
			const int m = (l % n_matmul);

			if (!sweep_up) { // wrap for down sweep
				#pragma omp parallel sections
				{
				#pragma omp section
				{
				profile_begin(wrap);
				matmul(tmpNN1u, gu, Bu[l]);
				matmul(gu, iBu[l], tmpNN1u);
				profile_end(wrap);
				}
				#pragma omp section
				{
				profile_begin(wrap);
				matmul(tmpNN1d, gd, Bd[l]);
				matmul(gd, iBd[l], tmpNN1d);
				profile_end(wrap);
				}
				}
			}

			profile_begin(updates);
			shuffle(rng, N, site_order);
			update_delayed(N, n_delay, del, site_order,
			               rng, hs + N*l, gu, gd, &phase,
			               tmpNN1u, tmpNN2u, tmpN1u,
			               tmpNN1d, tmpNN2d, tmpN1d);
			profile_end(updates);

			const int recalc = sweep_up ? (m == n_matmul - 1) : (m == 0);
			num phaseu, phased;
			#pragma omp parallel sections
			{
			#pragma omp section
			{
			profile_begin(calcb);
			calcBu(l, N, exp_lambda, hs, exp_Ku, Bu[l]);
			calciBu(l, N, exp_lambda, hs, inv_exp_Ku, iBu[l]);
			profile_end(calcb);
			if (recalc) {
				profile_begin(multb);
				mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L,
				        1.0, Bu, Cu[f], N, tmpNN1u);
				profile_end(multb);
				profile_begin(recalc);
				if (sweep_up) {
					if (f == 0)
						calc_QdX_first(0, N, Cu[f], &QdX0u[f],
						               tmpN1u, pvtu, worku, lwork);
					else
						calc_QdX(0, N, Cu[f], &QdX0u[f - 1], &QdX0u[f],
						         tmpN1u, pvtu, worku, lwork);
					if (f == F - 1)
						phaseu = calc_Gtt_last(0, N, &QdX0u[f], gu,
						                       tmpNN1u, tmpN1u, pvtu, worku, lwork);
					else
						phaseu = calc_Gtt(N, &QdX0u[f], &QdXLu[f + 1], gu,
						                  tmpNN1u, tmpNN2u, tmpN1u, tmpN2u, pvtu, worku, lwork);
				} else {
					if (f == F - 1)
						calc_QdX_first(1, N, Cu[f], &QdXLu[f],
						               tmpN1u, pvtu, worku, lwork);
					else
						calc_QdX(1, N, Cu[f], &QdXLu[f + 1], &QdXLu[f],
						         tmpN1u, pvtu, worku, lwork);
					if (f == 0)
						phaseu = calc_Gtt_last(1, N, &QdXLu[f], gu,
						                       tmpNN1u, tmpN1u, pvtu, worku, lwork);
					else
						phaseu = calc_Gtt(N, &QdX0u[f - 1], &QdXLu[f], gu,
						                  tmpNN1u, tmpNN2u, tmpN1u, tmpN2u, pvtu, worku, lwork);
				}
				profile_end(recalc);
			} else {
				if (sweep_up) { // wrap for up sweep
					profile_begin(wrap);
					matmul(tmpNN1u, gu, iBu[l]);
					matmul(gu, Bu[l], tmpNN1u);
					profile_end(wrap);
				}
			}
			}
			#pragma omp section
			{
			profile_begin(calcb);
			calcBd(l, N, exp_lambda, hs, exp_Kd, Bd[l]);
			calciBd(l, N, exp_lambda, hs, inv_exp_Kd, iBd[l]);
			profile_end(calcb);
			if (recalc) {
				profile_begin(multb);
				mul_seq(N, L, f*n_matmul, ((f + 1)*n_matmul) % L,
				        1.0, Bd, Cd[f], N, tmpNN1d);
				profile_end(multb);
				profile_begin(recalc);
				if (sweep_up) {
					if (f == 0)
						calc_QdX_first(0, N, Cd[f], &QdX0d[f],
						               tmpN1d, pvtd, workd, lwork);
					else
						calc_QdX(0, N, Cd[f], &QdX0d[f - 1], &QdX0d[f],
						         tmpN1d, pvtd, workd, lwork);
					if (f == F - 1)
						phased = calc_Gtt_last(0, N, &QdX0d[f], gd,
						                       tmpNN1d, tmpN1d, pvtd, workd, lwork);
					else
						phased = calc_Gtt(N, &QdX0d[f], &QdXLd[f + 1], gd,
						                  tmpNN1d, tmpNN2d, tmpN1d, tmpN2d, pvtd, workd, lwork);
				} else {
					if (f == F - 1)
						calc_QdX_first(1, N, Cd[f], &QdXLd[f],
						               tmpN1d, pvtd, workd, lwork);
					else
						calc_QdX(1, N, Cd[f], &QdXLd[f + 1], &QdXLd[f],
						         tmpN1d, pvtd, workd, lwork);
					if (f == 0)
						phased = calc_Gtt_last(1, N, &QdXLd[f], gd,
						                       tmpNN1d, tmpN1d, pvtd, workd, lwork);
					else
						phased = calc_Gtt(N, &QdX0d[f - 1], &QdXLd[f], gd,
						                  tmpNN1d, tmpNN2d, tmpN1d, tmpN2d, pvtd, workd, lwork);
				}
				profile_end(recalc);
			} else {
				if (sweep_up) {
					profile_begin(wrap);
					matmul(tmpNN1d, gd, iBd[l]);
					matmul(gd, Bd[l], tmpNN1d);
					profile_end(wrap);
				}
			}
			}
			}

			if (recalc) phase = phaseu*phased;

			if (enabled_eqlt && (l + sweep_up) % sim->p.period_eqlt == 0) {
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

		if (enabled_uneqlt && (sim->s.sweep % sim->p.period_uneqlt == 0) ) {
			#pragma omp parallel sections
			{
			#pragma omp section
			calc_ue_g(N, L, F, N_MUL, Bu, iBu, Cu, Gu0t, Gutt, Gut0,
			          Gredu, tauu, Qu, worku, lwork);
			#pragma omp section
			calc_ue_g(N, L, F, N_MUL, Bd, iBd, Cd, Gd0t, Gdtt, Gdt0,
			          Gredd, taud, Qd, workd, lwork);
			}

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
		}
	}

	pool_free(mp);
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
		status = sim_data_save(sim);
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

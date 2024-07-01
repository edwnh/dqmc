#include "dqmc.h"
#include <tgmath.h>
#include <stdio.h>
#include "data.h"
#include "greens.h"
#include "linalg.h"
#include "meas.h"
#include "mem.h"
#include "prof.h"
#include "rand.h"
#include "sig.h"
#include "time_.h"
#include "updates.h"

FILE *log_f;

static int dqmc(struct sim_data *sim)
{
	const int N = sim->p.N;
	// N*N matrices use padded ld*N storage for better alignment, slightly better performance.
	const int ld = best_ld(N); // ideally should use different ld different types... whatever
	__builtin_assume(ld % MEM_ALIGN_NUM == 0);

	const int L = sim->p.L;
	const int F = sim->p.F;
	const int n_matmul = sim->p.n_matmul;
	const int n_delay = sim->p.n_delay;
	uint64_t *const rng = sim->s.rng;
	int *const hs = sim->s.hs;

	// lapack work array size
	int lwork = get_lwork(N, ld);

#define ALLOC_TABLE(XX, FOR, ENDFOR) \
	XX(int *const site_order, N * sizeof(int)) \
	XX(double *const del, ld*2 * sizeof(double)) \
	XX(double *const exp_lambda, ld*2 * sizeof(double)) \
	XX(num *const inv_exp_Ku, ld*N * sizeof(num)) \
	XX(num **const iBu, L * sizeof(num *)) \
	FOR(l, L) \
		XX(iBu[l], ld*N * sizeof(num)) \
	ENDFOR \
	XX(num *const exp_Ku, ld*N * sizeof(num)) \
	XX(num *const exp_Vu, N * sizeof(num)) \
	XX(num **const Bu, L * sizeof(num *)) \
	FOR(l, L) \
		XX(Bu[l], ld*N * sizeof(num)) \
	ENDFOR \
	XX(num **const Cu, F * sizeof(num *)) \
	FOR(f, F) \
		XX(Cu[f], ld*N * sizeof(num)) \
	ENDFOR \
	XX(struct QdX *QdXLu, F * sizeof(struct QdX)) \
	XX(struct QdX *QdX0u, F * sizeof(struct QdX)) \
	FOR(f, F) \
		XX(QdXLu[f].Q, ld*N * sizeof(num)) \
		XX(QdXLu[f].d, N * sizeof(num)) \
		XX(QdXLu[f].X, ld*N * sizeof(num)) \
		XX(QdXLu[f].iL, ld*N * sizeof(num)) \
		XX(QdXLu[f].R, ld*N * sizeof(num)) \
		XX(QdX0u[f].Q, ld*N * sizeof(num)) \
		XX(QdX0u[f].d, N * sizeof(num)) \
		XX(QdX0u[f].X, ld*N * sizeof(num)) \
		XX(QdX0u[f].iL, ld*N * sizeof(num)) \
		XX(QdX0u[f].R, ld*N * sizeof(num)) \
	ENDFOR \
	XX(num *const gu, ld*N * sizeof(num)) \
	XX(num *const exp_halfKu, ld*N * sizeof(num)) \
	XX(num *const inv_exp_halfKu, ld*N * sizeof(num)) \
	XX(num *const inv_exp_Kd, ld*N * sizeof(num)) \
	XX(num **const iBd, L * sizeof(num *)) \
	FOR(l, L) \
		XX(iBd[l], ld*N * sizeof(num)) \
	ENDFOR \
	XX(num *const exp_Kd, ld*N * sizeof(num)) \
	XX(num *const exp_Vd, N * sizeof(num)) \
	XX(num **const Bd, L * sizeof(num *)) \
	FOR(l, L) \
		XX(Bd[l], ld*N * sizeof(num)) \
	ENDFOR \
	XX(num **const Cd, F * sizeof(num *)) \
	FOR(f, F) \
		XX(Cd[f], ld*N * sizeof(num)) \
	ENDFOR \
	XX(struct QdX *QdXLd, F * sizeof(struct QdX)) \
	XX(struct QdX *QdX0d, F * sizeof(struct QdX)) \
	FOR(f, F) \
		XX(QdXLd[f].Q, ld*N * sizeof(num)) \
		XX(QdXLd[f].d, N * sizeof(num)) \
		XX(QdXLd[f].X, ld*N * sizeof(num)) \
		XX(QdXLd[f].iL, ld*N * sizeof(num)) \
		XX(QdXLd[f].R, ld*N * sizeof(num)) \
		XX(QdX0d[f].Q, ld*N * sizeof(num)) \
		XX(QdX0d[f].d, N * sizeof(num)) \
		XX(QdX0d[f].X, ld*N * sizeof(num)) \
		XX(QdX0d[f].iL, ld*N * sizeof(num)) \
		XX(QdX0d[f].R, ld*N * sizeof(num)) \
	ENDFOR \
	XX(num *const gd, ld*N * sizeof(num)) \
	XX(num *const exp_halfKd, ld*N * sizeof(num)) \
	XX(num *const inv_exp_halfKd, ld*N * sizeof(num)) \
	XX(num *const tmpNN1u, ld*N * sizeof(num)) \
	XX(num *const tmpNN2u, ld*N * sizeof(num)) \
	XX(num *const tmpN1u, N * sizeof(num)) \
	XX(num *const tmpN2u, N * sizeof(num)) \
	XX(int *const pvtu, N * sizeof(int)) \
	XX(num *const tmpNN1d, ld*N * sizeof(num)) \
	XX(num *const tmpNN2d, ld*N * sizeof(num)) \
	XX(num *const tmpN1d, N * sizeof(num)) \
	XX(num *const tmpN2d, N * sizeof(num)) \
	XX(int *const pvtd, N * sizeof(int)) \
	XX(num *const worku, lwork * sizeof(num)) \
	XX(num *const workd, lwork * sizeof(num)) \
	XX(num **const Gu0t, (sim->p.period_uneqlt > 0)*L * sizeof(num *)) \
	XX(num **const Gutt, (sim->p.period_uneqlt > 0)*L * sizeof(num *)) \
	XX(num **const Gut0, (sim->p.period_uneqlt > 0)*L * sizeof(num *)) \
	FOR(l, (sim->p.period_uneqlt > 0)*L) \
		XX(Gu0t[l], ld*N * sizeof(num)) \
	ENDFOR\
	FOR(l, (sim->p.period_uneqlt > 0)*L) \
		XX(Gutt[l], ld*N * sizeof(num)) \
	ENDFOR\
	FOR(l, (sim->p.period_uneqlt > 0)*L) \
		XX(Gut0[l], ld*N * sizeof(num)) \
	ENDFOR \
	XX(num **const Gd0t, (sim->p.period_uneqlt > 0)*L * sizeof(num *)) \
	XX(num **const Gdtt, (sim->p.period_uneqlt > 0)*L * sizeof(num *)) \
	XX(num **const Gdt0, (sim->p.period_uneqlt > 0)*L * sizeof(num *)) \
	FOR(l, (sim->p.period_uneqlt > 0)*L) \
		XX(Gd0t[l], ld*N * sizeof(num)) \
	ENDFOR\
	FOR(l, (sim->p.period_uneqlt > 0)*L) \
		XX(Gdtt[l], ld*N * sizeof(num)) \
	ENDFOR\
	FOR(l, (sim->p.period_uneqlt > 0)*L) \
		XX(Gdt0[l], ld*N * sizeof(num)) \
	ENDFOR

	void *pool = my_calloc(POOL_GET_SIZE(ALLOC_TABLE));
	POOL_DO_ALLOC(pool, ALLOC_TABLE);
#undef ALLOC_TABLE

	// copy into matrices with leading dimension ld
	xomatcopy('N', N, 2, 1.0, sim->p.del,            N, del,            ld);
	xomatcopy('N', N, 2, 1.0, sim->p.exp_lambda,     N, exp_lambda,     ld);
	xomatcopy('N', N, N, 1.0, sim->p.exp_Ku,         N, exp_Ku,         ld);
	xomatcopy('N', N, N, 1.0, sim->p.inv_exp_Ku,     N, inv_exp_Ku,     ld);
	xomatcopy('N', N, N, 1.0, sim->p.exp_halfKu,     N, exp_halfKu,     ld);
	xomatcopy('N', N, N, 1.0, sim->p.inv_exp_halfKu, N, inv_exp_halfKu, ld);
	xomatcopy('N', N, N, 1.0, sim->p.exp_Kd,         N, exp_Kd,         ld);
	xomatcopy('N', N, N, 1.0, sim->p.inv_exp_Kd,     N, inv_exp_Kd,     ld);
	xomatcopy('N', N, N, 1.0, sim->p.exp_halfKd,     N, exp_halfKd,     ld);
	xomatcopy('N', N, N, 1.0, sim->p.inv_exp_halfKd, N, inv_exp_halfKd, ld);

	num phase;
	{
	for (int l = 0; l < L; l++) {
		for (int i = 0; i < N; i++) {
			const int hsil = hs[i + N*l];
			exp_Vu[i] = exp_lambda[i + ld*hsil];
			exp_Vd[i] = exp_lambda[i + ld*!hsil];
		}
		mul_mat_diag(N, ld, exp_Ku, exp_Vu, Bu[l]);
		mul_diag_mat(N, ld, exp_Vd, inv_exp_Ku, iBu[l]);
		mul_mat_diag(N, ld, exp_Kd, exp_Vd, Bd[l]);
		mul_diag_mat(N, ld, exp_Vu, inv_exp_Kd, iBd[l]);
	}
	num phaseu, phased;
	#pragma omp parallel sections
	{
	#pragma omp section
	{
	for (int f = 0; f < F; f++)
		mul_seq(N, f*n_matmul, (f + 1)*n_matmul, 1.0,
		        Bu, ld, Cu[f], ld, tmpNN1u);
	if (sim->s.sweep % 2 == 0) { // first sweep is up, initialize QdXL
		for (int f = F - 1; f >= 0; f--)
			calc_QdX(1, N, ld, Cu[f], (f == F - 1) ? NULL : &QdXLu[f + 1], &QdXLu[f],
			         tmpN1u, pvtu, worku, lwork);
		phaseu = calc_Gtt(N, ld, NULL, &QdXLu[0], gu, tmpNN1u, pvtu);
	} else { // first sweep is down, initialize QdX0
		for (int f = 0; f < F; f++)
			calc_QdX(0, N, ld, Cu[f], (f == 0) ? NULL : &QdX0u[f - 1], &QdX0u[f],
			         tmpN1u, pvtu, worku, lwork);
		phaseu = calc_Gtt(N, ld, &QdX0u[F - 1], NULL, gu, tmpNN1u, pvtu);
	}
	}
	#pragma omp section
	{
	for (int f = 0; f < F; f++)
		mul_seq(N, f*n_matmul, (f + 1)*n_matmul, 1.0,
		        Bd, ld, Cd[f], ld, tmpNN1d);
	if (sim->s.sweep % 2 == 0) { // first sweep is up, initialize QdXL
		for (int f = F - 1; f >= 0; f--)
			calc_QdX(1, N, ld, Cd[f], (f == F - 1) ? NULL : &QdXLd[f + 1], &QdXLd[f],
			         tmpN1d, pvtd, workd, lwork);
		phased = calc_Gtt(N, ld, NULL, &QdXLd[0], gd, tmpNN1d, pvtd);
	} else { // first sweep is down, initialize QdX0
		for (int f = 0; f < F; f++)
			calc_QdX(0, N, ld, Cd[f], (f == 0) ? NULL : &QdX0d[f - 1], &QdX0d[f],
			         tmpN1d, pvtd, workd, lwork);
		phased = calc_Gtt(N, ld, &QdX0d[F - 1], NULL, gd, tmpNN1d, pvtd);
	}
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
		const int sweep_up = (sim->s.sweep % 2 == 0);

		for (int _l = 0; _l < L; _l++) {
			// on even sweeps, sweep up from l = 0 to L-1
			// on odd sweeps, sweep down from l = L-1 to 0
			const int l = sweep_up ? _l : (L - 1 - _l);
			const int f = (l / n_matmul);
			const int m = (l % n_matmul);

			if (!sweep_up) { // wrap for down sweep
				#pragma omp parallel sections
				{
				#pragma omp section
				{
				profile_begin(wrap);
				wrap(N, ld, gu, iBu[l], Bu[l], tmpNN1u);
				profile_end(wrap);
				}
				#pragma omp section
				{
				profile_begin(wrap);
				wrap(N, ld, gd, iBd[l], Bd[l], tmpNN1d);
				profile_end(wrap);
				}
				}
			}

			profile_begin(updates);
			shuffle(rng, N, site_order);
			update_delayed(N, ld, n_delay, del, site_order,
			               rng, hs + N*l, gu, gd, &phase,
			               tmpNN1u, tmpNN2u, tmpN1u,
			               tmpNN1d, tmpNN2d, tmpN1d);
			for (int i = 0; i < N; i++) {
				const int hsil = hs[i + N*l];
				exp_Vu[i] = exp_lambda[i + ld*hsil];
				exp_Vd[i] = exp_lambda[i + ld*!hsil];
			}
			profile_end(updates);

			const int recalc = sweep_up ? (m == n_matmul - 1) : (m == 0);
			num phaseu, phased;
			#pragma omp parallel sections
			{
			#pragma omp section
			{
			profile_begin(calcb);
			mul_mat_diag(N, ld, exp_Ku, exp_Vu, Bu[l]);
			mul_diag_mat(N, ld, exp_Vd, inv_exp_Ku, iBu[l]);
			profile_end(calcb);
			if (recalc) {
				profile_begin(multb);
				mul_seq(N, f*n_matmul, (f + 1)*n_matmul,
				        1.0, Bu, ld, Cu[f], ld, tmpNN1u);
				profile_end(multb);
				profile_begin(recalc);
				if (sweep_up) {
					calc_QdX(0, N, ld, Cu[f], (f == 0) ? NULL : &QdX0u[f - 1], &QdX0u[f],
					         tmpN1u, pvtu, worku, lwork);
					phaseu = calc_Gtt(N, ld, &QdX0u[f], (f == F - 1) ? NULL : &QdXLu[f + 1], gu, tmpNN1u, pvtu);
				} else {
					calc_QdX(1, N, ld, Cu[f], (f == F - 1) ? NULL : &QdXLu[f + 1], &QdXLu[f],
					         tmpN1u, pvtu, worku, lwork);
					phaseu = calc_Gtt(N, ld, (f == 0) ? NULL : &QdX0u[f - 1], &QdXLu[f], gu, tmpNN1u, pvtu);
				}
				profile_end(recalc);
			} else {
				if (sweep_up) {// wrap for up sweep
					profile_begin(wrap);
					wrap(N, ld, gu, Bu[l], iBu[l], tmpNN1u);
					profile_end(wrap);
				}
			}
			}
			#pragma omp section
			{
			profile_begin(calcb);
			mul_mat_diag(N, ld, exp_Kd, exp_Vd, Bd[l]);
			mul_diag_mat(N, ld, exp_Vu, inv_exp_Kd, iBd[l]);
			profile_end(calcb);
			if (recalc) {
				profile_begin(multb);
				mul_seq(N, f*n_matmul, (f + 1)*n_matmul,
				        1.0, Bd, ld, Cd[f], ld, tmpNN1d);
				profile_end(multb);
				profile_begin(recalc);
				if (sweep_up) {
					calc_QdX(0, N, ld, Cd[f], (f == 0) ? NULL : &QdX0d[f - 1], &QdX0d[f],
					         tmpN1d, pvtd, workd, lwork);
					phased = calc_Gtt(N, ld, &QdX0d[f], (f == F - 1) ? NULL : &QdXLd[f + 1], gd, tmpNN1d, pvtd);
				} else {
					calc_QdX(1, N, ld, Cd[f], (f == F - 1) ? NULL : &QdXLd[f + 1], &QdXLd[f],
					         tmpN1d, pvtd, workd, lwork);
					phased = calc_Gtt(N, ld, (f == 0) ? NULL : &QdX0d[f - 1], &QdXLd[f], gd, tmpNN1d, pvtd);
				}
				profile_end(recalc);
			} else {
				if (sweep_up) {
					profile_begin(wrap);
					wrap(N, ld, gd, Bd[l], iBd[l], tmpNN1d);
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
				xgemm("N", "N", N, N, N, 1.0, gu, ld, exp_halfKu, ld, 0.0, tmpNN1u, ld);
				xgemm("N", "N", N, N, N, 1.0, inv_exp_halfKu, ld, tmpNN1u, ld, 0.0, tmpNN2u, ld);
				profile_end(half_wrap);
				}
				#pragma omp section
				{
				profile_begin(half_wrap);
				xgemm("N", "N", N, N, N, 1.0, gd, ld, exp_halfKd, ld, 0.0, tmpNN1d, ld);
				xgemm("N", "N", N, N, N, 1.0, inv_exp_halfKd, ld, tmpNN1d, ld, 0.0, tmpNN2d, ld);
				profile_end(half_wrap);
				}
				}
				profile_begin(meas_eq);
				measure_eqlt(&sim->p, phase, ld, tmpNN2u, tmpNN2d, &sim->m_eq);
				profile_end(meas_eq);
			}
		}

		if (enabled_uneqlt && (sim->s.sweep % sim->p.period_uneqlt == 0) ) {
			#pragma omp parallel sections
			{
			#pragma omp section
			{
			profile_begin(calc_ue);
			xomatcopy('N', N, N, 1.0, gu, ld, Gu0t[0], ld);
			xomatcopy('N', N, N, 1.0, gu, ld, Gutt[0], ld);
			xomatcopy('N', N, N, 1.0, gu, ld, Gut0[0], ld);
			if (sweep_up) { // then QdX0 is fresh, QdXL is old
				for (int f = F - 1; f >= 0; f--)
					calc_QdX(1, N, ld, Cu[f], (f == F - 1) ? NULL : &QdXLu[f + 1], &QdXLu[f],
					         tmpN1u, pvtu, worku, lwork);
			} else {
				for (int f = 0; f < F; f++)
					calc_QdX(0, N, ld, Cu[f], (f == 0) ? NULL : &QdX0u[f - 1], &QdX0u[f],
					         tmpN1u, pvtu, worku, lwork);
			}
			calc_ue_g(N, ld, L, F, n_matmul, Bu, iBu, QdX0u, QdXLu, Gu0t, Gutt, Gut0, tmpNN1u, pvtu);
			profile_end(calc_ue);
			profile_begin(half_wrap_ue);
			wrap(N, ld, Gu0t[0], inv_exp_halfKu, exp_halfKu, tmpNN1u);
			xomatcopy('N', N, N, 1.0, Gu0t[0], ld, Gutt[0], ld);
			xomatcopy('N', N, N, 1.0, Gu0t[0], ld, Gut0[0], ld);
			for (int l = 1; l < L; l++) {
				wrap(N, ld, Gu0t[l], inv_exp_halfKu, exp_halfKu, tmpNN1d);
				wrap(N, ld, Gutt[l], inv_exp_halfKu, exp_halfKu, tmpNN1d);
				wrap(N, ld, Gut0[l], inv_exp_halfKu, exp_halfKu, tmpNN1d);
			}
			profile_end(half_wrap_ue);
			}
			#pragma omp section
			{
			profile_begin(calc_ue);
			xomatcopy('N', N, N, 1.0, gd, ld, Gd0t[0], ld);
			xomatcopy('N', N, N, 1.0, gd, ld, Gdtt[0], ld);
			xomatcopy('N', N, N, 1.0, gd, ld, Gdt0[0], ld);
			if (sweep_up) { // then QdX0 is fresh, QdXL is old
				for (int f = F - 1; f >= 0; f--)
					calc_QdX(1, N, ld, Cd[f], (f == F - 1) ? NULL : &QdXLd[f + 1], &QdXLd[f],
							tmpN1d, pvtd, workd, lwork);
			} else {
				for (int f = 0; f < F; f++)
					calc_QdX(0, N, ld, Cd[f], (f == 0) ? NULL : &QdX0d[f - 1], &QdX0d[f],
							tmpN1d, pvtd, workd, lwork);
			}
			calc_ue_g(N, ld, L, F, n_matmul, Bd, iBd, QdX0d, QdXLd, Gd0t, Gdtt, Gdt0, tmpNN1d, pvtd);
			profile_end(calc_ue);
			profile_begin(half_wrap_ue);
			wrap(N, ld, Gd0t[0], inv_exp_halfKd, exp_halfKd, tmpNN1d);
			xomatcopy('N', N, N, 1.0, Gd0t[0], ld, Gdtt[0], ld);
			xomatcopy('N', N, N, 1.0, Gd0t[0], ld, Gdt0[0], ld);
			for (int l = 1; l < L; l++) {
				wrap(N, ld, Gd0t[l], inv_exp_halfKd, exp_halfKd, tmpNN1d);
				wrap(N, ld, Gdtt[l], inv_exp_halfKd, exp_halfKd, tmpNN1d);
				wrap(N, ld, Gdt0[l], inv_exp_halfKd, exp_halfKd, tmpNN1d);
			}
			profile_end(half_wrap_ue);
			}
			}
			profile_begin(meas_uneq);
			// GuXX is actually a pointer to pointer, but below we
			// abuse the fact that GuXX[l] is allocated contiguously.
			// compiler optimizes better this way than if measure_uneqlt
			// takes GuXX and GdXX as arguments.
			measure_uneqlt(&sim->p, phase, ld,
			               Gu0t[0], Gutt[0], Gut0[0], Gd0t[0], Gdtt[0], Gdt0[0],
			               &sim->m_ue);
			profile_end(meas_uneq);
		}
	}

	my_free(pool);
	return 0;
}

static void print_cpu_model(void)
{
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

	// initialize signal handling
	sig_init(wall_start, save_interval, max_time);

	// open and read simulation file
	struct sim_data *sim = my_calloc(sizeof(struct sim_data));
	fprintf(log_f, "opening %s\n", sim_file);
	status = sim_data_read_alloc(sim, sim_file);
	if (status < 0) {
		fprintf(stderr, "read_file() failed: %d\n", status);
		status = -1;
		goto cleanup;
	}

	// check existing progress
	fprintf(log_f, "%d/%d sweeps completed\n", sim->s.sweep, sim->p.n_sweep);
	if (sim->s.sweep >= sim->p.n_sweep) {
		fprintf(log_f, "already finished\n");
		goto cleanup;
	}

	// run dqmc
	fprintf(log_f, "starting dqmc\n");
	status = dqmc(sim);
	if (status < 0) {
		fprintf(stderr, "dqmc() failed to allocate memory\n");
		status = -1;
		goto cleanup;
	}
	fprintf(log_f, "%d/%d sweeps completed\n", sim->s.sweep, sim->p.n_sweep);

	// save to simulation file (if not in benchmarking mode)
	if (!bench) {
		fprintf(log_f, "saving data\n");
		status = sim_data_save(sim);
		if (status < 0) {
			fprintf(stderr, "save_file() failed: %d\n", status);
			status = -1;
			goto cleanup;
		}
	} else {
		fprintf(log_f, "benchmark mode enabled; not saving data\n");
	}

	status = (sim->s.sweep == sim->p.n_sweep) ? 0 : 1;

cleanup:
	sim_data_free(sim);
	my_free(sim);

	const tick_t wall_time = time_wall() - wall_start;
	print_cpu_model();
	fprintf(log_f, "wall time: %.3f\n", wall_time * SEC_PER_TICK);
	profile_print(wall_time);

	if (log_f != stdout)
		fclose(log_f);

	return status;
}

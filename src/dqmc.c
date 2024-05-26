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

#define matmul(C, A, B) do { \
	xgemm("N", "N", N, N, N, 1.0, (A), ld, (B), ld, 0.0, (C), ld); \
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

// B = A diag(d)
static inline void mul_mat_diag(const int N, const int ld, const num *const A, const num *const d, num *const B)
{
	__builtin_assume(ld % MEM_ALIGN_NUM == 0);
	(void)__builtin_assume_aligned(A, MEM_ALIGN);
	(void)__builtin_assume_aligned(d, MEM_ALIGN);
	(void)__builtin_assume_aligned(B, MEM_ALIGN);

	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			B[i + ld*j] = A[i + ld*j] * d[j];
}


// B = diag(d) A
static inline void mul_diag_mat(const int N, const int ld, const num *const d, const num *const A, num *const B)
{
	__builtin_assume(ld % MEM_ALIGN_NUM == 0);
	(void)__builtin_assume_aligned(d, MEM_ALIGN);
	(void)__builtin_assume_aligned(A, MEM_ALIGN);
	(void)__builtin_assume_aligned(B, MEM_ALIGN);

	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			B[i + ld*j] = d[i] * A[i + ld*j];
}

static int dqmc(struct sim_data *sim)
{
	const int N = sim->p.N;
	// N*N matrices use padded ld*N storage for better alignment, slightly better performance.
	const int ld = mem_best_ld(N); // ideally should use different ld different types... whatever
	__builtin_assume(ld % MEM_ALIGN_NUM == 0);

	const int L = sim->p.L;
	const int F = sim->p.F;
	const int E = 1 + (F - 1) / 2;
	const int n_matmul = sim->p.n_matmul;
	const int n_delay = sim->p.n_delay;
	uint64_t *const rng = sim->s.rng;
	int *const hs = sim->s.hs;

	// lapack work array size
	int lwork = get_lwork(N, ld);

#define ALLOC_TABLE(XX, FOR, ENDFOR) \
	XX(int *const site_order, mp, N * sizeof(int)) \
	XX(double *const del, mp, ld*2 * sizeof(double)) \
	XX(double *const exp_lambda, mp, ld*2 * sizeof(double)) \
	XX(num *const inv_exp_Ku, mp, ld*N * sizeof(num)) \
	XX(num **const iBu, mp, L * sizeof(num *)) \
	FOR(l, L) \
		XX(iBu[l], mp, ld*N * sizeof(num)) \
	ENDFOR \
	XX(num *const exp_Ku, mp, ld*N * sizeof(num)) \
	XX(num *const exp_Vu, mp, N * sizeof(num)) \
	XX(num **const Bu, mp, L * sizeof(num *)) \
	FOR(l, L) \
		XX(Bu[l], mp, ld*N * sizeof(num)) \
	ENDFOR \
	XX(num **const Cu, mp, F * sizeof(num *)) \
	FOR(f, F) \
		XX(Cu[f], mp, ld*N * sizeof(num)) \
	ENDFOR \
	XX(struct QdX *QdXLu, mp, F * sizeof(struct QdX)) \
	XX(struct QdX *QdX0u, mp, F * sizeof(struct QdX)) \
	FOR(f, F) \
		XX(QdXLu[f].Q, mp, ld*N * sizeof(num)) \
		XX(QdXLu[f].d, mp, N * sizeof(num)) \
		XX(QdXLu[f].X, mp, ld*N * sizeof(num)) \
		XX(QdXLu[f].iL, mp, ld*N * sizeof(num)) \
		XX(QdXLu[f].R, mp, ld*N * sizeof(num)) \
		XX(QdX0u[f].Q, mp, ld*N * sizeof(num)) \
		XX(QdX0u[f].d, mp, N * sizeof(num)) \
		XX(QdX0u[f].X, mp, ld*N * sizeof(num)) \
		XX(QdX0u[f].iL, mp, ld*N * sizeof(num)) \
		XX(QdX0u[f].R, mp, ld*N * sizeof(num)) \
	ENDFOR \
	XX(num *const gu, mp, ld*N * sizeof(num)) \
	XX(num *const exp_halfKu, mp, ld*N * sizeof(num)) \
	XX(num *const inv_exp_halfKu, mp, ld*N * sizeof(num)) \
	XX(num *const inv_exp_Kd, mp, ld*N * sizeof(num)) \
	XX(num **const iBd, mp, L * sizeof(num *)) \
	FOR(l, L) \
		XX(iBd[l], mp, ld*N * sizeof(num)) \
	ENDFOR \
	XX(num *const exp_Kd, mp, ld*N * sizeof(num)) \
	XX(num *const exp_Vd, mp, N * sizeof(num)) \
	XX(num **const Bd, mp, L * sizeof(num *)) \
	FOR(l, L) \
		XX(Bd[l], mp, ld*N * sizeof(num)) \
	ENDFOR \
	XX(num **const Cd, mp, F * sizeof(num *)) \
	FOR(f, F) \
		XX(Cd[f], mp, ld*N * sizeof(num)) \
	ENDFOR \
	XX(struct QdX *QdXLd, mp, F * sizeof(struct QdX)) \
	XX(struct QdX *QdX0d, mp, F * sizeof(struct QdX)) \
	FOR(f, F) \
		XX(QdXLd[f].Q, mp, ld*N * sizeof(num)) \
		XX(QdXLd[f].d, mp, N * sizeof(num)) \
		XX(QdXLd[f].X, mp, ld*N * sizeof(num)) \
		XX(QdXLd[f].iL, mp, ld*N * sizeof(num)) \
		XX(QdXLd[f].R, mp, ld*N * sizeof(num)) \
		XX(QdX0d[f].Q, mp, ld*N * sizeof(num)) \
		XX(QdX0d[f].d, mp, N * sizeof(num)) \
		XX(QdX0d[f].X, mp, ld*N * sizeof(num)) \
		XX(QdX0d[f].iL, mp, ld*N * sizeof(num)) \
		XX(QdX0d[f].R, mp, ld*N * sizeof(num)) \
	ENDFOR \
	XX(num *const gd, mp, ld*N * sizeof(num)) \
	XX(num *const exp_halfKd, mp, ld*N * sizeof(num)) \
	XX(num *const inv_exp_halfKd, mp, ld*N * sizeof(num)) \
	XX(num *const tmpNN1u, mp, ld*N * sizeof(num)) \
	XX(num *const tmpNN2u, mp, ld*N * sizeof(num)) \
	XX(num *const tmpN1u, mp, N * sizeof(num)) \
	XX(num *const tmpN2u, mp, N * sizeof(num)) \
	XX(int *const pvtu, mp, N * sizeof(int)) \
	XX(num *const tmpNN1d, mp, ld*N * sizeof(num)) \
	XX(num *const tmpNN2d, mp, ld*N * sizeof(num)) \
	XX(num *const tmpN1d, mp, N * sizeof(num)) \
	XX(num *const tmpN2d, mp, N * sizeof(num)) \
	XX(int *const pvtd, mp, N * sizeof(int)) \
	XX(num *const worku, mp, lwork * sizeof(num)) \
	XX(num *const workd, mp, lwork * sizeof(num)) \
	XX(num **const Gu0t, mp, (sim->p.period_uneqlt > 0)*L * sizeof(num *)) \
	XX(num **const Gutt, mp, (sim->p.period_uneqlt > 0)*L * sizeof(num *)) \
	XX(num **const Gut0, mp, (sim->p.period_uneqlt > 0)*L * sizeof(num *)) \
	FOR(l, (sim->p.period_uneqlt > 0)*L) \
		XX(Gu0t[l], mp, ld*N * sizeof(num)) \
	ENDFOR\
	FOR(l, (sim->p.period_uneqlt > 0)*L) \
		XX(Gutt[l], mp, ld*N * sizeof(num)) \
	ENDFOR\
	FOR(l, (sim->p.period_uneqlt > 0)*L) \
		XX(Gut0[l], mp, ld*N * sizeof(num)) \
	ENDFOR \
	XX(num **const Gd0t, mp, (sim->p.period_uneqlt > 0)*L * sizeof(num *)) \
	XX(num **const Gdtt, mp, (sim->p.period_uneqlt > 0)*L * sizeof(num *)) \
	XX(num **const Gdt0, mp, (sim->p.period_uneqlt > 0)*L * sizeof(num *)) \
	FOR(l, (sim->p.period_uneqlt > 0)*L) \
		XX(Gd0t[l], mp, ld*N * sizeof(num)) \
	ENDFOR\
	FOR(l, (sim->p.period_uneqlt > 0)*L) \
		XX(Gdtt[l], mp, ld*N * sizeof(num)) \
	ENDFOR\
	FOR(l, (sim->p.period_uneqlt > 0)*L) \
		XX(Gdt0[l], mp, ld*N * sizeof(num)) \
	ENDFOR

	struct mem_pool *mp = pool_new(POOL_GET_SIZE(ALLOC_TABLE));
	POOL_DO_ALLOC(ALLOC_TABLE);
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
		calc_QdX_first(1, N, ld, Cu[F - 1], &QdXLu[F - 1],
		               tmpN1u, pvtu, worku, lwork);
		for (int f = F - 2; f >= 0; f--)
			calc_QdX(1, N, ld, Cu[f], &QdXLu[f + 1], &QdXLu[f],
			         tmpN1u, pvtu, worku, lwork);
		phaseu = calc_Gtt_last(1, N, ld, &QdXLu[0], gu, tmpNN1u, pvtu);
	} else { // first sweep is down, initialize QdX0
		calc_QdX_first(0, N, ld, Cu[0], &QdX0u[0],
		               tmpN1u, pvtu, worku, lwork);
		for (int f = 1; f < F; f++)
			calc_QdX(0, N, ld, Cu[f], &QdX0u[f - 1], &QdX0u[f],
			         tmpN1u, pvtu, worku, lwork);
		phaseu = calc_Gtt_last(0, N, ld, &QdX0u[F - 1], gu, tmpNN1u, pvtu);
	}
	}
	#pragma omp section
	{
	for (int f = 0; f < F; f++)
		mul_seq(N, f*n_matmul, (f + 1)*n_matmul, 1.0,
		        Bd, ld, Cd[f], ld, tmpNN1d);
	if (sim->s.sweep % 2 == 0) { // first sweep is up, initialize QdXL
		calc_QdX_first(1, N, ld, Cd[F - 1], &QdXLd[F - 1],
		               tmpN1d, pvtd, workd, lwork);
		for (int f = F - 2; f >= 0; f--)
			calc_QdX(1, N, ld, Cd[f], &QdXLd[f + 1], &QdXLd[f],
			         tmpN1d, pvtd, workd, lwork);
		phased = calc_Gtt_last(1, N, ld, &QdXLd[0], gd, tmpNN1d, pvtd);
	} else { // first sweep is down, initialize QdX0
		calc_QdX_first(0, N, ld, Cd[0], &QdX0d[0],
		               tmpN1d, pvtd, workd, lwork);
		for (int f = 1; f < F; f++)
			calc_QdX(0, N, ld, Cd[f], &QdX0d[f - 1], &QdX0d[f],
			         tmpN1d, pvtd, workd, lwork);
		phased = calc_Gtt_last(0, N, ld, &QdX0d[F - 1], gd, tmpNN1d, pvtd);
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
					if (f == 0)
						calc_QdX_first(0, N, ld, Cu[f], &QdX0u[f],
						               tmpN1u, pvtu, worku, lwork);
					else
						calc_QdX(0, N, ld, Cu[f], &QdX0u[f - 1], &QdX0u[f],
						         tmpN1u, pvtu, worku, lwork);
					if (f == F - 1)
						phaseu = calc_Gtt_last(0, N, ld, &QdX0u[f], gu, tmpNN1u, pvtu);
					else
						phaseu = calc_Gtt(N, ld, &QdX0u[f], &QdXLu[f + 1], gu, tmpNN1u, pvtu);
				} else {
					if (f == F - 1)
						calc_QdX_first(1, N, ld, Cu[f], &QdXLu[f],
						               tmpN1u, pvtu, worku, lwork);
					else
						calc_QdX(1, N, ld, Cu[f], &QdXLu[f + 1], &QdXLu[f],
						         tmpN1u, pvtu, worku, lwork);
					if (f == 0)
						phaseu = calc_Gtt_last(1, N, ld, &QdXLu[f], gu, tmpNN1u, pvtu);
					else
						phaseu = calc_Gtt(N, ld, &QdX0u[f - 1], &QdXLu[f], gu, tmpNN1u, pvtu);
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
					if (f == 0)
						calc_QdX_first(0, N, ld, Cd[f], &QdX0d[f],
						               tmpN1d, pvtd, workd, lwork);
					else
						calc_QdX(0, N, ld, Cd[f], &QdX0d[f - 1], &QdX0d[f],
						         tmpN1d, pvtd, workd, lwork);
					if (f == F - 1)
						phased = calc_Gtt_last(0, N, ld, &QdX0d[f], gd, tmpNN1d, pvtd);
					else
						phased = calc_Gtt(N, ld, &QdX0d[f], &QdXLd[f + 1], gd, tmpNN1d, pvtd);
				} else {
					if (f == F - 1)
						calc_QdX_first(1, N, ld, Cd[f], &QdXLd[f],
						               tmpN1d, pvtd, workd, lwork);
					else
						calc_QdX(1, N, ld, Cd[f], &QdXLd[f + 1], &QdXLd[f],
						         tmpN1d, pvtd, workd, lwork);
					if (f == 0)
						phased = calc_Gtt_last(1, N, ld, &QdXLd[f], gd, tmpNN1d, pvtd);
					else
						phased = calc_Gtt(N, ld, &QdX0d[f - 1], &QdXLd[f], gd, tmpNN1d, pvtd);
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
				measure_eqlt(&sim->p, phase, ld, tmpNN2u, tmpNN2d, &sim->m_eq);
				profile_end(meas_eq);
			}
		}

		if (enabled_uneqlt && (sim->s.sweep % sim->p.period_uneqlt == 0) ) {
			// todo: skip this by setting GuXX[0] = tmpNN2u if last eqlt meas was at l=0
			xomatcopy('N', N, N, 1.0, gu, ld, Gu0t[0], ld);
			xomatcopy('N', N, N, 1.0, gu, ld, Gutt[0], ld);
			xomatcopy('N', N, N, 1.0, gu, ld, Gut0[0], ld);
			xomatcopy('N', N, N, 1.0, gd, ld, Gd0t[0], ld);
			xomatcopy('N', N, N, 1.0, gd, ld, Gdtt[0], ld);
			xomatcopy('N', N, N, 1.0, gd, ld, Gdt0[0], ld);

			#pragma omp parallel sections
			{
			#pragma omp section
			{
			profile_begin(calc_ue);
			if (sweep_up) { // then QdX0 is fresh, QdXL is old
				calc_QdX_first(1, N, ld, Cu[F - 1], &QdXLu[F - 1],
							tmpN1u, pvtu, worku, lwork);
				for (int f = F - 2; f >= 0; f--)
					calc_QdX(1, N, ld, Cu[f], &QdXLu[f + 1], &QdXLu[f],
							tmpN1u, pvtu, worku, lwork);
			} else {
				calc_QdX_first(0, N, ld, Cu[0], &QdX0u[0],
							tmpN1u, pvtu, worku, lwork);
				for (int f = 1; f < F; f++)
					calc_QdX(0, N, ld, Cu[f], &QdX0u[f - 1], &QdX0u[f],
							tmpN1u, pvtu, worku, lwork);
			}
			calc_ue_g(N, ld, L, F, n_matmul, Bu, iBu, QdX0u, QdXLu, Gu0t, Gutt, Gut0, tmpNN1u, pvtu);
			profile_end(calc_ue);
			profile_begin(half_wrap);
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1u, Gu0t[l], exp_halfKu);
				matmul(Gu0t[l], inv_exp_halfKu, tmpNN1u);
			}
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1u, Gutt[l], exp_halfKu);
				matmul(Gutt[l], inv_exp_halfKu, tmpNN1u);
			}
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1u, Gut0[l], exp_halfKu);
				matmul(Gut0[l], inv_exp_halfKu, tmpNN1u);
			}
			profile_end(half_wrap);
			}
			#pragma omp section
			{
			profile_begin(calc_ue);
			if (sweep_up) { // then QdX0 is fresh, QdXL is old
				calc_QdX_first(1, N, ld, Cd[F - 1], &QdXLd[F - 1],
							tmpN1d, pvtd, workd, lwork);
				for (int f = F - 2; f >= 0; f--)
					calc_QdX(1, N, ld, Cd[f], &QdXLd[f + 1], &QdXLd[f],
							tmpN1d, pvtd, workd, lwork);
			} else {
				calc_QdX_first(0, N, ld, Cd[0], &QdX0d[0],
							tmpN1d, pvtd, workd, lwork);
				for (int f = 1; f < F; f++)
					calc_QdX(0, N, ld, Cd[f], &QdX0d[f - 1], &QdX0d[f],
							tmpN1d, pvtd, workd, lwork);
			}
			calc_ue_g(N, ld, L, F, n_matmul, Bd, iBd, QdX0d, QdXLd, Gd0t, Gdtt, Gdt0, tmpNN1d, pvtd);
			profile_end(calc_ue);
			profile_begin(half_wrap);
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1d, Gd0t[l], exp_halfKd);
				matmul(Gd0t[l], inv_exp_halfKd, tmpNN1d);
			}
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1d, Gdtt[l], exp_halfKd);
				matmul(Gdtt[l], inv_exp_halfKd, tmpNN1d);
			}
			for (int l = 0; l < L; l++) {
				matmul(tmpNN1d, Gdt0[l], exp_halfKd);
				matmul(Gdt0[l], inv_exp_halfKd, tmpNN1d);
			}
			profile_end(half_wrap);
			}
			}
			profile_begin(meas_uneq);
			// abuse fact that GuXX[l] is allocated contiguously
			measure_uneqlt(&sim->p, phase, ld,
			               Gu0t[0], Gutt[0], Gut0[0], Gd0t[0], Gdtt[0], Gdt0[0],
			               &sim->m_ue);
			profile_end(meas_uneq);
		}
	}

	pool_free(mp);
	return 0;
}

static void print_cpu_model(FILE *log)
{
	FILE *cpuinfo = fopen("/proc/cpuinfo", "r");
	if (!cpuinfo) {
		fprintf(log, "couldn't open /proc/cpuinfo\n");
		return;
	}

	char *line = NULL;
	size_t len = 0;
	while (getline(&line, &len, cpuinfo) != -1) {
		if (strncmp(line, "model name", 10) == 0) {
			fprintf(log, "cpu %s", line);
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
	FILE *log = (log_file != NULL) ? fopen(log_file, "a") : stdout;
	if (log == NULL) {
		fprintf(stderr, "fopen() failed to open: %s\n", log_file);
		return -1;
	}

	fprintf(log, "commit id %s\n", GIT_ID);
	fprintf(log, "compiled on %s %s\n", __DATE__, __TIME__);

	// initialize signal handling
	sig_init(log, wall_start, save_interval, max_time);

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
	print_cpu_model(log);
	fprintf(log, "wall time: %.3f\n", wall_time * SEC_PER_TICK);
	profile_print(log, wall_time);

	if (log != stdout)
		fclose(log);
	else
		fflush(log);

	return status;
}

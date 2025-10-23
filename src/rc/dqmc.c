#include <tgmath.h>
#include "data.h"
#include "greens.h"
#include "linalg.h"
#include "meas.h"
#include "prof.h"
#include "rand.h"
#include "sig.h"
#include "updates.h"
#include "wrapper.h"

#define N_DOF 2

// returns -1 for failure, 0 for completion, 1 for partial completion
int RC(dqmc)(struct RC(sim_data) *sim)
{
	// check existing progress
	fprintf(log_f, "%d/%d sweeps completed\n", sim->s.sweep, sim->p.n_sweep);
	if (sim->s.sweep >= sim->p.n_sweep) {
		fprintf(log_f, "already finished\n");
		return 0;
	}

	const int N = sim->p.N;
	// N*N matrices use padded ld*N storage for better alignment, slightly better performance.
	const int ld = best_ld(N);
	__builtin_assume(ld % MEM_ALIGN_NUM == 0);
	int lwork = RC(get_lwork)(N, ld); // lapack work array size

	const int L = sim->p.L;
	const int F = sim->p.F;
	const int n_matmul = sim->p.n_matmul;
	const int n_delay = sim->p.n_delay;
	uint64_t *const rng = sim->s.rng;
	int *const hs = sim->s.hs;
	const double *const del = sim->p.del;
	const double *const exp_lambda = sim->p.exp_lambda;

	struct RC(workspace) *w[N_DOF] = {&sim->up, &sim->dn};

	int site_order[N]; // N should be small enough to put on stack

	const struct RC(QdX) QdX_NULL = {NULL, NULL, NULL};
	const struct RC(LR) LR_NULL = {NULL, NULL, NULL};
	#define QdX0(f) (((f) < 0 || (f) >= F) ? QdX_NULL : \
		(struct RC(QdX)){w[s]->Q_0 + (f)*ld*N, w[s]->d_0 + (f)*ld, w[s]->X_0 + (f)*ld*N})
	#define QdXL(f) (((f) < 0 || (f) >= F) ? QdX_NULL : \
		(struct RC(QdX)){w[s]->Q_L + (f)*ld*N, w[s]->d_L + (f)*ld, w[s]->X_L + (f)*ld*N})
	#define LR0(f) (((f) < 0 || (f) >= F) ? LR_NULL : \
		(struct RC(LR)){w[s]->iL_0 + (f)*ld*N, w[s]->R_0 + (f)*ld*N, w[s]->phase_iL_0 + (f)})
	#define LRL(f) (((f) < 0 || (f) >= F) ? LR_NULL : \
		(struct RC(LR)){w[s]->iL_L + (f)*ld*N, w[s]->R_L + (f)*ld*N, w[s]->phase_iL_L + (f)})

	// copy into matrices with leading dimension ld
	xomatcopy('N', N, N, 1.0, sim->p.exp_Ku,         N, w[0]->exp_K,         ld);
	xomatcopy('N', N, N, 1.0, sim->p.inv_exp_Ku,     N, w[0]->inv_exp_K,     ld);
	xomatcopy('N', N, N, 1.0, sim->p.exp_halfKu,     N, w[0]->exp_halfK,     ld);
	xomatcopy('N', N, N, 1.0, sim->p.inv_exp_halfKu, N, w[0]->inv_exp_halfK, ld);
	xomatcopy('N', N, N, 1.0, sim->p.exp_Kd,         N, w[1]->exp_K,         ld);
	xomatcopy('N', N, N, 1.0, sim->p.inv_exp_Kd,     N, w[1]->inv_exp_K,     ld);
	xomatcopy('N', N, N, 1.0, sim->p.exp_halfKd,     N, w[1]->exp_halfK,     ld);
	xomatcopy('N', N, N, 1.0, sim->p.inv_exp_halfKd, N, w[1]->inv_exp_halfK, ld);

	num phase;
	num phases[N_DOF] = {0};

	for (int l = 0; l < L; l++) {
		for (int i = 0; i < N; i++) {
			const int hsil = hs[i + N*l];
			w[0]->exp_V[i] = exp_lambda[i + N*hsil];
			w[1]->exp_V[i] = exp_lambda[i + N*!hsil];
		}
		mul_mat_diag(N, ld, w[0]->exp_K, w[0]->exp_V,     w[0]->B + l*ld*N);
		mul_diag_mat(N, ld, w[1]->exp_V, w[0]->inv_exp_K, w[0]->iB + l*ld*N);
		mul_mat_diag(N, ld, w[1]->exp_K, w[1]->exp_V,     w[1]->B + l*ld*N);
		mul_diag_mat(N, ld, w[0]->exp_V, w[1]->inv_exp_K, w[1]->iB + l*ld*N);
	}

	#pragma omp parallel for schedule(static, 1)
	for (int s = 0; s < N_DOF; s++) {
		for (int f = 0; f < F; f++)
			RC(mul_seq)(N, ld, f*n_matmul, (f + 1)*n_matmul, 1.0,
					w[s]->B, w[s]->C + f*ld*N, w[s]->tmpNN1);
		if (sim->s.sweep % 2 == 0) { // first sweep is up, initialize QdXL
			for (int f = F - 1; f >= 0; f--)
				RC(calc_QdX)(1, N, ld, w[s]->C + f*ld*N, QdXL(f + 1), QdXL(f), LRL(f),
						w[s]->tmpN1, w[s]->pvt, w[s]->work, lwork);
			phases[s] = RC(calc_Gtt)(N, ld, LR_NULL, LRL(0), w[s]->g, w[s]->tmpNN1, w[s]->pvt);
		} else { // first sweep is down, initialize QdX0
			for (int f = 0; f < F; f++)
				RC(calc_QdX)(0, N, ld, w[s]->C + f*ld*N, QdX0(f - 1), QdX0(f), LR0(f),
						w[s]->tmpN1, w[s]->pvt, w[s]->work, lwork);
			phases[s] = RC(calc_Gtt)(N, ld, LR0(F - 1), LR_NULL, w[s]->g, w[s]->tmpNN1, w[s]->pvt);
		}
	}
	phase = phases[0]*phases[1];


	for (; sim->s.sweep < sim->p.n_sweep; sim->s.sweep++) {
		const int sig = sig_check_state(sim->s.sweep, sim->p.n_sweep_warm, sim->p.n_sweep);
		if (sig == 1) // stop flag
			break;
		else if (sig == 2) { // progress flag
			const int status = RC(sim_data_save)(sim);
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
				#pragma omp parallel for schedule(static, 1)
				for (int s = 0; s < N_DOF; s++) {
					profile_begin(wrap);
					RC(wrap)(N, ld, w[s]->g, w[s]->iB + l*ld*N, w[s]->B + l*ld*N, w[s]->tmpNN1);
					profile_end(wrap);
				}
			}

			profile_begin(updates);
			shuffle(rng, N, site_order);
			RC(update_delayed)(N, ld, n_delay, del, site_order,
			               rng, hs + N*l, w[0]->g, w[1]->g, &phase,
			               w[0]->tmpNN1, w[0]->tmpNN2, w[0]->tmpN1,
			               w[1]->tmpNN1, w[1]->tmpNN2, w[1]->tmpN1);
			for (int i = 0; i < N; i++) {
				const int hsil = hs[i + N*l];
				w[0]->exp_V[i] = exp_lambda[i + N*hsil];
				w[1]->exp_V[i] = exp_lambda[i + N*!hsil];
			}
			profile_end(updates);

			const int recalc = sweep_up ? (m == n_matmul - 1) : (m == 0);
			#pragma omp parallel for schedule(static, 1)
			for (int s = 0; s < N_DOF; s++) {
				profile_begin(calcb);
				mul_mat_diag(N, ld, w[s]->exp_K, w[s]->exp_V, w[s]->B + l*ld*N);
				mul_diag_mat(N, ld, w[1 - s]->exp_V, w[s]->inv_exp_K, w[s]->iB + l*ld*N);
				profile_end(calcb);
				if (recalc) {
					profile_begin(multb);
					RC(mul_seq)(N, ld, f*n_matmul, (f + 1)*n_matmul,
							1.0, w[s]->B, w[s]->C + f*ld*N, w[s]->tmpNN1);
					profile_end(multb);
					profile_begin(recalc);
					if (sweep_up) {
						RC(calc_QdX)(0, N, ld, w[s]->C + f*ld*N, QdX0(f - 1), QdX0(f), LR0(f),
								w[s]->tmpN1, w[s]->pvt, w[s]->work, lwork);
						phases[s] = RC(calc_Gtt)(N, ld, LR0(f), LRL(f + 1), w[s]->g, w[s]->tmpNN1, w[s]->pvt);
					} else {
						RC(calc_QdX)(1, N, ld, w[s]->C + f*ld*N, QdXL(f + 1), QdXL(f), LRL(f),
								w[s]->tmpN1, w[s]->pvt, w[s]->work, lwork);
						phases[s] = RC(calc_Gtt)(N, ld, LR0(f - 1), LRL(f), w[s]->g, w[s]->tmpNN1, w[s]->pvt);
					}
					profile_end(recalc);
				} else {
					if (sweep_up) {
						profile_begin(wrap);
						RC(wrap)(N, ld, w[s]->g, w[s]->B + l*ld*N, w[s]->iB + l*ld*N, w[s]->tmpNN1);
						profile_end(wrap);
					}
				}
			}

			if (recalc) phase = phases[0]*phases[1];

			if (enabled_eqlt && (l + sweep_up) % sim->p.period_eqlt == 0) {
				#pragma omp parallel for schedule(static, 1)
				for (int s = 0; s < N_DOF; s++) {
					profile_begin(half_wrap);
					xgemm("N", "N", N, N, N, 1.0, w[s]->g, ld, w[s]->exp_halfK, ld, 0.0, w[s]->tmpNN1, ld);
					xgemm("N", "N", N, N, N, 1.0, w[s]->inv_exp_halfK, ld, w[s]->tmpNN1, ld, 0.0, w[s]->tmpNN2, ld);
					profile_end(half_wrap);
				}
				profile_begin(meas_eq);
				RC(measure_eqlt)(&sim->p, phase, ld, w[0]->tmpNN2, w[1]->tmpNN2, &sim->m_eq);
				profile_end(meas_eq);
			}
		}

		if (enabled_uneqlt && (sim->s.sweep % sim->p.period_uneqlt == 0) ) {
			#pragma omp parallel for schedule(static, 1)
			for (int s = 0; s < N_DOF; s++) {
				profile_begin(calc_ue);
				xomatcopy('N', N, N, 1.0, w[s]->g, ld, w[s]->G0t, ld);
				xomatcopy('N', N, N, 1.0, w[s]->g, ld, w[s]->Gtt, ld);
				xomatcopy('N', N, N, 1.0, w[s]->g, ld, w[s]->Gt0, ld);
				if (sweep_up) { // then QdX0 is fresh, QdXL is old
					for (int f = F - 1; f >= 0; f--)
						RC(calc_QdX)(1, N, ld, w[s]->C + f*ld*N, QdXL(f + 1), QdXL(f), LRL(f),
						         w[s]->tmpN1, w[s]->pvt, w[s]->work, lwork);
				} else {
					for (int f = 0; f < F; f++)
						RC(calc_QdX)(0, N, ld, w[s]->C + f*ld*N, QdX0(f - 1), QdX0(f), LR0(f),
						         w[s]->tmpN1, w[s]->pvt, w[s]->work, lwork);
				}
				RC(calc_ue_g)(N, ld, L, F, n_matmul, w[s]->B, w[s]->iB,
				          w[s]->iL_0, w[s]->R_0, w[s]->iL_L, w[s]->R_L,
				          w[s]->G0t, w[s]->Gtt, w[s]->Gt0, w[s]->tmpNN1, w[s]->pvt);
				profile_end(calc_ue);
				profile_begin(half_wrap_ue);
				RC(wrap)(N, ld, w[s]->G0t, w[s]->inv_exp_halfK, w[s]->exp_halfK, w[s]->tmpNN1);
				xomatcopy('N', N, N, 1.0, w[s]->G0t, ld, w[s]->Gtt, ld);
				xomatcopy('N', N, N, 1.0, w[s]->G0t, ld, w[s]->Gt0, ld);
				for (int l = 1; l < L; l++) {
					RC(wrap)(N, ld, w[s]->G0t + l*ld*N, w[s]->inv_exp_halfK, w[s]->exp_halfK, w[s]->tmpNN1);
					RC(wrap)(N, ld, w[s]->Gtt + l*ld*N, w[s]->inv_exp_halfK, w[s]->exp_halfK, w[s]->tmpNN1);
					RC(wrap)(N, ld, w[s]->Gt0 + l*ld*N, w[s]->inv_exp_halfK, w[s]->exp_halfK, w[s]->tmpNN1);
				}
				profile_end(half_wrap_ue);
			}
			profile_begin(meas_uneq);
			RC(measure_uneqlt)(&sim->p, phase, ld,
			               w[0]->G0t, w[0]->Gtt, w[0]->Gt0, w[1]->G0t, w[1]->Gtt, w[1]->Gt0,
			               &sim->m_ue);
			profile_end(meas_uneq);
		}
	}

	fprintf(log_f, "%d/%d sweeps completed\n", sim->s.sweep, sim->p.n_sweep);
	return (sim->s.sweep == sim->p.n_sweep) ? 0 : 1;
}

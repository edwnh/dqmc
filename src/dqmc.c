#include "dqmc.h"
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <mkl.h>
#include "io.h"
#include "meas.h"
#include "prof.h"
#include "rand.h"
#include "time_.h"
#include "util.h"

//~ #define CHECK_G_WRP // check recalculated G against wrapped G
//~ #define CHECK_G_ACC // check recalculated G against using QR for every 2nd multiply

#define cint(x) &(const int){(x)}
#define cdbl(x) &(const double){(x)}

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

static int get_lwork(const int N)
{
	double lwork;

	int info;
	int max_lwork = N;

	dgeqp3(&N, &N, NULL, &N, NULL, NULL, &lwork, cint(-1), &info);
	if (lwork > max_lwork) max_lwork = (int)lwork;

	dgeqrf(&N, &N, NULL, &N, NULL, &lwork, cint(-1), &info);
	if (lwork > max_lwork) max_lwork = (int)lwork;

	dormqr("R", "N", &N, &N, &N, NULL, &N, NULL, NULL, &N, &lwork, cint(-1), &info);
	if (lwork > max_lwork) max_lwork = (int)lwork;

	dormqr("R", "T", &N, &N, &N, NULL, &N, NULL, NULL, &N, &lwork, cint(-1), &info);
	if (lwork > max_lwork) max_lwork = (int)lwork;

	return max_lwork;
}


static int calcG(const int ls, const int N, const int idk, const int n_Bs,
		const double *const restrict Bs, double *const restrict G,
		// work arrays
		double *const restrict Q, double *const restrict T,
		double *const restrict tau, double *const restrict d,
		double *const restrict v, int *const restrict pvt,
		double *const restrict work, const int lwork)
{
	__assume(idk % DBL_ALIGN == 0);
	_aa(Bs); _aa(G); _aa(Q); _aa(T); _aa(tau); _aa(d); _aa(v); _aa(pvt);

	int info;

	// algorithm 3 of 10.1109/IPDPS.2012.37
	// slightly modified; pairs of matrices are multiplied with dgemm
	// like (B5 B4)(B3 B2)(B1 B0) if n_Bs is even
	// or (B6 B5)(B4 B3)(B2 B1)(B0) if n_Bs is odd
	// (1)
	int l;
	if (n_Bs % 2 == 1) { // odd
		my_copy(Q, Bs + idk*ls, N*N);
		l = 1;
	} else { // even
		dgemm("N", "N", &N, &N, &N, cdbl(1.0), Bs + idk*((ls + 1)%n_Bs),
		      &N, Bs + idk*ls, &N, cdbl(0.0), Q, &N);
		l = 2;
	}

	for (int i = 0; i < N; i++) pvt[i] = 0;
	dgeqp3(&N, &N, Q, &N, pvt, tau, work, &lwork, &info);

	// (2)
	for (int i = 0; i < N; i++) {
		d[i] = Q[i + i*N];
		if (d[i] == 0.0) d[i] = 1.0;
		v[i] = 1.0/d[i];
	}

	for (int i = 0; i < N*N; i++) T[i] = 0.0;
	for (int j = 0; j < N; j++)
		for (int i = 0; i <= j; i++)
			T[i + (pvt[j]-1)*N] = v[i] * Q[i + j*N];


	for (; l < n_Bs; l += 2) {
		// (3a)
		dgemm("N", "N", &N, &N, &N, cdbl(1.0), Bs + idk*((ls + l + 1)%n_Bs),
		      &N, Bs + idk*((ls + l)%n_Bs), &N, cdbl(0.0), G, &N);

		dormqr("R", "N", &N, &N, &N, Q, &N, tau, G, &N, work, &lwork, &info);

		for (int j = 0; j < N; j++)
			for (int i = 0; i < N; i++)
				G[i + j*N] *= d[j];

		// (3b)
		for (int j = 0; j < N; j++) { // use d for norms
			d[j] = 0.0;
			for (int i = 0; i < N; i++)
				d[j] += G[i + j*N] * G[i + j*N];
		}

		pvt[0] = 0;
		for (int i = 1; i < N; i++) { // insertion sort
			int j;
			for (j = i; j > 0 && d[pvt[j-1]] < d[i]; j--)
				pvt[j] = pvt[j-1];
			pvt[j] = i;
		}

		for (int j = 0; j < N; j++) // pre-pivot
			my_copy(Q + j*N, G + pvt[j]*N, N);

		// (3c)
		dgeqrf(&N, &N, Q, &N, tau, work, &lwork, &info);

		// (3d)
		for (int i = 0; i < N; i++) {
			d[i] = Q[i + i*N];
			if (d[i] == 0.0) d[i] = 1.0;
			v[i] = 1.0/d[i];
		}

		for (int j = 0; j < N; j++)
			for (int i = 0; i <= j; i++)
				Q[i + j*N] *= v[i];

		for (int j = 0; j < N; j++) {
			for (int i = 0; i < N; i++)
				v[i] = T[pvt[i] + j*N];
			my_copy(T + j*N, v, N);
		}

		dtrmm("L", "U", "N", "N", &N, &N, cdbl(1.0), Q, &N, T, &N);
	}
	// construct G from Eq 2.12 of 10.1016/j.laa.2010.06.023
	for (int i = 0; i < N*N; i++) G[i] = 0.0;
	for (int i = 0; i < N; i++) {
		if (fabs(d[i]) > 1.0) { // v = 1/Db; d = Ds
			v[i] = 1.0/d[i];
			d[i] = 1.0;
		} else {
			v[i] = 1.0;
		}
		G[i+i*N] = v[i];
	}

	dormqr("R", "T", &N, &N, &N, Q, &N, tau, G, &N, work, &lwork, &info);

	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			T[i + j*N] *= d[i];

	for (int i = 0; i < N*N; i++) T[i] += G[i];

	dgetrf(&N, &N, T, &N, pvt, &info);
	dgetrs("N", &N, &N, T, &N, pvt, G, &N, &info);

	// calculate sign of det(G)
	int sign = 1;
	for (int i = 0; i < N; i++)
		if ((T[i + N*i] < 0) ^ (pvt[i] != i + 1) ^ (v[i] < 0) ^ (tau[i] > 0))
			sign *= -1;

	return sign;
}

static inline int update_delayed(const int N, const double *const restrict del,
		uint64_t *const restrict rng, int *const restrict site_order,
		int *const restrict hs,
		double *const restrict Gu, double *const restrict Gd,
		const int q,
		double *const restrict au, double *const restrict bu,
		double *const restrict du, double *const restrict ad,
		double *const restrict bd, double *const restrict dd)
{
	_aa(Gu); _aa(Gd); _aa(au); _aa(bu); _aa(du); _aa(ad); _aa(bd); _aa(dd);

	int sign = 1;
	int k = 0;
	for (int j = 0; j < N; j++) du[j] = Gu[j + N*j];
	for (int j = 0; j < N; j++) dd[j] = Gd[j + N*j];
	shuffle(rng, N, site_order);
	for (int ii = 0; ii < N; ii++) {
		const int i = site_order[ii];
		const double delu = del[i + N*hs[i]];
		const double deld = del[i + N*!hs[i]];
		const double ru = 1.0 + (1.0 - du[i]) * delu;
		const double rd = 1.0 + (1.0 - dd[i]) * deld;
		const double prob = ru * rd;
		if (rand_doub(rng) < fabs(prob)) {
			#pragma omp parallel sections
			{
			#pragma omp section
			{
			for (int j = 0; j < N; j++) au[j + N*k] = Gu[j + N*i];
			for (int j = 0; j < N; j++) bu[j + N*k] = Gu[i + N*j];
			dgemv("N", &N, &k, cdbl(1.0), au, &N, bu + i,
			      &N, cdbl(1.0), au + N*k, cint(1));
			dgemv("N", &N, &k, cdbl(1.0), bu, &N, au + i,
			      &N, cdbl(1.0), bu + N*k, cint(1));
			au[i + N*k] -= 1.0;
			for (int j = 0; j < N; j++) au[j + N*k] *= delu/ru;
			for (int j = 0; j < N; j++) du[j] += au[j + N*k] * bu[j + N*k];
			}
			#pragma omp section
			{
			for (int j = 0; j < N; j++) ad[j + N*k] = Gd[j + N*i];
			for (int j = 0; j < N; j++) bd[j + N*k] = Gd[i + N*j];
			dgemv("N", &N, &k, cdbl(1.0), ad, &N, bd + i,
			      &N, cdbl(1.0), ad + N*k, cint(1));
			dgemv("N", &N, &k, cdbl(1.0), bd, &N, ad + i,
			      &N, cdbl(1.0), bd + N*k, cint(1));
			ad[i + N*k] -= 1.0;
			for (int j = 0; j < N; j++) ad[j + N*k] *= deld/rd;
			for (int j = 0; j < N; j++) dd[j] += ad[j + N*k] * bd[j + N*k];
			}
			}
			k++;
			hs[i] = !hs[i];
			if (prob < 0) sign *= -1;
		}
		if (k == q) {
			k = 0;
			#pragma omp parallel sections
			{
			#pragma omp section
			{
			dgemm("N", "T", &N, &N, &q, cdbl(1.0),
			      au, &N, bu, &N, cdbl(1.0), Gu, &N);
			for (int j = 0; j < N; j++) du[j] = Gu[j + N*j];
			}
			#pragma omp section
			{
			dgemm("N", "T", &N, &N, &q, cdbl(1.0),
			      ad, &N, bd, &N, cdbl(1.0), Gd, &N);
			for (int j = 0; j < N; j++) dd[j] = Gd[j + N*j];
			}
			}
		}
	}
	#pragma omp parallel sections
	{
	#pragma omp section
	dgemm("N", "T", &N, &N, &k, cdbl(1.0), au, &N, bu, &N, cdbl(1.0), Gu, &N);
	#pragma omp section
	dgemm("N", "T", &N, &N, &k, cdbl(1.0), ad, &N, bd, &N, cdbl(1.0), Gd, &N);
	}
	return sign;
}

static void dqmc(FILE *log, const tick_t wall_start, const tick_t max_time,
		const struct params *p, struct state *s,
		struct meas_eqlt *m_eq, struct meas_uneqlt *m_ue)
{
	const int N = p->N;
	const int n_matmul = p->n_matmul;
	const int n_delay = p->n_delay;
	const int n_Bs = p->n_Bs;
	const double *const restrict exp_K = p->exp_K; _aa(exp_K);
	const double *const restrict inv_exp_K = p->inv_exp_K; _aa(inv_exp_K);
	const double *const restrict exp_lambda = p->exp_lambda; _aa(exp_lambda);
	const double *const restrict del = p->del; _aa(del);
	uint64_t *const restrict rng = s->rng; _aa(rng);
	int *const restrict hs = s->hs;

	// idk what to call this
	const int idk = DBL_ALIGN * ((N*N + DBL_ALIGN - 1) / DBL_ALIGN);
	__assume(idk % DBL_ALIGN == 0);

	double *const restrict Bu = my_calloc(idk*p->L * sizeof(double)); _aa(Bu);
	double *const restrict Bd = my_calloc(idk*p->L * sizeof(double)); _aa(Bd);
	double *const restrict Bsu = my_calloc(idk*n_Bs * sizeof(double)); _aa(Bsu);
	double *const restrict Bsd = my_calloc(idk*n_Bs * sizeof(double)); _aa(Bsd);
	double *const restrict Gu = my_calloc(N*N * sizeof(double)); _aa(Gu);
	double *const restrict Gd = my_calloc(N*N * sizeof(double)); _aa(Gd);
	#ifdef CHECK_G_WRP
	double *const restrict Guwrp = my_calloc(N*N * sizeof(double)); _aa(Guwrp);
	double *const restrict Gdwrp = my_calloc(N*N * sizeof(double)); _aa(Gdwrp);
	#endif
	#ifdef CHECK_G_ACC
	double *const restrict Guacc = my_calloc(N*N * sizeof(double)); _aa(Guwrp);
	double *const restrict Gdacc = my_calloc(N*N * sizeof(double)); _aa(Gdwrp);
	#endif
	int sign = 0;
	int *const site_order = my_calloc(N * sizeof(double)); _aa(site_order);

	// work arrays for calcG and stuff. two sets for easy 2x parallelization
	const int lwork = get_lwork(N);

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
	for (int ls = 0; ls < n_Bs; ls++) {
		calcBu(Bu + idk*ls*n_matmul, ls*n_matmul);
		my_copy(Bsu + idk*ls, Bu + idk*ls*n_matmul, N*N);
		for (int l = ls*n_matmul + 1 ; l < (ls + 1) * n_matmul; l++) {
			calcBu(Bu + idk*l, l);
			my_copy(tmpNN1u, Bsu + idk*ls, N*N);
			matmul(Bsu + idk*ls, Bu + idk*l, tmpNN1u);
		}
	}
	signu = calcG(0, N, idk, n_Bs, Bsu, Gu, tmpNN1u, tmpNN2u,
		      tmpN1u, tmpN2u, tmpN3u, pvtu, worku, lwork);
	}
	#pragma omp section
	{
	for (int ls = 0; ls < n_Bs; ls++) {
		calcBd(Bd + idk*ls*n_matmul, ls*n_matmul);
		my_copy(Bsd + idk*ls, Bd + idk*ls*n_matmul, N*N);
		for (int l = ls*n_matmul + 1 ; l < (ls + 1) * n_matmul; l++) {
			calcBd(Bd + idk*l, l);
			my_copy(tmpNN1d, Bsd + idk*ls, N*N);
			matmul(Bsd + idk*ls, Bd + idk*l, tmpNN1d);
		}
	}
	signd = calcG(0, N, idk, n_Bs, Bsd, Gd, tmpNN1d, tmpNN2d,
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
			sign *= update_delayed(N, del, rng, site_order, hs + N*l, Gu, Gd, n_delay,
			                       tmpNN1u, tmpNN2u, tmpN1u, tmpNN1d, tmpNN2d, tmpN1d);
			profile_end(updates);

			const int ls = l / n_matmul;
			const int recalc = ((l + 1) % n_matmul == 0);
			int signu, signd;
			#pragma omp parallel sections
			{
			#pragma omp section
			{
			profile_begin(multb);
			calcBu(Bu + idk*l, l);
			if (l % n_matmul == 0) {
				my_copy(Bsu + idk*ls, Bu + idk*l, N*N);
			} else {
				my_copy(tmpNN1u, Bsu + idk*ls, N*N);
				matmul(Bsu + idk*ls, Bu + idk*l, tmpNN1u);
			}
			profile_end(multb);
			if (recalc) {
				profile_begin(recalc);
				#ifdef CHECK_G_WRP
				calcinvBu(tmpNN1u, l);
				matmul(tmpNN2u, Gu, tmpNN1u);
				matmul(Guwrp, Bu + idk*l, tmpNN2u);
				#endif
				#ifdef CHECK_G_ACC
				calcG((l + 1) % p->L, N, idk, p->L, Bu, Guacc,
				      tmpNN1u, tmpNN2u, tmpN1u, tmpN2u,
				      tmpN3u, pvtu, worku, lwork);
				#endif
				signu = calcG((ls + 1) % n_Bs, N, idk, n_Bs, Bsu, Gu,
				              tmpNN1u, tmpNN2u, tmpN1u, tmpN2u,
				              tmpN3u, pvtu, worku, lwork);
				profile_end(recalc);
			} else {
				profile_begin(wrap);
				calcinvBu(tmpNN1u, l);
				matmul(tmpNN2u, Gu, tmpNN1u);
				matmul(Gu, Bu + idk*l, tmpNN2u);
				profile_end(wrap);
			}
			}
			#pragma omp section
			{
			profile_begin(multb);
			calcBd(Bd + idk*l, l)
			if (l % n_matmul == 0) {
				my_copy(Bsd + idk*ls, Bd + idk*l, N*N);
			} else {
				my_copy(tmpNN1d, Bsd + idk*ls, N*N);
				matmul(Bsd + idk*ls, Bd + idk*l, tmpNN1d);
			}
			profile_end(multb);
			if (recalc) {
				profile_begin(recalc);
				#ifdef CHECK_G_WRP
				calcinvBd(tmpNN1d, l);
				matmul(tmpNN2d, Gd, tmpNN1d);
				matmul(Gdwrp, Bd + idk*l, tmpNN2d);
				#endif
				#ifdef CHECK_G_ACC
				calcG((l + 1) % p->L, N, idk, p->L, Bd, Gdacc,
				      tmpNN1d, tmpNN2d, tmpN1d, tmpN2d,
				      tmpN3d, pvtd, workd, lwork);
				#endif
				signd = calcG((ls + 1) % n_Bs, N, idk, n_Bs, Bsd, Gd,
				              tmpNN1d, tmpNN2d, tmpN1d, tmpN2d,
				              tmpN3d, pvtd, workd, lwork);
				profile_end(recalc);
			} else {
				profile_begin(wrap);
				calcinvBd(tmpNN1d, l);
				matmul(tmpNN2d, Gd, tmpNN1d);
				matmul(Gd, Bd + idk*l, tmpNN2d);
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
				matdiff(Gu, Guwrp);
				matdiff(Gd, Gdwrp);
			}
			#endif
			#ifdef CHECK_G_ACC
			if (recalc) {
				matdiff(Gu, Guacc);
				matdiff(Gd, Gdacc);
			}
			#endif
			#if defined(CHECK_G_WRP) && defined(CHECK_G_ACC)
			if (recalc) {
				matdiff(Guwrp, Guacc);
				matdiff(Gdwrp, Gdacc);
			}
			#endif

			#undef matdiff

			if (recalc) sign = signu*signd;

			if (enabled_eqlt && (l + 1) % p->period_eqlt == 0) {
				profile_begin(meas_eq);
				measure_eqlt(p, sign, Gu, Gd, m_eq);
				profile_end(meas_eq);
			}
		}

		if (enabled_uneqlt && s->sweep % p->period_uneqlt == 0) {
			profile_begin(meas_uneq);
			measure_uneqlt(p, sign, Gu, Gd, m_ue);
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
	my_free(Gdacc);
	my_free(Guacc);
	#endif
	#ifdef CHECK_G_WRP
	my_free(Gdwrp);
	my_free(Guwrp);
	#endif
	my_free(Gd);
	my_free(Gu);
	my_free(Bsd);
	my_free(Bsu);
	my_free(Bd);
	my_free(Bu);
}

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
	my_free(m_ue->zz);
	my_free(m_ue->xx);
	my_free(m_ue->nn);
	my_free(m_ue->gt0);
	my_free(m_ue->g0t);
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

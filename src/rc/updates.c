#include "updates.h"
#include <tgmath.h>
#include "linalg.h"
#include "mem.h"
#include "rand.h"

void RC(update_delayed)(const int N, const int ld, const int n_delay, const double *const del,
		const int *const site_order,
		uint64_t *const rng, int *const hs,
		num *const gu, num *const gd, num *const phase,
		num *const au, num *const bu, num *const du,
		num *const ad, num *const bd, num *const dd)
{
	__builtin_assume(ld % MEM_ALIGN_NUM == 0);
	(void)__builtin_assume_aligned(gu, MEM_ALIGN);
	(void)__builtin_assume_aligned(gd, MEM_ALIGN);
	(void)__builtin_assume_aligned(au, MEM_ALIGN);
	(void)__builtin_assume_aligned(bu, MEM_ALIGN);
	(void)__builtin_assume_aligned(du, MEM_ALIGN);
	(void)__builtin_assume_aligned(ad, MEM_ALIGN);
	(void)__builtin_assume_aligned(bd, MEM_ALIGN);
	(void)__builtin_assume_aligned(dd, MEM_ALIGN);

	int k = 0;
	for (int j = 0; j < N; j++) du[j] = gu[j + ld*j];
	for (int j = 0; j < N; j++) dd[j] = gd[j + ld*j];
	for (int ii = 0; ii < N; ii++) {
		const int i = site_order[ii];
		const double delu = del[i + N*hs[i]];
		const double deld = del[i + N*!hs[i]];
		if (delu == 0.0 && deld == 0.0) continue;
		const num ru = 1.0 + (1.0 - du[i]) * delu;
		const num rd = 1.0 + (1.0 - dd[i]) * deld;
		const num prob = ru * rd;
		const double absprob = fabs(prob);
		if (rand_doub(rng) < absprob) {
			#pragma omp parallel sections
			{
			#pragma omp section
			{
			for (int j = 0; j < N; j++) au[j + ld*k] = gu[j + ld*i];
			for (int j = 0; j < N; j++) bu[j + ld*k] = gu[i + ld*j];
			xgemv("N", N, k, 1.0, au, ld, bu + i,
			      ld, 1.0, au + ld*k, 1);
			xgemv("N", N, k, 1.0, bu, ld, au + i,
			      ld, 1.0, bu + ld*k, 1);
			au[i + ld*k] -= 1.0;
			for (int j = 0; j < N; j++) au[j + ld*k] *= delu/ru;
			for (int j = 0; j < N; j++) du[j] += au[j + ld*k] * bu[j + ld*k];
			}
			#pragma omp section
			{
			for (int j = 0; j < N; j++) ad[j + ld*k] = gd[j + ld*i];
			for (int j = 0; j < N; j++) bd[j + ld*k] = gd[i + ld*j];
			xgemv("N", N, k, 1.0, ad, ld, bd + i,
			      ld, 1.0, ad + ld*k, 1);
			xgemv("N", N, k, 1.0, bd, ld, ad + i,
			      ld, 1.0, bd + ld*k, 1);
			ad[i + ld*k] -= 1.0;
			for (int j = 0; j < N; j++) ad[j + ld*k] *= deld/rd;
			for (int j = 0; j < N; j++) dd[j] += ad[j + ld*k] * bd[j + ld*k];
			}
			}
			k++;
			hs[i] = !hs[i];
			*phase *= prob/absprob;
		}
		if (k == n_delay) {
			k = 0;
			#pragma omp parallel sections
			{
			#pragma omp section
			{
			xgemm("N", "T", N, N, n_delay, 1.0,
			      au, ld, bu, ld, 1.0, gu, ld);
			for (int j = 0; j < N; j++) du[j] = gu[j + ld*j];
			}
			#pragma omp section
			{
			xgemm("N", "T", N, N, n_delay, 1.0,
			      ad, ld, bd, ld, 1.0, gd, ld);
			for (int j = 0; j < N; j++) dd[j] = gd[j + ld*j];
			}
			}
		}
	}
	#pragma omp parallel sections
	{
	#pragma omp section
	xgemm("N", "T", N, N, k, 1.0, au, ld, bu, ld, 1.0, gu, ld);
	#pragma omp section
	xgemm("N", "T", N, N, k, 1.0, ad, ld, bd, ld, 1.0, gd, ld);
	}
}

// below is unmaintained
/*
void update_shermor(const int N, const double *const del,
		const int *const site_order,
		uint64_t *const rng, int *const hs,
		double *const gu, double *const gd, int *const sign,
		double *const cu, double *const du,
		double *const cd, double *const dd)
{
	for (int ii = 0; ii < N; ii++) {
		const int i = site_order[ii];
		const double delu = del[i + N*hs[i]];
		const double deld = del[i + N*!hs[i]];
		const double pu = (1 + (1 - gu[i + N*i])*delu);
		const double pd = (1 + (1 - gd[i + N*i])*deld);
		const double prob = pu*pd;
		if (rand_doub(rng) < fabs(prob)) {
			for (int j = 0; j < N; j++) cu[j] = gu[j + N*i];
			cu[i] -= 1.0;
			for (int j = 0; j < N; j++) du[j] = gu[i + N*j];
			const double au = delu/pu;
			dger(&N, &N, &au, cu, cint(1), du, cint(1), gu, &N);

			for (int j = 0; j < N; j++) cd[j] = gd[j + N*i];
			cd[i] -= 1.0;
			for (int j = 0; j < N; j++) dd[j] = gd[i + N*j];
			const double ad = deld/pd;
			dger(&N, &N, &ad, cd, cint(1), dd, cint(1), gd, &N);

			hs[i] = !hs[i];
			if (prob < 0) *sign *= -1;
		}
	}
}

void update_submat(const int N, const int q, const double *const del,
		const int *const site_order,
		uint64_t *const rng, int *const hs,
		double *const gu, double *const gd, int *const sign,
		double *const gr_u, double *const g_ru,
		double *const DDu, double *const yu, double *const xu,
		double *const gr_d, double *const g_rd,
		double *const DDd, double *const yd, double *const xd)
{
	int *const r = my_calloc(q * sizeof(int)); _aa(r);
	double *const LUu = my_calloc(q*q * sizeof(double)); _aa(LUu);
	double *const LUd = my_calloc(q*q * sizeof(double)); _aa(LUd);

	int k = 0;
	for (int ii = 0; ii < N; ii++) {
		const int i = site_order[ii];
		const double delu = del[i + N*hs[i]];
		const double deld = del[i + N*!hs[i]];
		double du = gu[i + N*i] - (1 + delu)/delu;
		double dd = gd[i + N*i] - (1 + deld)/deld;
		if (k > 0) {
			for (int j = 0; j < k; j++) yu[j] = gr_u[j + N*i];
			dtrtrs("L", "N", "U", &k, cint(1), LUu, &q, yu, &k, &(int){0});
			for (int j = 0; j < k; j++) xu[j] = g_ru[i + N*j];
			dtrtrs("U", "T", "N", &k, cint(1), LUu, &q, xu, &k, &(int){0});
			for (int j = 0; j < k; j++) du -= yu[j]*xu[j];

			for (int j = 0; j < k; j++) yd[j] = gr_d[j + N*i];
			dtrtrs("L", "N", "U", &k, cint(1), LUd, &q, yd, &k, &(int){0});
			for (int j = 0; j < k; j++) xd[j] = g_rd[i + N*j];
			dtrtrs("U", "T", "N", &k, cint(1), LUd, &q, xd, &k, &(int){0});
			for (int j = 0; j < k; j++) dd -= yd[j]*xd[j];
		}

		const double prob = du*delu * dd*deld;
		if (rand_doub(rng) < fabs(prob)) {
			r[k] = i;
			DDu[k] = 1.0 / (1.0 + delu);
			DDd[k] = 1.0 / (1.0 + deld);
			for (int j = 0; j < N; j++) gr_u[k + N*j] = gu[i + N*j];
			for (int j = 0; j < N; j++) g_ru[j + N*k] = gu[j + N*i];
			for (int j = 0; j < N; j++) gr_d[k + N*j] = gd[i + N*j];
			for (int j = 0; j < N; j++) g_rd[j + N*k] = gd[j + N*i];
			for (int j = 0; j < k; j++) LUu[j + q*k] = yu[j];
			for (int j = 0; j < k; j++) LUu[k + q*j] = xu[j];
			for (int j = 0; j < k; j++) LUd[j + q*k] = yd[j];
			for (int j = 0; j < k; j++) LUd[k + q*j] = xd[j];
			LUu[k + q*k] = du;
			LUd[k + q*k] = dd;
			k++;
			hs[i] = !hs[i];
			if (prob < 0) *sign *= -1;
		}

		if (k == q || (ii == N - 1 && k > 0)) {
			dtrtrs("L", "N", "U", &k, &N, LUu, &q, gr_u, &N, &(int){0});
			dtrtrs("U", "N", "N", &k, &N, LUu, &q, gr_u, &N, &(int){0});
			dgemm("N", "N", &N, &N, &k, cdbl(-1.0), g_ru, &N, gr_u, &N, cdbl(1.0), gu, &N);
			for (int j = 0; j < k; j++) {
				const int rj = r[j];
				const double DDk = DDu[j];
				for (int iii = 0; iii < N; iii++)
					gu[rj + iii*N] *= DDk;
			}
			dtrtrs("L", "N", "U", &k, &N, LUd, &q, gr_d, &N, &(int){0});
			dtrtrs("U", "N", "N", &k, &N, LUd, &q, gr_d, &N, &(int){0});
			dgemm("N", "N", &N, &N, &k, cdbl(-1.0), g_rd, &N, gr_d, &N, cdbl(1.0), gd, &N);
			for (int j = 0; j < k; j++) {
				const int rj = r[j];
				const double DDk = DDd[j];
				for (int iii = 0; iii < N; iii++)
					gd[rj + iii*N] *= DDk;
			}
			k = 0;
		}
	}
	my_free(LUd);
	my_free(LUu);
	my_free(r);
}
*/

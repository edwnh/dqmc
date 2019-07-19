#include "updates.h"
#include <tgmath.h>
#include "rand.h"
#include "util.h"

void update_delayed(const int N, const int n_delay, const double *const restrict del,
		const int *const restrict site_order,
		uint64_t *const restrict rng, int *const restrict hs,
		complex double *const restrict gu, complex double *const restrict gd, complex double *const restrict phase,
		complex double *const restrict au, complex double *const restrict bu, complex double *const restrict du,
		complex double *const restrict ad, complex double *const restrict bd, complex double *const restrict dd)
{
	int k = 0;
	for (int j = 0; j < N; j++) du[j] = gu[j + N*j];
	for (int j = 0; j < N; j++) dd[j] = gd[j + N*j];
	for (int ii = 0; ii < N; ii++) {
		const int i = site_order[ii];
		const double delu = del[i + N*hs[i]];
		const double deld = del[i + N*!hs[i]];
		if (delu == 0.0 && deld == 0.0) continue;
		const complex double ru = 1.0 + (1.0 - du[i]) * delu;
		const complex double rd = 1.0 + (1.0 - dd[i]) * deld;
		const complex double prob = ru * rd;
		const double absprob = fabs(prob);
		if (rand_doub(rng) < absprob) {
			#pragma omp parallel sections
			{
			#pragma omp section
			{
			for (int j = 0; j < N; j++) au[j + N*k] = gu[j + N*i];
			for (int j = 0; j < N; j++) bu[j + N*k] = gu[i + N*j];
			zgemv("N", &N, &k, ccplx(1.0), au, &N, bu + i,
			      &N, ccplx(1.0), au + N*k, cint(1));
			zgemv("N", &N, &k, ccplx(1.0), bu, &N, au + i,
			      &N, ccplx(1.0), bu + N*k, cint(1));
			au[i + N*k] -= 1.0;
			for (int j = 0; j < N; j++) au[j + N*k] *= delu/ru;
			for (int j = 0; j < N; j++) du[j] += au[j + N*k] * bu[j + N*k];
			}
			#pragma omp section
			{
			for (int j = 0; j < N; j++) ad[j + N*k] = gd[j + N*i];
			for (int j = 0; j < N; j++) bd[j + N*k] = gd[i + N*j];
			zgemv("N", &N, &k, ccplx(1.0), ad, &N, bd + i,
			      &N, ccplx(1.0), ad + N*k, cint(1));
			zgemv("N", &N, &k, ccplx(1.0), bd, &N, ad + i,
			      &N, ccplx(1.0), bd + N*k, cint(1));
			ad[i + N*k] -= 1.0;
			for (int j = 0; j < N; j++) ad[j + N*k] *= deld/rd;
			for (int j = 0; j < N; j++) dd[j] += ad[j + N*k] * bd[j + N*k];
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
			zgemm("N", "T", &N, &N, &n_delay, ccplx(1.0),
			      au, &N, bu, &N, ccplx(1.0), gu, &N);
			for (int j = 0; j < N; j++) du[j] = gu[j + N*j];
			}
			#pragma omp section
			{
			zgemm("N", "T", &N, &N, &n_delay, ccplx(1.0),
			      ad, &N, bd, &N, ccplx(1.0), gd, &N);
			for (int j = 0; j < N; j++) dd[j] = gd[j + N*j];
			}
			}
		}
	}
	#pragma omp parallel sections
	{
	#pragma omp section
	zgemm("N", "T", &N, &N, &k, ccplx(1.0), au, &N, bu, &N, ccplx(1.0), gu, &N);
	#pragma omp section
	zgemm("N", "T", &N, &N, &k, ccplx(1.0), ad, &N, bd, &N, ccplx(1.0), gd, &N);
	}
}

/*
void update_shermor(const int N, const double *const restrict del,
		const int *const restrict site_order,
		uint64_t *const restrict rng, int *const restrict hs,
		double *const restrict gu, double *const restrict gd, int *const restrict sign,
		double *const restrict cu, double *const restrict du,
		double *const restrict cd, double *const restrict dd)
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

void update_submat(const int N, const int q, const double *const restrict del,
		const int *const restrict site_order,
		uint64_t *const restrict rng, int *const restrict hs,
		double *const restrict gu, double *const restrict gd, int *const restrict sign,
		double *const restrict gr_u, double *const restrict g_ru,
		double *const restrict DDu, double *const restrict yu, double *const restrict xu,
		double *const restrict gr_d, double *const restrict g_rd,
		double *const restrict DDd, double *const restrict yd, double *const restrict xd)
{
	int *const restrict r = my_calloc(q * sizeof(int)); _aa(r);
	double *const restrict LUu = my_calloc(q*q * sizeof(double)); _aa(LUu);
	double *const restrict LUd = my_calloc(q*q * sizeof(double)); _aa(LUd);

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

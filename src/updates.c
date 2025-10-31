#include "updates.h"
#include <tgmath.h>
#include "linalg.h"
#include "mem.h"
#include "rand.h"

/*
void update_delayed(const int N, const int ld, const int n_delay, const double *const del,
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
*/



void update_shermor(const int N, const int ld, const int N_inter, const int *const bonds_inter,
	const int *const site_order, uint64_t *const rng, int *const hs,
	num *const gu, num *const gd, num *const phase,
	double *const Delta_p, double *const Delta_q, double *const weight_comparison_matrice,
	num *const cu_p_col, num *const cu_q_col, num *const cu_p_row, 
	num *const cu_q_row, num *const cu_col, num *const cu_row, 
	num *const cd_p_col, num *const cd_q_col, num *const cd_p_row, 
	num *const cd_q_row, num *const cd_col, num *const cd_row, 
	num *const tmpuld2, num *const tmpdld2
)

{
	for (int ii = 0; ii < N_inter; ii++) {
		const int i = site_order[ii];
		int all_nums[4] = {0, 1, 2, 3};
		int other_states[3];
		int count = 0;
		for (int i1 = 0; i1 < 4; i1++) {
        if (all_nums[i1] != hs[i]) {
            other_states[count++] = all_nums[i1];
        }
		}

		int after_jump;
		const double rand_jump_para = rand_doub(rng);
		if (rand_jump_para < 1.0/3) {
			after_jump = other_states[0];
		} else if (rand_jump_para < 2.0/3) {
			after_jump = other_states[1];
		} else {
			after_jump = other_states[2];
		}

		int p = bonds_inter[0 * N_inter + i];
		int q = bonds_inter[1 * N_inter + i];
		const num Deltap = Delta_p[i*16 + after_jump*4 + hs[i]];
		const num Deltaq = Delta_q[i*16 + after_jump*4 + hs[i]];
		const num A_divided = weight_comparison_matrice[i*16 + after_jump*4 + hs[i]];
		const num pu = (1 + Deltap * (1-gu[p + ld*p]) + Deltaq * (1-gu[q + ld*q]) + Deltap * Deltaq * (
            1-gu[p + ld*p]) * (1-gu[q + ld*q]) - Deltap * Deltaq * gu[p + ld*q] * gu[q + ld*p]);
        const num pd = (1 + Deltap * (1-gd[p + ld*p]) + Deltaq * (1-gd[q + ld*q]) + Deltap * Deltaq * (
            1-gd[p + ld*p]) * (1-gd[q + ld*q]) - Deltap * Deltaq * gd[p + ld*q] * gd[q + ld*p]);
		const num prob = A_divided * pu * pd;

		const double absprob = fabs(prob);
		if (rand_doub(rng) < absprob) {
			if (p < q) {
				const num Fu_11 = Deltaq * (1. + Deltaq * (1.0 - gu[q + ld*q]));
                const num Fu_12 = Deltap * Deltaq * gu[p + ld*q];
                const num Fu_21 = Deltap * Deltaq * gu[q + ld*p];
                const num Fu_22 = Deltap * (1.0 + Deltap * (1.0 - gu[p + ld*p]));
                const num Denou = Fu_11 * Fu_22 - Fu_12 * Fu_21;
                const num Fu_renor_11 = Deltap * Deltap * Fu_11 / Denou;
                const num Fu_renor_12 = Deltap * Deltaq * Fu_12 / Denou;
                const num Fu_renor_21 = Deltap * Deltaq * Fu_21 / Denou;
                const num Fu_renor_22 = Deltaq * Deltaq * Fu_22 / Denou;
				const num Fu_renor[4] = {Fu_renor_11, Fu_renor_21, Fu_renor_12, Fu_renor_22}; 

				for (int j = 0; j < N; j++) cu_p_col[j] = gu[j + ld*p];
				cu_p_col[p] -= 1.0;
				for (int j = 0; j < N; j++) cu_q_col[j] = gu[j + ld*q];
				cu_q_col[q] -= 1.0;

				for (int j = 0; j < N; j++) cu_p_row[j] = gu[p + ld*j];
				for (int j = 0; j < N; j++) cu_q_row[j] = gu[q + ld*j];

				memcpy(cu_col, cu_p_col, ld * sizeof(num));
				memcpy(cu_col + ld, cu_q_col, ld * sizeof(num));

				for (int j = 0; j < N; j++) {
					cu_row[j*2] = cu_p_row[j];
					cu_row[j*2 + 1] = cu_q_row[j];
				}

				xgemm("N", "N", N, 2, 2, 1.0, cu_col, ld, Fu_renor, 2, 0.0, tmpuld2, ld);
				xgemm("N", "N", N, N, 2, 1.0, tmpuld2, ld, cu_row, 2, 1.0, gu, ld);

				const num Fd_11 = Deltaq * (1. + Deltaq * (1.0 - gd[q + ld*q]));
				const num Fd_12 = Deltap * Deltaq * gd[p + ld*q];
				const num Fd_21 = Deltap * Deltaq * gd[q + ld*p];
				const num Fd_22 = Deltap * (1.0 + Deltap * (1.0 - gd[p + ld*p]));
				const num Denod = Fd_11 * Fd_22 - Fd_12 * Fd_21;

				const num Fd_renor_11 = Deltap * Deltap * Fd_11 / Denod;
				const num Fd_renor_12 = Deltap * Deltaq * Fd_12 / Denod;
				const num Fd_renor_21 = Deltap * Deltaq * Fd_21 / Denod;
				const num Fd_renor_22 = Deltaq * Deltaq * Fd_22 / Denod;
				const num Fd_renor[4] = {Fd_renor_11, Fd_renor_21, Fd_renor_12, Fd_renor_22}; 

				for (int j = 0; j < N; j++) cd_p_col[j] = gd[j + ld*p];
				cd_p_col[p] -= 1.0;
				for (int j = 0; j < N; j++) cd_q_col[j] = gd[j + ld*q];
				cd_q_col[q] -= 1.0;

				for (int j = 0; j < N; j++) cd_p_row[j] = gd[p + ld*j];
				for (int j = 0; j < N; j++) cd_q_row[j] = gd[q + ld*j];

				memcpy(cd_col, cd_p_col, ld * sizeof(num));
				memcpy(cd_col + ld, cd_q_col, ld * sizeof(num));

				for (int j = 0; j < N; j++) {
					cd_row[j*2] = cd_p_row[j];
					cd_row[j*2 + 1] = cd_q_row[j];
				}

				xgemm("N", "N", N, 2, 2, 1.0, cd_col, ld, Fd_renor, 2, 0.0, tmpdld2, ld);
				xgemm("N", "N", N, N, 2, 1.0, tmpdld2, ld, cd_row, 2, 1.0, gd, ld);

			} else if (p > q) {

				const num Fu_11 = Deltap * (1. + Deltap * (1.0 - gu[p + ld*p]));
				const num Fu_12 = Deltaq * Deltap * gu[q + ld*p];
				const num Fu_21 = Deltaq * Deltap * gu[p + ld*q];
				const num Fu_22 = Deltaq * (1.0 + Deltaq * (1.0 - gu[q + ld*q]));
				const num Denou = Fu_11 * Fu_22 - Fu_12 * Fu_21;

				const num Fu_renor_11 = Deltaq * Deltaq * Fu_11 / Denou;
				const num Fu_renor_12 = Deltaq * Deltap * Fu_12 / Denou;
				const num Fu_renor_21 = Deltaq * Deltap * Fu_21 / Denou;
				const num Fu_renor_22 = Deltap * Deltap * Fu_22 / Denou;
				const num Fu_renor[4] = {Fu_renor_11, Fu_renor_21, Fu_renor_12, Fu_renor_22}; 

				for (int j = 0; j < N; j++) cu_q_col[j] = gu[j + ld*q];
				cu_q_col[q] -= 1.0;
				for (int j = 0; j < N; j++) cu_p_col[j] = gu[j + ld*p];
				cu_p_col[p] -= 1.0;

				for (int j = 0; j < N; j++) cu_q_row[j] = gu[q + ld*j];
				for (int j = 0; j < N; j++) cu_p_row[j] = gu[p + ld*j];

				memcpy(cu_col, cu_q_col, ld * sizeof(num));
				memcpy(cu_col + ld, cu_p_col, ld * sizeof(num));

				for (int j = 0; j < N; j++) {
					cu_row[j*2] = cu_q_row[j];
					cu_row[j*2 + 1] = cu_p_row[j];
				}

				xgemm("N", "N", N, 2, 2, 1.0, cu_col, ld, Fu_renor, 2, 0.0, tmpuld2, ld);
				xgemm("N", "N", N, N, 2, 1.0, tmpuld2, ld, cu_row, 2, 1.0, gu, ld);

				const num Fd_11 = Deltap * (1. + Deltap * (1.0 - gd[p + ld*p]));
				const num Fd_12 = Deltaq * Deltap * gd[q + ld*p];
				const num Fd_21 = Deltaq * Deltap * gd[p + ld*q];
				const num Fd_22 = Deltaq * (1.0 + Deltaq * (1.0 - gd[q + ld*q]));
				const num Denod = Fd_11 * Fd_22 - Fd_12 * Fd_21;

				const num Fd_renor_11 = Deltaq * Deltaq * Fd_11 / Denod;
				const num Fd_renor_12 = Deltaq * Deltap * Fd_12 / Denod;
				const num Fd_renor_21 = Deltaq * Deltap * Fd_21 / Denod;
				const num Fd_renor_22 = Deltap * Deltap * Fd_22 / Denod;
				const num Fd_renor[4] = {Fd_renor_11, Fd_renor_21, Fd_renor_12, Fd_renor_22}; 

				for (int j = 0; j < N; j++) cd_q_col[j] = gd[j + ld*q];
				cd_q_col[q] -= 1.0;
				for (int j = 0; j < N; j++) cd_p_col[j] = gd[j + ld*p];
				cd_p_col[p] -= 1.0;

				for (int j = 0; j < N; j++) cd_q_row[j] = gd[q + ld*j];
				for (int j = 0; j < N; j++) cd_p_row[j] = gd[p + ld*j];

				memcpy(cd_col, cd_q_col, ld * sizeof(num));
				memcpy(cd_col + ld, cd_p_col, ld * sizeof(num));

				for (int j = 0; j < N; j++) {
					cd_row[j*2] = cd_q_row[j];
					cd_row[j*2 + 1] = cd_p_row[j];
				}

				xgemm("N", "N", N, 2, 2, 1.0, cd_col, ld, Fd_renor, 2, 0.0, tmpdld2, ld);
				xgemm("N", "N", N, N, 2, 1.0, tmpdld2, ld, cd_row, 2, 1.0, gd, ld);
			
			}

			*phase *= prob/absprob;
			hs[i] = after_jump;
		}
	}
}




/*
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

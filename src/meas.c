#include "meas.h"
#include "data.h"
#include "linalg.h"

// number of types of bonds kept for 4-particle nematic correlators.
// 2 by default since these are slow measurerments
#define NEM_BONDS 2

// if complex numbers are being used, multiple some measurements by Peierls
// phases to preserve gauge invariance
#ifdef USE_CPLX
#define USE_PEIERLS
#else
// if not, these are equal to 1 anyway. multiplying by the variables costs a
// little performance, so #define them away at compile time
// TODO: exception: if using twisted boundaries, these are not always 1
#define pui0i1 1
#define pui1i0 1
#define pdi0i1 1
#define pdi1i0 1
#define puj0j1 1
#define puj1j0 1
#define pdj0j1 1
#define pdj1j0 1
#endif

void measure_eqlt(const struct params *const restrict p, const num phase,
		const num *const restrict gu,
		const num *const restrict gd,
		struct meas_eqlt *const restrict m)
{
	m->n_sample++;
	m->sign += phase;
	const int N = p->N, num_i = p->num_i, num_ij = p->num_ij;
	const int num_b = p->num_b, num_bs = p->num_bs, num_bb = p->num_bb;
	const int meas_energy_corr = p->meas_energy_corr;

	// 1 site measurements
	for (int i = 0; i < N; i++) {
		const int r = p->map_i[i];
		const num pre = phase / p->degen_i[r];
		const num guii = gu[i + i*N], gdii = gd[i + i*N];
		m->density[r] += pre*(2. - guii - gdii);
		m->double_occ[r] += pre*(1. - guii)*(1. - gdii);
	}

	// 2 site measurements
	for (int j = 0; j < N; j++)
	for (int i = 0; i < N; i++) {
		const int delta = (i == j);
		const int r = p->map_ij[i + j*N];
		const num pre = phase / p->degen_ij[r];
		const num guii = gu[i + i*N], gdii = gd[i + i*N];
		const num guij = gu[i + j*N], gdij = gd[i + j*N];
		const num guji = gu[j + i*N], gdji = gd[j + i*N];
		const num gujj = gu[j + j*N], gdjj = gd[j + j*N];
#ifdef USE_PEIERLS
		m->g00[r] += 0.5*pre*(guij*p->peierlsu[j + i*N] + gdij*p->peierlsd[j + i*N]);
#else
		m->g00[r] += 0.5*pre*(guij + gdij);
#endif
		const num x = delta*(guii + gdii) - (guji*guij + gdji*gdij);
		m->nn[r] += pre*((2. - guii - gdii)*(2. - gujj - gdjj) + x);
		m->xx[r] += 0.25*pre*(delta*(guii + gdii) - (guji*gdij + gdji*guij));
		m->zz[r] += 0.25*pre*((gdii - guii)*(gdjj - gujj) + x);
		m->pair_sw[r] += pre*guij*gdij;
		if (meas_energy_corr) {
			const double nuinuj = (1. - guii)*(1. - gujj) + (delta - guji)*guij;
			const double ndindj = (1. - gdii)*(1. - gdjj) + (delta - gdji)*gdij;
			m->vv[r] += pre*nuinuj*ndindj;
			m->vn[r] += pre*(nuinuj*(1. - gdii) + (1. - guii)*ndindj);
		}
	}

	if (!meas_energy_corr)
		return;

	// 1 bond 1 site measurements
	for (int j = 0; j < N; j++)
	for (int b = 0; b < num_b; b++) {
		const int i0 = p->bonds[b];
		const int i1 = p->bonds[b + num_b];
#ifdef USE_PEIERLS
		const num pui0i1 = p->peierlsu[i0 + N*i1];
		const num pui1i0 = p->peierlsu[i1 + N*i0];
		const num pdi0i1 = p->peierlsd[i0 + N*i1];
		const num pdi1i0 = p->peierlsd[i1 + N*i0];
#endif
		const int bs = p->map_bs[b + num_b*j];
		const num pre = phase / p->degen_bs[bs];
		const int delta_i0i1 = 0;
		const int delta_i0j = (i0 == j);
		const int delta_i1j = (i1 == j);
		const num gui0j = gu[i0 + N*j];
		const num guji0 = gu[j + N*i0];
		const num gdi0j = gd[i0 + N*j];
		const num gdji0 = gd[j + N*i0];
		const num gui1j = gu[i1 + N*j];
		const num guji1 = gu[j + N*i1];
		const num gdi1j = gd[i1 + N*j];
		const num gdji1 = gd[j + N*i1];
		const num gui0i1 = gu[i0 + N*i1];
		const num gui1i0 = gu[i1 + N*i0];
		const num gdi0i1 = gd[i0 + N*i1];
		const num gdi1i0 = gd[i1 + N*i0];
		const num gujj = gu[j + N*j];
		const num gdjj = gd[j + N*j];

		const num ku = pui1i0*(delta_i0i1 - gui0i1) + pui0i1*(delta_i0i1 - gui1i0);
		const num kd = pdi1i0*(delta_i0i1 - gdi0i1) + pdi0i1*(delta_i0i1 - gdi1i0);
		const num xu = pui0i1*(delta_i0j - guji0)*gui1j + pui1i0*(delta_i1j - guji1)*gui0j;
		const num xd = pdi0i1*(delta_i0j - gdji0)*gdi1j + pdi1i0*(delta_i1j - gdji1)*gdi0j;
		m->kv[bs] += pre*((ku*(1. - gujj) + xu)*(1. - gdjj)
		                + (kd*(1. - gdjj) + xd)*(1. - gujj));
		m->kn[bs] += pre*((ku + kd)*(2. - gujj - gdjj) + xu + xd);
	}

	// 2 bond measurements
	for (int c = 0; c < num_b; c++) {
		const int j0 = p->bonds[c];
		const int j1 = p->bonds[c + num_b];
#ifdef USE_PEIERLS
		const num puj0j1 = p->peierlsu[j0 + N*j1];
		const num puj1j0 = p->peierlsu[j1 + N*j0];
		const num pdj0j1 = p->peierlsd[j0 + N*j1];
		const num pdj1j0 = p->peierlsd[j1 + N*j0];
#endif
	for (int b = 0; b < num_b; b++) {
		const int i0 = p->bonds[b];
		const int i1 = p->bonds[b + num_b];
#ifdef USE_PEIERLS
		const num pui0i1 = p->peierlsu[i0 + N*i1];
		const num pui1i0 = p->peierlsu[i1 + N*i0];
		const num pdi0i1 = p->peierlsd[i0 + N*i1];
		const num pdi1i0 = p->peierlsd[i1 + N*i0];
#endif
		const int bb = p->map_bb[b + c*num_b];
		const num pre = phase / p->degen_bb[bb];
		const int delta_i0j0 = (i0 == j0);
		const int delta_i1j0 = (i1 == j0);
		const int delta_i0j1 = (i0 == j1);
		const int delta_i1j1 = (i1 == j1);
		const num gui1i0 = gu[i1 + i0*N];
		const num gui0i1 = gu[i0 + i1*N];
		const num gui0j0 = gu[i0 + j0*N];
		const num gui1j0 = gu[i1 + j0*N];
		const num gui0j1 = gu[i0 + j1*N];
		const num gui1j1 = gu[i1 + j1*N];
		const num guj0i0 = gu[j0 + i0*N];
		const num guj1i0 = gu[j1 + i0*N];
		const num guj0i1 = gu[j0 + i1*N];
		const num guj1i1 = gu[j1 + i1*N];
		const num guj1j0 = gu[j1 + j0*N];
		const num guj0j1 = gu[j0 + j1*N];
		const num gdi1i0 = gd[i1 + i0*N];
		const num gdi0i1 = gd[i0 + i1*N];
		const num gdi0j0 = gd[i0 + j0*N];
		const num gdi1j0 = gd[i1 + j0*N];
		const num gdi0j1 = gd[i0 + j1*N];
		const num gdi1j1 = gd[i1 + j1*N];
		const num gdj0i0 = gd[j0 + i0*N];
		const num gdj1i0 = gd[j1 + i0*N];
		const num gdj0i1 = gd[j0 + i1*N];
		const num gdj1i1 = gd[j1 + i1*N];
		const num gdj1j0 = gd[j1 + j0*N];
		const num gdj0j1 = gd[j0 + j1*N];
		const num x = pui0i1*puj0j1*(delta_i0j1 - guj1i0)*gui1j0 + pui1i0*puj1j0*(delta_i1j0 - guj0i1)*gui0j1
		            + pdi0i1*pdj0j1*(delta_i0j1 - gdj1i0)*gdi1j0 + pdi1i0*pdj1j0*(delta_i1j0 - gdj0i1)*gdi0j1;
		const num y = pui0i1*puj1j0*(delta_i0j0 - guj0i0)*gui1j1 + pui1i0*puj0j1*(delta_i1j1 - guj1i1)*gui0j0
		            + pdi0i1*pdj1j0*(delta_i0j0 - gdj0i0)*gdi1j1 + pdi1i0*pdj0j1*(delta_i1j1 - gdj1i1)*gdi0j0;
		m->kk[bb] += pre*((pui1i0*gui0i1 + pui0i1*gui1i0 + pdi1i0*gdi0i1 + pdi0i1*gdi1i0)
		                 *(puj1j0*guj0j1 + puj0j1*guj1j0 + pdj1j0*gdj0j1 + pdj0j1*gdj1j0) + x + y);
	}
	}
}

void measure_uneqlt(const struct params *const restrict p, const num phase,
		const num *const Gu0t,
		const num *const Gutt,
		const num *const Gut0,
		const num *const Gd0t,
		const num *const Gdtt,
		const num *const Gdt0,
		struct meas_uneqlt *const restrict m)
{
	m->n_sample++;
	m->sign += phase;
	const int N = p->N, L = p->L, num_i = p->num_i, num_ij = p->num_ij;
	const int num_b = p->num_b, num_bs = p->num_bs, num_bb = p->num_bb;
	const int meas_bond_corr = p->meas_bond_corr;
	const int meas_energy_corr = p->meas_energy_corr;
	const int meas_nematic_corr = p->meas_nematic_corr;

	const num *const restrict Gu00 = Gutt;
	const num *const restrict Gd00 = Gdtt;

	// 2 site measurements
	#pragma omp parallel for
	for (int t = 0; t < L; t++) {
		const int delta_t = (t == 0);
		const num *const restrict Gu0t_t = Gu0t + N*N*t;
		const num *const restrict Gutt_t = Gutt + N*N*t;
		const num *const restrict Gut0_t = Gut0 + N*N*t;
		const num *const restrict Gd0t_t = Gd0t + N*N*t;
		const num *const restrict Gdtt_t = Gdtt + N*N*t;
		const num *const restrict Gdt0_t = Gdt0 + N*N*t;
	for (int j = 0; j < N; j++)
	for (int i = 0; i < N; i++) {
		const int r = p->map_ij[i + j*N];
		const int delta_tij = delta_t * (i == j);
		const num pre = phase / p->degen_ij[r];
		const num guii = Gutt_t[i + N*i];
		const num guij = Gut0_t[i + N*j];
		const num guji = Gu0t_t[j + N*i];
		const num gujj = Gu00[j + N*j];
		const num gdii = Gdtt_t[i + N*i];
		const num gdij = Gdt0_t[i + N*j];
		const num gdji = Gd0t_t[j + N*i];
		const num gdjj = Gd00[j + N*j];
#ifdef USE_PEIERLS
		m->gt0[r + num_ij*t] += 0.5*pre*(guij*p->peierlsu[j + i*N] + gdij*p->peierlsd[j + i*N]);
#else
		m->gt0[r + num_ij*t] += 0.5*pre*(guij + gdij);
#endif
		const num x = delta_tij*(guii + gdii) - (guji*guij + gdji*gdij);
		m->nn[r + num_ij*t] += pre*((2. - guii - gdii)*(2. - gujj - gdjj) + x);
		m->xx[r + num_ij*t] += 0.25*pre*(delta_tij*(guii + gdii) - (guji*gdij + gdji*guij));
		m->zz[r + num_ij*t] += 0.25*pre*((gdii - guii)*(gdjj - gujj) + x);
		m->pair_sw[r + num_ij*t] += pre*guij*gdij;
		if (meas_energy_corr) {
			const num nuinuj = (1. - guii)*(1. - gujj) + (delta_tij - guji)*guij;
			const num ndindj = (1. - gdii)*(1. - gdjj) + (delta_tij - gdji)*gdij;
			m->vv[r + num_ij*t] += pre*nuinuj*ndindj;
			m->vn[r + num_ij*t] += pre*(nuinuj*(1. - gdii) + (1. - guii)*ndindj);
		}
	}
	}

	// 1 bond 1 site measurements
	if (meas_energy_corr)
	#pragma omp parallel for
	for (int t = 0; t < L; t++) {
		const int delta_t = (t == 0);
		const num *const restrict Gu0t_t = Gu0t + N*N*t;
		const num *const restrict Gutt_t = Gutt + N*N*t;
		const num *const restrict Gut0_t = Gut0 + N*N*t;
		const num *const restrict Gd0t_t = Gd0t + N*N*t;
		const num *const restrict Gdtt_t = Gdtt + N*N*t;
		const num *const restrict Gdt0_t = Gdt0 + N*N*t;
	for (int j = 0; j < N; j++)
	for (int b = 0; b < num_b; b++) {
		const int i0 = p->bonds[b];
		const int i1 = p->bonds[b + num_b];
#ifdef USE_PEIERLS
		const num pui0i1 = p->peierlsu[i0 + N*i1];
		const num pui1i0 = p->peierlsu[i1 + N*i0];
		const num pdi0i1 = p->peierlsd[i0 + N*i1];
		const num pdi1i0 = p->peierlsd[i1 + N*i0];
#endif
		const int bs = p->map_bs[b + num_b*j];
		const num pre = phase / p->degen_bs[bs];
		const int delta_i0i1 = 0;
		const int delta_i0j = delta_t*(i0 == j);
		const int delta_i1j = delta_t*(i1 == j);
		const num gui0j = Gut0_t[i0 + N*j];
		const num guji0 = Gu0t_t[j + N*i0];
		const num gdi0j = Gdt0_t[i0 + N*j];
		const num gdji0 = Gd0t_t[j + N*i0];
		const num gui1j = Gut0_t[i1 + N*j];
		const num guji1 = Gu0t_t[j + N*i1];
		const num gdi1j = Gdt0_t[i1 + N*j];
		const num gdji1 = Gd0t_t[j + N*i1];
		const num gui0i1 = Gutt_t[i0 + N*i1];
		const num gui1i0 = Gutt_t[i1 + N*i0];
		const num gdi0i1 = Gdtt_t[i0 + N*i1];
		const num gdi1i0 = Gdtt_t[i1 + N*i0];
		const num gujj = Gu00[j + N*j];
		const num gdjj = Gd00[j + N*j];

		const num ku = pui1i0*(delta_i0i1 - gui0i1) + pui0i1*(delta_i0i1 - gui1i0);
		const num kd = pdi1i0*(delta_i0i1 - gdi0i1) + pdi0i1*(delta_i0i1 - gdi1i0);
		const num xu = pui0i1*(delta_i0j - guji0)*gui1j + pui1i0*(delta_i1j - guji1)*gui0j;
		const num xd = pdi0i1*(delta_i0j - gdji0)*gdi1j + pdi1i0*(delta_i1j - gdji1)*gdi0j;
		m->kv[bs + num_bs*t] += pre*((ku*(1. - gujj) + xu)*(1. - gdjj)
		                           + (kd*(1. - gdjj) + xd)*(1. - gujj));
		m->kn[bs + num_bs*t] += pre*((ku + kd)*(2. - gujj - gdjj) + xu + xd);
	}
	}

	// 2 bond measurements
	// minor optimization: handle t = 0 separately, since there are no delta
	// functions for t > 0. not really needed in 2-site measurements above
	// as those are fast anyway
	if (meas_bond_corr)
	for (int c = 0; c < num_b; c++) {
		const int j0 = p->bonds[c];
		const int j1 = p->bonds[c + num_b];
#ifdef USE_PEIERLS
		const num puj0j1 = p->peierlsu[j0 + N*j1];
		const num puj1j0 = p->peierlsu[j1 + N*j0];
		const num pdj0j1 = p->peierlsd[j0 + N*j1];
		const num pdj1j0 = p->peierlsd[j1 + N*j0];
#endif
	for (int b = 0; b < num_b; b++) {
		const int i0 = p->bonds[b];
		const int i1 = p->bonds[b + num_b];
#ifdef USE_PEIERLS
		const num pui0i1 = p->peierlsu[i0 + N*i1];
		const num pui1i0 = p->peierlsu[i1 + N*i0];
		const num pdi0i1 = p->peierlsd[i0 + N*i1];
		const num pdi1i0 = p->peierlsd[i1 + N*i0];
#endif
		const int bb = p->map_bb[b + c*num_b];
		const num pre = phase / p->degen_bb[bb];
		const int delta_i0j0 = (i0 == j0);
		const int delta_i1j0 = (i1 == j0);
		const int delta_i0j1 = (i0 == j1);
		const int delta_i1j1 = (i1 == j1);
		const num gui1i0 = Gu00[i1 + i0*N];
		const num gui0i1 = Gu00[i0 + i1*N];
		const num gui0j0 = Gu00[i0 + j0*N];
		const num gui1j0 = Gu00[i1 + j0*N];
		const num gui0j1 = Gu00[i0 + j1*N];
		const num gui1j1 = Gu00[i1 + j1*N];
		const num guj0i0 = Gu00[j0 + i0*N];
		const num guj1i0 = Gu00[j1 + i0*N];
		const num guj0i1 = Gu00[j0 + i1*N];
		const num guj1i1 = Gu00[j1 + i1*N];
		const num guj1j0 = Gu00[j1 + j0*N];
		const num guj0j1 = Gu00[j0 + j1*N];
		const num gdi1i0 = Gd00[i1 + i0*N];
		const num gdi0i1 = Gd00[i0 + i1*N];
		const num gdi0j0 = Gd00[i0 + j0*N];
		const num gdi1j0 = Gd00[i1 + j0*N];
		const num gdi0j1 = Gd00[i0 + j1*N];
		const num gdi1j1 = Gd00[i1 + j1*N];
		const num gdj0i0 = Gd00[j0 + i0*N];
		const num gdj1i0 = Gd00[j1 + i0*N];
		const num gdj0i1 = Gd00[j0 + i1*N];
		const num gdj1i1 = Gd00[j1 + i1*N];
		const num gdj1j0 = Gd00[j1 + j0*N];
		const num gdj0j1 = Gd00[j0 + j1*N];
		m->pair_bb[bb] += 0.5*pre*(gui0j0*gdi1j1 + gui1j0*gdi0j1 + gui0j1*gdi1j0 + gui1j1*gdi0j0);
		const num x = pui0i1*puj0j1*(delta_i0j1 - guj1i0)*gui1j0 + pui1i0*puj1j0*(delta_i1j0 - guj0i1)*gui0j1
		            + pdi0i1*pdj0j1*(delta_i0j1 - gdj1i0)*gdi1j0 + pdi1i0*pdj1j0*(delta_i1j0 - gdj0i1)*gdi0j1;
		const num y = pui0i1*puj1j0*(delta_i0j0 - guj0i0)*gui1j1 + pui1i0*puj0j1*(delta_i1j1 - guj1i1)*gui0j0
		            + pdi0i1*pdj1j0*(delta_i0j0 - gdj0i0)*gdi1j1 + pdi1i0*pdj0j1*(delta_i1j1 - gdj1i1)*gdi0j0;
		m->jj[bb]   += pre*((pui1i0*gui0i1 - pui0i1*gui1i0 + pdi1i0*gdi0i1 - pdi0i1*gdi1i0)
		                   *(puj1j0*guj0j1 - puj0j1*guj1j0 + pdj1j0*gdj0j1 - pdj0j1*gdj1j0) + x - y);
		m->jsjs[bb] += pre*((pui1i0*gui0i1 - pui0i1*gui1i0 - pdi1i0*gdi0i1 + pdi0i1*gdi1i0)
		                   *(puj1j0*guj0j1 - puj0j1*guj1j0 - pdj1j0*gdj0j1 + pdj0j1*gdj1j0) + x - y);
		m->kk[bb]   += pre*((pui1i0*gui0i1 + pui0i1*gui1i0 + pdi1i0*gdi0i1 + pdi0i1*gdi1i0)
		                   *(puj1j0*guj0j1 + puj0j1*guj1j0 + pdj1j0*gdj0j1 + pdj0j1*gdj1j0) + x + y);
		m->ksks[bb] += pre*((pui1i0*gui0i1 + pui0i1*gui1i0 - pdi1i0*gdi0i1 - pdi0i1*gdi1i0)
		                   *(puj1j0*guj0j1 + puj0j1*guj1j0 - pdj1j0*gdj0j1 - pdj0j1*gdj1j0) + x + y);
	}
	}

	if (meas_nematic_corr)
	for (int c = 0; c < NEM_BONDS*N; c++) {
		const int j0 = p->bonds[c];
		const int j1 = p->bonds[c + num_b];
#ifdef USE_PEIERLS
		const num puj0j1 = p->peierlsu[j0 + N*j1];
		const num puj1j0 = p->peierlsu[j1 + N*j0];
		const num pdj0j1 = p->peierlsd[j0 + N*j1];
		const num pdj1j0 = p->peierlsd[j1 + N*j0];
#endif
	for (int b = 0; b < NEM_BONDS*N; b++) {
		const int i0 = p->bonds[b];
		const int i1 = p->bonds[b + num_b];
#ifdef USE_PEIERLS
		const num pui0i1 = p->peierlsu[i0 + N*i1];
		const num pui1i0 = p->peierlsu[i1 + N*i0];
		const num pdi0i1 = p->peierlsd[i0 + N*i1];
		const num pdi1i0 = p->peierlsd[i1 + N*i0];
#endif
		const int bb = p->map_bb[b + c*num_b];
		const num pre = phase / p->degen_bb[bb];
		const int delta_i0j0 = (i0 == j0);
		const int delta_i1j0 = (i1 == j0);
		const int delta_i0j1 = (i0 == j1);
		const int delta_i1j1 = (i1 == j1);
		const num gui0i0 = Gu00[i0 + i0*N];
		const num gui1i0 = Gu00[i1 + i0*N];
		const num gui0i1 = Gu00[i0 + i1*N];
		const num gui1i1 = Gu00[i1 + i1*N];
		const num gui0j0 = Gu00[i0 + j0*N];
		const num gui1j0 = Gu00[i1 + j0*N];
		const num gui0j1 = Gu00[i0 + j1*N];
		const num gui1j1 = Gu00[i1 + j1*N];
		const num guj0i0 = Gu00[j0 + i0*N];
		const num guj1i0 = Gu00[j1 + i0*N];
		const num guj0i1 = Gu00[j0 + i1*N];
		const num guj1i1 = Gu00[j1 + i1*N];
		const num guj0j0 = Gu00[j0 + j0*N];
		const num guj1j0 = Gu00[j1 + j0*N];
		const num guj0j1 = Gu00[j0 + j1*N];
		const num guj1j1 = Gu00[j1 + j1*N];
		const num gdi0i0 = Gd00[i0 + i0*N];
		const num gdi1i0 = Gd00[i1 + i0*N];
		const num gdi0i1 = Gd00[i0 + i1*N];
		const num gdi1i1 = Gd00[i1 + i1*N];
		const num gdi0j0 = Gd00[i0 + j0*N];
		const num gdi1j0 = Gd00[i1 + j0*N];
		const num gdi0j1 = Gd00[i0 + j1*N];
		const num gdi1j1 = Gd00[i1 + j1*N];
		const num gdj0i0 = Gd00[j0 + i0*N];
		const num gdj1i0 = Gd00[j1 + i0*N];
		const num gdj0i1 = Gd00[j0 + i1*N];
		const num gdj1i1 = Gd00[j1 + i1*N];
		const num gdj0j0 = Gd00[j0 + j0*N];
		const num gdj1j0 = Gd00[j1 + j0*N];
		const num gdj0j1 = Gd00[j0 + j1*N];
		const num gdj1j1 = Gd00[j1 + j1*N];
		const int delta_i0i1 = 0;
		const int delta_j0j1 = 0;
		const num uuuu = +(1.-gui0i0)*(1.-gui1i1)*(1.-guj0j0)*(1.-guj1j1)+(1.-gui0i0)*(1.-gui1i1)*(delta_j0j1-guj1j0)*guj0j1+(1.-gui0i0)*(delta_i1j0-guj0i1)*gui1j0*(1.-guj1j1)-(1.-gui0i0)*(delta_i1j0-guj0i1)*gui1j1*(delta_j0j1-guj1j0)+(1.-gui0i0)*(delta_i1j1-guj1i1)*gui1j0*guj0j1+(1.-gui0i0)*(delta_i1j1-guj1i1)*gui1j1*(1.-guj0j0)+(delta_i0i1-gui1i0)*gui0i1*(1.-guj0j0)*(1.-guj1j1)+(delta_i0i1-gui1i0)*gui0i1*(delta_j0j1-guj1j0)*guj0j1-(delta_i0i1-gui1i0)*gui0j0*(delta_i1j0-guj0i1)*(1.-guj1j1)-(delta_i0i1-gui1i0)*gui0j0*(delta_i1j1-guj1i1)*guj0j1+(delta_i0i1-gui1i0)*gui0j1*(delta_i1j0-guj0i1)*(delta_j0j1-guj1j0)-(delta_i0i1-gui1i0)*gui0j1*(delta_i1j1-guj1i1)*(1.-guj0j0)+(delta_i0j0-guj0i0)*gui0i1*gui1j0*(1.-guj1j1)-(delta_i0j0-guj0i0)*gui0i1*gui1j1*(delta_j0j1-guj1j0)+(delta_i0j0-guj0i0)*gui0j0*(1.-gui1i1)*(1.-guj1j1)+(delta_i0j0-guj0i0)*gui0j0*(delta_i1j1-guj1i1)*gui1j1-(delta_i0j0-guj0i0)*gui0j1*(1.-gui1i1)*(delta_j0j1-guj1j0)-(delta_i0j0-guj0i0)*gui0j1*(delta_i1j1-guj1i1)*gui1j0+(delta_i0j1-guj1i0)*gui0i1*gui1j0*guj0j1+(delta_i0j1-guj1i0)*gui0i1*gui1j1*(1.-guj0j0)+(delta_i0j1-guj1i0)*gui0j0*(1.-gui1i1)*guj0j1-(delta_i0j1-guj1i0)*gui0j0*(delta_i1j0-guj0i1)*gui1j1+(delta_i0j1-guj1i0)*gui0j1*(1.-gui1i1)*(1.-guj0j0)+(delta_i0j1-guj1i0)*gui0j1*(delta_i1j0-guj0i1)*gui1j0;
		const num uuud = +(1.-gui0i0)*(1.-gui1i1)*(1.-guj0j0)*(1.-gdj1j1)+(1.-gui0i0)*(delta_i1j0-guj0i1)*gui1j0*(1.-gdj1j1)+(delta_i0i1-gui1i0)*gui0i1*(1.-guj0j0)*(1.-gdj1j1)-(delta_i0i1-gui1i0)*gui0j0*(delta_i1j0-guj0i1)*(1.-gdj1j1)+(delta_i0j0-guj0i0)*gui0i1*gui1j0*(1.-gdj1j1)+(delta_i0j0-guj0i0)*gui0j0*(1.-gui1i1)*(1.-gdj1j1);
		const num uudu = +(1.-gui0i0)*(1.-gui1i1)*(1.-gdj0j0)*(1.-guj1j1)+(1.-gui0i0)*(delta_i1j1-guj1i1)*gui1j1*(1.-gdj0j0)+(delta_i0i1-gui1i0)*gui0i1*(1.-gdj0j0)*(1.-guj1j1)-(delta_i0i1-gui1i0)*gui0j1*(delta_i1j1-guj1i1)*(1.-gdj0j0)+(delta_i0j1-guj1i0)*gui0i1*gui1j1*(1.-gdj0j0)+(delta_i0j1-guj1i0)*gui0j1*(1.-gui1i1)*(1.-gdj0j0);
		const num uudd = +(1.-gui0i0)*(1.-gui1i1)*(1.-gdj0j0)*(1.-gdj1j1)+(1.-gui0i0)*(1.-gui1i1)*(delta_j0j1-gdj1j0)*gdj0j1+(delta_i0i1-gui1i0)*gui0i1*(1.-gdj0j0)*(1.-gdj1j1)+(delta_i0i1-gui1i0)*gui0i1*(delta_j0j1-gdj1j0)*gdj0j1;
		const num uduu = +(1.-gui0i0)*(1.-gdi1i1)*(1.-guj0j0)*(1.-guj1j1)+(1.-gui0i0)*(1.-gdi1i1)*(delta_j0j1-guj1j0)*guj0j1+(delta_i0j0-guj0i0)*gui0j0*(1.-gdi1i1)*(1.-guj1j1)-(delta_i0j0-guj0i0)*gui0j1*(1.-gdi1i1)*(delta_j0j1-guj1j0)+(delta_i0j1-guj1i0)*gui0j0*(1.-gdi1i1)*guj0j1+(delta_i0j1-guj1i0)*gui0j1*(1.-gdi1i1)*(1.-guj0j0);
		const num udud = +(1.-gui0i0)*(1.-gdi1i1)*(1.-guj0j0)*(1.-gdj1j1)+(1.-gui0i0)*(delta_i1j1-gdj1i1)*gdi1j1*(1.-guj0j0)+(delta_i0j0-guj0i0)*gui0j0*(1.-gdi1i1)*(1.-gdj1j1)+(delta_i0j0-guj0i0)*gui0j0*(delta_i1j1-gdj1i1)*gdi1j1;
		const num uddu = +(1.-gui0i0)*(1.-gdi1i1)*(1.-gdj0j0)*(1.-guj1j1)+(1.-gui0i0)*(delta_i1j0-gdj0i1)*gdi1j0*(1.-guj1j1)+(delta_i0j1-guj1i0)*gui0j1*(1.-gdi1i1)*(1.-gdj0j0)+(delta_i0j1-guj1i0)*gui0j1*(delta_i1j0-gdj0i1)*gdi1j0;
		const num uddd = +(1.-gui0i0)*(1.-gdi1i1)*(1.-gdj0j0)*(1.-gdj1j1)+(1.-gui0i0)*(1.-gdi1i1)*(delta_j0j1-gdj1j0)*gdj0j1+(1.-gui0i0)*(delta_i1j0-gdj0i1)*gdi1j0*(1.-gdj1j1)-(1.-gui0i0)*(delta_i1j0-gdj0i1)*gdi1j1*(delta_j0j1-gdj1j0)+(1.-gui0i0)*(delta_i1j1-gdj1i1)*gdi1j0*gdj0j1+(1.-gui0i0)*(delta_i1j1-gdj1i1)*gdi1j1*(1.-gdj0j0);
		const num duuu = +(1.-gdi0i0)*(1.-gui1i1)*(1.-guj0j0)*(1.-guj1j1)+(1.-gdi0i0)*(1.-gui1i1)*(delta_j0j1-guj1j0)*guj0j1+(1.-gdi0i0)*(delta_i1j0-guj0i1)*gui1j0*(1.-guj1j1)-(1.-gdi0i0)*(delta_i1j0-guj0i1)*gui1j1*(delta_j0j1-guj1j0)+(1.-gdi0i0)*(delta_i1j1-guj1i1)*gui1j0*guj0j1+(1.-gdi0i0)*(delta_i1j1-guj1i1)*gui1j1*(1.-guj0j0);
		const num duud = +(1.-gdi0i0)*(1.-gui1i1)*(1.-guj0j0)*(1.-gdj1j1)+(1.-gdi0i0)*(delta_i1j0-guj0i1)*gui1j0*(1.-gdj1j1)+(delta_i0j1-gdj1i0)*gdi0j1*(1.-gui1i1)*(1.-guj0j0)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i1j0-guj0i1)*gui1j0;
		const num dudu = +(1.-gdi0i0)*(1.-gui1i1)*(1.-gdj0j0)*(1.-guj1j1)+(1.-gdi0i0)*(delta_i1j1-guj1i1)*gui1j1*(1.-gdj0j0)+(delta_i0j0-gdj0i0)*gdi0j0*(1.-gui1i1)*(1.-guj1j1)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i1j1-guj1i1)*gui1j1;
		const num dudd = +(1.-gdi0i0)*(1.-gui1i1)*(1.-gdj0j0)*(1.-gdj1j1)+(1.-gdi0i0)*(1.-gui1i1)*(delta_j0j1-gdj1j0)*gdj0j1+(delta_i0j0-gdj0i0)*gdi0j0*(1.-gui1i1)*(1.-gdj1j1)-(delta_i0j0-gdj0i0)*gdi0j1*(1.-gui1i1)*(delta_j0j1-gdj1j0)+(delta_i0j1-gdj1i0)*gdi0j0*(1.-gui1i1)*gdj0j1+(delta_i0j1-gdj1i0)*gdi0j1*(1.-gui1i1)*(1.-gdj0j0);
		const num dduu = +(1.-gdi0i0)*(1.-gdi1i1)*(1.-guj0j0)*(1.-guj1j1)+(1.-gdi0i0)*(1.-gdi1i1)*(delta_j0j1-guj1j0)*guj0j1+(delta_i0i1-gdi1i0)*gdi0i1*(1.-guj0j0)*(1.-guj1j1)+(delta_i0i1-gdi1i0)*gdi0i1*(delta_j0j1-guj1j0)*guj0j1;
		const num ddud = +(1.-gdi0i0)*(1.-gdi1i1)*(1.-guj0j0)*(1.-gdj1j1)+(1.-gdi0i0)*(delta_i1j1-gdj1i1)*gdi1j1*(1.-guj0j0)+(delta_i0i1-gdi1i0)*gdi0i1*(1.-guj0j0)*(1.-gdj1j1)-(delta_i0i1-gdi1i0)*gdi0j1*(delta_i1j1-gdj1i1)*(1.-guj0j0)+(delta_i0j1-gdj1i0)*gdi0i1*gdi1j1*(1.-guj0j0)+(delta_i0j1-gdj1i0)*gdi0j1*(1.-gdi1i1)*(1.-guj0j0);
		const num dddu = +(1.-gdi0i0)*(1.-gdi1i1)*(1.-gdj0j0)*(1.-guj1j1)+(1.-gdi0i0)*(delta_i1j0-gdj0i1)*gdi1j0*(1.-guj1j1)+(delta_i0i1-gdi1i0)*gdi0i1*(1.-gdj0j0)*(1.-guj1j1)-(delta_i0i1-gdi1i0)*gdi0j0*(delta_i1j0-gdj0i1)*(1.-guj1j1)+(delta_i0j0-gdj0i0)*gdi0i1*gdi1j0*(1.-guj1j1)+(delta_i0j0-gdj0i0)*gdi0j0*(1.-gdi1i1)*(1.-guj1j1);
		const num dddd = +(1.-gdi0i0)*(1.-gdi1i1)*(1.-gdj0j0)*(1.-gdj1j1)+(1.-gdi0i0)*(1.-gdi1i1)*(delta_j0j1-gdj1j0)*gdj0j1+(1.-gdi0i0)*(delta_i1j0-gdj0i1)*gdi1j0*(1.-gdj1j1)-(1.-gdi0i0)*(delta_i1j0-gdj0i1)*gdi1j1*(delta_j0j1-gdj1j0)+(1.-gdi0i0)*(delta_i1j1-gdj1i1)*gdi1j0*gdj0j1+(1.-gdi0i0)*(delta_i1j1-gdj1i1)*gdi1j1*(1.-gdj0j0)+(delta_i0i1-gdi1i0)*gdi0i1*(1.-gdj0j0)*(1.-gdj1j1)+(delta_i0i1-gdi1i0)*gdi0i1*(delta_j0j1-gdj1j0)*gdj0j1-(delta_i0i1-gdi1i0)*gdi0j0*(delta_i1j0-gdj0i1)*(1.-gdj1j1)-(delta_i0i1-gdi1i0)*gdi0j0*(delta_i1j1-gdj1i1)*gdj0j1+(delta_i0i1-gdi1i0)*gdi0j1*(delta_i1j0-gdj0i1)*(delta_j0j1-gdj1j0)-(delta_i0i1-gdi1i0)*gdi0j1*(delta_i1j1-gdj1i1)*(1.-gdj0j0)+(delta_i0j0-gdj0i0)*gdi0i1*gdi1j0*(1.-gdj1j1)-(delta_i0j0-gdj0i0)*gdi0i1*gdi1j1*(delta_j0j1-gdj1j0)+(delta_i0j0-gdj0i0)*gdi0j0*(1.-gdi1i1)*(1.-gdj1j1)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i1j1-gdj1i1)*gdi1j1-(delta_i0j0-gdj0i0)*gdi0j1*(1.-gdi1i1)*(delta_j0j1-gdj1j0)-(delta_i0j0-gdj0i0)*gdi0j1*(delta_i1j1-gdj1i1)*gdi1j0+(delta_i0j1-gdj1i0)*gdi0i1*gdi1j0*gdj0j1+(delta_i0j1-gdj1i0)*gdi0i1*gdi1j1*(1.-gdj0j0)+(delta_i0j1-gdj1i0)*gdi0j0*(1.-gdi1i1)*gdj0j1-(delta_i0j1-gdj1i0)*gdi0j0*(delta_i1j0-gdj0i1)*gdi1j1+(delta_i0j1-gdj1i0)*gdi0j1*(1.-gdi1i1)*(1.-gdj0j0)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i1j0-gdj0i1)*gdi1j0;
		m->nem_nnnn[bb] += pre*(uuuu + uuud + uudu + uudd
				      + uduu + udud + uddu + uddd
				      + duuu + duud + dudu + dudd
				      + dduu + ddud + dddu + dddd);
		m->nem_ssss[bb] += pre*(uuuu - uuud - uudu + uudd
				      - uduu + udud + uddu - uddd
				      - duuu + duud + dudu - dudd
				      + dduu - ddud - dddu + dddd);
	}
	}

	// no delta functions here.
	if (meas_bond_corr)
	#pragma omp parallel for
	for (int t = 1; t < L; t++) {
		const num *const restrict Gu0t_t = Gu0t + N*N*t;
		const num *const restrict Gutt_t = Gutt + N*N*t;
		const num *const restrict Gut0_t = Gut0 + N*N*t;
		const num *const restrict Gd0t_t = Gd0t + N*N*t;
		const num *const restrict Gdtt_t = Gdtt + N*N*t;
		const num *const restrict Gdt0_t = Gdt0 + N*N*t;
	for (int c = 0; c < num_b; c++) {
		const int j0 = p->bonds[c];
		const int j1 = p->bonds[c + num_b];
#ifdef USE_PEIERLS
		const num puj0j1 = p->peierlsu[j0 + N*j1];
		const num puj1j0 = p->peierlsu[j1 + N*j0];
		const num pdj0j1 = p->peierlsd[j0 + N*j1];
		const num pdj1j0 = p->peierlsd[j1 + N*j0];
#endif
	for (int b = 0; b < num_b; b++) {
		const int i0 = p->bonds[b];
		const int i1 = p->bonds[b + num_b];
#ifdef USE_PEIERLS
		const num pui0i1 = p->peierlsu[i0 + N*i1];
		const num pui1i0 = p->peierlsu[i1 + N*i0];
		const num pdi0i1 = p->peierlsd[i0 + N*i1];
		const num pdi1i0 = p->peierlsd[i1 + N*i0];
#endif
		const int bb = p->map_bb[b + c*num_b];
		const num pre = phase / p->degen_bb[bb];
		const num gui0i0 = Gutt_t[i0 + i0*N];
		const num gui1i0 = Gutt_t[i1 + i0*N];
		const num gui0i1 = Gutt_t[i0 + i1*N];
		const num gui1i1 = Gutt_t[i1 + i1*N];
		const num gui0j0 = Gut0_t[i0 + j0*N];
		const num gui1j0 = Gut0_t[i1 + j0*N];
		const num gui0j1 = Gut0_t[i0 + j1*N];
		const num gui1j1 = Gut0_t[i1 + j1*N];
		const num guj0i0 = Gu0t_t[j0 + i0*N];
		const num guj1i0 = Gu0t_t[j1 + i0*N];
		const num guj0i1 = Gu0t_t[j0 + i1*N];
		const num guj1i1 = Gu0t_t[j1 + i1*N];
		const num guj0j0 = Gu00[j0 + j0*N];
		const num guj1j0 = Gu00[j1 + j0*N];
		const num guj0j1 = Gu00[j0 + j1*N];
		const num guj1j1 = Gu00[j1 + j1*N];
		const num gdi0i0 = Gdtt_t[i0 + i0*N];
		const num gdi1i0 = Gdtt_t[i1 + i0*N];
		const num gdi0i1 = Gdtt_t[i0 + i1*N];
		const num gdi1i1 = Gdtt_t[i1 + i1*N];
		const num gdi0j0 = Gdt0_t[i0 + j0*N];
		const num gdi1j0 = Gdt0_t[i1 + j0*N];
		const num gdi0j1 = Gdt0_t[i0 + j1*N];
		const num gdi1j1 = Gdt0_t[i1 + j1*N];
		const num gdj0i0 = Gd0t_t[j0 + i0*N];
		const num gdj1i0 = Gd0t_t[j1 + i0*N];
		const num gdj0i1 = Gd0t_t[j0 + i1*N];
		const num gdj1i1 = Gd0t_t[j1 + i1*N];
		const num gdj0j0 = Gd00[j0 + j0*N];
		const num gdj1j0 = Gd00[j1 + j0*N];
		const num gdj0j1 = Gd00[j0 + j1*N];
		const num gdj1j1 = Gd00[j1 + j1*N];
		m->pair_bb[bb + num_bb*t] += 0.5*pre*(gui0j0*gdi1j1 + gui1j0*gdi0j1 + gui0j1*gdi1j0 + gui1j1*gdi0j0);
		const num x = -pui0i1*puj0j1*guj1i0*gui1j0 - pui1i0*puj1j0*guj0i1*gui0j1
		             - pdi0i1*pdj0j1*gdj1i0*gdi1j0 - pdi1i0*pdj1j0*gdj0i1*gdi0j1;
		const num y = -pui0i1*puj1j0*guj0i0*gui1j1 - pui1i0*puj0j1*guj1i1*gui0j0
		             - pdi0i1*pdj1j0*gdj0i0*gdi1j1 - pdi1i0*pdj0j1*gdj1i1*gdi0j0;
		m->jj[bb + num_bb*t]   += pre*((pui1i0*gui0i1 - pui0i1*gui1i0 + pdi1i0*gdi0i1 - pdi0i1*gdi1i0)
		                              *(puj1j0*guj0j1 - puj0j1*guj1j0 + pdj1j0*gdj0j1 - pdj0j1*gdj1j0) + x - y);
		m->jsjs[bb + num_bb*t] += pre*((pui1i0*gui0i1 - pui0i1*gui1i0 - pdi1i0*gdi0i1 + pdi0i1*gdi1i0)
		                              *(puj1j0*guj0j1 - puj0j1*guj1j0 - pdj1j0*gdj0j1 + pdj0j1*gdj1j0) + x - y);
		m->kk[bb + num_bb*t]   += pre*((pui1i0*gui0i1 + pui0i1*gui1i0 + pdi1i0*gdi0i1 + pdi0i1*gdi1i0)
		                              *(puj1j0*guj0j1 + puj0j1*guj1j0 + pdj1j0*gdj0j1 + pdj0j1*gdj1j0) + x + y);
		m->ksks[bb + num_bb*t] += pre*((pui1i0*gui0i1 + pui0i1*gui1i0 - pdi1i0*gdi0i1 - pdi0i1*gdi1i0)
		                              *(puj1j0*guj0j1 + puj0j1*guj1j0 - pdj1j0*gdj0j1 - pdj0j1*gdj1j0) + x + y);
	}
	}
	}

	if (meas_nematic_corr)
	#pragma omp parallel for
	for (int t = 1; t < L; t++) {
		const num *const restrict Gu0t_t = Gu0t + N*N*t;
		const num *const restrict Gutt_t = Gutt + N*N*t;
		const num *const restrict Gut0_t = Gut0 + N*N*t;
		const num *const restrict Gd0t_t = Gd0t + N*N*t;
		const num *const restrict Gdtt_t = Gdtt + N*N*t;
		const num *const restrict Gdt0_t = Gdt0 + N*N*t;
	for (int c = 0; c < NEM_BONDS*N; c++) {
		const int j0 = p->bonds[c];
		const int j1 = p->bonds[c + num_b];
	for (int b = 0; b < NEM_BONDS*N; b++) {
		const int i0 = p->bonds[b];
		const int i1 = p->bonds[b + num_b];
		const int bb = p->map_bb[b + c*num_b];
		const num pre = phase / p->degen_bb[bb];
		const num gui0i0 = Gutt_t[i0 + i0*N];
		const num gui1i0 = Gutt_t[i1 + i0*N];
		const num gui0i1 = Gutt_t[i0 + i1*N];
		const num gui1i1 = Gutt_t[i1 + i1*N];
		const num gui0j0 = Gut0_t[i0 + j0*N];
		const num gui1j0 = Gut0_t[i1 + j0*N];
		const num gui0j1 = Gut0_t[i0 + j1*N];
		const num gui1j1 = Gut0_t[i1 + j1*N];
		const num guj0i0 = Gu0t_t[j0 + i0*N];
		const num guj1i0 = Gu0t_t[j1 + i0*N];
		const num guj0i1 = Gu0t_t[j0 + i1*N];
		const num guj1i1 = Gu0t_t[j1 + i1*N];
		const num guj0j0 = Gu00[j0 + j0*N];
		const num guj1j0 = Gu00[j1 + j0*N];
		const num guj0j1 = Gu00[j0 + j1*N];
		const num guj1j1 = Gu00[j1 + j1*N];
		const num gdi0i0 = Gdtt_t[i0 + i0*N];
		const num gdi1i0 = Gdtt_t[i1 + i0*N];
		const num gdi0i1 = Gdtt_t[i0 + i1*N];
		const num gdi1i1 = Gdtt_t[i1 + i1*N];
		const num gdi0j0 = Gdt0_t[i0 + j0*N];
		const num gdi1j0 = Gdt0_t[i1 + j0*N];
		const num gdi0j1 = Gdt0_t[i0 + j1*N];
		const num gdi1j1 = Gdt0_t[i1 + j1*N];
		const num gdj0i0 = Gd0t_t[j0 + i0*N];
		const num gdj1i0 = Gd0t_t[j1 + i0*N];
		const num gdj0i1 = Gd0t_t[j0 + i1*N];
		const num gdj1i1 = Gd0t_t[j1 + i1*N];
		const num gdj0j0 = Gd00[j0 + j0*N];
		const num gdj1j0 = Gd00[j1 + j0*N];
		const num gdj0j1 = Gd00[j0 + j1*N];
		const num gdj1j1 = Gd00[j1 + j1*N];
		const int delta_i0i1 = 0;
		const int delta_j0j1 = 0;
		const int delta_i0j0 = 0;
		const int delta_i1j0 = 0;
		const int delta_i0j1 = 0;
		const int delta_i1j1 = 0;
		const num uuuu = +(1.-gui0i0)*(1.-gui1i1)*(1.-guj0j0)*(1.-guj1j1)+(1.-gui0i0)*(1.-gui1i1)*(delta_j0j1-guj1j0)*guj0j1+(1.-gui0i0)*(delta_i1j0-guj0i1)*gui1j0*(1.-guj1j1)-(1.-gui0i0)*(delta_i1j0-guj0i1)*gui1j1*(delta_j0j1-guj1j0)+(1.-gui0i0)*(delta_i1j1-guj1i1)*gui1j0*guj0j1+(1.-gui0i0)*(delta_i1j1-guj1i1)*gui1j1*(1.-guj0j0)+(delta_i0i1-gui1i0)*gui0i1*(1.-guj0j0)*(1.-guj1j1)+(delta_i0i1-gui1i0)*gui0i1*(delta_j0j1-guj1j0)*guj0j1-(delta_i0i1-gui1i0)*gui0j0*(delta_i1j0-guj0i1)*(1.-guj1j1)-(delta_i0i1-gui1i0)*gui0j0*(delta_i1j1-guj1i1)*guj0j1+(delta_i0i1-gui1i0)*gui0j1*(delta_i1j0-guj0i1)*(delta_j0j1-guj1j0)-(delta_i0i1-gui1i0)*gui0j1*(delta_i1j1-guj1i1)*(1.-guj0j0)+(delta_i0j0-guj0i0)*gui0i1*gui1j0*(1.-guj1j1)-(delta_i0j0-guj0i0)*gui0i1*gui1j1*(delta_j0j1-guj1j0)+(delta_i0j0-guj0i0)*gui0j0*(1.-gui1i1)*(1.-guj1j1)+(delta_i0j0-guj0i0)*gui0j0*(delta_i1j1-guj1i1)*gui1j1-(delta_i0j0-guj0i0)*gui0j1*(1.-gui1i1)*(delta_j0j1-guj1j0)-(delta_i0j0-guj0i0)*gui0j1*(delta_i1j1-guj1i1)*gui1j0+(delta_i0j1-guj1i0)*gui0i1*gui1j0*guj0j1+(delta_i0j1-guj1i0)*gui0i1*gui1j1*(1.-guj0j0)+(delta_i0j1-guj1i0)*gui0j0*(1.-gui1i1)*guj0j1-(delta_i0j1-guj1i0)*gui0j0*(delta_i1j0-guj0i1)*gui1j1+(delta_i0j1-guj1i0)*gui0j1*(1.-gui1i1)*(1.-guj0j0)+(delta_i0j1-guj1i0)*gui0j1*(delta_i1j0-guj0i1)*gui1j0;
		const num uuud = +(1.-gui0i0)*(1.-gui1i1)*(1.-guj0j0)*(1.-gdj1j1)+(1.-gui0i0)*(delta_i1j0-guj0i1)*gui1j0*(1.-gdj1j1)+(delta_i0i1-gui1i0)*gui0i1*(1.-guj0j0)*(1.-gdj1j1)-(delta_i0i1-gui1i0)*gui0j0*(delta_i1j0-guj0i1)*(1.-gdj1j1)+(delta_i0j0-guj0i0)*gui0i1*gui1j0*(1.-gdj1j1)+(delta_i0j0-guj0i0)*gui0j0*(1.-gui1i1)*(1.-gdj1j1);
		const num uudu = +(1.-gui0i0)*(1.-gui1i1)*(1.-gdj0j0)*(1.-guj1j1)+(1.-gui0i0)*(delta_i1j1-guj1i1)*gui1j1*(1.-gdj0j0)+(delta_i0i1-gui1i0)*gui0i1*(1.-gdj0j0)*(1.-guj1j1)-(delta_i0i1-gui1i0)*gui0j1*(delta_i1j1-guj1i1)*(1.-gdj0j0)+(delta_i0j1-guj1i0)*gui0i1*gui1j1*(1.-gdj0j0)+(delta_i0j1-guj1i0)*gui0j1*(1.-gui1i1)*(1.-gdj0j0);
		const num uudd = +(1.-gui0i0)*(1.-gui1i1)*(1.-gdj0j0)*(1.-gdj1j1)+(1.-gui0i0)*(1.-gui1i1)*(delta_j0j1-gdj1j0)*gdj0j1+(delta_i0i1-gui1i0)*gui0i1*(1.-gdj0j0)*(1.-gdj1j1)+(delta_i0i1-gui1i0)*gui0i1*(delta_j0j1-gdj1j0)*gdj0j1;
		const num uduu = +(1.-gui0i0)*(1.-gdi1i1)*(1.-guj0j0)*(1.-guj1j1)+(1.-gui0i0)*(1.-gdi1i1)*(delta_j0j1-guj1j0)*guj0j1+(delta_i0j0-guj0i0)*gui0j0*(1.-gdi1i1)*(1.-guj1j1)-(delta_i0j0-guj0i0)*gui0j1*(1.-gdi1i1)*(delta_j0j1-guj1j0)+(delta_i0j1-guj1i0)*gui0j0*(1.-gdi1i1)*guj0j1+(delta_i0j1-guj1i0)*gui0j1*(1.-gdi1i1)*(1.-guj0j0);
		const num udud = +(1.-gui0i0)*(1.-gdi1i1)*(1.-guj0j0)*(1.-gdj1j1)+(1.-gui0i0)*(delta_i1j1-gdj1i1)*gdi1j1*(1.-guj0j0)+(delta_i0j0-guj0i0)*gui0j0*(1.-gdi1i1)*(1.-gdj1j1)+(delta_i0j0-guj0i0)*gui0j0*(delta_i1j1-gdj1i1)*gdi1j1;
		const num uddu = +(1.-gui0i0)*(1.-gdi1i1)*(1.-gdj0j0)*(1.-guj1j1)+(1.-gui0i0)*(delta_i1j0-gdj0i1)*gdi1j0*(1.-guj1j1)+(delta_i0j1-guj1i0)*gui0j1*(1.-gdi1i1)*(1.-gdj0j0)+(delta_i0j1-guj1i0)*gui0j1*(delta_i1j0-gdj0i1)*gdi1j0;
		const num uddd = +(1.-gui0i0)*(1.-gdi1i1)*(1.-gdj0j0)*(1.-gdj1j1)+(1.-gui0i0)*(1.-gdi1i1)*(delta_j0j1-gdj1j0)*gdj0j1+(1.-gui0i0)*(delta_i1j0-gdj0i1)*gdi1j0*(1.-gdj1j1)-(1.-gui0i0)*(delta_i1j0-gdj0i1)*gdi1j1*(delta_j0j1-gdj1j0)+(1.-gui0i0)*(delta_i1j1-gdj1i1)*gdi1j0*gdj0j1+(1.-gui0i0)*(delta_i1j1-gdj1i1)*gdi1j1*(1.-gdj0j0);
		const num duuu = +(1.-gdi0i0)*(1.-gui1i1)*(1.-guj0j0)*(1.-guj1j1)+(1.-gdi0i0)*(1.-gui1i1)*(delta_j0j1-guj1j0)*guj0j1+(1.-gdi0i0)*(delta_i1j0-guj0i1)*gui1j0*(1.-guj1j1)-(1.-gdi0i0)*(delta_i1j0-guj0i1)*gui1j1*(delta_j0j1-guj1j0)+(1.-gdi0i0)*(delta_i1j1-guj1i1)*gui1j0*guj0j1+(1.-gdi0i0)*(delta_i1j1-guj1i1)*gui1j1*(1.-guj0j0);
		const num duud = +(1.-gdi0i0)*(1.-gui1i1)*(1.-guj0j0)*(1.-gdj1j1)+(1.-gdi0i0)*(delta_i1j0-guj0i1)*gui1j0*(1.-gdj1j1)+(delta_i0j1-gdj1i0)*gdi0j1*(1.-gui1i1)*(1.-guj0j0)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i1j0-guj0i1)*gui1j0;
		const num dudu = +(1.-gdi0i0)*(1.-gui1i1)*(1.-gdj0j0)*(1.-guj1j1)+(1.-gdi0i0)*(delta_i1j1-guj1i1)*gui1j1*(1.-gdj0j0)+(delta_i0j0-gdj0i0)*gdi0j0*(1.-gui1i1)*(1.-guj1j1)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i1j1-guj1i1)*gui1j1;
		const num dudd = +(1.-gdi0i0)*(1.-gui1i1)*(1.-gdj0j0)*(1.-gdj1j1)+(1.-gdi0i0)*(1.-gui1i1)*(delta_j0j1-gdj1j0)*gdj0j1+(delta_i0j0-gdj0i0)*gdi0j0*(1.-gui1i1)*(1.-gdj1j1)-(delta_i0j0-gdj0i0)*gdi0j1*(1.-gui1i1)*(delta_j0j1-gdj1j0)+(delta_i0j1-gdj1i0)*gdi0j0*(1.-gui1i1)*gdj0j1+(delta_i0j1-gdj1i0)*gdi0j1*(1.-gui1i1)*(1.-gdj0j0);
		const num dduu = +(1.-gdi0i0)*(1.-gdi1i1)*(1.-guj0j0)*(1.-guj1j1)+(1.-gdi0i0)*(1.-gdi1i1)*(delta_j0j1-guj1j0)*guj0j1+(delta_i0i1-gdi1i0)*gdi0i1*(1.-guj0j0)*(1.-guj1j1)+(delta_i0i1-gdi1i0)*gdi0i1*(delta_j0j1-guj1j0)*guj0j1;
		const num ddud = +(1.-gdi0i0)*(1.-gdi1i1)*(1.-guj0j0)*(1.-gdj1j1)+(1.-gdi0i0)*(delta_i1j1-gdj1i1)*gdi1j1*(1.-guj0j0)+(delta_i0i1-gdi1i0)*gdi0i1*(1.-guj0j0)*(1.-gdj1j1)-(delta_i0i1-gdi1i0)*gdi0j1*(delta_i1j1-gdj1i1)*(1.-guj0j0)+(delta_i0j1-gdj1i0)*gdi0i1*gdi1j1*(1.-guj0j0)+(delta_i0j1-gdj1i0)*gdi0j1*(1.-gdi1i1)*(1.-guj0j0);
		const num dddu = +(1.-gdi0i0)*(1.-gdi1i1)*(1.-gdj0j0)*(1.-guj1j1)+(1.-gdi0i0)*(delta_i1j0-gdj0i1)*gdi1j0*(1.-guj1j1)+(delta_i0i1-gdi1i0)*gdi0i1*(1.-gdj0j0)*(1.-guj1j1)-(delta_i0i1-gdi1i0)*gdi0j0*(delta_i1j0-gdj0i1)*(1.-guj1j1)+(delta_i0j0-gdj0i0)*gdi0i1*gdi1j0*(1.-guj1j1)+(delta_i0j0-gdj0i0)*gdi0j0*(1.-gdi1i1)*(1.-guj1j1);
		const num dddd = +(1.-gdi0i0)*(1.-gdi1i1)*(1.-gdj0j0)*(1.-gdj1j1)+(1.-gdi0i0)*(1.-gdi1i1)*(delta_j0j1-gdj1j0)*gdj0j1+(1.-gdi0i0)*(delta_i1j0-gdj0i1)*gdi1j0*(1.-gdj1j1)-(1.-gdi0i0)*(delta_i1j0-gdj0i1)*gdi1j1*(delta_j0j1-gdj1j0)+(1.-gdi0i0)*(delta_i1j1-gdj1i1)*gdi1j0*gdj0j1+(1.-gdi0i0)*(delta_i1j1-gdj1i1)*gdi1j1*(1.-gdj0j0)+(delta_i0i1-gdi1i0)*gdi0i1*(1.-gdj0j0)*(1.-gdj1j1)+(delta_i0i1-gdi1i0)*gdi0i1*(delta_j0j1-gdj1j0)*gdj0j1-(delta_i0i1-gdi1i0)*gdi0j0*(delta_i1j0-gdj0i1)*(1.-gdj1j1)-(delta_i0i1-gdi1i0)*gdi0j0*(delta_i1j1-gdj1i1)*gdj0j1+(delta_i0i1-gdi1i0)*gdi0j1*(delta_i1j0-gdj0i1)*(delta_j0j1-gdj1j0)-(delta_i0i1-gdi1i0)*gdi0j1*(delta_i1j1-gdj1i1)*(1.-gdj0j0)+(delta_i0j0-gdj0i0)*gdi0i1*gdi1j0*(1.-gdj1j1)-(delta_i0j0-gdj0i0)*gdi0i1*gdi1j1*(delta_j0j1-gdj1j0)+(delta_i0j0-gdj0i0)*gdi0j0*(1.-gdi1i1)*(1.-gdj1j1)+(delta_i0j0-gdj0i0)*gdi0j0*(delta_i1j1-gdj1i1)*gdi1j1-(delta_i0j0-gdj0i0)*gdi0j1*(1.-gdi1i1)*(delta_j0j1-gdj1j0)-(delta_i0j0-gdj0i0)*gdi0j1*(delta_i1j1-gdj1i1)*gdi1j0+(delta_i0j1-gdj1i0)*gdi0i1*gdi1j0*gdj0j1+(delta_i0j1-gdj1i0)*gdi0i1*gdi1j1*(1.-gdj0j0)+(delta_i0j1-gdj1i0)*gdi0j0*(1.-gdi1i1)*gdj0j1-(delta_i0j1-gdj1i0)*gdi0j0*(delta_i1j0-gdj0i1)*gdi1j1+(delta_i0j1-gdj1i0)*gdi0j1*(1.-gdi1i1)*(1.-gdj0j0)+(delta_i0j1-gdj1i0)*gdi0j1*(delta_i1j0-gdj0i1)*gdi1j0;
		m->nem_nnnn[bb + num_bb*t] += pre*(uuuu + uuud + uudu + uudd
		                                 + uduu + udud + uddu + uddd
		                                 + duuu + duud + dudu + dudd
		                                 + dduu + ddud + dddu + dddd);
		m->nem_ssss[bb + num_bb*t] += pre*(uuuu - uuud - uudu + uudd
		                                 - uduu + udud + uddu - uddd
		                                 - duuu + duud + dudu - dudd
		                                 + dduu - ddud - dddu + dddd);
	}
	}
	}
}

// unmaintained, outdated
void measure_uneqlt_full(const struct params *const restrict p, const int sign,
		const double *const restrict Gu,
		const double *const restrict Gd,
		struct meas_uneqlt *const restrict m)
{
	m->n_sample++;
	m->sign += sign;
	const int N = p->N, L = p->L, num_i = p->num_i, num_ij = p->num_ij;
	const int num_b = p->num_b, num_bb = p->num_bb;

	// 2-site measurements
	#pragma omp parallel for
	for (int t = 0; t < L; t++)
	for (int l = 0; l < L; l++) {
		const int k = (l + t) % L;
		const int T_sign = ((k >= l) ? 1.0 : -1.0); // for fermionic
		const int delta_t = (t == 0);
	for (int j = 0; j < N; j++)
	for (int i = 0; i < N; i++) {
		const int r = p->map_ij[i + j*N];
		const int delta_tij = delta_t * (i == j);
		const double pre = (double)sign / p->degen_ij[r] / L;
		const double guii = Gu[(i + N*i) + N*N*(k + L*k)];
		const double guij = Gu[(i + N*j) + N*N*(k + L*l)];
		const double guji = Gu[(j + N*i) + N*N*(l + L*k)];
		const double gujj = Gu[(j + N*j) + N*N*(l + L*l)];
		const double gdii = Gd[(i + N*i) + N*N*(k + L*k)];
		const double gdij = Gd[(i + N*j) + N*N*(k + L*l)];
		const double gdji = Gd[(j + N*i) + N*N*(l + L*k)];
		const double gdjj = Gd[(j + N*j) + N*N*(l + L*l)];
		m->gt0[r + num_ij*t] += 0.5*T_sign*pre*(guij + gdij);
		const double x = delta_tij*(guii + gdii) - (guji*guij + gdji*gdij);
		m->nn[r + num_ij*t] += pre*((2. - guii - gdii)*(2. - gujj - gdjj) + x);
		m->xx[r + num_ij*t] += 0.25*pre*(delta_tij*(guii + gdii) - (guji*gdij + gdji*guij));
		m->zz[r + num_ij*t] += 0.25*pre*((gdii - guii)*(gdjj - gujj) + x);
		m->pair_sw[r + num_ij*t] += pre*guij*gdij;
	}
	}

	// 2-bond measurements
	#pragma omp parallel for
	for (int t = 0; t < L; t++)
	for (int l = 0; l < L; l++) {
		const int k = (l + t) % L;
		const int delta_t = (t == 0);
	for (int c = 0; c < num_b; c++) {
		const int j0 = p->bonds[c];
		const int j1 = p->bonds[c + num_b];
	for (int b = 0; b < num_b; b++) {
		const int i0 = p->bonds[b];
		const int i1 = p->bonds[b + num_b];
		const int bb = p->map_bb[b + c*num_b];
		const double pre = (double)sign / p->degen_bb[bb] / L;
		const int delta_ti0j0 = delta_t * (i0 == j0);
		const int delta_ti1j0 = delta_t * (i1 == j0);
		const int delta_ti0j1 = delta_t * (i0 == j1);
		const int delta_ti1j1 = delta_t * (i1 == j1);
		const double gui0i0 = Gu[i0 + i0*N + N*N*(k + L*k)];
		const double gui1i0 = Gu[i1 + i0*N + N*N*(k + L*k)];
		const double gui0i1 = Gu[i0 + i1*N + N*N*(k + L*k)];
		const double gui1i1 = Gu[i1 + i1*N + N*N*(k + L*k)];
		const double gui0j0 = Gu[i0 + j0*N + N*N*(k + L*l)];
		const double gui1j0 = Gu[i1 + j0*N + N*N*(k + L*l)];
		const double gui0j1 = Gu[i0 + j1*N + N*N*(k + L*l)];
		const double gui1j1 = Gu[i1 + j1*N + N*N*(k + L*l)];
		const double guj0i0 = Gu[j0 + i0*N + N*N*(l + L*k)];
		const double guj1i0 = Gu[j1 + i0*N + N*N*(l + L*k)];
		const double guj0i1 = Gu[j0 + i1*N + N*N*(l + L*k)];
		const double guj1i1 = Gu[j1 + i1*N + N*N*(l + L*k)];
		const double guj0j0 = Gu[j0 + j0*N + N*N*(l + L*l)];
		const double guj1j0 = Gu[j1 + j0*N + N*N*(l + L*l)];
		const double guj0j1 = Gu[j0 + j1*N + N*N*(l + L*l)];
		const double guj1j1 = Gu[j1 + j1*N + N*N*(l + L*l)];
		const double gdi0i0 = Gd[i0 + i0*N + N*N*(k + L*k)];
		const double gdi1i0 = Gd[i1 + i0*N + N*N*(k + L*k)];
		const double gdi0i1 = Gd[i0 + i1*N + N*N*(k + L*k)];
		const double gdi1i1 = Gd[i1 + i1*N + N*N*(k + L*k)];
		const double gdi0j0 = Gd[i0 + j0*N + N*N*(k + L*l)];
		const double gdi1j0 = Gd[i1 + j0*N + N*N*(k + L*l)];
		const double gdi0j1 = Gd[i0 + j1*N + N*N*(k + L*l)];
		const double gdi1j1 = Gd[i1 + j1*N + N*N*(k + L*l)];
		const double gdj0i0 = Gd[j0 + i0*N + N*N*(l + L*k)];
		const double gdj1i0 = Gd[j1 + i0*N + N*N*(l + L*k)];
		const double gdj0i1 = Gd[j0 + i1*N + N*N*(l + L*k)];
		const double gdj1i1 = Gd[j1 + i1*N + N*N*(l + L*k)];
		const double gdj0j0 = Gd[j0 + j0*N + N*N*(l + L*l)];
		const double gdj1j0 = Gd[j1 + j0*N + N*N*(l + L*l)];
		const double gdj0j1 = Gd[j0 + j1*N + N*N*(l + L*l)];
		const double gdj1j1 = Gd[j1 + j1*N + N*N*(l + L*l)];
		m->pair_bb[bb + num_bb*t] += 0.5*pre*(gui0j0*gdi1j1 + gui1j0*gdi0j1 + gui0j1*gdi1j0 + gui1j1*gdi0j0);
		m->jj[bb + num_bb*t] += pre*((gui0i1 - gui1i0 + gdi0i1 - gdi1i0)*(guj0j1 - guj1j0 + gdj0j1 - gdj1j0)
		                             + (delta_ti0j1 - guj1i0)*gui1j0 - (delta_ti0j0 - guj0i0)*gui1j1
		                             - (delta_ti1j1 - guj1i1)*gui0j0 + (delta_ti1j0 - guj0i1)*gui0j1
		                             + (delta_ti0j1 - gdj1i0)*gdi1j0 - (delta_ti0j0 - gdj0i0)*gdi1j1
		                             - (delta_ti1j1 - gdj1i1)*gdi0j0 + (delta_ti1j0 - gdj0i1)*gdi0j1);
		m->kk[bb + num_bb*t] += pre*((gui0i1 + gui1i0 + gdi0i1 + gdi1i0)*(guj0j1 + guj1j0 + gdj0j1 + gdj1j0)
		                             + (delta_ti0j1 - guj1i0)*gui1j0 + (delta_ti0j0 - guj0i0)*gui1j1
		                             + (delta_ti1j1 - guj1i1)*gui0j0 + (delta_ti1j0 - guj0i1)*gui0j1
		                             + (delta_ti0j1 - gdj1i0)*gdi1j0 + (delta_ti0j0 - gdj0i0)*gdi1j1
		                             + (delta_ti1j1 - gdj1i1)*gdi0j0 + (delta_ti1j0 - gdj0i1)*gdi0j1);
	}
	}
	}
}

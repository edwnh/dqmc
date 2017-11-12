#include "meas.h"
#include "data.h"

void measure_eqlt(const struct params *const restrict p, const int sign,
		const double *const restrict gu,
		const double *const restrict gd,
		struct meas_eqlt *const restrict m)
{
	m->n_sample++;
	m->sign += sign;
	const int N = p->N, num_i = p->num_i, num_ij = p->num_ij;
	const int num_b = p->num_b;

	// 1-site measurements
	for (int i = 0; i < N; i++) {
		const int r = p->map_i[i];
		const double pre = (double)sign / p->degen_i[r];
		const double guii = gu[i + i*N], gdii = gd[i + i*N];
		m->density[r] += pre*(2. - guii - gdii);
		m->double_occ[r] += pre*(1. - guii)*(1. - gdii);
	}

	// 2-site measurements
	for (int j = 0; j < N; j++)
	for (int i = 0; i < N; i++) {
		const int delta = (i == j);
		const int r = p->map_ij[i + j*N];
		const double pre = (double)sign / p->degen_ij[r];
		const double guii = gu[i + i*N], gdii = gd[i + i*N];
		const double guij = gu[i + j*N], gdij = gd[i + j*N];
		const double guji = gu[j + i*N], gdji = gd[j + i*N];
		const double gujj = gu[j + j*N], gdjj = gd[j + j*N];
		m->g00[r] += 0.5*pre*(guij + gdij);
		const double x = delta*(guii + gdii) - (guji*guij + gdji*gdij);
		m->nn[r] += pre*((2. - guii - gdii)*(2. - gujj - gdjj) + x);
		m->xx[r] += 0.25*pre*(delta*(guii + gdii) - (guji*gdij + gdji*guij));
		m->zz[r] += 0.25*pre*((gdii - guii)*(gdjj - gujj) + x);
		m->pair_sw[r] += pre*guij*gdij;
	}
}

void measure_uneqlt(const struct params *const restrict p, const int sign,
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

	// 2-bond measurements: pairing
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
		m->rhorho[bb + num_bb*t] += pre*((gui0i1 + gui1i0 + gdi0i1 + gdi1i0)*(guj0j1 + guj1j0 + gdj0j1 + gdj1j0)
		                                 + (delta_ti0j1 - guj1i0)*gui1j0 + (delta_ti0j0 - guj0i0)*gui1j1
		                                 + (delta_ti1j1 - guj1i1)*gui0j0 + (delta_ti1j0 - guj0i1)*gui0j1
		                                 + (delta_ti0j1 - gdj1i0)*gdi1j0 + (delta_ti0j0 - gdj0i0)*gdi1j1
		                                 + (delta_ti1j1 - gdj1i1)*gdi0j0 + (delta_ti1j0 - gdj0i1)*gdi0j1);
	}
	}
	}
}

#include "meas.h"
#include "data.h"

void measure_eqlt(const struct params *const restrict p, const int sign,
		const double *const restrict Gu,
		const double *const restrict Gd,
		struct meas_eqlt *const restrict m)
{
	m->n_sample++;
	m->sign += sign;
	const int N = p->N, num_i = p->num_i, num_ij = p->num_ij;

	// 1-site measurements
	for (int i = 0; i < N; i++) {
		const int r = p->map_i[i];
		const double pre = (double)sign / p->degen_i[r];
		const double Guii = Gu[i + i*N], Gdii = Gd[i + i*N];
		m->density[r] += pre*(2. - Guii - Gdii);
		m->double_occ[r] += pre*(1. - Guii)*(1. - Gdii);
	}

	// 2-site measurements
	for (int j = 0; j < N; j++)
	for (int i = 0; i < N; i++) {
		const int delta = (i == j);
		const int r = p->map_ij[i + j*N];
		const int pre = (double)sign / p->degen_ij[r];
		const double Guii = Gu[i + i*N], Gdii = Gd[i + i*N];
		const double Guij = Gu[i + j*N], Gdij = Gd[i + j*N];
		const double Guji = Gu[j + i*N], Gdji = Gd[j + i*N];
		const double Gujj = Gu[j + j*N], Gdjj = Gd[j + j*N];
		m->g00[r] += 0.5*pre*(Guij + Gdij);
		const double x = delta*(Guii + Gdii) - (Guji*Guij + Gdji*Gdij);
		m->nn[r] += pre*((2. - Guii - Gdii)*(2. - Gujj - Gdjj) + x);
		m->xx[r] += 0.25*pre*(delta*(Guii + Gdii) - (Guji*Gdij + Gdji*Guij));
		m->zz[r] += 0.25*pre*((Gdii - Guii)*(Gdjj - Gujj) + x);
		m->pair_sw[r] += pre*(delta*(1. - Guii - Gdii) + 2.*Guij*Gdij);
	}

	// 2-bond measurements
}

void measure_uneqlt(const struct params *const restrict p, const int sign,
		const double *const restrict Gu,
		const double *const restrict Gd,
		struct meas_uneqlt *const restrict m)
{
	m->n_sample++;
	m->sign += sign;
}

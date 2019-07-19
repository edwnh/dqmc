#pragma once

#include <complex.h>
#include <stdint.h>

struct params {
	int N, L;
	int *map_i, *map_ij;
	int *bonds, *map_bs, *map_bb;
//	double *K, *U;
//	double dt;

	int n_matmul, n_delay;
	int n_sweep_warm, n_sweep_meas;
	int period_eqlt, period_uneqlt;
	int meas_bond_corr, meas_energy_corr, meas_nematic_corr;

	int num_i, num_ij;
	int num_b, num_bs, num_bb;
	int *degen_i, *degen_ij, *degen_bs, *degen_bb;
	complex double *exp_Ku, *exp_Kd, *inv_exp_Ku, *inv_exp_Kd;
	complex double *exp_halfKu, *exp_halfKd, *inv_exp_halfKu, *inv_exp_halfKd;
	double *exp_lambda, *del;
	int F, n_sweep;
};

struct state {
	uint64_t rng[17];
	int sweep;
	int *hs;
};

struct meas_eqlt {
	int n_sample;
	complex double sign;

	complex double *density;
	complex double *double_occ;

	complex double *g00;
	complex double *nn;
	complex double *xx;
	complex double *zz;
	complex double *pair_sw;
	complex double *kk, *kv, *kn, *vv, *vn;
};

struct meas_uneqlt {
	int n_sample;
	complex double sign;

	complex double *gt0;
	complex double *nn;
	complex double *xx;
	complex double *zz;
	complex double *pair_sw;
	complex double *pair_bb;
	complex double *jj, *jsjs;
	complex double *kk, *ksks;
	complex double *kv, *kn, *vv, *vn;
	complex double *nem_nnnn, *nem_ssss;
};

struct sim_data {
	struct params p;
	struct state s;
	struct meas_eqlt m_eq;
	struct meas_uneqlt m_ue;
};

int sim_data_read_alloc(struct sim_data *sim, const char *file);

int sim_data_save(const struct sim_data *sim, const char *file);

void sim_data_free(const struct sim_data *sim);

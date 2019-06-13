#pragma once

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

	int num_i, num_ij;
	int num_b, num_bs, num_bb;
	int *degen_i, *degen_ij, *degen_bs, *degen_bb;
	double *exp_K, *inv_exp_K;
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
	double sign;

	double *density;
	double *double_occ;

	double *g00;
	double *nn;
	double *xx;
	double *zz;
	double *pair_sw;
	double *kk, *kv, *kn, *vk, *vv, *vn;
};

struct meas_uneqlt {
	int n_sample;
	double sign;

	double *gt0;
	double *nn;
	double *xx;
	double *zz;
	double *pair_sw;
	double *pair_bb;
	double *jj, *jsjs;
	double *kk, *ksks;
	double *nem_nnnn, *nem_ssss;
	double *kv, *kn, *vv, *vn;
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

#pragma once

#include <stdint.h>
#include "linalg.h"

struct params {
	int N, L;
	int *map_i, *map_ij;
	int *bonds, *map_bs, *map_bb;
	num *peierlsu, *peierlsd;
//	double *K, *U;
//	double dt;

	int n_matmul, n_delay;
	int n_sweep_warm, n_sweep_meas;
	int period_eqlt, period_uneqlt;
	int meas_bond_corr, meas_energy_corr, meas_nematic_corr;

	int num_i, num_ij;
	int num_b, num_bs, num_bb;
	int *degen_i, *degen_ij, *degen_bs, *degen_bb;
	num *exp_Ku, *exp_Kd, *inv_exp_Ku, *inv_exp_Kd;
	num *exp_halfKu, *exp_halfKd, *inv_exp_halfKu, *inv_exp_halfKd;
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
	num sign;

	num *density;
	num *double_occ;

	num *g00;
	num *nn;
	num *xx;
	num *zz;
	num *pair_sw;
	num *kk, *kv, *kn, *vv, *vn;
};

struct meas_uneqlt {
	int n_sample;
	num sign;

	num *gt0;
	num *nn;
	num *xx;
	num *zz;
	num *pair_sw;
	num *pair_bb;
	num *jj, *jsjs;
	num *kk, *ksks;
	num *kv, *kn, *vv, *vn;
	num *nem_nnnn, *nem_ssss;
};

struct workspace {
	num *inv_exp_K, *exp_K;
	num *exp_V;
	num *iB, *B, *C;
	num *Q_L, *d_L, *X_L;
	num *iL_L, *R_L, *phase_iL_L;
	num *Q_0, *d_0, *X_0;
	num *iL_0, *R_0, *phase_iL_0;
	num *g;
	num *exp_halfK, *inv_exp_halfK;
	num *tmpNN1, *tmpNN2, *tmpN1, *tmpN2;
	int *pvt;
	num *work;
	num *G0t, *Gtt, *Gt0;
};

struct sim_data {
	const char *file;
	void *pool; // memory pool for everything below

	struct params p;
	struct state s;
	struct meas_eqlt m_eq;
	struct meas_uneqlt m_ue;

	struct workspace up, dn;
};

int sim_data_read_alloc(struct sim_data *sim, const char *file);

int sim_data_save(const struct sim_data *sim);

void sim_data_free(struct sim_data *sim);

#pragma once

#include <stdint.h>
#include "numeric.h"

#define N_FLAVORS 2 // number of fermion flavors

#define PARAMS_SCALAR_INT_LIST \
	X(N) \
	X(L) \
	X(F) \
	X(n_matmul) \
	X(n_delay) \
	X(n_sweep_warm) \
	X(n_sweep_meas) \
	X(n_sweep) \
	X(period_eqlt) \
	X(period_uneqlt) \
	X(meas_bond_corr) \
	X(meas_energy_corr) \
	X(meas_nematic_corr) \
	X(num_i) \
	X(num_ij) \
	X(num_b) \
	X(num_bs) \
	X(num_bb)

// format: X(name, type, size)
// using size assumes the following variables are in scope:
// N, num_b, num_i, num_ij, num_bb, num_bs
#define PARAMS_ARRAY_LIST \
	X(map_i,          int, N) \
	X(map_ij,         int, N*N) \
	X(bonds,          int, num_b*2) \
	X(map_bs,         int, num_b*N) \
	X(map_bb,         int, num_b*num_b) \
	X(degen_i,        int, num_i) \
	X(degen_ij,       int, num_ij) \
	X(degen_bs,       int, num_bs) \
	X(degen_bb,       int, num_bb) \
	X(peierlsu,       num, N*N) \
	X(peierlsd,       num, N*N) \
	X(exp_Ku,         num, N*N) \
	X(exp_Kd,         num, N*N) \
	X(inv_exp_Ku,     num, N*N) \
	X(inv_exp_Kd,     num, N*N) \
	X(exp_halfKu,     num, N*N) \
	X(exp_halfKd,     num, N*N) \
	X(inv_exp_halfKu, num, N*N) \
	X(inv_exp_halfKd, num, N*N) \
	X(exp_lambda,  double, N*2) \
	X(del,         double, N*2)

// format: X(name, size)
// using size assumes the following variables are also in scope:
// L, meas_energy_corr, meas_bond_corr, meas_nematic_corr
#define MEAS_EQLT_LIST \
	X(density,    num_i) \
	X(double_occ, num_i) \
	X(g00,        num_ij) \
	X(nn,         num_ij) \
	X(xx,         num_ij) \
	X(zz,         num_ij) \
	X(pair_sw,    num_ij) \
	X(kk,         (meas_energy_corr != 0)*num_bb) \
	X(kv,         (meas_energy_corr != 0)*num_bs) \
	X(kn,         (meas_energy_corr != 0)*num_bs) \
	X(vv,         (meas_energy_corr != 0)*num_ij) \
	X(vn,         (meas_energy_corr != 0)*num_ij)

#define MEAS_UNEQLT_LIST \
	X(gt0,      num_ij*L) \
	X(nn,       num_ij*L) \
	X(xx,       num_ij*L) \
	X(zz,       num_ij*L) \
	X(pair_sw,  num_ij*L) \
	X(pair_bb,  (meas_bond_corr != 0)*num_bb*L) \
	X(jj,       (meas_bond_corr != 0)*num_bb*L) \
	X(jsjs,     (meas_bond_corr != 0)*num_bb*L) \
	X(kk,       (meas_bond_corr != 0)*num_bb*L) \
	X(ksks,     (meas_bond_corr != 0)*num_bb*L) \
	X(kv,       (meas_energy_corr != 0)*num_bs*L) \
	X(kn,       (meas_energy_corr != 0)*num_bs*L) \
	X(vv,       (meas_energy_corr != 0)*num_ij*L) \
	X(vn,       (meas_energy_corr != 0)*num_ij*L) \
	X(nem_nnnn, (meas_nematic_corr != 0)*num_bb*L) \
	X(nem_ssss, (meas_nematic_corr != 0)*num_bb*L)

// format: X(name, type, size)
// using size assumes the following variables are in scope:
// ld, F, period_uneqlt
#define WORKSPACE_LIST \
	X(inv_exp_K,     num, ld*N) \
	X(iB,            num, L*ld*N) \
	X(exp_K,         num, ld*N) \
	X(exp_V,         num, N) \
	X(B,             num, L*ld*N) \
	X(C,             num, F*ld*N) \
	X(Q_L,           num, F*ld*N) \
	X(d_L,           num, F*ld) \
	X(X_L,           num, F*ld*N) \
	X(iL_L,          num, F*ld*N) \
	X(R_L,           num, F*ld*N) \
	X(phase_iL_L,    num, F) \
	X(Q_0,           num, F*ld*N) \
	X(d_0,           num, F*ld) \
	X(X_0,           num, F*ld*N) \
	X(iL_0,          num, F*ld*N) \
	X(R_0,           num, F*ld*N) \
	X(phase_iL_0,    num, F) \
	X(g,             num, ld*N) \
	X(exp_halfK,     num, ld*N) \
	X(inv_exp_halfK, num, ld*N) \
	X(tmpNN1,        num, ld*N) \
	X(tmpNN2,        num, ld*N) \
	X(tmpN1,         num, N) \
	X(tmpN2,         num, N) \
	X(pvt,           int, N) \
	X(work,          num, lwork) \
	X(G0t,           num, (period_uneqlt > 0)*L*ld*N) \
	X(Gtt,           num, (period_uneqlt > 0)*L*ld*N) \
	X(Gt0,           num, (period_uneqlt > 0)*L*ld*N)

struct RC(params) {
#define X(name) int name;
	PARAMS_SCALAR_INT_LIST
#undef X
#define X(name, type, size) type *name;
	PARAMS_ARRAY_LIST
#undef X
};

struct RC(meas_eqlt) {
	int n_sample;
	num sign;
#define X(name, size) num *name;
	MEAS_EQLT_LIST
#undef X
};

struct RC(meas_uneqlt) {
	int n_sample;
	num sign;
#define X(name, size) num *name;
	MEAS_UNEQLT_LIST
#undef X
};

struct RC(workspace) {
#define X(name, type, size) type *name;
	WORKSPACE_LIST
#undef X
};

struct RC(sim_data) {
	const char *file;
	void *pool; // memory pool for everything below

	struct RC(params) p;
	struct state {
		uint64_t rng[17];
		int sweep;
		int *hs;
	} s;
	struct RC(meas_eqlt) m_eq;
	struct RC(meas_uneqlt) m_ue;

	struct RC(workspace) ws[N_FLAVORS];
};

#include "data.h"
#include <stdio.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include "linalg.h"
#include "mem.h"

#define return_if(cond, val, ...) \
	do {if (cond) {fprintf(stderr, __VA_ARGS__); return (val);}} while (0)

static hid_t num_h5t;

int sim_data_read_alloc(struct sim_data *sim, const char *file)
{
	const hid_t file_id = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
	return_if(file_id < 0, -1, "H5Fopen() failed: %ld\n", file_id);

	sim->file = file;

	herr_t status;

#ifdef USE_CPLX
	num_h5t = H5Tcreate(H5T_COMPOUND, sizeof(num));
	status = H5Tinsert(num_h5t, "r", 0, H5T_NATIVE_DOUBLE);
	return_if(status < 0, -1, "H5Tinsert() failed: %d\n", status);
	status = H5Tinsert(num_h5t, "i", 8, H5T_NATIVE_DOUBLE);
	return_if(status < 0, -1, "H5Tinsert() failed: %d\n", status);
#else
	num_h5t = H5T_NATIVE_DOUBLE;
#endif


#define my_read(_type, name, ...) do { \
	status = H5LTread_dataset##_type(file_id, (name), __VA_ARGS__); \
	return_if(status < 0, -1, "H5LTread_dataset() failed for %s: %d\n", (name), status); \
} while (0);

	my_read(_int, "/params/N",      &sim->p.N);
	my_read(_int, "/params/L",      &sim->p.L);
	my_read(_int, "/params/num_i",  &sim->p.num_i);
	my_read(_int, "/params/num_ij", &sim->p.num_ij);
	my_read(_int, "/params/num_b", &sim->p.num_b);
	my_read(_int, "/params/num_bs", &sim->p.num_bs);
	my_read(_int, "/params/num_bb", &sim->p.num_bb);
	my_read(_int, "/params/period_uneqlt", &sim->p.period_uneqlt);
	my_read(_int, "/params/meas_bond_corr", &sim->p.meas_bond_corr);
	my_read(_int, "/params/meas_energy_corr", &sim->p.meas_energy_corr);
	my_read(_int, "/params/meas_nematic_corr", &sim->p.meas_nematic_corr);

	const int N = sim->p.N, L = sim->p.L;
	const int num_i = sim->p.num_i, num_ij = sim->p.num_ij;
	const int num_b = sim->p.num_b, num_bs = sim->p.num_bs, num_bb = sim->p.num_bb;

#define ALLOC_TABLE(XX, _UNUSED1, _UNUSED2) \
	XX(sim->p.map_i         , sim->mp, N        * sizeof(int)) \
	XX(sim->p.map_ij        , sim->mp, N*N      * sizeof(int)) \
	XX(sim->p.bonds         , sim->mp, num_b*2  * sizeof(int)) \
	XX(sim->p.map_bs        , sim->mp, num_b*N  * sizeof(int)) \
	XX(sim->p.map_bb        , sim->mp, num_b*num_b * sizeof(int)) \
	XX(sim->p.peierlsu      , sim->mp, N*N      * sizeof(num)) \
	XX(sim->p.peierlsd      , sim->mp, N*N      * sizeof(num)) \
	XX(sim->p.degen_i       , sim->mp, num_i    * sizeof(int)) \
	XX(sim->p.degen_ij      , sim->mp, num_ij   * sizeof(int)) \
	XX(sim->p.degen_bs      , sim->mp, num_bs   * sizeof(int)) \
	XX(sim->p.degen_bb      , sim->mp, num_bb   * sizeof(int)) \
	XX(sim->p.exp_lambda    , sim->mp, N*2      * sizeof(double)) \
	XX(sim->p.exp_Ku        , sim->mp, N*N      * sizeof(num)) \
	XX(sim->p.exp_Kd        , sim->mp, N*N      * sizeof(num)) \
	XX(sim->p.inv_exp_Ku    , sim->mp, N*N      * sizeof(num)) \
	XX(sim->p.inv_exp_Kd    , sim->mp, N*N      * sizeof(num)) \
	XX(sim->p.exp_halfKu    , sim->mp, N*N      * sizeof(num)) \
	XX(sim->p.exp_halfKd    , sim->mp, N*N      * sizeof(num)) \
	XX(sim->p.inv_exp_halfKu, sim->mp, N*N      * sizeof(num)) \
	XX(sim->p.inv_exp_halfKd, sim->mp, N*N      * sizeof(num)) \
	XX(sim->p.del           , sim->mp, N*2      * sizeof(double)) \
	XX(sim->s.hs            , sim->mp, N*L      * sizeof(int)) \
	XX(sim->m_eq.density    , sim->mp, num_i    * sizeof(num)) \
	XX(sim->m_eq.double_occ , sim->mp, num_i    * sizeof(num)) \
	XX(sim->m_eq.g00        , sim->mp, num_ij   * sizeof(num)) \
	XX(sim->m_eq.nn         , sim->mp, num_ij   * sizeof(num)) \
	XX(sim->m_eq.xx         , sim->mp, num_ij   * sizeof(num)) \
	XX(sim->m_eq.zz         , sim->mp, num_ij   * sizeof(num)) \
	XX(sim->m_eq.pair_sw    , sim->mp, num_ij   * sizeof(num)) \
	XX(sim->m_eq.kk         , sim->mp, (sim->p.meas_energy_corr != 0)*num_bb * sizeof(num)) \
	XX(sim->m_eq.kv         , sim->mp, (sim->p.meas_energy_corr != 0)*num_bs * sizeof(num)) \
	XX(sim->m_eq.kn         , sim->mp, (sim->p.meas_energy_corr != 0)*num_bs * sizeof(num)) \
	XX(sim->m_eq.vv         , sim->mp, (sim->p.meas_energy_corr != 0)*num_ij * sizeof(num)) \
	XX(sim->m_eq.vn         , sim->mp, (sim->p.meas_energy_corr != 0)*num_ij * sizeof(num)) \
	XX(sim->m_ue.gt0        , sim->mp, (sim->p.period_uneqlt != 0)*num_ij*L * sizeof(num)) \
	XX(sim->m_ue.nn         , sim->mp, (sim->p.period_uneqlt != 0)*num_ij*L * sizeof(num)) \
	XX(sim->m_ue.xx         , sim->mp, (sim->p.period_uneqlt != 0)*num_ij*L * sizeof(num)) \
	XX(sim->m_ue.zz         , sim->mp, (sim->p.period_uneqlt != 0)*num_ij*L * sizeof(num)) \
	XX(sim->m_ue.pair_sw    , sim->mp, (sim->p.period_uneqlt != 0)*num_ij*L * sizeof(num)) \
	XX(sim->m_ue.pair_bb    , sim->mp, (sim->p.period_uneqlt != 0)*(sim->p.meas_bond_corr != 0)*num_bb*L * sizeof(num)) \
	XX(sim->m_ue.jj         , sim->mp, (sim->p.period_uneqlt != 0)*(sim->p.meas_bond_corr != 0)*num_bb*L * sizeof(num)) \
	XX(sim->m_ue.jsjs       , sim->mp, (sim->p.period_uneqlt != 0)*(sim->p.meas_bond_corr != 0)*num_bb*L * sizeof(num)) \
	XX(sim->m_ue.kk         , sim->mp, (sim->p.period_uneqlt != 0)*(sim->p.meas_bond_corr != 0)*num_bb*L * sizeof(num)) \
	XX(sim->m_ue.ksks       , sim->mp, (sim->p.period_uneqlt != 0)*(sim->p.meas_bond_corr != 0)*num_bb*L * sizeof(num)) \
	XX(sim->m_ue.kv         , sim->mp, (sim->p.period_uneqlt != 0)*(sim->p.meas_energy_corr != 0)*num_bs*L * sizeof(num)) \
	XX(sim->m_ue.kn         , sim->mp, (sim->p.period_uneqlt != 0)*(sim->p.meas_energy_corr != 0)*num_bs*L * sizeof(num)) \
	XX(sim->m_ue.vv         , sim->mp, (sim->p.period_uneqlt != 0)*(sim->p.meas_energy_corr != 0)*num_ij*L * sizeof(num)) \
	XX(sim->m_ue.vn         , sim->mp, (sim->p.period_uneqlt != 0)*(sim->p.meas_energy_corr != 0)*num_ij*L * sizeof(num)) \
	XX(sim->m_ue.nem_nnnn   , sim->mp, (sim->p.period_uneqlt != 0)*(sim->p.meas_nematic_corr != 0)*num_bb*L * sizeof(num)) \
	XX(sim->m_ue.nem_ssss   , sim->mp, (sim->p.period_uneqlt != 0)*(sim->p.meas_nematic_corr != 0)*num_bb*L * sizeof(num))

	sim->mp = pool_new(POOL_GET_SIZE(ALLOC_TABLE));
	POOL_DO_ALLOC(ALLOC_TABLE);
#undef ALLOC_TABLE

	my_read(_int,    "/params/map_i",          sim->p.map_i);
	my_read(_int,    "/params/map_ij",         sim->p.map_ij);
	my_read(_int,    "/params/bonds",          sim->p.bonds);
	my_read(_int,    "/params/map_bs",         sim->p.map_bs);
	my_read(_int,    "/params/map_bb",         sim->p.map_bb);
	my_read( , "/params/peierlsu", num_h5t,    sim->p.peierlsu);
	my_read( , "/params/peierlsd", num_h5t,    sim->p.peierlsd);
//	my_read(_double, "/params/K",              sim->p.K);
//	my_read(_double, "/params/U",              sim->p.U);
//	my_read(_double, "/params/dt",            &sim->p.dt);
	my_read(_int,    "/params/n_matmul",      &sim->p.n_matmul);
	my_read(_int,    "/params/n_delay",       &sim->p.n_delay);
	my_read(_int,    "/params/n_sweep_warm",  &sim->p.n_sweep_warm);
	my_read(_int,    "/params/n_sweep_meas",  &sim->p.n_sweep_meas);
	my_read(_int,    "/params/period_eqlt",   &sim->p.period_eqlt);
	my_read(_int,    "/params/degen_i",        sim->p.degen_i);
	my_read(_int,    "/params/degen_ij",       sim->p.degen_ij);
	my_read(_int,    "/params/degen_bs",       sim->p.degen_bs);
	my_read(_int,    "/params/degen_bb",       sim->p.degen_bb);
	my_read( , "/params/exp_Ku",     num_h5t,   sim->p.exp_Ku);
	my_read( , "/params/exp_Kd",     num_h5t,   sim->p.exp_Kd);
	my_read( , "/params/inv_exp_Ku", num_h5t,   sim->p.inv_exp_Ku);
	my_read( , "/params/inv_exp_Kd", num_h5t,   sim->p.inv_exp_Kd);
	my_read( , "/params/exp_halfKu",     num_h5t,   sim->p.exp_halfKu);
	my_read( , "/params/exp_halfKd",     num_h5t,   sim->p.exp_halfKd);
	my_read( , "/params/inv_exp_halfKu", num_h5t,   sim->p.inv_exp_halfKu);
	my_read( , "/params/inv_exp_halfKd", num_h5t,   sim->p.inv_exp_halfKd);
	my_read(_double, "/params/exp_lambda",     sim->p.exp_lambda);
	my_read(_double, "/params/del",            sim->p.del);
	my_read(_int,    "/params/F",             &sim->p.F);
	my_read(_int,    "/params/n_sweep",       &sim->p.n_sweep);
	my_read( ,       "/state/rng", H5T_NATIVE_UINT64, sim->s.rng);
	my_read(_int,    "/state/sweep",          &sim->s.sweep);
	my_read(_int,    "/state/hs",              sim->s.hs);
	my_read(_int,    "/meas_eqlt/n_sample",   &sim->m_eq.n_sample);
	my_read( , "/meas_eqlt/sign",        num_h5t, &sim->m_eq.sign);
	my_read( , "/meas_eqlt/density",     num_h5t, sim->m_eq.density);
	my_read( , "/meas_eqlt/double_occ",  num_h5t, sim->m_eq.double_occ);
	my_read( , "/meas_eqlt/g00",         num_h5t, sim->m_eq.g00);
	my_read( , "/meas_eqlt/nn",          num_h5t, sim->m_eq.nn);
	my_read( , "/meas_eqlt/xx",          num_h5t, sim->m_eq.xx);
	my_read( , "/meas_eqlt/zz",          num_h5t, sim->m_eq.zz);
	my_read( , "/meas_eqlt/pair_sw",     num_h5t, sim->m_eq.pair_sw);
	if (sim->p.meas_energy_corr) {
		my_read( , "/meas_eqlt/kk", num_h5t, sim->m_eq.kk);
		my_read( , "/meas_eqlt/kv", num_h5t, sim->m_eq.kv);
		my_read( , "/meas_eqlt/kn", num_h5t, sim->m_eq.kn);
		my_read( , "/meas_eqlt/vv", num_h5t, sim->m_eq.vv);
		my_read( , "/meas_eqlt/vn", num_h5t, sim->m_eq.vn);
	}
	if (sim->p.period_uneqlt > 0) {
		my_read(_int,    "/meas_uneqlt/n_sample", &sim->m_ue.n_sample);
		my_read( , "/meas_uneqlt/sign",      num_h5t, &sim->m_ue.sign);
		my_read( , "/meas_uneqlt/gt0",       num_h5t, sim->m_ue.gt0);
		my_read( , "/meas_uneqlt/nn",        num_h5t, sim->m_ue.nn);
		my_read( , "/meas_uneqlt/xx",        num_h5t, sim->m_ue.xx);
		my_read( , "/meas_uneqlt/zz",        num_h5t, sim->m_ue.zz);
		my_read( , "/meas_uneqlt/pair_sw",   num_h5t, sim->m_ue.pair_sw);
		if (sim->p.meas_bond_corr) {
			my_read( , "/meas_uneqlt/pair_bb", num_h5t, sim->m_ue.pair_bb);
			my_read( , "/meas_uneqlt/jj",      num_h5t, sim->m_ue.jj);
			my_read( , "/meas_uneqlt/jsjs",    num_h5t, sim->m_ue.jsjs)
			my_read( , "/meas_uneqlt/kk",      num_h5t, sim->m_ue.kk);
			my_read( , "/meas_uneqlt/ksks",    num_h5t, sim->m_ue.ksks);
		}
		if (sim->p.meas_energy_corr) {
			my_read( , "/meas_uneqlt/kv", num_h5t, sim->m_ue.kv);
			my_read( , "/meas_uneqlt/kn", num_h5t, sim->m_ue.kn);
			my_read( , "/meas_uneqlt/vv", num_h5t, sim->m_ue.vv);
			my_read( , "/meas_uneqlt/vn", num_h5t, sim->m_ue.vn);
		}
		if (sim->p.meas_nematic_corr) {
			my_read( , "/meas_uneqlt/nem_nnnn", num_h5t, sim->m_ue.nem_nnnn);
			my_read( , "/meas_uneqlt/nem_ssss", num_h5t, sim->m_ue.nem_ssss);
		}
	}

#undef my_read

	status = H5Fclose(file_id);
	return_if(status < 0, -1, "H5Fclose() failed: %d\n", status);
	return 0;
}

int sim_data_save(const struct sim_data *sim)
{
	const hid_t file_id = H5Fopen(sim->file, H5F_ACC_RDWR, H5P_DEFAULT);
	return_if(file_id < 0, -1, "H5Fopen() failed: %ld\n", file_id);

	herr_t status;
	hid_t dset_id;

#define my_write(name, type, data) do { \
	dset_id = H5Dopen2(file_id, (name), H5P_DEFAULT); \
	return_if(dset_id < 0, -1, "H5Dopen2() failed for %s: %ld\n", name, dset_id); \
	status = H5Dwrite(dset_id, (type), H5S_ALL, H5S_ALL, H5P_DEFAULT, (data)); \
	return_if(status < 0, -1, "H5Dwrite() failed for %s: %d\n", name, status); \
	status = H5Dclose(dset_id); \
	return_if(status < 0, -1, "H5Dclose() failed for %s: %d\n", name, status); \
} while (0);

	my_write("/state/rng",            H5T_NATIVE_UINT64,  sim->s.rng);
	my_write("/state/sweep",          H5T_NATIVE_INT,    &sim->s.sweep);
	my_write("/state/hs",             H5T_NATIVE_INT,     sim->s.hs);
	my_write("/meas_eqlt/n_sample",   H5T_NATIVE_INT,    &sim->m_eq.n_sample);
	my_write("/meas_eqlt/sign",       num_h5t, &sim->m_eq.sign);
	my_write("/meas_eqlt/density",    num_h5t,  sim->m_eq.density);
	my_write("/meas_eqlt/double_occ", num_h5t,  sim->m_eq.double_occ);
	my_write("/meas_eqlt/g00",        num_h5t,  sim->m_eq.g00);
	my_write("/meas_eqlt/nn",         num_h5t,  sim->m_eq.nn);
	my_write("/meas_eqlt/xx",         num_h5t,  sim->m_eq.xx);
	my_write("/meas_eqlt/zz",         num_h5t,  sim->m_eq.zz);
	my_write("/meas_eqlt/pair_sw",    num_h5t,  sim->m_eq.pair_sw);
	if (sim->p.meas_energy_corr) {
		my_write("/meas_eqlt/kk", num_h5t, sim->m_eq.kk);
		my_write("/meas_eqlt/kv", num_h5t, sim->m_eq.kv);
		my_write("/meas_eqlt/kn", num_h5t, sim->m_eq.kn);
		my_write("/meas_eqlt/vv", num_h5t, sim->m_eq.vv);
		my_write("/meas_eqlt/vn", num_h5t, sim->m_eq.vn);
	}
	if (sim->p.period_uneqlt > 0) {
		my_write("/meas_uneqlt/n_sample", H5T_NATIVE_INT,    &sim->m_ue.n_sample);
		my_write("/meas_uneqlt/sign",     num_h5t, &sim->m_ue.sign);
		my_write("/meas_uneqlt/gt0",      num_h5t,  sim->m_ue.gt0);
		my_write("/meas_uneqlt/nn",       num_h5t,  sim->m_ue.nn);
		my_write("/meas_uneqlt/xx",       num_h5t,  sim->m_ue.xx);
		my_write("/meas_uneqlt/zz",       num_h5t,  sim->m_ue.zz);
		my_write("/meas_uneqlt/pair_sw",  num_h5t,  sim->m_ue.pair_sw);
		if (sim->p.meas_bond_corr) {
			my_write("/meas_uneqlt/pair_bb", num_h5t, sim->m_ue.pair_bb);
			my_write("/meas_uneqlt/jj",      num_h5t, sim->m_ue.jj);
			my_write("/meas_uneqlt/jsjs",    num_h5t, sim->m_ue.jsjs);
			my_write("/meas_uneqlt/kk",      num_h5t, sim->m_ue.kk);
			my_write("/meas_uneqlt/ksks",    num_h5t, sim->m_ue.ksks);
		}
		if (sim->p.meas_energy_corr) {
			my_write("/meas_uneqlt/kv", num_h5t, sim->m_ue.kv);
			my_write("/meas_uneqlt/kn", num_h5t, sim->m_ue.kn);
			my_write("/meas_uneqlt/vv", num_h5t, sim->m_ue.vv);
			my_write("/meas_uneqlt/vn", num_h5t, sim->m_ue.vn);
		}
		if (sim->p.meas_nematic_corr) {
			my_write("/meas_uneqlt/nem_nnnn", num_h5t, sim->m_ue.nem_nnnn);
			my_write("/meas_uneqlt/nem_ssss", num_h5t, sim->m_ue.nem_ssss);
		}
	}

#undef my_write

	status = H5Fclose(file_id);
	return_if(status < 0, -1, "H5Fclose() failed: %d\n", status);
	return 0;
}

void sim_data_free(struct sim_data *sim)
{
	pool_free(sim->mp);
	sim->mp = NULL;
}

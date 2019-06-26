#include "data.h"
#include <stdio.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include "util.h"

#define return_if(cond, val, ...) \
	do {if (cond) {fprintf(stderr, __VA_ARGS__); return (val);}} while (0)

int sim_data_read_alloc(struct sim_data *sim, const char *file)
{
	const hid_t file_id = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
	return_if(file_id < 0, -1, "H5Fopen() failed: %ld\n", file_id);

	herr_t status;

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

	sim->p.map_i         = my_calloc(N        * sizeof(int));
	sim->p.map_ij        = my_calloc(N*N      * sizeof(int));
	sim->p.bonds         = my_calloc(num_b*2  * sizeof(int));
	sim->p.map_bs        = my_calloc(num_b*N  * sizeof(int));
	sim->p.map_bb        = my_calloc(num_b*num_b * sizeof(int));
//	sim->p.K             = my_calloc(N*N      * sizeof(double));
//	sim->p.U             = my_calloc(num_i    * sizeof(double));
	sim->p.degen_i       = my_calloc(num_i    * sizeof(int));
	sim->p.degen_ij      = my_calloc(num_ij   * sizeof(int));
	sim->p.degen_bs      = my_calloc(num_bs   * sizeof(int));
	sim->p.degen_bb      = my_calloc(num_bb   * sizeof(int));
	sim->p.exp_K         = my_calloc(N*N      * sizeof(double));
	sim->p.inv_exp_K     = my_calloc(N*N      * sizeof(double));
	sim->p.exp_halfK     = my_calloc(N*N      * sizeof(double));
	sim->p.inv_exp_halfK = my_calloc(N*N      * sizeof(double));
	sim->p.exp_lambda    = my_calloc(N*2      * sizeof(double));
	sim->p.del           = my_calloc(N*2      * sizeof(double));
	sim->s.hs            = my_calloc(N*L      * sizeof(int));
	sim->m_eq.density    = my_calloc(num_i    * sizeof(double));
	sim->m_eq.double_occ = my_calloc(num_i    * sizeof(double));
	sim->m_eq.g00        = my_calloc(num_ij   * sizeof(double));
	sim->m_eq.nn         = my_calloc(num_ij   * sizeof(double));
	sim->m_eq.xx         = my_calloc(num_ij   * sizeof(double));
	sim->m_eq.zz         = my_calloc(num_ij   * sizeof(double));
	sim->m_eq.pair_sw    = my_calloc(num_ij   * sizeof(double));
	if (sim->p.meas_energy_corr) {
		sim->m_eq.kk = my_calloc(num_bb * sizeof(double));
		sim->m_eq.kv = my_calloc(num_bs * sizeof(double));
		sim->m_eq.kn = my_calloc(num_bs * sizeof(double));
		sim->m_eq.vv = my_calloc(num_ij * sizeof(double));
		sim->m_eq.vn = my_calloc(num_ij * sizeof(double));
	}
	if (sim->p.period_uneqlt > 0) {
		sim->m_ue.gt0     = my_calloc(num_ij*L * sizeof(double));
		sim->m_ue.nn      = my_calloc(num_ij*L * sizeof(double));
		sim->m_ue.xx      = my_calloc(num_ij*L * sizeof(double));
		sim->m_ue.zz      = my_calloc(num_ij*L * sizeof(double));
		sim->m_ue.pair_sw = my_calloc(num_ij*L * sizeof(double));
		if (sim->p.meas_bond_corr) {
			sim->m_ue.pair_bb = my_calloc(num_bb*L * sizeof(double));
			sim->m_ue.jj      = my_calloc(num_bb*L * sizeof(double));
			sim->m_ue.jsjs    = my_calloc(num_bb*L * sizeof(double));
			sim->m_ue.kk      = my_calloc(num_bb*L * sizeof(double));
			sim->m_ue.ksks    = my_calloc(num_bb*L * sizeof(double));
		}
		if (sim->p.meas_energy_corr) {
			sim->m_ue.kv      = my_calloc(num_bs*L * sizeof(double));
			sim->m_ue.kn      = my_calloc(num_bs*L * sizeof(double));
			sim->m_ue.vv      = my_calloc(num_ij*L * sizeof(double));
			sim->m_ue.vn      = my_calloc(num_ij*L * sizeof(double));
		}
		if (sim->p.meas_nematic_corr) {
			sim->m_ue.nem_nnnn = my_calloc(num_bb*L * sizeof(double));
			sim->m_ue.nem_ssss = my_calloc(num_bb*L * sizeof(double));
		}
	}
	// make sure anything appended here is free'd in sim_data_free()

	my_read(_int,    "/params/map_i",          sim->p.map_i);
	my_read(_int,    "/params/map_ij",         sim->p.map_ij);
	my_read(_int,    "/params/bonds",          sim->p.bonds);
	my_read(_int,    "/params/map_bs",         sim->p.map_bs);
	my_read(_int,    "/params/map_bb",         sim->p.map_bb);
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
	my_read(_double, "/params/exp_K",          sim->p.exp_K);
	my_read(_double, "/params/inv_exp_K",      sim->p.inv_exp_K);
	my_read(_double, "/params/exp_halfK",      sim->p.exp_halfK);
	my_read(_double, "/params/inv_exp_halfK",  sim->p.inv_exp_halfK);
	my_read(_double, "/params/exp_lambda",     sim->p.exp_lambda);
	my_read(_double, "/params/del",            sim->p.del);
	my_read(_int,    "/params/F",             &sim->p.F);
	my_read(_int,    "/params/n_sweep",       &sim->p.n_sweep);
	my_read( ,       "/state/rng", H5T_NATIVE_UINT64, sim->s.rng);
	my_read(_int,    "/state/sweep",          &sim->s.sweep);
	my_read(_int,    "/state/hs",              sim->s.hs);
	my_read(_int,    "/meas_eqlt/n_sample",   &sim->m_eq.n_sample);
	my_read(_double, "/meas_eqlt/sign",       &sim->m_eq.sign);
	my_read(_double, "/meas_eqlt/density",     sim->m_eq.density);
	my_read(_double, "/meas_eqlt/double_occ",  sim->m_eq.double_occ);
	my_read(_double, "/meas_eqlt/g00",         sim->m_eq.g00);
	my_read(_double, "/meas_eqlt/nn",          sim->m_eq.nn);
	my_read(_double, "/meas_eqlt/xx",          sim->m_eq.xx);
	my_read(_double, "/meas_eqlt/zz",          sim->m_eq.zz);
	my_read(_double, "/meas_eqlt/pair_sw",     sim->m_eq.pair_sw);
	if (sim->p.meas_energy_corr) {
		my_read(_double, "/meas_eqlt/kk", sim->m_eq.kk);
		my_read(_double, "/meas_eqlt/kv", sim->m_eq.kv);
		my_read(_double, "/meas_eqlt/kn", sim->m_eq.kn);
		my_read(_double, "/meas_eqlt/vv", sim->m_eq.vv);
		my_read(_double, "/meas_eqlt/vn", sim->m_eq.vn);
	}
	if (sim->p.period_uneqlt > 0) {
		my_read(_int,    "/meas_uneqlt/n_sample", &sim->m_ue.n_sample);
		my_read(_double, "/meas_uneqlt/sign",     &sim->m_ue.sign);
		my_read(_double, "/meas_uneqlt/gt0",       sim->m_ue.gt0);
		my_read(_double, "/meas_uneqlt/nn",        sim->m_ue.nn);
		my_read(_double, "/meas_uneqlt/xx",        sim->m_ue.xx);
		my_read(_double, "/meas_uneqlt/zz",        sim->m_ue.zz);
		my_read(_double, "/meas_uneqlt/pair_sw",   sim->m_ue.pair_sw);
		if (sim->p.meas_bond_corr) {
			my_read(_double, "/meas_uneqlt/pair_bb", sim->m_ue.pair_bb);
			my_read(_double, "/meas_uneqlt/jj",      sim->m_ue.jj);
			my_read(_double, "/meas_uneqlt/jsjs",    sim->m_ue.jsjs)
			my_read(_double, "/meas_uneqlt/kk",      sim->m_ue.kk);
			my_read(_double, "/meas_uneqlt/ksks",    sim->m_ue.ksks);
		}
		if (sim->p.meas_energy_corr) {
			my_read(_double, "/meas_uneqlt/kv", sim->m_ue.kv);
			my_read(_double, "/meas_uneqlt/kn", sim->m_ue.kn);
			my_read(_double, "/meas_uneqlt/vv", sim->m_ue.vv);
			my_read(_double, "/meas_uneqlt/vn", sim->m_ue.vn);
		}
		if (sim->p.meas_nematic_corr) {
			my_read(_double, "/meas_uneqlt/nem_nnnn", sim->m_ue.nem_nnnn);
			my_read(_double, "/meas_uneqlt/nem_ssss", sim->m_ue.nem_ssss);
		}
	}

#undef my_read

	status = H5Fclose(file_id);
	return_if(status < 0, -1, "H5Fclose() failed: %d\n", status);
	return 0;
}

int sim_data_save(const struct sim_data *sim, const char *file)
{
	const hid_t file_id = H5Fopen(file, H5F_ACC_RDWR, H5P_DEFAULT);
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
	my_write("/meas_eqlt/sign",       H5T_NATIVE_DOUBLE, &sim->m_eq.sign);
	my_write("/meas_eqlt/density",    H5T_NATIVE_DOUBLE,  sim->m_eq.density);
	my_write("/meas_eqlt/double_occ", H5T_NATIVE_DOUBLE,  sim->m_eq.double_occ);
	my_write("/meas_eqlt/g00",        H5T_NATIVE_DOUBLE,  sim->m_eq.g00);
	my_write("/meas_eqlt/nn",         H5T_NATIVE_DOUBLE,  sim->m_eq.nn);
	my_write("/meas_eqlt/xx",         H5T_NATIVE_DOUBLE,  sim->m_eq.xx);
	my_write("/meas_eqlt/zz",         H5T_NATIVE_DOUBLE,  sim->m_eq.zz);
	my_write("/meas_eqlt/pair_sw",    H5T_NATIVE_DOUBLE,  sim->m_eq.pair_sw);
	if (sim->p.meas_energy_corr) {
		my_write("/meas_eqlt/kk", H5T_NATIVE_DOUBLE, sim->m_eq.kk);
		my_write("/meas_eqlt/kv", H5T_NATIVE_DOUBLE, sim->m_eq.kv);
		my_write("/meas_eqlt/kn", H5T_NATIVE_DOUBLE, sim->m_eq.kn);
		my_write("/meas_eqlt/vv", H5T_NATIVE_DOUBLE, sim->m_eq.vv);
		my_write("/meas_eqlt/vn", H5T_NATIVE_DOUBLE, sim->m_eq.vn);
	}
	if (sim->p.period_uneqlt > 0) {
		my_write("/meas_uneqlt/n_sample", H5T_NATIVE_INT,    &sim->m_ue.n_sample);
		my_write("/meas_uneqlt/sign",     H5T_NATIVE_DOUBLE, &sim->m_ue.sign);
		my_write("/meas_uneqlt/gt0",      H5T_NATIVE_DOUBLE,  sim->m_ue.gt0);
		my_write("/meas_uneqlt/nn",       H5T_NATIVE_DOUBLE,  sim->m_ue.nn);
		my_write("/meas_uneqlt/xx",       H5T_NATIVE_DOUBLE,  sim->m_ue.xx);
		my_write("/meas_uneqlt/zz",       H5T_NATIVE_DOUBLE,  sim->m_ue.zz);
		my_write("/meas_uneqlt/pair_sw",  H5T_NATIVE_DOUBLE,  sim->m_ue.pair_sw);
		if (sim->p.meas_bond_corr) {
			my_write("/meas_uneqlt/pair_bb", H5T_NATIVE_DOUBLE, sim->m_ue.pair_bb);
			my_write("/meas_uneqlt/jj",      H5T_NATIVE_DOUBLE, sim->m_ue.jj);
			my_write("/meas_uneqlt/jsjs",    H5T_NATIVE_DOUBLE, sim->m_ue.jsjs);
			my_write("/meas_uneqlt/kk",      H5T_NATIVE_DOUBLE, sim->m_ue.kk);
			my_write("/meas_uneqlt/ksks",    H5T_NATIVE_DOUBLE, sim->m_ue.ksks);
		}
		if (sim->p.meas_energy_corr) {
			my_write("/meas_uneqlt/kv", H5T_NATIVE_DOUBLE, sim->m_ue.kv);
			my_write("/meas_uneqlt/kn", H5T_NATIVE_DOUBLE, sim->m_ue.kn);
			my_write("/meas_uneqlt/vv", H5T_NATIVE_DOUBLE, sim->m_ue.vv);
			my_write("/meas_uneqlt/vn", H5T_NATIVE_DOUBLE, sim->m_ue.vn);
		}
		if (sim->p.meas_nematic_corr) {
			my_write("/meas_uneqlt/nem_nnnn", H5T_NATIVE_DOUBLE, sim->m_ue.nem_nnnn);
			my_write("/meas_uneqlt/nem_ssss", H5T_NATIVE_DOUBLE, sim->m_ue.nem_ssss);
		}
	}

#undef my_write

	status = H5Fclose(file_id);
	return_if(status < 0, -1, "H5Fclose() failed: %d\n", status);
	return 0;
}

void sim_data_free(const struct sim_data *sim)
{
	if (sim->p.period_uneqlt > 0) {
		if (sim->p.meas_nematic_corr) {
			my_free(sim->m_ue.nem_ssss);
			my_free(sim->m_ue.nem_nnnn);
		}
		if (sim->p.meas_energy_corr) {
			my_free(sim->m_ue.vn);
			my_free(sim->m_ue.vv);
			my_free(sim->m_ue.kn);
			my_free(sim->m_ue.kv);
		}
		if (sim->p.meas_bond_corr) {
			my_free(sim->m_ue.ksks);
			my_free(sim->m_ue.kk);
			my_free(sim->m_ue.jsjs);
			my_free(sim->m_ue.jj);
			my_free(sim->m_ue.pair_bb);
		}
		my_free(sim->m_ue.pair_sw);
		my_free(sim->m_ue.zz);
		my_free(sim->m_ue.xx);
		my_free(sim->m_ue.nn);
		my_free(sim->m_ue.gt0);
	}
	if (sim->p.meas_energy_corr) {
		my_free(sim->m_eq.vn);
		my_free(sim->m_eq.vv);
		my_free(sim->m_eq.kn);
		my_free(sim->m_eq.kv);
		my_free(sim->m_eq.kk);
	}
	my_free(sim->m_eq.pair_sw);
	my_free(sim->m_eq.zz);
	my_free(sim->m_eq.xx);
	my_free(sim->m_eq.nn);
	my_free(sim->m_eq.g00);
	my_free(sim->m_eq.double_occ);
	my_free(sim->m_eq.density);
	my_free(sim->s.hs);
	my_free(sim->p.del);
	my_free(sim->p.exp_lambda);
	my_free(sim->p.inv_exp_halfK);
	my_free(sim->p.exp_halfK);
	my_free(sim->p.inv_exp_K);
	my_free(sim->p.exp_K);
	my_free(sim->p.degen_bb);
	my_free(sim->p.degen_bs);
	my_free(sim->p.degen_ij);
	my_free(sim->p.degen_i);
//	my_free(sim->p.U);
//	my_free(sim->p.K);
	my_free(sim->p.map_bb);
	my_free(sim->p.map_bs);
	my_free(sim->p.bonds);
	my_free(sim->p.map_ij);
	my_free(sim->p.map_i);
}

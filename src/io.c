#include "io.h"
#include <stdio.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include "util.h"

#define return_if(cond, val, ...) \
	do {if (cond) {fprintf(stderr, __VA_ARGS__); return (val);}} while (0)

int read_file(const char *file, struct params *p, struct state *s,
		struct meas_eqlt *m_eq, struct meas_uneqlt *m_ue)
{
	const hid_t file_id = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
	return_if(file_id < 0, -1, "H5Fopen() failed: %d\n", file_id);

	herr_t status;

#define my_read(_type, name, ...) do { \
	status = H5LTread_dataset##_type(file_id, (name), __VA_ARGS__); \
	return_if(status < 0, -1, "H5LTread_dataset() failed for %s: %d\n", (name), status); \
} while (0);

	my_read(_int, "/params/N",      &p->N);
	my_read(_int, "/params/L",      &p->L);
	my_read(_int, "/params/num_i",  &p->num_i);
	my_read(_int, "/params/num_ij", &p->num_ij);

	const int N = p->N, L = p->L, num_i = p->num_i, num_ij = p->num_ij;
	p->map_i         = my_calloc(N        * sizeof(int));
	p->map_ij        = my_calloc(N*N      * sizeof(int));
//	p->K             = my_calloc(N*N      * sizeof(double));
//	p->U             = my_calloc(num_i    * sizeof(double));
	p->degen_i       = my_calloc(num_i    * sizeof(int));
	p->degen_ij      = my_calloc(num_ij   * sizeof(int));
	p->exp_K         = my_calloc(N*N      * sizeof(double));
	p->inv_exp_K     = my_calloc(N*N      * sizeof(double));
	p->exp_lambda    = my_calloc(N*2      * sizeof(double));
	p->del           = my_calloc(N*2      * sizeof(double));
	s->hs            = my_calloc(N*L      * sizeof(int));
	m_eq->density    = my_calloc(num_i    * sizeof(double));
	m_eq->double_occ = my_calloc(num_i    * sizeof(double));
	m_eq->g00        = my_calloc(num_ij   * sizeof(double));
	m_eq->nn         = my_calloc(num_ij   * sizeof(double));
	m_eq->xx         = my_calloc(num_ij   * sizeof(double));
	m_eq->zz         = my_calloc(num_ij   * sizeof(double));
	m_eq->pair_sw    = my_calloc(num_ij   * sizeof(double));
	m_ue->g0t        = my_calloc(num_ij*L * sizeof(double));
	m_ue->gt0        = my_calloc(num_ij*L * sizeof(double));
	m_ue->nn         = my_calloc(num_ij*L * sizeof(double));
	m_ue->xx         = my_calloc(num_ij*L * sizeof(double));
	m_ue->zz         = my_calloc(num_ij*L * sizeof(double));
	m_ue->pair_sw    = my_calloc(num_ij*L * sizeof(double));
	// make sure anything appended here is free'd in dqmc_wrapper()

	my_read(_int,    "/params/map_i",          p->map_i);
	my_read(_int,    "/params/map_ij",         p->map_ij);
//	my_read(_double, "/params/K",              p->K);
//	my_read(_double, "/params/U",              p->U);
//	my_read(_double, "/params/dt",            &p->dt);
	my_read(_int,    "/params/n_matmul",      &p->n_matmul);
	my_read(_int,    "/params/n_delay",       &p->n_delay);
	my_read(_int,    "/params/n_sweep_warm",  &p->n_sweep_warm);
	my_read(_int,    "/params/n_sweep_meas",  &p->n_sweep_meas);
	my_read(_int,    "/params/period_eqlt",   &p->period_eqlt);
	my_read(_int,    "/params/period_uneqlt", &p->period_uneqlt);
	my_read(_int,    "/params/degen_i",        p->degen_i);
	my_read(_int,    "/params/degen_ij",       p->degen_ij);
	my_read(_double, "/params/exp_K",          p->exp_K);
	my_read(_double, "/params/inv_exp_K",      p->inv_exp_K);
	my_read(_double, "/params/exp_lambda",     p->exp_lambda);
	my_read(_double, "/params/del",            p->del);
	my_read(_int,    "/params/n_Bs",          &p->n_Bs);
	my_read(_int,    "/params/n_sweep",       &p->n_sweep);
	my_read( ,       "/state/rng", H5T_NATIVE_UINT64, s->rng);
	my_read(_int,    "/state/sweep",          &s->sweep);
	my_read(_int,    "/state/hs",              s->hs);
	my_read(_int,    "/meas_eqlt/n_sample",   &m_eq->n_sample);
	my_read(_double, "/meas_eqlt/sign",       &m_eq->sign);
	my_read(_double, "/meas_eqlt/density",     m_eq->density);
	my_read(_double, "/meas_eqlt/double_occ",  m_eq->double_occ);
	my_read(_double, "/meas_eqlt/g00",         m_eq->g00);
	my_read(_double, "/meas_eqlt/nn",          m_eq->nn);
	my_read(_double, "/meas_eqlt/xx",          m_eq->xx);
	my_read(_double, "/meas_eqlt/zz",          m_eq->zz);
	my_read(_double, "/meas_eqlt/pair_sw",     m_eq->pair_sw);
	my_read(_int,    "/meas_uneqlt/n_sample", &m_ue->n_sample);
	my_read(_double, "/meas_uneqlt/sign",     &m_ue->sign);
	my_read(_double, "/meas_uneqlt/g0t",       m_ue->g0t);
	my_read(_double, "/meas_uneqlt/gt0",       m_ue->gt0);
	my_read(_double, "/meas_uneqlt/nn",        m_ue->nn);
	my_read(_double, "/meas_uneqlt/xx",        m_ue->xx);
	my_read(_double, "/meas_uneqlt/zz",        m_ue->zz);
	my_read(_double, "/meas_uneqlt/pair_sw",   m_ue->pair_sw);

#undef my_read

	status = H5Fclose(file_id);
	return_if(status < 0, -1, "H5Fclose() failed: %d\n", status);
	return 0;
}

int save_file(const char *file, const struct state *s,
		const struct meas_eqlt *m_eq, const struct meas_uneqlt *m_ue)
{
	const hid_t file_id = H5Fopen(file, H5F_ACC_RDWR, H5P_DEFAULT);
	return_if(file_id < 0, -1, "H5Fopen() failed: %d\n", file_id);

	herr_t status;
	hid_t dset_id;

#define my_write(name, type, data) do { \
	dset_id = H5Dopen2(file_id, (name), H5P_DEFAULT); \
	return_if(dset_id < 0, -1, "H5Dopen2() failed for %s: %d\n", name, dset_id); \
	status = H5Dwrite(dset_id, (type), H5S_ALL, H5S_ALL, H5P_DEFAULT, (data)); \
	return_if(status < 0, -1, "H5Dwrite() failed for %s: %d\n", name, status); \
	status = H5Dclose(dset_id); \
	return_if(status < 0, -1, "H5Dclose() failed for %s: %d\n", name, status); \
} while (0);

	my_write("/state/rng",            H5T_NATIVE_UINT64,  s->rng);
	my_write("/state/sweep",          H5T_NATIVE_INT,    &s->sweep);
	my_write("/state/hs",             H5T_NATIVE_INT,     s->hs);
	my_write("/meas_eqlt/n_sample",   H5T_NATIVE_INT,    &m_eq->n_sample);
	my_write("/meas_eqlt/sign",       H5T_NATIVE_DOUBLE, &m_eq->sign);
	my_write("/meas_eqlt/density",    H5T_NATIVE_DOUBLE,  m_eq->density);
	my_write("/meas_eqlt/double_occ", H5T_NATIVE_DOUBLE,  m_eq->double_occ);
	my_write("/meas_eqlt/g00",        H5T_NATIVE_DOUBLE,  m_eq->g00);
	my_write("/meas_eqlt/nn",         H5T_NATIVE_DOUBLE,  m_eq->nn);
	my_write("/meas_eqlt/xx",         H5T_NATIVE_DOUBLE,  m_eq->xx);
	my_write("/meas_eqlt/zz",         H5T_NATIVE_DOUBLE,  m_eq->zz);
	my_write("/meas_eqlt/pair_sw",    H5T_NATIVE_DOUBLE,  m_eq->pair_sw);
	my_write("/meas_uneqlt/n_sample", H5T_NATIVE_INT,    &m_ue->n_sample);
	my_write("/meas_uneqlt/sign",     H5T_NATIVE_DOUBLE, &m_ue->sign);
	my_write("/meas_uneqlt/g0t",      H5T_NATIVE_DOUBLE,  m_ue->g0t);
	my_write("/meas_uneqlt/gt0",      H5T_NATIVE_DOUBLE,  m_ue->gt0);
	my_write("/meas_uneqlt/nn",       H5T_NATIVE_DOUBLE,  m_ue->nn);
	my_write("/meas_uneqlt/xx",       H5T_NATIVE_DOUBLE,  m_ue->xx);
	my_write("/meas_uneqlt/zz",       H5T_NATIVE_DOUBLE,  m_ue->zz);
	my_write("/meas_uneqlt/pair_sw",  H5T_NATIVE_DOUBLE,  m_ue->pair_sw);

#undef my_write

	status = H5Fclose(file_id);
	return_if(status < 0, -1, "H5Fclose() failed: %d\n", status);
	return 0;
}

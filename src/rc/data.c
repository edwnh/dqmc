#include "data.h"
#include <stdio.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include "linalg.h"
#include "mem.h"
#include "numeric.h"
#include "greens.h"
#include "sim_types.h"

#define H5TYPE_int H5T_NATIVE_INT
#define H5TYPE_double H5T_NATIVE_DOUBLE
#define H5TYPE_num num_h5t

#define return_if(cond, val, ...) \
	do {if (cond) {fprintf(stderr, __VA_ARGS__); return (val);}} while (0)

static hid_t num_h5t;

// does nothing if dataset doesn't exist
static herr_t my_read(hid_t loc_id,
	const char *name, hid_t type_id, void *buf)
{
	htri_t exists = H5Lexists(loc_id, name, H5P_DEFAULT);
	if (exists < 0)
		return (herr_t)exists;
	if (exists > 0)
		return H5LTread_dataset(loc_id, name, type_id, buf);
	return 0; // no error
}

// either creates data or overwrites existing.
// assumes any existing dataset has the correct size and type.
// rank 0 if size == 0.
static herr_t my_write(hid_t loc_id,
	const char *name, hsize_t size, hid_t type_id, const void *buf)
{
	htri_t exists = H5Lexists(loc_id, name, H5P_DEFAULT);
	if (exists < 0)
		return (herr_t)exists;
	if (exists == 0)
		return H5LTmake_dataset(loc_id, name, (size > 0), &size, type_id, buf);
	hid_t dset = H5Dopen2(loc_id, name, H5P_DEFAULT);
	if (dset < 0) return (herr_t)dset;
	herr_t status = H5Dwrite(dset, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
	status |= H5Dclose(dset);
	return status;
}

static int do_alloc(struct RC(sim_data) *sim)
{
	// these must be initialized in sim before any allocations
	const int N = sim->p.N, L = sim->p.L, F = sim->p.F;
	const int num_i = sim->p.num_i, num_ij = sim->p.num_ij;
	const int num_b = sim->p.num_b, num_bs = sim->p.num_bs, num_bb = sim->p.num_bb;
	const int meas_bond_corr = sim->p.meas_bond_corr;
	const int meas_energy_corr = sim->p.meas_energy_corr;
	const int meas_nematic_corr = sim->p.meas_nematic_corr;
	const int period_uneqlt = sim->p.period_uneqlt;

	const int ld = best_ld(N);
	const int lwork = RC(get_lwork)(N, ld);

	const struct alloc_entry tab[] = {
#define X(name, type, size) {(void **)&sim->p.name, (size) * sizeof(type)},
		PARAMS_ARRAY_LIST
#undef X
		{(void **)&sim->s.hs, N*L * sizeof(int)},
#define X(name, size) {(void **)&sim->m_eq.name, (size) * sizeof(num)},
		MEAS_EQLT_LIST
#undef X
#define X(name, size) {(void **)&sim->m_ue.name, (size) * sizeof(num)},
		MEAS_UNEQLT_LIST
#undef X
#define X(name, type, size) {(void **)&sim->ws[0].name, (size) * sizeof(type)},
		WORKSPACE_LIST
#undef X
#define X(name, type, size) {(void **)&sim->ws[1].name, (size) * sizeof(type)},
		WORKSPACE_LIST
#undef X
	};

	sim->pool = my_calloc_table(tab, sizeof(tab)/sizeof(tab[0]));
	return_if(sim->pool == NULL, -1, "my_calloc_table() failed\n");
	return 0;
}

struct RC(sim_data) *RC(sim_data_read_alloc)(const char *file)
{
	struct RC(sim_data) *sim = my_calloc(sizeof(*sim));

	const hid_t file_id = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
	return_if(file_id < 0, NULL, "H5Fopen() failed: %ld\n", file_id);

	sim->file = file;

	herr_t status = 0;

#ifdef USE_CPLX
	num_h5t = H5Tcreate(H5T_COMPOUND, sizeof(num));
	status |= H5Tinsert(num_h5t, "r", 0, H5T_NATIVE_DOUBLE);
	status |= H5Tinsert(num_h5t, "i", 8, H5T_NATIVE_DOUBLE);
#else
	num_h5t = H5T_NATIVE_DOUBLE;
#endif

#define X(name) \
	status |= my_read(file_id, "/params/" #name, H5T_NATIVE_INT, &sim->p.name);
	PARAMS_SCALAR_INT_LIST
#undef X

	do_alloc(sim);

	const int L = sim->p.L;
	const int num_i = sim->p.num_i, num_ij = sim->p.num_ij;
	const int num_b = sim->p.num_b, num_bs = sim->p.num_bs, num_bb = sim->p.num_bb;
	const int meas_bond_corr = sim->p.meas_bond_corr;
	const int meas_energy_corr = sim->p.meas_energy_corr;
	const int meas_nematic_corr = sim->p.meas_nematic_corr;

#define X(name, type, size) \
	status |= my_read(file_id, "/params/" #name, H5TYPE_##type, sim->p.name);
	PARAMS_ARRAY_LIST
#undef X

	status |= my_read(file_id, "/state/rng",   H5T_NATIVE_UINT64, sim->s.rng);
	status |= my_read(file_id, "/state/sweep", H5T_NATIVE_INT,   &sim->s.sweep);
	status |= my_read(file_id, "/state/hs",    H5T_NATIVE_INT,    sim->s.hs);

	status |= my_read(file_id, "/meas_eqlt/n_sample", H5T_NATIVE_INT,   &sim->m_eq.n_sample);
	status |= my_read(file_id, "/meas_eqlt/sign",     num_h5t,          &sim->m_eq.sign);
#define X(name, size) \
	if ((size) > 0) status |= my_read(file_id, "/meas_eqlt/" #name, num_h5t, sim->m_eq.name);
	MEAS_EQLT_LIST
#undef X
	if (sim->p.period_uneqlt > 0) {
		status |= my_read(file_id, "/meas_uneqlt/n_sample", H5T_NATIVE_INT, &sim->m_ue.n_sample);
		status |= my_read(file_id, "/meas_uneqlt/sign",     num_h5t,        &sim->m_ue.sign);
#define X(name, size) \
	if ((size) > 0) status |= my_read(file_id, "/meas_uneqlt/" #name, num_h5t, sim->m_ue.name);
		MEAS_UNEQLT_LIST
#undef X
	}

	status |= H5Fclose(file_id);
	return_if(status < 0, NULL, "my_read() or H5Fclose() failed: %d\n", status);
	return sim;
}

int RC(sim_data_save)(const struct RC(sim_data) *sim)
{
	const int N = sim->p.N, L = sim->p.L;
	const int num_i = sim->p.num_i, num_ij = sim->p.num_ij;
	const int num_b = sim->p.num_b, num_bs = sim->p.num_bs, num_bb = sim->p.num_bb;
	const int meas_bond_corr = sim->p.meas_bond_corr;
	const int meas_energy_corr = sim->p.meas_energy_corr;
	const int meas_nematic_corr = sim->p.meas_nematic_corr;

	const hid_t file_id = H5Fopen(sim->file, H5F_ACC_RDWR, H5P_DEFAULT);
	return_if(file_id < 0, -1, "H5Fopen() failed: %ld\n", file_id);

	herr_t status = 0;

	status |= my_write(file_id, "/state/rng", 17, H5T_NATIVE_UINT64, sim->s.rng);
	status |= my_write(file_id, "/state/sweep", 0, H5T_NATIVE_INT, &sim->s.sweep);
	status |= my_write(file_id, "/state/hs", N*L, H5T_NATIVE_INT, sim->s.hs);

	if (sim->m_eq.n_sample > 0) {
		status |= my_write(file_id, "/meas_eqlt/n_sample", 0, H5T_NATIVE_INT, &sim->m_eq.n_sample);
		status |= my_write(file_id, "/meas_eqlt/sign", 0, num_h5t, &sim->m_eq.sign);
#define X(name, size) \
	if ((size) > 0) status |= my_write(file_id, "/meas_eqlt/" #name, size, num_h5t, sim->m_eq.name);
		MEAS_EQLT_LIST
#undef X
	}
	if (sim->p.period_uneqlt > 0 && sim->m_ue.n_sample > 0) {
		status |= my_write(file_id, "/meas_uneqlt/n_sample", 0, H5T_NATIVE_INT, &sim->m_ue.n_sample);
		status |= my_write(file_id, "/meas_uneqlt/sign", 0, num_h5t, &sim->m_ue.sign);
#define X(name, size) \
	if ((size) > 0) status |= my_write(file_id, "/meas_uneqlt/" #name, size, num_h5t, sim->m_ue.name);
		MEAS_UNEQLT_LIST
#undef X
	}

	status |= H5Fclose(file_id);
	return_if(status < 0, -1, "my_write() or H5Fclose() failed: %d\n", status);
	return 0;
}

void RC(sim_data_free)(struct RC(sim_data) *sim)
{
	if (sim != NULL) {
		my_free(sim->pool);
		my_free(sim);
	}
}

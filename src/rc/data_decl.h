struct RC(sim_data);
struct RC(sim_data) *RC(sim_data_read_alloc)(const char *file, const int print_mem_only);
int RC(sim_data_save)(const struct RC(sim_data) *sim);
void RC(sim_data_free)(struct RC(sim_data) *sim);

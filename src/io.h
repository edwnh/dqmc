#ifndef _IO_H
#define _IO_H

#include "util.h"

int read_file(const char *file, struct params *p, struct state *s,
		struct meas_eqlt *m_eq, struct meas_uneqlt *m_ue);

int save_file(const char *file, const struct state *s,
		const struct meas_eqlt *m_eq,
		const struct meas_uneqlt *m_ue);

#endif

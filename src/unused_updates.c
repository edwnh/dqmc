static inline int update_shermor(const int N, const double *const restrict del,
		uint64_t *const restrict rng, int *const restrict site_order,
		int *const restrict hs,
		double *const restrict Gu, double *const restrict Gd,
		double *const restrict cu, double *const restrict du,
		double *const restrict cd, double *const restrict dd)
{
	//~ sign *= update_shermor(N, del, rng, site_order, hs + N*l, Gu, Gd,
		//~ tmpN1u, tmpN2u, tmpN1d, tmpN2d);
	_aa(Gu); _aa(Gd); _aa(cu); _aa(du); _aa(cd); _aa(dd);

	int sign = 1;
	shuffle(rng, N, site_order);
	for (int ii = 0; ii < N; ii++) {
		const int i = site_order[ii];
		const double delu = del[i + N*hs[i]];
		const double deld = del[i + N*!hs[i]];
		const double pu = (1 + (1 - Gu[i + N*i])*delu);
		const double pd = (1 + (1 - Gd[i + N*i])*deld);
		const double prob = pu*pd;
		if (rand_doub(rng) < fabs(prob)) {
			for (int j = 0; j < N; j++) cu[j] = Gu[j + N*i];
			cu[i] -= 1.0;
			for (int j = 0; j < N; j++) du[j] = Gu[i + N*j];
			const double au = delu/pu;
			dger(&N, &N, &au, cu, pconst(1), du, pconst(1), Gu, &N);

			for (int j = 0; j < N; j++) cd[j] = Gd[j + N*i];
			cd[i] -= 1.0;
			for (int j = 0; j < N; j++) dd[j] = Gd[i + N*j];
			const double ad = deld/pd;
			dger(&N, &N, &ad, cd, pconst(1), dd, pconst(1), Gd, &N);

			hs[i] = !hs[i];
			if (prob < 0) sign *= -1;
		}
	}
	return sign;
}

static inline int update_submat(const int N, const double *const restrict del,
		uint64_t *const restrict rng, int *const restrict site_order,
		int *const restrict hs,
		double *const restrict Gu, double *const restrict Gd,
		const int q,
		double *const restrict Gr_u, double *const restrict G_ru,
		double *const restrict DDu, double *const restrict yu,
		double *const restrict xu, double *const restrict Gr_d,
		double *const restrict G_rd, double *const restrict DDd,
		double *const restrict yd, double *const restrict xd)
{
	//~ sign *= update_submat(N, del, rng, site_order, hs + N*l, Gu, Gd, n_delay,
		//~ tmpNN1u, tmpNN2u, tmpN1u, tmpN2u, tmpN3u,
		//~ tmpNN1d, tmpNN2d, tmpN1d, tmpN2d, tmpN3d);
	int *const restrict r = my_calloc(q * sizeof(int)); _aa(r);
	double *const restrict LUu = my_calloc(q*q * sizeof(double)); _aa(LUu);
	double *const restrict LUd = my_calloc(q*q * sizeof(double)); _aa(LUd);
	int sign = 1;
	int k = 0;
	shuffle(rng, N, site_order);
	for (int ii = 0; ii < N; ii++) {
		const int i = site_order[ii];
		const double delu = del[i + N*hs[i]];
		const double deld = del[i + N*!hs[i]];
		double du = Gu[i + N*i] - (1 + delu)/delu;
		double dd = Gd[i + N*i] - (1 + deld)/deld;
		if (k > 0) {
			for (int j = 0; j < k; j++) yu[j] = Gr_u[j + N*i];
			dtrtrs("L", "N", "U", &k, pconst(1), LUu, &q, yu, &k, &(int){0});
			for (int j = 0; j < k; j++) xu[j] = G_ru[i + N*j];
			dtrtrs("U", "T", "N", &k, pconst(1), LUu, &q, xu, &k, &(int){0});
			for (int j = 0; j < k; j++) du -= yu[j]*xu[j];

			for (int j = 0; j < k; j++) yd[j] = Gr_d[j + N*i];
			dtrtrs("L", "N", "U", &k, pconst(1), LUd, &q, yd, &k, &(int){0});
			for (int j = 0; j < k; j++) xd[j] = G_rd[i + N*j];
			dtrtrs("U", "T", "N", &k, pconst(1), LUd, &q, xd, &k, &(int){0});
			for (int j = 0; j < k; j++) dd -= yd[j]*xd[j];
		}

		const double prob = du*delu * dd*deld;
		if (rand_doub(rng) < fabs(prob)) {
			r[k] = i;
			DDu[k] = 1.0 / (1.0 + delu);
			DDd[k] = 1.0 / (1.0 + deld);
			for (int j = 0; j < N; j++) Gr_u[k + N*j] = Gu[i + N*j];
			for (int j = 0; j < N; j++) G_ru[j + N*k] = Gu[j + N*i];
			for (int j = 0; j < N; j++) Gr_d[k + N*j] = Gd[i + N*j];
			for (int j = 0; j < N; j++) G_rd[j + N*k] = Gd[j + N*i];
			for (int j = 0; j < k; j++) LUu[j + q*k] = yu[j];
			for (int j = 0; j < k; j++) LUu[k + q*j] = xu[j];
			for (int j = 0; j < k; j++) LUd[j + q*k] = yd[j];
			for (int j = 0; j < k; j++) LUd[k + q*j] = xd[j];
			LUu[k + q*k] = du;
			LUd[k + q*k] = dd;
			k++;
			hs[i] = !hs[i];
			if (prob < 0) sign *= -1;
		}

		if (k == q || (ii == N - 1 && k > 0)) {
			dtrtrs("L", "N", "U", &k, &N, LUu, &q, Gr_u, &N, &(int){0});
			dtrtrs("U", "N", "N", &k, &N, LUu, &q, Gr_u, &N, &(int){0});
			dgemm("N", "N", &N, &N, &k, pconst(-1.0), G_ru, &N, Gr_u, &N, pconst(1.0), Gu, &N);
			for (int j = 0; j < k; j++) {
				const int rj = r[j];
				const double DDk = DDu[j];
				for (int iii = 0; iii < N; iii++)
					Gu[rj + iii*N] *= DDk;
			}
			dtrtrs("L", "N", "U", &k, &N, LUd, &q, Gr_d, &N, &(int){0});
			dtrtrs("U", "N", "N", &k, &N, LUd, &q, Gr_d, &N, &(int){0});
			dgemm("N", "N", &N, &N, &k, pconst(-1.0), G_rd, &N, Gr_d, &N, pconst(1.0), Gd, &N);
			for (int j = 0; j < k; j++) {
				const int rj = r[j];
				const double DDk = DDd[j];
				for (int iii = 0; iii < N; iii++)
					Gd[rj + iii*N] *= DDk;
			}
			k = 0;
		}
	}
	my_free(LUd);
	my_free(LUu);
	my_free(r);
	return sign;
}

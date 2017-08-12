#pragma once

#include <stdint.h>

// http://xoroshiro.di.unimi.it/xorshift1024star.c
static inline uint64_t rand_uint(uint64_t *rng) {
	const uint64_t s0 = rng[rng[16]];
	const int p = (rng[16] + 1) & 15;
	rng[16] = p;
	uint64_t s1 = rng[p];
	s1 ^= s1 << 31; // a
	rng[p] = s1 ^ s0 ^ (s1 >> 11) ^ (s0 >> 30); // b,c
	return rng[p] * UINT64_C(1181783497276652981);
}

static inline double rand_doub(uint64_t *rng)
{
	const union { uint64_t i; double d; } u = {
		.i = UINT64_C(0x3FF) << 52 | rand_uint(rng) >> 12
	};
	return u.d - 1.0;
}

// fisher-yates
static inline void shuffle(uint64_t *rng, const int n, int *a)
{
	for (int i = 0; i < n; i++) {
		const int j = (rand_uint(rng) >> 3) % (i + 1);
		if (i != j) a[i] = a[j];
		a[j] = i;
	}
}

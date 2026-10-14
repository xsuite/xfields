/*
 *  elegant_rng.h - Elegant-compatible RNG utilities with explicit state.
 *
 *  Overview
 *  --------
 *  Re-implements the random-number utilities used by Elegant/SDDS so that
 *  kernels compiled via xobjects can reproduce the same Touschek Monte Carlo
 *  sequences in Xsuite/xfields. Includes the LAPACK DLARAN core (48-bit LCG),
 *  the Elegant seed bit permutation, and the two streams used by the Touschek
 *  scatter kernel.
 *
 *  Provenance (portions adapted from)
 *  ----------------------------------
 *  - SDDS: mdbmth/drand.c      (random_* streams, randomizeOrder, seeding)
 *  - SDDS: mdbmth/dlaran.c     (C translation of LAPACK's DLARAN, via f2c)
 *  - Elegant: src/drand_oag.c  (random_1_elegant and seed behavior)
 *  - LAPACK: DLARAN            (48-bit LCG RNG core)
 *
 *  Purpose / Exposed API
 *  ---------------------
 *  - LAPACK-compatible DLARAN core (48-bit, 4x12-bit seed)
 *  - Explicit TouschekRNGState storage for stream 1 and stream 4
 *  - TouschekRNGState_seed() for Elegant-compatible seeding
 *  - touschek_random_1_elegant() for the event variates
 *  - touschek_random_4() for the random keys consumed by randomization
 *  - touschek_randomize_order() (qsort + random keys to match Elegant)
 *  - touschek_permute_seed_bit_order(), including permutation inhibition
 *
 *  Usage Notes
 *  -----------
 *  - There are no process-global static RNG streams in this header. The RNG
 *    state is stored in the TouschekRNGState xobject and passed explicitly to
 *    the seeding and scattering kernels.
 *  - The TouschekStudy object owns one RNG state and passes it through the
 *    selected TouschekScattering elements, so the stream continues from one
 *    element to the next.
 *  - Direct TouschekScattering.scatter() calls can receive an explicit state;
 *    if none is supplied, Python creates a temporary TouschekRNGState seeded
 *    from numpy.random.
 *  - MPI seed diversification from Elegant is intentionally omitted here.
 *  - Special behavior for seed 987654321 (seed-permutation inhibition) is
 *    preserved.
 *  - Elegant-compatible seeding consumes one value from stream 1 and stream 4;
 *    TouschekRNGState_seed() reproduces that side effect before storing the
 *    state, which is needed for bitwise agreement with the frozen reference.
 *
 *  References
 *  ----------
 *  - M. Borland, "elegant: A Flexible SDDS-Compliant Code for Accelerator
 *    Simulation," APS LS-287 (2000).
 */
#ifndef ELEGANT_RNG_H
#define ELEGANT_RNG_H

#include "xobjects/headers/common.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* LAPACK DLARAN core. The four state chunks are 12-bit integers and the last
 * chunk must be odd. The state is advanced in place. */
static inline double dlaran_core(int32_t iseed[4]) {
    int32_t it1, it2, it3, it4;
    it4 = iseed[3] * 2549;
    it3 = it4 / 4096;
    it4 -= (it3 << 12);
    it3 = it3 + iseed[2] * 2549 + iseed[3] * 2508;
    it2 = it3 / 4096;
    it3 -= (it2 << 12);
    it2 = it2 + iseed[1] * 2549 + iseed[2] * 2508 + iseed[3] * 322;
    it1 = it2 / 4096;
    it2 -= (it1 << 12);
    it1 = it1 + iseed[0] * 2549 + iseed[1] * 2508
        + iseed[2] * 322 + iseed[3] * 494;
    it1 %= 4096;

    iseed[0] = it1;
    iseed[1] = it2;
    iseed[2] = it3;
    iseed[3] = it4;

    const double twoneg12 = 2.44140625e-4; /* 2^-12 */
    return ((double)it1
            + ((double)it2
               + ((double)it3 + (double)it4 * twoneg12) * twoneg12)
              * twoneg12)
           * twoneg12;
}

static inline uint32_t touschek_permute_seed_bit_order(
        uint32_t input0, short inhibit_permute) {
    if (inhibit_permute) return input0;

    uint32_t new_value = 0u;
    uint32_t offset = input0 % 1000u;
    static const uint32_t bit_mask[32] = {
        0x00000001u, 0x00000002u, 0x00000004u, 0x00000008u,
        0x00000010u, 0x00000020u, 0x00000040u, 0x00000080u,
        0x00000100u, 0x00000200u, 0x00000400u, 0x00000800u,
        0x00001000u, 0x00002000u, 0x00004000u, 0x00008000u,
        0x00010000u, 0x00020000u, 0x00040000u, 0x00080000u,
        0x00100000u, 0x00200000u, 0x00400000u, 0x00800000u,
        0x01000000u, 0x02000000u, 0x04000000u, 0x08000000u,
        0x10000000u, 0x20000000u, 0x40000000u, 0x80000000u
    };

    for (int i = 0; i < 31; i++) {
        if (input0 & bit_mask[i]) new_value |= bit_mask[(i + offset) % 31];
    }

    if (new_value == input0) {
        offset++;
        new_value = 0u;
        for (int i = 0; i < 31; i++) {
            if (input0 & bit_mask[i]) {
                new_value |= bit_mask[(i + offset) % 31];
            }
        }
    }

    return new_value;
}

static inline void touschek_pack_seed(
        int32_t seed[4], long iseed_in, int force_odd_last,
        short inhibit_permute) {
    uint32_t s = (uint32_t)(iseed_in < 0 ? -iseed_in : iseed_in);
    s = touschek_permute_seed_bit_order(s, inhibit_permute);

    seed[3] = (int32_t)(s & 4095u);
    s >>= 12;
    if (force_odd_last) seed[3] = (seed[3] | 1);
    seed[2] = (int32_t)(s & 4095u);
    s >>= 12;
    seed[1] = (int32_t)(s & 4095u);
    s >>= 12;
    seed[0] = (int32_t)(s & 4095u);
}

static inline void touschek_pack_seed_1_elegant(
        int32_t seed[4], long iseed_in, short inhibit_permute) {
    long base = labs(iseed_in);
    base = (long)touschek_permute_seed_bit_order(
        (uint32_t)base, inhibit_permute);
    base = (base / 2) * 2 + 1;

    uint32_t s = (uint32_t)base;
    seed[3] = (int32_t)(s & 4095u);
    s >>= 12;
    seed[2] = (int32_t)(s & 4095u);
    s >>= 12;
    seed[1] = (int32_t)(s & 4095u);
    s >>= 12;
    seed[0] = (int32_t)(s & 4095u);
}

static inline void touschek_rng_state_get_seed_1(
        TouschekRNGStateData rng_state, int32_t seed[4]) {
    seed[0] = (int32_t)TouschekRNGStateData_get_seed_1_0(rng_state);
    seed[1] = (int32_t)TouschekRNGStateData_get_seed_1_1(rng_state);
    seed[2] = (int32_t)TouschekRNGStateData_get_seed_1_2(rng_state);
    seed[3] = (int32_t)TouschekRNGStateData_get_seed_1_3(rng_state);
}

static inline void touschek_rng_state_set_seed_1(
        TouschekRNGStateData rng_state, int32_t seed[4]) {
    TouschekRNGStateData_set_seed_1_0(rng_state, seed[0]);
    TouschekRNGStateData_set_seed_1_1(rng_state, seed[1]);
    TouschekRNGStateData_set_seed_1_2(rng_state, seed[2]);
    TouschekRNGStateData_set_seed_1_3(rng_state, seed[3]);
}

static inline void touschek_rng_state_get_seed_4(
        TouschekRNGStateData rng_state, int32_t seed[4]) {
    seed[0] = (int32_t)TouschekRNGStateData_get_seed_4_0(rng_state);
    seed[1] = (int32_t)TouschekRNGStateData_get_seed_4_1(rng_state);
    seed[2] = (int32_t)TouschekRNGStateData_get_seed_4_2(rng_state);
    seed[3] = (int32_t)TouschekRNGStateData_get_seed_4_3(rng_state);
}

static inline void touschek_rng_state_set_seed_4(
        TouschekRNGStateData rng_state, int32_t seed[4]) {
    TouschekRNGStateData_set_seed_4_0(rng_state, seed[0]);
    TouschekRNGStateData_set_seed_4_1(rng_state, seed[1]);
    TouschekRNGStateData_set_seed_4_2(rng_state, seed[2]);
    TouschekRNGStateData_set_seed_4_3(rng_state, seed[3]);
}

GPUKERN
void TouschekRNGState_seed(TouschekRNGStateData rng_state,
                           int64_t seed,
                           int64_t inhibit_permute) {
    int32_t seed_1[4];
    int32_t seed_4[4];
    long s0 = labs((long)seed);
    long s4 = labs((long)seed + 6);
    short inhibit = (s0 == 987654321)
        ? 1
        : (short)(inhibit_permute ? 1 : 0);

    TouschekRNGStateData_set_inhibit_permute(rng_state, inhibit);
    touschek_pack_seed_1_elegant(seed_1, s0, inhibit);
    touschek_pack_seed(seed_4, s4, 1, inhibit);
    dlaran_core(seed_1);
    dlaran_core(seed_4);
    touschek_rng_state_set_seed_1(rng_state, seed_1);
    touschek_rng_state_set_seed_4(rng_state, seed_4);
}

static inline double touschek_random_1_elegant(
        TouschekRNGStateData rng_state) {
    int32_t seed[4];
    double value;

    touschek_rng_state_get_seed_1(rng_state, seed);
    value = dlaran_core(seed);
    touschek_rng_state_set_seed_1(rng_state, seed);

    return value;
}

static inline double touschek_random_4(TouschekRNGStateData rng_state) {
    int32_t seed[4];
    double value;

    touschek_rng_state_get_seed_4(rng_state, seed);
    value = dlaran_core(seed);
    touschek_rng_state_set_seed_4(rng_state, seed);

    return value;
}

typedef struct RANDOMIZATION_HOLDER_ {
    void* buffer;
    double random_value;
} RANDOMIZATION_HOLDER;

static int randomizeOrderCmp(const void *p1, const void *p2) {
    const RANDOMIZATION_HOLDER *rh1 = (const RANDOMIZATION_HOLDER *)p1;
    const RANDOMIZATION_HOLDER *rh2 = (const RANDOMIZATION_HOLDER *)p2;
    if (rh1->random_value > rh2->random_value) return 1;
    if (rh1->random_value < rh2->random_value) return -1;
    return 0;
}

static long touschek_randomize_order(
        char *ptr, long size, long length, TouschekRNGStateData rng_state) {
    if (!ptr || size <= 0) return 0;
    if (length < 2) return 1;

    RANDOMIZATION_HOLDER *rh =
        (RANDOMIZATION_HOLDER*)malloc(sizeof(*rh) * (size_t)length);
    if (!rh) return 0;

    for (long i = 0; i < length; i++) {
        rh[i].buffer = malloc((size_t)size);
        if (!rh[i].buffer) {
            for (long k = 0; k < i; k++) free(rh[k].buffer);
            free(rh);
            return 0;
        }
        memcpy(rh[i].buffer, ptr + i * size, (size_t)size);
        rh[i].random_value = touschek_random_4(rng_state);
    }

    qsort((void*)rh, (size_t)length, sizeof(*rh), randomizeOrderCmp);

    for (long i = 0; i < length; i++) {
        memcpy(ptr + i * size, rh[i].buffer, (size_t)size);
        free(rh[i].buffer);
    }
    free(rh);
    return 1;
}

#endif /* ELEGANT_RNG_H */

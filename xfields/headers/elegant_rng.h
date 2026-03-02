/*************************************************************************\
* Portions adapted from Elegant and the SDDS Toolkit.
* Copyright (c) 2002 The University of Chicago.
* Copyright (c) 2002 The Regents of the University of California.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution.
\*************************************************************************/
/*
 *  elegant_rng.h — Elegant-compatible RNG utilities (C99)
 *
 *  Overview
 *  --------
 *  Re-implements the random-number utilities used by Elegant/SDDS so that
 *  kernels compiled via xobjects can reproduce (where applicable) the same
 *  sequences in Xsuite/xfields. Includes the LAPACK DLARAN core (48-bit LCG)
 *  and the streams random_1..random_6 with Elegant-compatible seeding rules.
 *  Provides random_1_elegant, randomizeOrder, and related helpers.
 *
 *  Provenance (portions adapted from)
 *  ----------------------------------
 *  - SDDS: mdbmth/drand.c            (random_* streams, randomizeOrder, seeding)
 *  - SDDS: mdbmth/dlaran.c           (C translation of LAPACK's DLARAN, via f2c)
 *  - Elegant: src/drand_oag.c        (random_1_elegant and seed behavior)
 *  - LAPACK: DLARAN (48-bit LCG RNG core)
 *
 *  Licenses & Notices
 *  ------------------
 *  - Upstream notice preserved above (Elegant/SDDS).
 *  - License texts:
 *      xfields/third_party/elegant/LICENSE
 *      xfields/third_party/SDDS/LICENSE
 *      xfields/third_party/lapack/LICENSE
 *  - This file is a derivative work.
 *    Modifications © 2025 Giacomo Broggi / CERN.
 *
 *  Purpose / Exposed API
 *  ---------------------
 *  - LAPACK-compatible DLARAN core (48-bit, 4×12-bit seed)
 *  - Streams: random_1 .. random_6 (Elegant-style seeding conventions)
 *  - random_1_elegant() and seedElegantRandomNumbers() (for Elegant compatibility)
 *  - permuteSeedBitOrder(), inhibitRandomSeedPermutation()
 *  - randomizeOrder() (qsort + random keys to match Elegant’s consumption)
 *
 *  Usage Notes
 *  -----------
 *  - Single-threaded, static state; synchronize if used from multiple threads.
 *  - Calls with negative seeds reinitialize the stream; non-negative seeds consume.
 *  - MPI seed diversification is intentionally omitted here.
 *  - Special behavior for seed 987654321 (permute inhibition) is preserved.
 *  - Goal: reproduce Elegant sequences (bitwise where possible).
 *
 *  References
 *  ----------
 *  - M. Borland, “elegant: A Flexible SDDS-Compliant Code for Accelerator Simulation,”
 *    APS LS-287 (2000).
 */
#ifndef ELEGANT_RNG_H
#define ELEGANT_RNG_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// -----------------------------------------------------------------------------
// LAPACK-style DLARAN core
// ------------------------
// This is the 48-bit multiplicative LCG used by DLARAN, with the 48-bit state
// stored in 4 integers of 12 bits each. The last chunk (iseed[3]) must be odd.
// Returns a uniform double in (0,1). The recurrence and normalization constants
// reproduce LAPACK DLARAN bit-for-bit.
// -----------------------------------------------------------------------------
static inline double dlaran_core(int32_t iseed[4]) {
    // Seed chunks: iseed[0..3], iseed[3] must be odd
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
    it1 = it1 + iseed[0] * 2549 + iseed[1] * 2508 + iseed[2] * 322 + iseed[3] * 494;
    it1 %= 4096;

    iseed[0] = it1;
    iseed[1] = it2;
    iseed[2] = it3;
    iseed[3] = it4;

    const double twoneg12 = 2.44140625e-4; // 2^-12
    return ( (double)it1 +
            ( (double)it2 + ( (double)it3 + (double)it4 * twoneg12 ) * twoneg12 ) * twoneg12
           ) * twoneg12; // in (0,1)
}

/* -----------------------------------------------------------------------------
   Seed permutation control (Elegant-compatible)
   --------------------------------------------
   Elegant permutes the bit order of integer seeds before packing 12-bit chunks.
   This helps decorrelate “close” seeds. A global flag can inhibit this step.

   - inhibitRandomSeedPermutation(state>=0) sets the global flag.
   - permuteSeedBitOrder(x) permutes bit positions unless inhibited.
   - Elegant disables permutation when random_number_seed == 987654321.
----------------------------------------------------------------------------- */
static short g_inhibitPermute = 0;
static inline short inhibitRandomSeedPermutation(short state){
    if (state >= 0) g_inhibitPermute = state;
    return g_inhibitPermute;
}

static inline uint32_t permuteSeedBitOrder(uint32_t input0){
    if (g_inhibitPermute) return input0;
    uint32_t input = input0;
    uint32_t newValue = 0u;
    uint32_t offset = input0 % 1000u;
    static const uint32_t bitMask[32] = {
        0x00000001u,0x00000002u,0x00000004u,0x00000008u,
        0x00000010u,0x00000020u,0x00000040u,0x00000080u,
        0x00000100u,0x00000200u,0x00000400u,0x00000800u,
        0x00001000u,0x00002000u,0x00004000u,0x00008000u,
        0x00010000u,0x00020000u,0x00040000u,0x00080000u,
        0x00100000u,0x00200000u,0x00400000u,0x00800000u,
        0x01000000u,0x02000000u,0x04000000u,0x08000000u,
        0x10000000u,0x20000000u,0x40000000u,0x80000000u
    };
    for (int i=0;i<31;i++)
        if (input & bitMask[i]) newValue |= bitMask[(i + offset) % 31];
    if (newValue == input){
        offset++;
        newValue = 0u;
        for (int i=0;i<31;i++)
            if (input & bitMask[i]) newValue |= bitMask[(i + offset) % 31];
    }
    return newValue;
}

/* -----------------------------------------------------------------------------
   Packing helper: split a 32-bit integer into 4×12-bit chunks (DLARAN format).
   The last chunk must be odd for DLARAN to work correctly (Elegant behavior).
----------------------------------------------------------------------------- */
static inline void seed_from_long(int32_t seed[4], long iseed_in, int force_odd_last){
    uint32_t s = (uint32_t)(iseed_in < 0 ? -iseed_in : iseed_in);
    s = permuteSeedBitOrder(s);
    // pack into 4x12-bit chunks, last must be odd
    seed[3] = (int32_t)(s & 4095u); s >>= 12;
    if (force_odd_last) seed[3] = (seed[3] | 1); // ensure odd
    seed[2] = (int32_t)(s & 4095u); s >>= 12;
    seed[1] = (int32_t)(s & 4095u); s >>= 12;
    seed[0] = (int32_t)(s & 4095u);
}

/* -----------------------------------------------------------------------------
   RNG streams random_1 .. random_6
   --------------------------------
   These reproduce the SDDS/Elegant API and seeding semantics:

   - Calling with a *negative* iseed re-initializes that stream from |iseed|.
   - Calling with a non-negative iseed consumes the next variate.
   - random_1 (on (re)seed) also re-seeds streams 2..6 using |base|+{2,4,6,8,10}.
   - All streams use the same DLARAN core, independent 48-bit states.

   Important:
   * This file intentionally omits MPI diversification (modes 1..4 in Elegant).
   * Keep static state: not thread-safe by design (matches Elegant).
----------------------------------------------------------------------------- */
double random_2(long iseed);
double random_3(long iseed);
double random_4(long iseed);
double random_5(long iseed);
double random_6(long iseed);

double random_1(long iseed){
    static int initialized = 0;
    static int32_t seed[4] = {0,0,0,0};
    if (!initialized || iseed < 0){
        long base = (iseed < 0) ? -iseed : iseed;        // abs
        base = (long)permuteSeedBitOrder((uint32_t)base); // permute
        // SDDS-like reseed
        random_2(-(base + 2));
        random_3(-(base + 4));
        random_4(-(base + 6));
        random_5(-(base + 8));
        random_6(-(base + 10));
        // force odd and split in 4×12 bit
        base = (base/2)*2 + 1;
        uint32_t s = (uint32_t)base;
        seed[3] = (int32_t)(s & 4095u); s >>= 12;
        seed[2] = (int32_t)(s & 4095u); s >>= 12;
        seed[1] = (int32_t)(s & 4095u); s >>= 12;
        seed[0] = (int32_t)(s & 4095u);
        initialized = 1;
    }
    return dlaran_core(seed);
}

double random_2(long iseed){
    static int initialized = 0;
    static int32_t seed[4] = {0,0,0,0};
    if (!initialized || iseed < 0){
        if (iseed >= 0) iseed = -1;
        seed_from_long(seed, -iseed, /*force_odd_last=*/1);
        initialized = 1;
    }
    return dlaran_core(seed);
}
double random_3(long iseed){
    static int initialized = 0;
    static int32_t seed[4] = {0,0,0,0};
    if (!initialized || iseed < 0){
        if (iseed >= 0) iseed = -1;
        seed_from_long(seed, -iseed, /*force_odd_last=*/1);
        initialized = 1;
    }
    return dlaran_core(seed);
}
double random_4(long iseed){
    static int initialized = 0;
    static int32_t seed[4] = {0,0,0,0};
    if (!initialized || iseed < 0){
        if (iseed >= 0) iseed = -1;
        seed_from_long(seed, -iseed, /*force_odd_last=*/1);
        initialized = 1;
    }
    return dlaran_core(seed);
}
double random_5(long iseed){
    static int initialized = 0;
    static int32_t seed[4] = {0,0,0,0};
    if (!initialized || iseed < 0){
        if (iseed >= 0) iseed = -1;
        seed_from_long(seed, -iseed, /*force_odd_last=*/1);
        initialized = 1;
    }
    return dlaran_core(seed);
}
double random_6(long iseed){
    static int initialized = 0;
    static int32_t seed[4] = {0,0,0,0};
    if (!initialized || iseed < 0){
        if (iseed >= 0) iseed = -1;
        seed_from_long(seed, -iseed, /*force_odd_last=*/1);
        initialized = 1;
    }
    return dlaran_core(seed);
}

static inline double random_1_elegant(long iseed) {
    static int initialized = 0;
    static int32_t seed[4] = {0,0,0,0};
    if (!initialized || iseed < 0){
        long base = (iseed < 0) ? -iseed : iseed;          // abs
        // Respect current inhibit flag (seedElegantRandomNumbers sets it)
        base = (long)permuteSeedBitOrder((uint32_t)base);  // no-op if inhibited
        base = (base/2)*2 + 1;                             // force odd
        uint32_t s = (uint32_t)base;
        seed[3] = (int32_t)(s & 4095u); s >>= 12;
        seed[2] = (int32_t)(s & 4095u); s >>= 12;
        seed[1] = (int32_t)(s & 4095u); s >>= 12;
        seed[0] = (int32_t)(s & 4095u);
        initialized = 1;
    }
    return dlaran_core(seed);
}

/* -----------------------------------------------------------------------------
   randomizeOrder
   ------------------------------
   Elegant shuffles arrays by:
     1) creating an array of (buffer copy, random key) pairs,
     2) sorting by the random key (qsort),
     3) copying buffers back in sorted order.

   This consumes RNG like Elegant does and matches its order exactly.
   It allocates O(N*size) memory.
----------------------------------------------------------------------------- */
typedef struct RANDOMIZATION_HOLDER_ {
    void*   buffer;
    double  randomValue;
} RANDOMIZATION_HOLDER;

static int randomizeOrderCmp(const void *p1, const void *p2) {
    const RANDOMIZATION_HOLDER *rh1 = (const RANDOMIZATION_HOLDER *)p1;
    const RANDOMIZATION_HOLDER *rh2 = (const RANDOMIZATION_HOLDER *)p2;
    if (rh1->randomValue > rh2->randomValue) return  1;
    if (rh1->randomValue < rh2->randomValue) return -1;
    return 0;
}

static long randomizeOrder(char *ptr, long size, long length,
                           long iseed, double (*urandom)(long iseed1)) {
    if (!ptr || size<=0 || !urandom) return 0;
    if (length < 2) return 1;
    if (iseed < 0) urandom(iseed);

    RANDOMIZATION_HOLDER *rh =
        (RANDOMIZATION_HOLDER*)malloc(sizeof(*rh) * (size_t)length);
    if (!rh) return 0;

    for (long i=0; i<length; i++) {
        rh[i].buffer = malloc((size_t)size);
        if (!rh[i].buffer) {
            for (long k=0; k<i; k++) free(rh[k].buffer);
            free(rh);
            return 0;
        }
        memcpy(rh[i].buffer, ptr + i*size, (size_t)size);
        rh[i].randomValue = urandom(0);
    }

    qsort((void*)rh, (size_t)length, sizeof(*rh), randomizeOrderCmp);

    for (long i=0; i<length; i++) {
        memcpy(ptr + i*size, rh[i].buffer, (size_t)size);
        free(rh[i].buffer);
    }
    free(rh);
    return 1;
}

/* -----------------------------------------------------------------------------
   seedElegantRandomNumbers
   ------------------------
   Drop-in replacement for Elegant’s global RNG seeding (without MPI modes):
     - random_1 gets -|seed|
     - random_2 gets -|seed+2|
     - random_3 gets -|seed+4|
     - random_4 gets -|seed+6|
   Also applies Elegant’s “inhibit permutation” convention:
     if seed == 987654321 -> inhibit permutation globally.

   Call this exactly once (per process) before any RNG use to mimic Elegant’s
   &run_setup random_number_seed behavior.
----------------------------------------------------------------------------- */
static inline void seedElegantRandomNumbers(long seed, short inhibit_permute){
    long s0 = labs(seed), s1 = labs(seed + 2), s2 = labs(seed + 4), s3 = labs(seed + 6);

    if (s0 == 987654321) inhibitRandomSeedPermutation(1);
    else                 inhibitRandomSeedPermutation(inhibit_permute ? 1 : 0);

    random_1_elegant(-s0);
    random_2(-s1);
    random_3(-s2);
    random_4(-s3);
}

#endif // ELEGANT_RNG_H
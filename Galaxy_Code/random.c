#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "constants.h"
#include "structs.h"
#include "random.h"

/* --------------------------------------------------------------------------
   Thread-safe RNG, OpenMP-aware

   We keep one RNG state per OpenMP thread, plus a separate Box¨CMuller
   "spare" value per thread for random_gaussian.

   This replaces the old version that used rand() and static globals, which
   were not safe under OpenMP.
   -------------------------------------------------------------------------- */

static int           rng_initialized = 0;
static int           rng_nthreads    = 1;
static unsigned int *rng_seeds       = NULL;
static int          *rng_has_spare   = NULL;
static double       *rng_spare       = NULL;

/* Simple LCG step: returns next 32-bit value for a given thread. */
static unsigned int rng_next(int tid)
{
    /* Numerical Recipes LCG parameters */
    rng_seeds[tid] = 1664525u * rng_seeds[tid] + 1013904223u;
    return rng_seeds[tid];
}

static void rng_init(void)
{
    if (rng_initialized) return;

    /* Double-checked locking so that only one thread does the malloc. */
#ifdef _OPENMP
#pragma omp critical(random_init)
#endif
    {
        if (rng_initialized) return;

#ifdef _OPENMP
        rng_nthreads = omp_get_max_threads();
#else
        rng_nthreads = 1;
#endif

        rng_seeds     = (unsigned int*)malloc((size_t)rng_nthreads * sizeof(unsigned int));
        rng_has_spare = (int*)         malloc((size_t)rng_nthreads * sizeof(int));
        rng_spare     = (double*)      malloc((size_t)rng_nthreads * sizeof(double));

        if (!rng_seeds || !rng_has_spare || !rng_spare) {
            fprintf(stderr, "random.c: failed to allocate RNG state\n");
            exit(EXIT_FAILURE);
        }

        /* Deterministic initialization (fixed seed) for reproducibility. */
        for (int i = 0; i < rng_nthreads; ++i) {
            rng_seeds[i]     = 123456789u + 362437u * (unsigned int)i;
            rng_has_spare[i] = 0;
            rng_spare[i]     = 0.0;
        }

        rng_initialized = 1;
    }
}

static int rng_thread_id(void)
{
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
    if (tid >= rng_nthreads) {
        /* Should not normally happen, but clamp just in case. */
        tid = rng_nthreads - 1;
    }
#endif
    return tid;
}

/* Uniform in [min, max]. */
double random_uniform(double min, double max)
{
    rng_init();
    int tid = rng_thread_id();
    unsigned int x = rng_next(tid);

    /* Map 32-bit integer to [0,1]. */
    double u = (double)x / (double)UINT_MAX;

    return min + (max - min) * u;
}

/* Gaussian with mean and sigma using Box¨CMuller, per-thread spare. */
double random_gaussian(double mean, double sigma)
{
    rng_init();
    int tid = rng_thread_id();

    if (rng_has_spare[tid]) {
        rng_has_spare[tid] = 0;
        return mean + sigma * rng_spare[tid];
    }

    double u, v, s;
    do {
        u = random_uniform(-1.0, 1.0);
        v = random_uniform(-1.0, 1.0);
        s = u*u + v*v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);

    rng_spare[tid]     = v * s;
    rng_has_spare[tid] = 1;

    return mean + sigma * u * s;
}

/* Sample radius from a Plummer profile with scale length a. */
double sample_plummer_radius(double a)
{
    double u    = random_uniform(0.0, 1.0);
    double term = pow(u, -2.0/3.0) - 1.0;
    if (term <= 0.0) term = 1e-6;
    return a / sqrt(term);
}

/* Random point on a sphere of radius r. */
Vector3 random_point_on_sphere(double r)
{
    Vector3 pos;

    double theta   = random_uniform(0.0, 2.0 * PI);
    double cos_phi = random_uniform(-1.0, 1.0);
    double sin_phi = sqrt(1.0 - cos_phi * cos_phi);

    pos.x = r * sin_phi * cos(theta);
    pos.y = r * sin_phi * sin(theta);
    pos.z = r * cos_phi;

    return pos;
}

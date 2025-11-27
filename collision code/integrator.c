#include <math.h>
#include <stddef.h>
#include "constants.h"   // for L
#include "structs.h"
#include "integrator.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* Map x to [0, L) with periodic boundaries. */
static inline double wrap_periodic(double x)
{
    if (!isfinite(x)) return x;   // if NaN/Inf, leave it (we can detect later)

    x = fmod(x, L);
    if (x < 0.0) x += L;
    return x;
}

void leapfrog_step(ParticleSystem *sys,
                   double          dt,
                   ForceFunc       force,
                   void           *ctx)
{
    if (!sys || !force) return;
    if (sys->N <= 0)    return;
    if (dt == 0.0)      return;   // nothing to do, avoids useless FFT calls

    const int N = sys->N;
    const double half_dt = 0.5 * dt;

    /* --------- half kick: v^{n+1/2} = v^n + 0.5 dt a^n --------- */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; ++i) {
        sys->velocities[i].x += half_dt * sys->accelerations[i].x;
        sys->velocities[i].y += half_dt * sys->accelerations[i].y;
        sys->velocities[i].z += half_dt * sys->accelerations[i].z;
    }

    /* --------- drift: x^{n+1} = x^n + dt v^{n+1/2} --------- */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; ++i) {
        sys->positions[i].x += dt * sys->velocities[i].x;
        sys->positions[i].y += dt * sys->velocities[i].y;
        sys->positions[i].z += dt * sys->velocities[i].z;

        /* periodic BCs */
        sys->positions[i].x = wrap_periodic(sys->positions[i].x);
        sys->positions[i].y = wrap_periodic(sys->positions[i].y);
        sys->positions[i].z = wrap_periodic(sys->positions[i].z);
    }

    /* --------- recompute accelerations at new positions --------- */
    /* Important: be careful if your FFT code also uses OpenMP.
     * If you see oversubscription, set OMP_NESTED=FALSE or disable
     * parallel regions inside force() when called from here.
     */
    force(sys, ctx);   // fills sys->accelerations with a^{n+1}

    /* --------- half kick: v^{n+1} = v^{n+1/2} + 0.5 dt a^{n+1} --------- */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; ++i) {
        sys->velocities[i].x += half_dt * sys->accelerations[i].x;
        sys->velocities[i].y += half_dt * sys->accelerations[i].y;
        sys->velocities[i].z += half_dt * sys->accelerations[i].z;
    }
}

/*
 * 4th-order symplectic integrator via Yoshida composition of leapfrog.
 */
void symplectic4_step(ParticleSystem *sys,
                      double          dt,
                      ForceFunc       force,
                      void           *ctx)
{
    if (!sys || !force) return;
    if (sys->N <= 0)    return;
    if (dt == 0.0)      return;

    /* Yoshida coefficients */
    const double r  = cbrt(2.0);   // 2^{1/3}
    const double w1 = 1.0 / (2.0 - r);
    const double w0 = -r * w1;

    const double c1 = 0.5 * w1;
    const double c2 = 0.5 * (w0 + w1);
    const double c3 = c2;
    const double c4 = c1;

    leapfrog_step(sys, c1 * dt, force, ctx);
    leapfrog_step(sys, c2 * dt, force, ctx);
    leapfrog_step(sys, c3 * dt, force, ctx);
    leapfrog_step(sys, c4 * dt, force, ctx);
}

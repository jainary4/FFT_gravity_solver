#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "structs.h"   // for ParticleSystem, Vector3

/* 
 * A callback that recomputes accelerations for all particles.
 * 
 * On entry:
 *   - sys->positions and sys->velocities contain the current state.
 * On exit:
 *   - sys->accelerations must contain a(q) for each particle.
 *
 * 'ctx' is an opaque pointer you can use to pass in mesh / FFT / etc.
 * For example, ctx might wrap:
 *   - padded rho grid
 *   - FFT plans
 *   - force_x, force_y, force_z grids
 */
typedef void (*ForceFunc)(ParticleSystem *sys, void *ctx);

/*
 * Perform one 2nd-order symplectic **leapfrog** (velocity-Verlet) step:
 * 
 *   v^{n+1/2} = v^n   + 0.5 dt a^n
 *   x^{n+1}   = x^n   + dt v^{n+1/2}
 *   (recompute a^{n+1} from x^{n+1} via force callback)
 *   v^{n+1}   = v^{n+1/2} + 0.5 dt a^{n+1}
 *
 * Requirements:
 *   - On first call, sys->accelerations must already contain a^n
 *     (so call your FFT gravity once before the first step).
 *   - Positions are wrapped into [0, L) with periodic BCs.
 */
void leapfrog_step(ParticleSystem *sys,
                   double          dt,
                   ForceFunc       force,
                   void           *ctx);

/*
 * 4th-order symplectic integrator via Yoshida composition of leapfrog.
 *
 * It calls leapfrog_step four times with scaled substeps that sum to dt.
 * This is time-reversible and 4th-order accurate for Hamiltonians of the 
 * form H(p,q) = T(p) + V(q) (your gravity-only N-body fits this).
 *
 * Same requirements as leapfrog_step regarding sys and force().
 */
void symplectic4_step(ParticleSystem *sys,
                      double          dt,
                      ForceFunc       force,
                      void           *ctx);

#endif /* INTEGRATOR_H */

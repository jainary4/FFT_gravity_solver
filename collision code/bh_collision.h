#ifndef BH_COLLISION_H
#define BH_COLLISION_H

#include "structs.h"

/*
 * Simple BH collision / accretion model.
 *
 * We treat:
 *   - TYPE_BH  particles as black holes
 *   - TYPE_STAR particles as ※swallowable§ (stars / clusters)
 *
 * Collisions:
 *   - BH每star:   star inside BH_STAR_COLLISION_RADIUS is swallowed.
 *   - BH每BH:     BHs within BH_BH_COLLISION_RADIUS are merged.
 *
 * Radii are in code units (kpc);
 */
#ifndef BH_STAR_COLLISION_RADIUS
#define BH_STAR_COLLISION_RADIUS   0.5   /* BH每star swallow radius (kpc) */
#endif

#ifndef BH_BH_COLLISION_RADIUS
#define BH_BH_COLLISION_RADIUS     0.2   /* BH每BH merge radius (kpc) */
#endif

#ifndef BH_MAX_COLLISIONS_PER_STEP
#define BH_MAX_COLLISIONS_PER_STEP 1000000  /* safety cap */
#endif

/*
 * One collision step:
 *   - sys:        particle system (positions, velocities, masses, types, N)
 *   - bh_indices: array of BH indices of length *n_bh
 *   - n_bh:       in/out: number of BHs (may decrease if BHs merge)
 *
 * This function:
 *   - modifies sys->N, positions, velocities, masses, types
 *   - keeps bh_indices consistent when particles are swapped/removed
 */
void bh_collision_step(ParticleSystem *sys,
                       int *bh_indices,
                       int *n_bh);

#endif

#ifndef BH_COLLISION_H
#define BH_COLLISION_H

#include "structs.h"
#include "constants.h"

/* Match initialise.c:
 *   0 = stars
 *   1 = dark matter
 *   2 = black holes
 */
#define TYPE_STAR 0
#define TYPE_DM   1
#define TYPE_BH   2

/* Fixed collision radii in code units (kpc). */
#define BH_STAR_COLLISION_RADIUS  (1.0f * (L / (float)NMESH))
#define BH_BH_COLLISION_RADIUS    (1.5f * (L / (float)NMESH))

/*
 * Perform one black hole collision / merge step.
 *
 * Inputs:
 *   sys        : particle system
 *   bh_indices : array of indices into sys->[...] that are BHs
 *   n_bh       : in/out, number of active BHs in bh_indices
 *
 * Behaviour:
 *   - Uses NMESH^3 grid geometry + a cell-linked list to find neighbours.
 *   - Stars within BH_STAR_COLLISION_RADIUS of a BH are swallowed.
 *   - BHs within BH_BH_COLLISION_RADIUS of each other are merged.
 *   - Dark matter never collides.
 *   - Removed particles are actually removed from the ParticleSystem by
 *     swapping with the last active particle and decrementing sys->N.
 *   - bh_indices[] and *n_bh are kept consistent with these removals.
 */
void bh_collision_step(ParticleSystem *sys,
                       int           *bh_indices,
                       int           *n_bh);

#endif /* BH_COLLISION_H */

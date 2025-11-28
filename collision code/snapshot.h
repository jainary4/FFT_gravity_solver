#ifndef SNAPSHOT_H
#define SNAPSHOT_H

#include "structs.h"

/*
 * Side-on projection (x¨Cz plane), but we keep the old function names
 * for compatibility with main.c.
 *
 * write_xy_density_pgm:
 *   - grayscale density map in the (x,z) plane
 *
 * write_xy_density_ppm_color:
 *   - color (P6) density map in the (x,z) plane
 *   - stars:      white
 *   - dark matter: purple
 *   - black holes: bright red, overpainted as larger dots
 *   - background: black
 */

/* Grayscale projection onto x¨Cz plane */
int write_xy_density_pgm(const ParticleSystem *sys,
                         int imgN,
                         const char *filename);

/* Color projection onto x¨Cz plane */
int write_xy_density_ppm_color(const ParticleSystem *sys,
                               const int *bh_indices,
                               int n_bh,
                               int imgN,
                               const char *filename);

#endif /* SNAPSHOT_H */

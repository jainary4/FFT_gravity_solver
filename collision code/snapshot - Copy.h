#ifndef SNAPSHOT_H
#define SNAPSHOT_H

#include "structs.h"

/* Grayscale projection (old version; still available if you want it) */
int write_xy_density_pgm(const ParticleSystem *sys,
                         int imgN,
                         const char *filename);

/*
 * Color projection onto x¨Cy plane:
 *   - ¡°normal¡± particles (non-BH) ¡ú white
 *   - BH particles (by index list) ¡ú red
 *   - background ¡ú black
 *
 * DM vs stars are not distinguished here (we¡¯d need your type
 * conventions for that), but BHs are highlighted red.
 */
int write_xy_density_ppm_color(const ParticleSystem *sys,
                               const int *bh_indices,
                               int n_bh,
                               int imgN,
                               const char *filename);

#endif /* SNAPSHOT_H */

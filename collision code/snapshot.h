#ifndef SNAPSHOT_H
#define SNAPSHOT_H

#include "structs.h"

/* Write x¨Cy projected mass density to a PGM image. */
int write_xy_density_pgm(const ParticleSystem *sys, int imgN, const char *filename);

#endif /* SNAPSHOT_H */

#ifndef MESH_H
#define MESH_H

#include "structs.h"

// Mesh creation/destruction
ParticleMesh* create_particle_mesh(void);
void destroy_particle_mesh(ParticleMesh *pm);

// Grid operations
double*** allocate_3d_array(int n);
void free_3d_array(double ***array, int n);

// Mass assignment into a zero-padded grid (no periodic wrapping).
// rho_pad: padded array of size Np x Np x Np
// Np: padded grid size (NMESH_PADDED)
// sys: particle system
// N: original grid size (NMESH)
// h: cell size
// offset: starting index in padded array where original grid block is placed (usually 0 or (Np-N)/2)
void assign_mass_cic_padded(double ***rho_pad, int Np, ParticleSystem *sys, int N, double h, int offset);

// CIC gather: interpolate grid forces to particle positions and compute accelerations.
// force_x, force_y, force_z: force grids (padded, size Np x Np x Np)
// Np: padded grid size
// sys: particle system (positions and masses)
// N: original grid size (NMESH)
// h: cell size
// offset: starting index in padded array where original grid block is placed
// Output: particle accelerations stored in sys->accelerations[] (must be pre-allocated)
void gather_forces_to_particles(
    double ***force_x, double ***force_y, double ***force_z,
    int Np, ParticleSystem *sys, int N, double h, int offset
);

#endif
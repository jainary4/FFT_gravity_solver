#ifndef MESH_H
#define MESH_H

#include "structs.h"

// Mesh creation/destruction
ParticleMesh* create_particle_mesh(void);
void destroy_particle_mesh(ParticleMesh *pm);

// Grid operations
float*** allocate_3d_array(int n);
void free_3d_array(float ***array, int n);

// Mass assignment
void assign_mass_cic(ParticleMesh *pm, ParticleSystem *sys);

#endif
#ifndef PARTICLES_H
#define PARTICLES_H

#include "structs.h"

// Initialization functions
void initialize_plummer_positions(Vector3 *positions, int N, double a, Vector3 center);
void initialize_velocities(Vector3 *velocities, int N, double sigma);

// System operations
void destroy_particle_system(ParticleSystem *sys);
ParticleSystem* initialize_particle_system(void);

// Utilities
void remove_bulk_motion(ParticleSystem *sys);
void center_in_box(ParticleSystem *sys, Vector3 box_center);

// Helpers
// Return pointer to the first black-hole position (array of length N_BH)
Vector3* get_blackhole_positions(ParticleSystem *sys);


#endif
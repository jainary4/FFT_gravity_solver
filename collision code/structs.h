
#ifndef STRUCTS_H
#define STRUCTS_H

typedef struct {
    float x, y, z;
} Vector3;

typedef struct {
    Vector3 *positions;
    Vector3 *velocities;
    float *masses;
    int   *types; 
    int N;
} ParticleSystem;

typedef struct {
    float ***rho; // 3-d vector 
    int N;
    float cell_size;
    float box_size;
} ParticleMesh;

#endif
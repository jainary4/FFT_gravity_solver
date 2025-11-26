
#ifndef STRUCTS_H
#define STRUCTS_H

typedef struct {
    double x, y, z;
} Vector3;

typedef struct {
    Vector3 *positions;
    Vector3 *velocities;
    Vector3 *accelerations;
    double *masses;
    int   *types; 
    int N;
} ParticleSystem;

typedef struct {
    double ***rho; // 3-d vector 
    int N;
    double cell_size;
    double box_size;
} ParticleMesh;



#endif
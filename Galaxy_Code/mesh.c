#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "constants.h"
#include "structs.h"
#include "mesh.h"

float*** allocate_3d_array(int n) {
    float *data = (float*)calloc(n * n * n, sizeof(float));
    float ***array = (float***)malloc(n * sizeof(float**));
    
    for (int i = 0; i < n; i++) {
        array[i] = (float**)malloc(n * sizeof(float*));
        for (int j = 0; j < n; j++) {
            array[i][j] = &data[i * n * n + j * n];
        }
    }
    
    return array;
}

void free_3d_array(float ***array, int n) {
    if (!array) return;
    free(array[0][0]);
    for (int i = 0; i < n; i++) {
        free(array[i]);
    }
    free(array);
}

ParticleMesh* create_particle_mesh() {
    ParticleMesh *pm = (ParticleMesh*)malloc(sizeof(ParticleMesh));
    pm->N = NMESH;
    pm->cell_size = L / NMESH;
    pm->box_size = L;
    pm->rho = allocate_3d_array(NMESH);
    return pm;
}

void destroy_particle_mesh(ParticleMesh *pm) {
    free_3d_array(pm->rho, pm->N);
    free(pm);
}

void assign_mass_cic(ParticleMesh *pm, ParticleSystem *sys) {
    double t_start = omp_get_wtime();
    
    float h = pm->cell_size;
    float h3 = h * h * h;
    int N = pm->N;
    
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                pm->rho[i][j][k] = 0.0f;
            }
        }
    }
    
    #pragma omp parallel for
    for (int p = 0; p < sys->N; p++) {
        float x_grid = sys->positions[p].x / h;
        float y_grid = sys->positions[p].y / h;
        float z_grid = sys->positions[p].z / h;
        
        int i0 = (int)floorf(x_grid);
        int j0 = (int)floorf(y_grid);
        int k0 = (int)floorf(z_grid);
        
        float dx = x_grid - i0;
        float dy = y_grid - j0;
        float dz = z_grid - k0;
        
        float wx0 = 1.0f - dx;
        float wx1 = dx;
        float wy0 = 1.0f - dy;
        float wy1 = dy;
        float wz0 = 1.0f - dz;
        float wz1 = dz;
        
        float density = sys->masses[p] / h3;
        
        for (int di = 0; di <= 1; di++) {
            for (int dj = 0; dj <= 1; dj++) {
                for (int dk = 0; dk <= 1; dk++) {
                    int i = (i0 + di + N) % N;
                    int j = (j0 + dj + N) % N;
                    int k = (k0 + dk + N) % N;
                    
                    float wx = (di == 0) ? wx0 : wx1;
                    float wy = (dj == 0) ? wy0 : wy1;
                    float wz = (dk == 0) ? wz0 : wz1;
                    float weight = wx * wy * wz;
                    
                    #pragma omp atomic
                    pm->rho[i][j][k] += density * weight;
                }
            }
        }
    }
    
    double t_end = omp_get_wtime();
    printf("CIC mass assignment: %.2f seconds\n", t_end - t_start);
}
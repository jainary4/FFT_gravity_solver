#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "constants.h"
#include "structs.h"
#include "mesh.h"

double*** allocate_3d_array(int n) {
    double *data = (double*)calloc(n * n * n, sizeof(double));
    double ***array = (double***)malloc(n * sizeof(double**));
    
    for (int i = 0; i < n; i++) {
        array[i] = (double**)malloc(n * sizeof(double*));
        for (int j = 0; j < n; j++) {
            array[i][j] = &data[i * n * n + j * n];
        }
    }
    
    return array;
}

void free_3d_array(double ***array, int n) {
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

// (periodic assign_mass_cic removed - simulation uses padded isolated solver)

// Assign masses into a zero-padded grid without periodic wrapping.
// The original N x N x N block is written starting at index 'offset' in each dimension
// inside the padded Np x Np x Np array.
void assign_mass_cic_padded(double ***rho_pad, int Np, ParticleSystem *sys, int N, double h, int offset) {
    double t_start = omp_get_wtime();

    double h3 = h * h * h;

    // zero padded array
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Np; i++) {
        for (int j = 0; j < Np; j++) {
            for (int k = 0; k < Np; k++) {
                rho_pad[i][j][k] = 0.0f;
            }
        }
    }

    // mass assignment into the padded array: write into indices offset..offset+N-1
    #pragma omp parallel for
    for (int p = 0; p < sys->N; p++) {
        double x_grid = sys->positions[p].x / h;
        double y_grid = sys->positions[p].y / h;
        double z_grid = sys->positions[p].z / h;

        int i0 = (int)floorf(x_grid);
        int j0 = (int)floorf(y_grid);
        int k0 = (int)floorf(z_grid);

        double dx = x_grid - i0;
        double dy = y_grid - j0;
        double dz = z_grid - k0;

        double wx0 = 1.0f - dx;
        double wx1 = dx;
        double wy0 = 1.0f - dy;
        double wy1 = dy;
        double wz0 = 1.0f - dz;
        double wz1 = dz;

        double density = sys->masses[p] / h3;

        for (int di = 0; di <= 1; di++) {
            for (int dj = 0; dj <= 1; dj++) {
                for (int dk = 0; dk <= 1; dk++) {
                    int i = offset + i0 + di;
                    int j = offset + j0 + dj;
                    int k = offset + k0 + dk;

                    // Allow writes anywhere inside the padded array bounds [0..Np-1].
                    // For particles near the original N block edge, contributions may
                    // fall into the first padded cell (index offset+N); accept those
                    // so the padded convolution is correct. Defensive check against
                    // the full padded size Np.
                    if (i < 0 || i >= Np) continue;
                    if (j < 0 || j >= Np) continue;
                    if (k < 0 || k >= Np) continue;

                    double wx = (di == 0) ? wx0 : wx1;
                    double wy = (dj == 0) ? wy0 : wy1;
                    double wz = (dk == 0) ? wz0 : wz1;
                    double weight = wx * wy * wz;

                    #pragma omp atomic
                    rho_pad[i][j][k] += density * weight;
                }
            }
        }
    }

    double t_end = omp_get_wtime();
    printf("CIC padded mass assignment: %.2f seconds\n", t_end - t_start);
}
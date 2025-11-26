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
                    int i_orig = i0 + di;
                    int j_orig = j0 + dj;
                    int k_orig = k0 + dk;
                    
                    // Apply periodic wrapping to the original N grid
                    i_orig = (i_orig % N + N) % N;
                    j_orig = (j_orig % N + N) % N;
                    k_orig = (k_orig % N + N) % N;
                    
                    // Map to padded array
                    int i = offset + i_orig;
                    int j = offset + j_orig;
                    int k = offset + k_orig;
                    
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

// CIC Gather: Interpolate grid forces to particles and compute accelerations
// F = force from grid, a = F/m = acceleration of particle
void gather_forces_to_particles(
    double ***force_x, double ***force_y, double ***force_z,
    int Np, ParticleSystem *sys, int N, double h, int offset
) {
    double t_start = omp_get_wtime();

    if (!force_x || !force_y || !force_z || !sys || N <= 0 || h <= 0.0) {
        fprintf(stderr, "gather_forces_to_particles: invalid arguments\n");
        return;
    }

    // Parallel loop over all particles
    #pragma omp parallel for
    for (int p = 0; p < sys->N; p++) {
        // 1. Get particle position
        double x = sys->positions[p].x;
        double y = sys->positions[p].y;
        double z = sys->positions[p].z;

        // 2. Convert to grid units
        double x_grid = x / h;
        double y_grid = y / h;
        double z_grid = z / h;

        // 3. Find enclosing cell (physical grid coordinates)
        int i0 = (int)floorf(x_grid);
        int j0 = (int)floorf(y_grid);
        int k0 = (int)floorf(z_grid);

        // Clamp to physical grid bounds [0, N-1]
        // (Particles should be within [0, L], but allow slight overshoot due to numerics)
        i0 = (i0 < 0) ? 0 : (i0 >= N - 1) ? N - 1 : i0;
        j0 = (j0 < 0) ? 0 : (j0 >= N - 1) ? N - 1 : j0;
        k0 = (k0 < 0) ? 0 : (k0 >= N - 1) ? N - 1 : k0;

        // 4. Compute fractional offsets within cell
        double dx = x_grid - (double)i0;
        double dy = y_grid - (double)j0;
        double dz = z_grid - (double)k0;

        // Clamp fractional parts to [0, 1) in case of rounding errors
        dx = (dx < 0.0) ? 0.0 : (dx >= 1.0) ? 0.999999 : dx;
        dy = (dy < 0.0) ? 0.0 : (dy >= 1.0) ? 0.999999 : dy;
        dz = (dz < 0.0) ? 0.0 : (dz >= 1.0) ? 0.999999 : dz;

        // 5. Compute weights for 8 corners
        double wx[2], wy[2], wz[2];
        wx[0] = 1.0 - dx;  wx[1] = dx;
        wy[0] = 1.0 - dy;  wy[1] = dy;
        wz[0] = 1.0 - dz;  wz[1] = dz;

        // 6. Interpolate forces from 8 corners
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (int di = 0; di <= 1; di++) {
            for (int dj = 0; dj <= 1; dj++) {
                for (int dk = 0; dk <= 1; dk++) {
                    // Grid index in physical domain
                    int i_phys = i0 + di;
                    int j_phys = j0 + dj;
                    int k_phys = k0 + dk;

                    // Map to padded array indices
                    int i = offset + i_phys;
                    int j = offset + j_phys;
                    int k = offset + k_phys;

                    // Compute weight: product of 1D weights
                    double weight = wx[di] * wy[dj] * wz[dk];

                    // Accumulate force contributions
                    Fx += weight * force_x[i][j][k];
                    Fy += weight * force_y[i][j][k];
                    Fz += weight * force_z[i][j][k];
                }
            }
        }

        // 7. Compute acceleration: a = F / m
        double mass = sys->masses[p];
        if (mass > 0.0) {
            sys->accelerations[p].x = Fx / mass;
            sys->accelerations[p].y = Fy / mass;
            sys->accelerations[p].z = Fz / mass;
        } else {
            // Massless particle or error: zero acceleration
            sys->accelerations[p].x = 0.0;
            sys->accelerations[p].y = 0.0;
            sys->accelerations[p].z = 0.0;
        }
    }

    double t_end = omp_get_wtime();
    printf("CIC gather forces to particles: %.2f seconds\n", t_end - t_start);
}
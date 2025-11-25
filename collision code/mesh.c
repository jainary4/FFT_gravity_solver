#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "constants.h"
#include "structs.h"
#include "mesh.h"

// ----------------------------------------------------------------------
// Allocate a contiguous 3D array [n][n][n] of double.
//
// We allocate a single block of n^3 doubles, then build a triple-pointer
// so that pm->rho[i][j][k] works as expected.
// ----------------------------------------------------------------------
double*** allocate_3d_array(int n)
{
    double *data   = (double*)calloc((size_t)n * n * n, sizeof(double));
    double ***array = (double***)malloc((size_t)n * sizeof(double**));

    if (!data || !array) {
        fprintf(stderr, "allocate_3d_array: allocation failed\n");
        free(data);
        free(array);
        return NULL;
    }

    for (int i = 0; i < n; ++i) {
        array[i] = (double**)malloc((size_t)n * sizeof(double*));
        if (!array[i]) {
            fprintf(stderr, "allocate_3d_array: allocation failed (level 2)\n");
            // simple cleanup
            for (int k = 0; k < i; ++k) free(array[k]);
            free(array);
            free(data);
            return NULL;
        }
        for (int j = 0; j < n; ++j) {
            array[i][j] = &data[(size_t)i * n * n + (size_t)j * n];
        }
    }

    return array;
}

void free_3d_array(double ***array, int n)
{
    if (!array) return;

    // data block is at array[0][0]
    free(array[0][0]);

    // free second-level pointers
    for (int i = 0; i < n; ++i) {
        free(array[i]);
    }

    // free top-level pointer
    free(array);
}

// ----------------------------------------------------------------------
// Create / destroy ParticleMesh
// ----------------------------------------------------------------------
ParticleMesh* create_particle_mesh(void)
{
    ParticleMesh *pm = (ParticleMesh*)malloc(sizeof(ParticleMesh));
    if (!pm) {
        fprintf(stderr, "create_particle_mesh: allocation failed\n");
        return NULL;
    }

    pm->N         = NMESH;
    pm->cell_size = L / (double)NMESH;
    pm->box_size  = L;
    pm->rho       = allocate_3d_array(pm->N);

    if (!pm->rho) {
        free(pm);
        return NULL;
    }

    return pm;
}

void destroy_particle_mesh(ParticleMesh *pm)
{
    if (!pm) return;
    free_3d_array(pm->rho, pm->N);
    free(pm);
}

// ----------------------------------------------------------------------
// CIC mass assignment: put particle masses on the mesh with periodic BC.
//
//   pm->rho is a density field [mass / volume]. So we divide particle mass
//   by h^3 before distributing to the 8 neighbouring cells.
// ----------------------------------------------------------------------
void assign_mass_cic(ParticleMesh *pm, ParticleSystem *sys)
{
    double t_start = omp_get_wtime();

    const double h  = pm->cell_size;
    const double h3 = h * h * h;
    const int    N  = pm->N;

    // Zero out density field
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                pm->rho[i][j][k] = 0.0;
            }
        }
    }

    // Loop over particles
    #pragma omp parallel for
    for (int p = 0; p < sys->N; ++p) {
        // Physical position -> grid coordinates
        double x_grid = sys->positions[p].x / h;
        double y_grid = sys->positions[p].y / h;
        double z_grid = sys->positions[p].z / h;

        int i0 = (int)floor(x_grid);
        int j0 = (int)floor(y_grid);
        int k0 = (int)floor(z_grid);

        double dx = x_grid - i0;
        double dy = y_grid - j0;
        double dz = z_grid - k0;

        double wx0 = 1.0 - dx;
        double wx1 = dx;
        double wy0 = 1.0 - dy;
        double wy1 = dy;
        double wz0 = 1.0 - dz;
        double wz1 = dz;

        // Convert mass to density
        double density = sys->masses[p] / h3;

        for (int di = 0; di <= 1; ++di) {
            for (int dj = 0; dj <= 1; ++dj) {
                for (int dk = 0; dk <= 1; ++dk) {
                    int i = (i0 + di + N) % N;
                    int j = (j0 + dj + N) % N;
                    int k = (k0 + dk + N) % N;

                    double wx = (di == 0) ? wx0 : wx1;
                    double wy = (dj == 0) ? wy0 : wy1;
                    double wz = (dk == 0) ? wz0 : wz1;
                    double weight = wx * wy * wz;

                    #pragma omp atomic
                    pm->rho[i][j][k] += density * weight;
                }
            }
        }
    }

    double t_end = omp_get_wtime();
    printf("CIC mass assignment: %.2f seconds\n", t_end - t_start);
}

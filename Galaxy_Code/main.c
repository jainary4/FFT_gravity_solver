#include <stdio.h>
#include <stdlib.h>

#include "constants.h"
#include "structs.h"
#include "particles.h"
#include "mesh.h"

int main()
{
    printf("Initializing particle system...\n");
    ParticleSystem *sys = initialize_particle_system();

    if (!sys) {
        fprintf(stderr, "ERROR: initialize_particle_system() returned NULL\n");
        return 1;
    }

    printf("  Loaded %d particles\n", sys->N);

    printf("\nCreating particle mesh...\n");
    ParticleMesh *pm = create_particle_mesh();

    if (!pm) {
        fprintf(stderr, "ERROR: create_particle_mesh() returned NULL\n");
        return 1;
    }

    printf("  Grid: %d³\n", pm->N);
    printf("  Cell size: %.3f kpc\n", pm->cell_size);

    printf("\nRunning CIC mass assignment...\n");
    assign_mass_cic(pm, sys);

    printf("CIC complete.\n");

    /* Print black hole positions (use helper added in particles.c/h) */
    Vector3 *bh_pos = get_blackhole_positions(sys);
    int n_bh = N_BH;
    printf("\nBlack hole positions (initial):\n");
    for (int b = 0; b < n_bh; ++b) {
        printf("  BH %d: (%.6f, %.6f, %.6f)\n", b,
               bh_pos[b].x, bh_pos[b].y, bh_pos[b].z);
    }

    // Optional: print total mass on the grid
    double total_mass = 0.0;
    for (int i = 0; i < pm->N; i++)
        for (int j = 0; j < pm->N; j++)
            for (int k = 0; k < pm->N; k++)
                total_mass += pm->rho[i][j][k]*(pm->cell_size * pm->cell_size * pm->cell_size);

    printf("Total mass in grid = %.6f (should be ≈ 1.0)\n", total_mass);

    destroy_particle_mesh(pm);
    destroy_particle_system(sys);
    return 0;
}

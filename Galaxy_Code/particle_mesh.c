#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "constants.h"
#include "structs.h"
#include "particles.h"
#include "mesh.h"

int main() {
    srand(time(NULL));
    
    printf("Initializing particle system...\n");
    ParticleSystem *sys = initialize_particle_system();
    
    printf("\nCreating particle mesh...\n");
    ParticleMesh *pm = create_particle_mesh();
    printf("  Grid: %d³\n", pm->N);
    printf("  Cell size: %.3f kpc\n", pm->cell_size);
    
    printf("\nCIC mass assignment (padded)...\n");
    // Allocate padded density array
    int Np = NMESH_PADDED;
    double ***rho_pad = allocate_3d_array(Np);
    int offset = 0; // place original N^3 block at corner; change to (Np - NMESH)/2 to center
    assign_mass_cic_padded(rho_pad, Np, sys, pm->N, pm->cell_size, offset);

    // Verify total mass (sum over padded array * cell volume)
    double total_mass = 0.0;
    double h3 = (double)pm->cell_size * pm->cell_size * pm->cell_size;
    for (int i = 0; i < Np; i++)
        for (int j = 0; j < Np; j++)
            for (int k = 0; k < Np; k++)
                total_mass += rho_pad[i][j][k] * h3;

    printf("Total mass in padded grid = %.6f (should be ≈ 1.0)\n", total_mass);

    // Free padded array now (FFT step will use a later allocated padded array)
    free_3d_array(rho_pad, Np);
    
    destroy_particle_mesh(pm);
    destroy_particle_system(sys);
    
    return 0;
}
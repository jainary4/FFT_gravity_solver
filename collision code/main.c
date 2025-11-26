#include <stdio.h>
#include <stdlib.h>
#include "constants.h"
#include "structs.h"
#include "particles.h"
#include "mesh.h"
#include "fft_solver.h"
#include "poisson.h"
#include "force.h"


int write_to_txt(double ***phi, int N, const char *filepath) {
    if (!phi || N <= 0 || !filepath) return -1;

    FILE *f = fopen(filepath, "w");
    if (!f) {
        perror("write_phi_to_txt: fopen");
        return -1;
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                /* Use a high-precision text format */
                if (fprintf(f, "%.17g\n", phi[i][j][k]) < 0) {
                    perror("write_phi_to_txt: fprintf");
                    fclose(f);
                    return -1;
                }
            }
        }
    }

    if (fclose(f) != 0) {
        perror("write_phi_to_txt: fclose");
        return -1;
    }

    return 0;
}

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

    printf("\nRunning CIC mass assignment (padded)...\n");

    // Allocate padded density array and assign masses into it
    int Np = NMESH_PADDED;
    double ***rho_pad = allocate_3d_array(Np);
    int offset = 0; // place original N^3 block at corner; use (Np - NMESH)/2 to center
    assign_mass_cic_padded(rho_pad, Np, sys, pm->N, pm->cell_size, offset);

    double ***laplacian_phi_pad = create_laplacian_equation(Np, rho_pad);
    double ***phi_pad = solve_poisson_fftw( laplacian_phi_pad, Np, pm->cell_size);

    if (!phi_pad) {
        fprintf(stderr, "ERROR: solve_poisson_fftw returned NULL!\n");
        return 1;
    }
    printf("\nWriting potential to phi_pad.txt...\n");
    if (write_to_txt(phi_pad, Np, "phi_pad.txt") !=0) {
        fprintf(stderr, "ERROR: write_phi_to_txt failed!\n");
        return 1;
    }
    // Now we need to create 3 force arrays
    double ***force_x = allocate_3d_array(Np);
    double ***force_y = allocate_3d_array(Np);
    double ***force_z = allocate_3d_array(Np);

    // now compute forces from potential

    compute_forces_from_potential(phi_pad, Np, pm->cell_size, force_x, force_y, force_z);

    printf("\nWriting force_x to force_x.txt...\n");

    if (write_to_txt(force_x, Np, "force_x.txt") !=0) {
        fprintf(stderr, "ERROR: write_to_txt for force_x failed!\n");
        return 1;
    }

     printf("\nWriting force_y to force_y.txt...\n");
    if (write_to_txt(force_y, Np, "force_y.txt") !=0) {
        fprintf(stderr, "ERROR: write_to_txt for force_y failed!\n");
        return 1;
    }

     printf("\nWriting force_z to force_z.txt...\n");
    if (write_to_txt(force_x, Np, "force_z.txt") !=0) {
        fprintf(stderr, "ERROR: write_to_txt for force_z failed!\n");
        return 1;
    }
    
    // now we calculate the accelerations on particles from the force grids
    printf("\nGathering forces to particles...\n");
    gather_forces_to_particles(force_x, force_y, force_z, Np, sys, pm->N, pm->cell_size, offset);
    printf("Done gathering forces to particles.\n");
    // now we have acceleration of th system in sys->accelerations
    // now the problem is to just use these accelerations to update velocities and positions
    // this part is not implemented yet
    
    free_3d_array(rho_pad, Np);
    destroy_particle_mesh(pm);
    destroy_particle_system(sys);
    free_3d_array(laplacian_phi_pad, Np);
    free_3d_array(phi_pad, Np);
    free_3d_array(force_x, Np);
    free_3d_array(force_y, Np);
    free_3d_array(force_z, Np);
    return 0;
}

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
    
    printf("\nCIC mass assignment...\n");
    assign_mass_cic(pm, sys);
    
    destroy_particle_mesh(pm);
    destroy_particle_system(sys);
    
    return 0;
}
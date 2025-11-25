#include <stdio.h>
#include <stdlib.h>

#include "constants.h"
#include "structs.h"
#include "bh_collision.h"

int main(void)
{
    ParticleSystem sys;

    sys.N          = 3;
    sys.positions  = (Vector3*)malloc(sys.N * sizeof(Vector3));
    sys.velocities = (Vector3*)malloc(sys.N * sizeof(Vector3));
    sys.masses     = (double*)  malloc(sys.N * sizeof(double));
    sys.types      = (int*)     malloc(sys.N * sizeof(int));

    if (!sys.positions || !sys.velocities || !sys.masses || !sys.types) {
        fprintf(stderr, "Allocation failed in bh_collision_example\n");
        return 1;
    }

    /* Particle 0: black hole at centre */
    sys.types[0] = TYPE_BH;
    sys.masses[0] = 10.0;
    sys.positions[0].x = 50.0;
    sys.positions[0].y = 50.0;
    sys.positions[0].z = 50.0;
    sys.velocities[0].x = 0.0;
    sys.velocities[0].y = 0.0;
    sys.velocities[0].z = 0.0;

    /* Particle 1: star inside capture radius */
    sys.types[1] = TYPE_STAR;
    sys.masses[1] = 1.0;
    sys.positions[1].x = 50.0 + 0.5 * BH_STAR_COLLISION_RADIUS;
    sys.positions[1].y = 50.0;
    sys.positions[1].z = 50.0;
    sys.velocities[1].x = 1.0;
    sys.velocities[1].y = 0.0;
    sys.velocities[1].z = 0.0;

    /* Particle 2: dark matter somewhere else */
    sys.types[2] = TYPE_DM;
    sys.masses[2] = 3.0;
    sys.positions[2].x = 10.0;
    sys.positions[2].y = 10.0;
    sys.positions[2].z = 10.0;
    sys.velocities[2].x = 0.0;
    sys.velocities[2].y = 0.1;
    sys.velocities[2].z = 0.0;

    int bh_indices[1] = {0};
    int n_bh          = 1;

    printf("Before BH collisions:\n");
    for (int i = 0; i < sys.N; ++i) {
        printf("  p%d: type=%d mass=%.3f pos=(%.3f,%.3f,%.3f) "
               "vel=(%.3f,%.3f,%.3f)\n",
               i, sys.types[i], sys.masses[i],
               sys.positions[i].x, sys.positions[i].y, sys.positions[i].z,
               sys.velocities[i].x, sys.velocities[i].y, sys.velocities[i].z);
    }

    bh_collision_step(&sys, bh_indices, &n_bh);

    printf("\nAfter BH collisions:\n");
    for (int i = 0; i < sys.N; ++i) {
        printf("  p%d: type=%d mass=%.3f pos=(%.3f,%.3f,%.3f) "
               "vel=(%.3f,%.3f,%.3f)\n",
               i, sys.types[i], sys.masses[i],
               sys.positions[i].x, sys.positions[i].y, sys.positions[i].z,
               sys.velocities[i].x, sys.velocities[i].y, sys.velocities[i].z);
    }

    free(sys.positions);
    free(sys.velocities);
    free(sys.masses);
    free(sys.types);

    return 0;
}

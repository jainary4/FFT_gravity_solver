#include <stdio.h>
#include <stdlib.h>

#include "constants.h"
#include "structs.h"
#include "bh_collision.h"

int main(void)
{
    ParticleSystem sys;

    sys.N = 3;
    sys.positions  = (Vector3*)malloc(sys.N * sizeof(Vector3));
    sys.velocities = (Vector3*)malloc(sys.N * sizeof(Vector3));
    sys.masses     = (float*)  malloc(sys.N * sizeof(float));
    sys.types      = (int*)    malloc(sys.N * sizeof(int));

    if (!sys.positions || !sys.velocities || !sys.masses || !sys.types) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    float cell_size = (float)(L / (float)NMESH);
    printf("Box size L = %.3f, NMESH = %d, cell_size = %.3f\n",
           (float)L, NMESH, cell_size);
    printf("BH_STAR_COLLISION_RADIUS = %.3f\n", BH_STAR_COLLISION_RADIUS);
    printf("BH_BH_COLLISION_RADIUS   = %.3f\n\n", BH_BH_COLLISION_RADIUS);

    /* p0: star close to BH -> should be eaten */
    sys.types[0] = TYPE_STAR;
    sys.masses[0] = 1.0f;
    sys.positions[0].x = 50.0f + 0.25f * BH_STAR_COLLISION_RADIUS;
    sys.positions[0].y = 50.0f;
    sys.positions[0].z = 50.0f;
    sys.velocities[0].x = 1.0f;
    sys.velocities[0].y = 0.0f;
    sys.velocities[0].z = 0.0f;

    /* p1: BH at center */
    sys.types[1] = TYPE_BH;
    sys.masses[1] = 10.0f;
    sys.positions[1].x = 50.0f;
    sys.positions[1].y = 50.0f;
    sys.positions[1].z = 50.0f;
    sys.velocities[1].x = 0.0f;
    sys.velocities[1].y = 0.0f;
    sys.velocities[1].z = 0.0f;

    /* p2: star far away -> should NOT be eaten */
    sys.types[2] = TYPE_STAR;
    sys.masses[2] = 1.0f;
    sys.positions[2].x = 50.0f + 5.0f * BH_STAR_COLLISION_RADIUS;
    sys.positions[2].y = 50.0f;
    sys.positions[2].z = 50.0f;
    sys.velocities[2].x = -1.0f;
    sys.velocities[2].y =  0.0f;
    sys.velocities[2].z =  0.0f;

    int bh_indices[1] = {1};
    int n_bh = 1;

    printf("Initial state:\n");
    for (int i = 0; i < sys.N; ++i) {
        printf("  p%d: type=%d mass=%.3f pos=(%.3f,%.3f,%.3f) vel=(%.3f,%.3f,%.3f)\n",
               i, sys.types[i], sys.masses[i],
               sys.positions[i].x, sys.positions[i].y, sys.positions[i].z,
               sys.velocities[i].x, sys.velocities[i].y, sys.velocities[i].z);
    }
    printf("\n");

    bh_collision_step(&sys, bh_indices, &n_bh);

    printf("After bh_collision_step:\n");
    for (int i = 0; i < sys.N; ++i) {
        printf("  p%d: type=%d mass=%.3f pos=(%.3f,%.3f,%.3f) vel=(%.3f,%.3f,%.3f)\n",
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

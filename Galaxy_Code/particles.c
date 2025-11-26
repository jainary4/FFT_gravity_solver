#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "constants.h"
#include "structs.h"
#include "particles.h"
#include "random.h"

void initialize_plummer_positions(Vector3 *positions, int N, double a, Vector3 center) {
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; i++) {
            double r = sample_plummer_radius(a);
            Vector3 offset = random_point_on_sphere(r);
            
            positions[i].x = center.x + offset.x;
            positions[i].y = center.y + offset.y;
            positions[i].z = center.z + offset.z;
        }
    }
}

void initialize_velocities(Vector3 *velocities, int N, double sigma) {
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; i++) {
            velocities[i].x = random_gaussian(0.0f, sigma);
            velocities[i].y = random_gaussian(0.0f, sigma);
            velocities[i].z = random_gaussian(0.0f, sigma);
        }
    }
}

void remove_bulk_motion(ParticleSystem *sys) {
    // Use separate scalar variables instead of Vector3
    float momentum_x = 0.0f;
    float momentum_y = 0.0f;
    float momentum_z = 0.0f;
    float total_mass = 0.0f;
    
    // OpenMP can reduce these scalars just fine
    #pragma omp parallel for reduction(+:momentum_x, momentum_y, momentum_z, total_mass)
    for (int i = 0; i < sys->N; i++) {
        momentum_x += sys->masses[i] * sys->velocities[i].x;
        momentum_y += sys->masses[i] * sys->velocities[i].y;
        momentum_z += sys->masses[i] * sys->velocities[i].z;
        total_mass += sys->masses[i];
    }
    
    // Compute center-of-mass velocity
    Vector3 v_cm;
    v_cm.x = momentum_x / total_mass;
    v_cm.y = momentum_y / total_mass;
    v_cm.z = momentum_z / total_mass;
    
    printf("  Center-of-mass velocity before: (%.3e, %.3e, %.3e)\n",
           v_cm.x, v_cm.y, v_cm.z);
    
    // Subtract v_cm from each particle's velocity
    #pragma omp parallel for
    for (int i = 0; i < sys->N; i++) {
        sys->velocities[i].x -= v_cm.x;
        sys->velocities[i].y -= v_cm.y;
        sys->velocities[i].z -= v_cm.z;
    }
    
    // Recompute to check
    momentum_x = momentum_y = momentum_z = 0.0f;
    total_mass = 0.0f;
    
    #pragma omp parallel for reduction(+:momentum_x, momentum_y, momentum_z, total_mass)
    for (int i = 0; i < sys->N; i++) {
        momentum_x += sys->masses[i] * sys->velocities[i].x;
        momentum_y += sys->masses[i] * sys->velocities[i].y;
        momentum_z += sys->masses[i] * sys->velocities[i].z;
        total_mass += sys->masses[i];
    }
    
    v_cm.x = momentum_x / total_mass;
    v_cm.y = momentum_y / total_mass;
    v_cm.z = momentum_z / total_mass;
    
    printf("  Center-of-mass velocity after:  (%.3e, %.3e, %.3e)\n",
           v_cm.x, v_cm.y, v_cm.z);
}

void center_in_box(ParticleSystem *sys, Vector3 box_center) {
    double pos_cm_x = 0.0;
    double pos_cm_y = 0.0;
    double pos_cm_z = 0.0;
    double total_mass = 0.0;
    
    #pragma omp parallel for reduction(+:pos_cm_x, pos_cm_y, pos_cm_z, total_mass)
    for (int i = 0; i < sys->N; i++) {
        pos_cm_x += sys->masses[i] * sys->positions[i].x;
        pos_cm_y += sys->masses[i] * sys->positions[i].y;
        pos_cm_z += sys->masses[i] * sys->positions[i].z;
        total_mass += sys->masses[i];
    }
    
    Vector3 pos_cm;
    pos_cm.x = pos_cm_x / total_mass;
    pos_cm.y = pos_cm_y / total_mass;
    pos_cm.z = pos_cm_z / total_mass;
    
    // Shift to box center
    Vector3 shift;
    shift.x = box_center.x - pos_cm.x;
    shift.y = box_center.y - pos_cm.y;
    shift.z = box_center.z - pos_cm.z;
    
    printf("  Center-of-mass position before: (%.3e, %.3e, %.3e)\n",
           pos_cm.x, pos_cm.y, pos_cm.z);
    printf("  Shifting system by: (%.3e, %.3e, %.3e)\n",
           shift.x, shift.y, shift.z);
    
    #pragma omp parallel for
    for (int i = 0; i < sys->N; i++) {
        sys->positions[i].x += shift.x;
        sys->positions[i].y += shift.y;
        sys->positions[i].z += shift.z;
    }
    
    // Recompute COM to confirm it is at box center
    pos_cm_x = pos_cm_y = pos_cm_z = 0.0;
    total_mass = 0.0;
    
    #pragma omp parallel for reduction(+:pos_cm_x, pos_cm_y, pos_cm_z, total_mass)
    for (int i = 0; i < sys->N; i++) {
        pos_cm_x += sys->masses[i] * sys->positions[i].x;
        pos_cm_y += sys->masses[i] * sys->positions[i].y;
        pos_cm_z += sys->masses[i] * sys->positions[i].z;
        total_mass += sys->masses[i];
    }
    
    pos_cm.x = pos_cm_x / total_mass;
    pos_cm.y = pos_cm_y / total_mass;
    pos_cm.z = pos_cm_z / total_mass;
    
    printf("  Center-of-mass position after:  (%.3e, %.3e, %.3e)\n",
           pos_cm.x, pos_cm.y, pos_cm.z);
}

ParticleSystem* initialize_particle_system(void) {
    ParticleSystem *sys = (ParticleSystem*)malloc(sizeof(ParticleSystem));
    if (!sys) {
        fprintf(stderr, "ERROR: initialize_particle_system: malloc(sys) failed\n");
        return NULL;
    }

    sys->N = N_TOTAL;
    printf("  Allocating %d particles\n", sys->N);

    sys->positions     = (Vector3*)malloc(sys->N * sizeof(Vector3));
    sys->velocities    = (Vector3*)malloc(sys->N * sizeof(Vector3));
    sys->accelerations = (Vector3*)malloc(sys->N * sizeof(Vector3));
    sys->masses        = (double*) malloc(sys->N * sizeof(double));
    sys->types         = (int*)    malloc(sys->N * sizeof(int));

    if (!sys->positions || !sys->velocities || !sys->accelerations ||
        !sys->masses || !sys->types) 
    {
        fprintf(stderr, "ERROR: initialize_particle_system: malloc of arrays failed\n");
        destroy_particle_system(sys);
        return NULL;
    }

    // Initialize accelerations to zero
    #pragma omp parallel for
    for (int i = 0; i < sys->N; i++) {
        sys->accelerations[i].x = 0.0;
        sys->accelerations[i].y = 0.0;
        sys->accelerations[i].z = 0.0;
    }

    printf("  Setting Plummer positions for stars...\n");
    Vector3 center = { L * 0.5, L * 0.5, L * 0.5 };

    initialize_plummer_positions(sys->positions, N_STARS, L / 16.0, center);

    printf("  Setting Plummer positions for dark matter halo...\n");
    initialize_plummer_positions(sys->positions + N_STARS, N_DM, L / 8.0, center);

    printf("  Placing supermassive black holes...\n");
    for (int i = 0; i < N_BH; i++) {
        int idx = N_STARS + N_DM + i;
        sys->positions[idx] = center;
    }

    printf("  Initializing velocities...\n");
    initialize_velocities(sys->velocities, sys->N, 50.0);

    printf("  Assigning masses and types...\n");
    #pragma omp parallel for
    for (int i = 0; i < sys->N; i++) {
        if (i < N_STARS) {
            sys->masses[i] = M_STAR;
            sys->types[i]  = TYPE_STAR;
        } else if (i < N_STARS + N_DM) {
            sys->masses[i] = M_DM;
            sys->types[i]  = TYPE_DM;
        } else {
            sys->masses[i] = M_BH;
            sys->types[i]  = TYPE_BH;
        }
    }

    printf("  Removing bulk motion...\n");
    remove_bulk_motion(sys);

    printf("  Centering system in box...\n");
    center_in_box(sys, center);

    return sys;
}

void destroy_particle_system(ParticleSystem *sys) {
    if (!sys) return;

    free(sys->positions);
    free(sys->velocities);
    free(sys->accelerations);
    free(sys->masses);
    free(sys->types);
    free(sys);
}

// Return pointer to the first black-hole position (contiguous block)
Vector3* get_blackhole_positions(ParticleSystem *sys) {
    if (!sys) return NULL;
    return sys->positions + (N_STARS + N_DM);
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "constants.h"
#include "structs.h"
#include "particles.h"
#include "random.h"

void initialize_plummer_positions(Vector3 *positions, int N, double a, Vector3 center) {
    #pragma omp parallel
    {
        unsigned int seed = time(NULL) + omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < N; i++) {
            float r = sample_plummer_radius(a);
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
        unsigned int seed = time(NULL) + omp_get_thread_num() * 1000;
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
    double momentum_x = 0.0;
    double momentum_y = 0.0;
    double momentum_z = 0.0;
    double total_mass = 0.0;
    
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
    
    // Subtract from all particles
    #pragma omp parallel for
    for (int i = 0; i < sys->N; i++) {
        sys->velocities[i].x -= v_cm.x;
        sys->velocities[i].y -= v_cm.y;
        sys->velocities[i].z -= v_cm.z;
    }
}

void center_in_box(ParticleSystem *sys, Vector3 box_center) {
    // Use separate scalar variables instead of Vector3
    double pos_cm_x = 0.0;
    double pos_cm_y = 0.0;
    double pos_cm_z = 0.0;
    double total_mass = 0.0;
    
    // Calculate center of mass
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

    #pragma omp parallel for
    for (int i = 0; i < sys->N; i++) {
        sys->positions[i].x += shift.x;
        sys->positions[i].y += shift.y;
        sys->positions[i].z += shift.z;
        
        // Apply periodic boundaries
        sys->positions[i].x = fmod(sys->positions[i].x + L, L);
        sys->positions[i].y = fmod(sys->positions[i].y + L, L);
        sys->positions[i].z = fmod(sys->positions[i].z + L, L);
    }
}

ParticleSystem* initialize_particle_system() {
    ParticleSystem *sys = (ParticleSystem*)malloc(sizeof(ParticleSystem));
    sys->N = N_TOTAL;
    sys->positions = (Vector3*)malloc(N_TOTAL * sizeof(Vector3));
    sys->velocities = (Vector3*)malloc(N_TOTAL * sizeof(Vector3));
    sys->masses = (double*)malloc(N_TOTAL * sizeof(double));
    sys->types = (int*)malloc(N_TOTAL * sizeof(int));
    
    Vector3 center = {L/2.0, L/2.0, L/2.0};
    
    double t_start, t_end;
    
    // Positions
    t_start = omp_get_wtime();
    initialize_plummer_positions(sys->positions, N_STARS, A_STARS, center);
    initialize_plummer_positions(sys->positions + N_STARS, N_DM, A_DM, center);
    initialize_plummer_positions(sys->positions + N_STARS + N_DM, N_BH, A_BH, center);
    t_end = omp_get_wtime();
    printf("Position generation: %.2f seconds\n", t_end - t_start);
    
    // Velocities
    t_start = omp_get_wtime();
    initialize_velocities(sys->velocities, N_STARS, SIGMA_V_STARS);
    initialize_velocities(sys->velocities + N_STARS, N_DM, SIGMA_V_DM);
    initialize_velocities(sys->velocities + N_STARS + N_DM, N_BH, SIGMA_V_BH);
    t_end = omp_get_wtime();
    printf("Velocity generation: %.2f seconds\n", t_end - t_start);
    
    // Masses and types
    t_start = omp_get_wtime();
    double m_star = (F_STARS * M_TOTAL) / N_STARS;
    double m_dm = (F_DM * M_TOTAL) / N_DM;
    double m_bh = (F_BH * M_TOTAL) / N_BH;
    
    #pragma omp parallel for
    for (int i = 0; i < N_STARS; i++) {
        sys->masses[i] = m_star;
        sys->types[i] = 0;
    }
    #pragma omp parallel for
    for (int i = N_STARS; i < N_STARS + N_DM; i++) {
        sys->masses[i] = m_dm;
        sys->types[i] = 1;
    }
    #pragma omp parallel for
    for (int i = N_STARS + N_DM; i < N_TOTAL; i++) {
        sys->masses[i] = m_bh;
        sys->types[i] = 2;
    }
    t_end = omp_get_wtime();
    printf("Mass assignment: %.2f seconds\n", t_end - t_start);
    
    t_start = omp_get_wtime();
    remove_bulk_motion(sys);
    center_in_box(sys, center);
    t_end = omp_get_wtime();
    printf("Bulk motion & centering: %.2f seconds\n", t_end - t_start);
    
    return sys;
}

void destroy_particle_system(ParticleSystem *sys) {
    if (!sys) return;
    free(sys->positions);
    free(sys->velocities);
    free(sys->masses);
    free(sys->types);
    free(sys);
}

// Return pointer to the first black-hole position (contiguous block)
Vector3* get_blackhole_positions(ParticleSystem *sys) {
    if (!sys) return NULL;
    return sys->positions + (N_STARS + N_DM);
}


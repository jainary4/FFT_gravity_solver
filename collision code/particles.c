#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "constants.h"
#include "structs.h"
#include "particles.h"
#include "random.h"

/* ----------------------------------------------------------------------
   Position initialisation: Plummer spheres
   ---------------------------------------------------------------------- */

void initialize_plummer_positions(Vector3 *positions, int N, double a, Vector3 center)
{
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        /* Sample radius from Plummer profile and random direction on sphere */
        double r      = sample_plummer_radius(a);
        Vector3 off   = random_point_on_sphere(r);

        double x = center.x + off.x;
        double y = center.y + off.y;
        double z = center.z + off.z;

        /* Apply periodic boundaries in [0, L) */
        x = fmod(x, L); if (x < 0.0) x += L;
        y = fmod(y, L); if (y < 0.0) y += L;
        z = fmod(z, L); if (z < 0.0) z += L;

        positions[i].x = x;
        positions[i].y = y;
        positions[i].z = z;
    }
}

/* ----------------------------------------------------------------------
   Velocity initialisation: Gaussian with dispersion sigma
   ---------------------------------------------------------------------- */

void initialize_velocities(Vector3 *velocities, int N, double sigma)
{
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        velocities[i].x = random_gaussian(0.0, sigma);
        velocities[i].y = random_gaussian(0.0, sigma);
        velocities[i].z = random_gaussian(0.0, sigma);
    }
}

/* ----------------------------------------------------------------------
   Remove bulk motion (center-of-mass velocity)
   ---------------------------------------------------------------------- */

void remove_bulk_motion(ParticleSystem *sys)
{
    double Px = 0.0;
    double Py = 0.0;
    double Pz = 0.0;
    double M  = 0.0;

    #pragma omp parallel for reduction(+:Px,Py,Pz,M)
    for (int i = 0; i < sys->N; i++) {
        double m = sys->masses[i];
        Px += m * sys->velocities[i].x;
        Py += m * sys->velocities[i].y;
        Pz += m * sys->velocities[i].z;
        M  += m;
    }

    Vector3 v_cm = { Px / M, Py / M, Pz / M };

    printf("  Center-of-mass velocity before: (%.3e, %.3e, %.3e)\n",
           v_cm.x, v_cm.y, v_cm.z);

    #pragma omp parallel for
    for (int i = 0; i < sys->N; i++) {
        sys->velocities[i].x -= v_cm.x;
        sys->velocities[i].y -= v_cm.y;
        sys->velocities[i].z -= v_cm.z;
    }

    /* Recompute to check */
    Px = Py = Pz = 0.0;
    M  = 0.0;

    #pragma omp parallel for reduction(+:Px,Py,Pz,M)
    for (int i = 0; i < sys->N; i++) {
        double m = sys->masses[i];
        Px += m * sys->velocities[i].x;
        Py += m * sys->velocities[i].y;
        Pz += m * sys->velocities[i].z;
        M  += m;
    }

    v_cm.x = Px / M;
    v_cm.y = Py / M;
    v_cm.z = Pz / M;

    printf("  Center-of-mass velocity after:  (%.3e, %.3e, %.3e)\n",
           v_cm.x, v_cm.y, v_cm.z);
}

/* ----------------------------------------------------------------------
   Recentre positions so COM is at a chosen box_center
   ---------------------------------------------------------------------- */

void center_in_box(ParticleSystem *sys, Vector3 box_center)
{
    double X = 0.0;
    double Y = 0.0;
    double Z = 0.0;
    double M = 0.0;

    #pragma omp parallel for reduction(+:X,Y,Z,M)
    for (int i = 0; i < sys->N; i++) {
        double m = sys->masses[i];
        X += m * sys->positions[i].x;
        Y += m * sys->positions[i].y;
        Z += m * sys->positions[i].z;
        M += m;
    }

    Vector3 com = { X / M, Y / M, Z / M };
    Vector3 shift = {
        box_center.x - com.x,
        box_center.y - com.y,
        box_center.z - com.z
    };

    printf("  Center-of-mass position before: (%.3e, %.3e, %.3e)\n",
           com.x, com.y, com.z);
    printf("  Shifting system by: (%.3e, %.3e, %.3e)\n",
           shift.x, shift.y, shift.z);

    #pragma omp parallel for
    for (int i = 0; i < sys->N; i++) {
        sys->positions[i].x += shift.x;
        sys->positions[i].y += shift.y;
        sys->positions[i].z += shift.z;

        /* Keep everything inside [0, L) just in case */
        double x = fmod(sys->positions[i].x, L);
        double y = fmod(sys->positions[i].y, L);
        double z = fmod(sys->positions[i].z, L);
        if (x < 0.0) x += L;
        if (y < 0.0) y += L;
        if (z < 0.0) z += L;
        sys->positions[i].x = x;
        sys->positions[i].y = y;
        sys->positions[i].z = z;
    }

    /* Recompute COM to confirm */
    X = Y = Z = M = 0.0;
    #pragma omp parallel for reduction(+:X,Y,Z,M)
    for (int i = 0; i < sys->N; i++) {
        double m = sys->masses[i];
        X += m * sys->positions[i].x;
        Y += m * sys->positions[i].y;
        Z += m * sys->positions[i].z;
        M += m;
    }

    com.x = X / M;
    com.y = Y / M;
    com.z = Z / M;

    printf("  Center-of-mass position after:  (%.3e, %.3e, %.3e)\n",
           com.x, com.y, com.z);
}

/* ----------------------------------------------------------------------
   Build a full ParticleSystem with Plummer stars+DM+BH
   ---------------------------------------------------------------------- */

ParticleSystem* initialize_particle_system(void)
{
    ParticleSystem *sys = (ParticleSystem*)malloc(sizeof(ParticleSystem));
    if (!sys) {
        fprintf(stderr, "ERROR: initialize_particle_system: malloc(sys) failed\n");
        return NULL;
    }

    sys->N = N_TOTAL;

    sys->positions     = (Vector3*)malloc(sys->N * sizeof(Vector3));
    sys->velocities    = (Vector3*)malloc(sys->N * sizeof(Vector3));
    sys->accelerations = (Vector3*)malloc(sys->N * sizeof(Vector3));
    sys->masses        = (double*) malloc(sys->N * sizeof(double));
    sys->types         = (int*)    malloc(sys->N * sizeof(int));

    if (!sys->positions || !sys->velocities || !sys->accelerations ||
        !sys->masses    || !sys->types) {
        fprintf(stderr, "ERROR: initialize_particle_system: malloc of arrays failed\n");
        destroy_particle_system(sys);
        return NULL;
    }

    /* Zero accelerations */
    #pragma omp parallel for
    for (int i = 0; i < sys->N; i++) {
        sys->accelerations[i].x = 0.0;
        sys->accelerations[i].y = 0.0;
        sys->accelerations[i].z = 0.0;
    }

    Vector3 center = { 0.5 * L, 0.5 * L, 0.5 * L };

    /* Positions: stars, then DM, then BHs */
    printf("  Initialising Plummer positions...\n");
    initialize_plummer_positions(sys->positions,
                                 N_STARS, A_STARS, center);
    initialize_plummer_positions(sys->positions + N_STARS,
                                 N_DM,    A_DM,    center);
    initialize_plummer_positions(sys->positions + N_STARS + N_DM,
                                 N_BH,    A_BH,    center);

    /* Velocities */
    printf("  Initialising velocities...\n");
    initialize_velocities(sys->velocities,
                          N_STARS, SIGMA_V_STARS);
    initialize_velocities(sys->velocities + N_STARS,
                          N_DM,    SIGMA_V_DM);
    initialize_velocities(sys->velocities + N_STARS + N_DM,
                          N_BH,    SIGMA_V_BH);

    /* Masses: split M_TOTAL into three fractions */
    double m_star = (F_STARS * M_TOTAL) / (double)N_STARS;
    double m_dm   = (F_DM    * M_TOTAL) / (double)N_DM;
    double m_bh   = (F_BH    * M_TOTAL) / (double)N_BH;

    /* Types: 0 = stars, 1 = dark matter, 2 = black holes
       (matches TYPE_* definitions in bh_collision.h) */

    /* Stars */
    #pragma omp parallel for
    for (int i = 0; i < N_STARS; i++) {
        sys->masses[i] = m_star;
        sys->types[i]  = 0;
    }

    /* Dark matter */
    #pragma omp parallel for
    for (int i = N_STARS; i < N_STARS + N_DM; i++) {
        sys->masses[i] = m_dm;
        sys->types[i]  = 1;
    }

    /* Black holes */
    #pragma omp parallel for
    for (int i = N_STARS + N_DM; i < N_TOTAL; i++) {
        sys->masses[i] = m_bh;
        sys->types[i]  = 2;
    }

    printf("  Removing bulk motion...\n");
    remove_bulk_motion(sys);

    printf("  Centering system in box...\n");
    center_in_box(sys, center);

    return sys;
}

/* ----------------------------------------------------------------------
   Destroyer and helper
   ---------------------------------------------------------------------- */

void destroy_particle_system(ParticleSystem *sys)
{
    if (!sys) return;

    free(sys->positions);
    free(sys->velocities);
    free(sys->accelerations);
    free(sys->masses);
    free(sys->types);
    free(sys);
}

/* Return pointer to the first BH position (contiguous block of length N_BH) */
Vector3* get_blackhole_positions(ParticleSystem *sys)
{
    if (!sys) return NULL;
    return sys->positions + (N_STARS + N_DM);
}

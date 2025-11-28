#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "constants.h"
#include "structs.h"
#include "particles.h"
#include "mesh.h"
#include "bh_collision.h"

/* Small epsilon for comparisons in tests.*/
static const double EPS_MASS   = 1e-8;
static const double EPS_MOMENT = 1e-6;

/* ----------------------------------------------------------------------
 * Utility: compute total mass and momentum of a ParticleSystem
 * ---------------------------------------------------------------------- */
static void total_mass_momentum(const ParticleSystem *sys,
                                double *M,
                                Vector3 *P)
{
    double m_sum = 0.0;
    Vector3 p_sum = (Vector3){0.0, 0.0, 0.0};

    for (int i = 0; i < sys->N; ++i) {
        double m = sys->masses[i];
        m_sum += m;
        p_sum.x += m * sys->velocities[i].x;
        p_sum.y += m * sys->velocities[i].y;
        p_sum.z += m * sys->velocities[i].z;
    }

    *M = m_sum;
    *P = p_sum;
}

/* Simple helper for pass/fail printing. */
static void print_result(const char *name, int ok)
{
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
}

/* ----------------------------------------------------------------------
 * Test 0: Empty system, no crash, nothing happens.
 * ---------------------------------------------------------------------- */
static int test_empty_system(void)
{
    ParticleSystem sys;
    sys.N          = 0;
    sys.positions  = NULL;
    sys.velocities = NULL;
    sys.masses     = NULL;
    sys.types      = NULL;

    int  n_bh       = 0;
    int *bh_indices = NULL;

    bh_collision_step(&sys, bh_indices, &n_bh);

    if (sys.N != 0 || n_bh != 0) {
        print_result("Empty system: no particles, no BHs", 0);
        return 0;
    }

    print_result("Empty system: no particles, no BHs", 1);
    return 1;
}

/* ----------------------------------------------------------------------
 * Test 1: System with stars + DM but no BHs 每 nothing should change.
 * ---------------------------------------------------------------------- */
static int test_no_blackholes(void)
{
    ParticleSystem sys;
    sys.N          = 3;
    sys.positions  = (Vector3*)malloc(sys.N * sizeof(Vector3));
    sys.velocities = (Vector3*)malloc(sys.N * sizeof(Vector3));
    sys.masses     = (double*)  malloc(sys.N * sizeof(double));
    sys.types      = (int*)     malloc(sys.N * sizeof(int));

    if (!sys.positions || !sys.velocities || !sys.masses || !sys.types) {
        fprintf(stderr, "Allocation failed in test_no_blackholes\n");
        return 0;
    }

    /* Star 1 */
    sys.types[0] = TYPE_STAR;
    sys.masses[0] = 1.0;
    sys.positions[0].x = 10.0;
    sys.positions[0].y = 10.0;
    sys.positions[0].z = 10.0;
    sys.velocities[0].x = 0.1;
    sys.velocities[0].y = 0.0;
    sys.velocities[0].z = 0.0;

    /* Star 2 */
    sys.types[1] = TYPE_STAR;
    sys.masses[1] = 2.0;
    sys.positions[1].x = 20.0;
    sys.positions[1].y = 20.0;
    sys.positions[1].z = 20.0;
    sys.velocities[1].x = -0.1;
    sys.velocities[1].y =  0.1;
    sys.velocities[1].z =  0.0;

    /* Dark matter */
    sys.types[2] = TYPE_DM;
    sys.masses[2] = 3.0;
    sys.positions[2].x = 30.0;
    sys.positions[2].y = 30.0;
    sys.positions[2].z = 30.0;
    sys.velocities[2].x = 0.0;
    sys.velocities[2].y = 0.0;
    sys.velocities[2].z = 0.1;

    /* No BHs at all */
    int *bh_indices = NULL;
    int  n_bh       = 0;

    double  M_before, M_after;
    Vector3 P_before, P_after;
    total_mass_momentum(&sys, &M_before, &P_before);

    bh_collision_step(&sys, bh_indices, &n_bh);

    total_mass_momentum(&sys, &M_after, &P_after);

    int ok = 1;
    if (fabs(M_before - M_after) > EPS_MASS) {
        printf("  test1: total mass changed %.6f -> %.6f\n", M_before, M_after);
        ok = 0;
    }
    if (fabs(P_before.x - P_after.x) > EPS_MOMENT ||
        fabs(P_before.y - P_after.y) > EPS_MOMENT ||
        fabs(P_before.z - P_after.z) > EPS_MOMENT) {
        printf("  test1: total momentum changed\n");
        ok = 0;
    }
    if (n_bh != 0) {
        printf("  test1: n_bh changed from 0 to %d\n", n_bh);
        ok = 0;
    }

    print_result("No BHs in system: no changes", ok);

    free(sys.positions);
    free(sys.velocities);
    free(sys.masses);
    free(sys.types);

    return ok;
}

/* ----------------------------------------------------------------------
 * Test 2: One BH + one star inside BH_STAR_COLLISION_RADIUS 每 star eaten.
 * ---------------------------------------------------------------------- */
static int test_star_swallow_inside_radius(void)
{
    ParticleSystem sys;
    sys.N          = 2;
    sys.positions  = (Vector3*)malloc(sys.N * sizeof(Vector3));
    sys.velocities = (Vector3*)malloc(sys.N * sizeof(Vector3));
    sys.masses     = (double*)  malloc(sys.N * sizeof(double));
    sys.types      = (int*)     malloc(sys.N * sizeof(int));

    if (!sys.positions || !sys.velocities || !sys.masses || !sys.types) {
        fprintf(stderr, "Allocation failed in test_star_swallow_inside_radius\n");
        return 0;
    }

    /* BH at center of the box */
    sys.types[0] = TYPE_BH;
    sys.masses[0] = 10.0;
    sys.positions[0].x = 50.0;
    sys.positions[0].y = 50.0;
    sys.positions[0].z = 50.0;
    sys.velocities[0].x = 0.0;
    sys.velocities[0].y = 0.0;
    sys.velocities[0].z = 0.0;

    /* Star within capture radius */
    sys.types[1] = TYPE_STAR;
    sys.masses[1] = 1.0;
    sys.positions[1].x = 50.0 + 0.5 * BH_STAR_COLLISION_RADIUS;
    sys.positions[1].y = 50.0;
    sys.positions[1].z = 50.0;
    sys.velocities[1].x = 1.0;
    sys.velocities[1].y = 0.0;
    sys.velocities[1].z = 0.0;

    int bh_indices[1];
    bh_indices[0] = 0;
    int n_bh      = 1;

    double  M_before, M_after;
    Vector3 P_before, P_after;
    total_mass_momentum(&sys, &M_before, &P_before);

    bh_collision_step(&sys, bh_indices, &n_bh);

    total_mass_momentum(&sys, &M_after, &P_after);

    int ok = 1;

    /* Mass should be conserved and number of particles reduced to 1. */
    if (fabs(M_before - M_after) > EPS_MASS) {
        printf("  test2: mass changed %.6f -> %.6f\n", M_before, M_after);
        ok = 0;
    }
    if (sys.N != 1) {
        printf("  test2: expected sys.N=1 after swallowing, got %d\n", sys.N);
        ok = 0;
    }
    if (n_bh != 1) {
        printf("  test2: expected n_bh=1, got %d\n", n_bh);
        ok = 0;
    }
    if (sys.types[0] != TYPE_BH) {
        printf("  test2: remaining particle is not BH (type=%d)\n", sys.types[0]);
        ok = 0;
    }

    print_result("BH swallows star inside radius", ok);

    free(sys.positions);
    free(sys.velocities);
    free(sys.masses);
    free(sys.types);

    return ok;
}

/* ----------------------------------------------------------------------
 * Test 3: One BH + one star just outside radius 每 no collision.
 * ---------------------------------------------------------------------- */
static int test_star_no_collision_outside_radius(void)
{
    ParticleSystem sys;
    sys.N          = 2;
    sys.positions  = (Vector3*)malloc(sys.N * sizeof(Vector3));
    sys.velocities = (Vector3*)malloc(sys.N * sizeof(Vector3));
    sys.masses     = (double*)  malloc(sys.N * sizeof(double));
    sys.types      = (int*)     malloc(sys.N * sizeof(int));

    if (!sys.positions || !sys.velocities || !sys.masses || !sys.types) {
        fprintf(stderr, "Allocation failed in test_star_no_collision_outside_radius\n");
        return 0;
    }

    /* Star just outside the capture radius */
    sys.types[0]   = TYPE_STAR;
    sys.masses[0]  = 1.0;
    sys.positions[0].x = 50.0 + 1.01 * BH_STAR_COLLISION_RADIUS;
    sys.positions[0].y = 50.0;
    sys.positions[0].z = 50.0;
    sys.velocities[0].x = 1.0;
    sys.velocities[0].y = 0.0;
    sys.velocities[0].z = 0.0;

    /* BH at center */
    sys.types[1]   = TYPE_BH;
    sys.masses[1]  = 10.0;
    sys.positions[1].x = 50.0;
    sys.positions[1].y = 50.0;
    sys.positions[1].z = 50.0;
    sys.velocities[1].x = 0.0;
    sys.velocities[1].y = 0.0;
    sys.velocities[1].z = 0.0;

    int bh_indices[1];
    bh_indices[0] = 1;
    int n_bh      = 1;

    double  M_before, M_after;
    Vector3 P_before, P_after;
    total_mass_momentum(&sys, &M_before, &P_before);

    bh_collision_step(&sys, bh_indices, &n_bh);

    total_mass_momentum(&sys, &M_after, &P_after);

    int ok = 1;

    if (sys.N != 2) {
        printf("  test3: expected sys.N=2 (no swallowing), got %d\n", sys.N);
        ok = 0;
    }
    if (sys.types[0] != TYPE_STAR || sys.types[1] != TYPE_BH) {
        printf("  test3: particle types changed unexpectedly\n");
        ok = 0;
    }
    if (fabs(M_before - M_after) > EPS_MASS) {
        printf("  test3: mass changed %.6f -> %.6f\n", M_before, M_after);
        ok = 0;
    }

    print_result("No collision when star outside radius", ok);

    free(sys.positions);
    free(sys.velocities);
    free(sys.masses);
    free(sys.types);

    return ok;
}

/* ----------------------------------------------------------------------
 * Test 4: Two BHs within BH_BH_COLLISION_RADIUS 每 they should merge.
 * ---------------------------------------------------------------------- */
static int test_bh_bh_merge(void)
{
    ParticleSystem sys;
    sys.N          = 2;
    sys.positions  = (Vector3*)malloc(sys.N * sizeof(Vector3));
    sys.velocities = (Vector3*)malloc(sys.N * sizeof(Vector3));
    sys.masses     = (double*)  malloc(sys.N * sizeof(double));
    sys.types      = (int*)     malloc(sys.N * sizeof(int));

    if (!sys.positions || !sys.velocities || !sys.masses || !sys.types) {
        fprintf(stderr, "Allocation failed in test_bh_bh_merge\n");
        return 0;
    }

    /* BH1 */
    sys.types[0] = TYPE_BH;
    sys.masses[0] = 5.0;
    sys.positions[0].x = 50.0;
    sys.positions[0].y = 50.0;
    sys.positions[0].z = 50.0;
    sys.velocities[0].x = 0.0;
    sys.velocities[0].y = 0.0;
    sys.velocities[0].z = 0.0;

    /* BH2 within BH_BH_COLLISION_RADIUS */
    sys.types[1] = TYPE_BH;
    sys.masses[1] = 7.0;
    sys.positions[1].x = 50.0 + 0.5 * BH_BH_COLLISION_RADIUS;
    sys.positions[1].y = 50.0;
    sys.positions[1].z = 50.0;
    sys.velocities[1].x = 1.0;
    sys.velocities[1].y = 0.0;
    sys.velocities[1].z = 0.0;

    int bh_indices[2] = {0, 1};
    int n_bh          = 2;

    double  M_before, M_after;
    Vector3 P_before, P_after;
    total_mass_momentum(&sys, &M_before, &P_before);

    bh_collision_step(&sys, bh_indices, &n_bh);

    total_mass_momentum(&sys, &M_after, &P_after);

    int ok = 1;

    if (fabs(M_before - M_after) > EPS_MASS) {
        printf("  test4: mass changed %.6f -> %.6f\n", M_before, M_after);
        ok = 0;
    }
    if (sys.N != 1) {
        printf("  test4: expected sys.N=1 (two BHs merge), got %d\n", sys.N);
        ok = 0;
    }
    if (n_bh != 1) {
        printf("  test4: expected n_bh=1, got %d\n", n_bh);
        ok = 0;
    }
    if (sys.types[0] != TYPE_BH) {
        printf("  test4: remaining particle is not BH (type=%d)\n", sys.types[0]);
        ok = 0;
    }

    print_result("BH每BH merge inside radius (no stars)", ok);

    free(sys.positions);
    free(sys.velocities);
    free(sys.masses);
    free(sys.types);

    return ok;
}

/* ----------------------------------------------------------------------
 * Test 5: Integration test:
 *   - Use initialize_particle_system() to generate Plummer ICs
 *   - Build a mesh and assign masses via CIC
 *   - Run bh_collision_step()
 *   - Reassign masses to the mesh
 *   - Check that total particle mass and grid mass are conserved
 * ---------------------------------------------------------------------- */
static int test_plummer_with_mesh_and_bh(void)
{
    printf("\n[Integration test] Using initialize_particle_system() and mesh...\n");

    ParticleSystem *sys = initialize_particle_system();
    if (!sys) {
        fprintf(stderr, "  test5: initialize_particle_system() failed\n");
        return 0;
    }

    /* Build BH index list from types */
    int *bh_indices = (int*)malloc(sys->N * sizeof(int));
    int n_bh = 0;
    for (int i = 0; i < sys->N; ++i) {
        if (sys->types[i] == TYPE_BH) {
            bh_indices[n_bh++] = i;
        }
    }
    printf("  Found %d BH particles (expected ~N_BH=%d)\n", n_bh, N_BH);

    ParticleMesh *pm = create_particle_mesh();
    printf("  Mesh grid: %d^3, cell size = %.3f\n", pm->N, pm->cell_size);
    printf("  (For boundary testing, set NMESH=64 in constants.h for this run)\n");

    /* Mass on grid before collisions */
    assign_mass_cic_padded(pm->rho, pm->N, sys, pm->N, pm->cell_size, 0);

    double grid_mass_before = 0.0;
    for (int i = 0; i < pm->N; ++i)
        for (int j = 0; j < pm->N; ++j)
            for (int k = 0; k < pm->N; ++k)
                grid_mass_before += pm->rho[i][j][k];

    double  M_before;
    Vector3 P_before;
    total_mass_momentum(sys, &M_before, &P_before);

    printf("  Particle mass before collisions     = %.6f\n", M_before);
    printf("  Grid mass from CIC (before)         = %.6f\n", grid_mass_before);

    /* Run BH collision step */
    bh_collision_step(sys, bh_indices, &n_bh);

    /* Mass on grid after collisions */
    assign_mass_cic_padded(pm->rho, pm->N, sys, pm->N, pm->cell_size, 0);

    double grid_mass_after = 0.0;
    for (int i = 0; i < pm->N; ++i)
        for (int j = 0; j < pm->N; ++j)
            for (int k = 0; k < pm->N; ++k)
                grid_mass_after += pm->rho[i][j][k];

    double  M_after;
    Vector3 P_after;
    total_mass_momentum(sys, &M_after, &P_after);

    printf("  Particle mass after collisions      = %.6f\n", M_after);
    printf("  Grid mass from CIC (after)          = %.6f\n", grid_mass_after);

    int ok = 1;

    if (fabs(M_before - M_after) > 1e-6) {
        printf("  test5: particle mass not conserved!\n");
        ok = 0;
    }
    if (fabs(grid_mass_before - grid_mass_after) > 1e-6 * grid_mass_before) {
        printf("  test5: grid mass (from CIC) not conserved!\n");
        ok = 0;
    }

    print_result("Integration: Plummer init + mesh + BH collisions", ok);

    free(bh_indices);
    destroy_particle_mesh(pm);
    destroy_particle_system(sys);

    return ok;
}

int main(void)
{
    int pass0 = test_empty_system();
    int pass1 = test_no_blackholes();
    int pass2 = test_star_swallow_inside_radius();
    int pass3 = test_star_no_collision_outside_radius();
    int pass4 = test_bh_bh_merge();
    int pass5 = test_plummer_with_mesh_and_bh();

    int all_ok = pass0 && pass1 && pass2 && pass3 && pass4 && pass5;

    printf("\nOverall BH collision tests: %s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}

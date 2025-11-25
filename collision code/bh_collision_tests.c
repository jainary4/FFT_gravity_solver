#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "constants.h"
#include "structs.h"
#include "particles.h"
#include "mesh.h"
#include "bh_collision.h"

/* Small epsilon for float comparisons */
static const float EPS = 1e-5f;

/* Utility: compute total mass and momentum of a ParticleSystem */
static void total_mass_momentum(const ParticleSystem *sys,
                                float *M,
                                Vector3 *P)
{
    float m_sum = 0.0f;
    Vector3 p_sum = {0.0f, 0.0f, 0.0f};

    for (int i = 0; i < sys->N; ++i) {
        float m = sys->masses[i];
        m_sum += m;
        p_sum.x += m * sys->velocities[i].x;
        p_sum.y += m * sys->velocities[i].y;
        p_sum.z += m * sys->velocities[i].z;
    }

    *M = m_sum;
    *P = p_sum;
}

/* Print a simple pass/fail header */
static void print_result(const char *name, int ok)
{
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
}

/* --------------- Test 0: completely empty system ---------------- */

static int test_empty_system(void)
{
    ParticleSystem sys;
    sys.N = 0;
    sys.positions  = NULL;
    sys.velocities = NULL;
    sys.masses     = NULL;
    sys.types      = NULL;

    int n_bh = 0;
    int *bh_indices = NULL;

    /* Should simply return, no crash */
    bh_collision_step(&sys, bh_indices, &n_bh);

    int ok = 1;
    if (sys.N != 0) {
        printf("  test0: expected sys.N = 0, got %d\n", sys.N);
        ok = 0;
    }
    if (n_bh != 0) {
        printf("  test0: expected n_bh = 0, got %d\n", n_bh);
        ok = 0;
    }

    print_result("Empty system: no particles, no BHs", ok);
    return ok;
}

/* --------------- Test 1: no black holes at all ------------------ */

static int test_no_blackholes(void)
{
    ParticleSystem sys;
    sys.N = 3;
    sys.positions  = (Vector3*)malloc(sys.N * sizeof(Vector3));
    sys.velocities = (Vector3*)malloc(sys.N * sizeof(Vector3));
    sys.masses     = (float*)  malloc(sys.N * sizeof(float));
    sys.types      = (int*)    malloc(sys.N * sizeof(int));

    if (!sys.positions || !sys.velocities || !sys.masses || !sys.types) {
        fprintf(stderr, "Allocation failed in test 1\n");
        return 0;
    }

    /* p0: star */
    sys.types[0] = TYPE_STAR;
    sys.masses[0] = 1.0f;
    sys.positions[0].x = 10.0f;
    sys.positions[0].y = 10.0f;
    sys.positions[0].z = 10.0f;
    sys.velocities[0].x = 0.1f;
    sys.velocities[0].y = 0.0f;
    sys.velocities[0].z = 0.0f;

    /* p1: star */
    sys.types[1] = TYPE_STAR;
    sys.masses[1] = 2.0f;
    sys.positions[1].x = 20.0f;
    sys.positions[1].y = 20.0f;
    sys.positions[1].z = 20.0f;
    sys.velocities[1].x = -0.1f;
    sys.velocities[1].y =  0.1f;
    sys.velocities[1].z =  0.0f;

    /* p2: dark matter */
    sys.types[2] = TYPE_DM;
    sys.masses[2] = 3.0f;
    sys.positions[2].x = 30.0f;
    sys.positions[2].y = 30.0f;
    sys.positions[2].z = 30.0f;
    sys.velocities[2].x = 0.0f;
    sys.velocities[2].y = 0.0f;
    sys.velocities[2].z = 0.1f;

    int *bh_indices = NULL;
    int n_bh = 0;

    float M_before;
    Vector3 P_before;
    total_mass_momentum(&sys, &M_before, &P_before);

    /* Should do nothing since n_bh = 0 */
    bh_collision_step(&sys, bh_indices, &n_bh);

    float M_after;
    Vector3 P_after;
    total_mass_momentum(&sys, &M_after, &P_after);

    int ok = 1;

    if (sys.N != 3) {
        printf("  test1: expected sys.N = 3, got %d\n", sys.N);
        ok = 0;
    }
    if (fabsf(M_before - M_after) > EPS) {
        printf("  test1: total mass changed %.6f -> %.6f\n", M_before, M_after);
        ok = 0;
    }
    if (fabsf(P_before.x - P_after.x) > EPS ||
        fabsf(P_before.y - P_after.y) > EPS ||
        fabsf(P_before.z - P_after.z) > EPS) {
        printf("  test1: total momentum changed\n");
        ok = 0;
    }

    free(sys.positions);
    free(sys.velocities);
    free(sys.masses);
    free(sys.types);

    print_result("No BHs in system: no changes", ok);
    return ok;
}

/* ---------------- Test 2: BH swallows star inside radius ---------------- */

static int test_star_swallow_inside_radius(void)
{
    ParticleSystem sys;
    sys.N = 3;
    sys.positions  = (Vector3*)malloc(sys.N * sizeof(Vector3));
    sys.velocities = (Vector3*)malloc(sys.N * sizeof(Vector3));
    sys.masses     = (float*)  malloc(sys.N * sizeof(float));
    sys.types      = (int*)    malloc(sys.N * sizeof(int));

    if (!sys.positions || !sys.velocities || !sys.masses || !sys.types) {
        fprintf(stderr, "Allocation failed in test 2\n");
        return 0;
    }

    /* Close star (should be eaten) */
    sys.types[0] = TYPE_STAR;
    sys.masses[0] = 1.0f;
    sys.positions[0].x = 50.0f + 0.25f * BH_STAR_COLLISION_RADIUS;
    sys.positions[0].y = 50.0f;
    sys.positions[0].z = 50.0f;
    sys.velocities[0].x = 1.0f;
    sys.velocities[0].y = 0.0f;
    sys.velocities[0].z = 0.0f;

    /* BH at center */
    sys.types[1] = TYPE_BH;
    sys.masses[1] = 10.0f;
    sys.positions[1].x = 50.0f;
    sys.positions[1].y = 50.0f;
    sys.positions[1].z = 50.0f;
    sys.velocities[1].x = 0.0f;
    sys.velocities[1].y = 0.0f;
    sys.velocities[1].z = 0.0f;

    /* Far star (should survive) */
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

    float M_before;
    Vector3 P_before;
    total_mass_momentum(&sys, &M_before, &P_before);

    bh_collision_step(&sys, bh_indices, &n_bh);

    float M_after;
    Vector3 P_after;
    total_mass_momentum(&sys, &M_after, &P_after);

    int ok = 1;

    if (sys.N != 2) {
        printf("  test2: expected sys.N = 2, got %d\n", sys.N);
        ok = 0;
    }

    /* Find BH and star by type */
    int idx_bh = -1, idx_star = -1;
    for (int i = 0; i < sys.N; ++i) {
        if (sys.types[i] == TYPE_BH) idx_bh = i;
        if (sys.types[i] == TYPE_STAR && idx_star < 0) idx_star = i;
    }

    if (idx_bh < 0 || idx_star < 0) {
        printf("  test2: could not find BH or star after collision\n");
        ok = 0;
    } else {
        if (fabsf(sys.masses[idx_bh] - 11.0f) > 1e-3f) {
            printf("  test2: BH mass expected ~11, got %.3f\n", sys.masses[idx_bh]);
            ok = 0;
        }
        if (fabsf(sys.masses[idx_star] - 1.0f) > 1e-3f) {
            printf("  test2: surviving star mass expected 1, got %.3f\n", sys.masses[idx_star]);
            ok = 0;
        }
    }

    if (fabsf(M_before - M_after) > 1e-4f) {
        printf("  test2: total mass not conserved (%.6f -> %.6f)\n", M_before, M_after);
        ok = 0;
    }

    float dPx = fabsf(P_before.x - P_after.x);
    float dPy = fabsf(P_before.y - P_after.y);
    float dPz = fabsf(P_before.z - P_after.z);
    if (dPx > 1e-4f || dPy > 1e-4f || dPz > 1e-4f) {
        printf("  test2: total momentum not conserved dP=(%.6e,%.6e,%.6e)\n", dPx, dPy, dPz);
        ok = 0;
    }

    free(sys.positions);
    free(sys.velocities);
    free(sys.masses);
    free(sys.types);

    print_result("BH swallows star inside radius", ok);
    return ok;
}

/* ------------- Test 3: Star just outside radius (no collision) ---------- */

static int test_star_no_collision_outside_radius(void)
{
    ParticleSystem sys;
    sys.N = 2;
    sys.positions  = (Vector3*)malloc(sys.N * sizeof(Vector3));
    sys.velocities = (Vector3*)malloc(sys.N * sizeof(Vector3));
    sys.masses     = (float*)  malloc(sys.N * sizeof(float));
    sys.types      = (int*)    malloc(sys.N * sizeof(int));

    if (!sys.positions || !sys.velocities || !sys.masses || !sys.types) {
        fprintf(stderr, "Allocation failed in test 3\n");
        return 0;
    }

    /* Star just outside the capture radius */
    sys.types[0] = TYPE_STAR;
    sys.masses[0] = 1.0f;
    sys.positions[0].x = 50.0f + 1.01f * BH_STAR_COLLISION_RADIUS;
    sys.positions[0].y = 50.0f;
    sys.positions[0].z = 50.0f;
    sys.velocities[0].x = 1.0f;
    sys.velocities[0].y = 0.0f;
    sys.velocities[0].z = 0.0f;

    /* BH at center */
    sys.types[1] = TYPE_BH;
    sys.masses[1] = 10.0f;
    sys.positions[1].x = 50.0f;
    sys.positions[1].y = 50.0f;
    sys.positions[1].z = 50.0f;
    sys.velocities[1].x = 0.0f;
    sys.velocities[1].y = 0.0f;
    sys.velocities[1].z = 0.0f;

    int bh_indices[1] = {1};
    int n_bh = 1;

    float M_before;
    Vector3 P_before;
    total_mass_momentum(&sys, &M_before, &P_before);

    bh_collision_step(&sys, bh_indices, &n_bh);

    float M_after;
    Vector3 P_after;
    total_mass_momentum(&sys, &M_after, &P_after);

    int ok = 1;

    if (sys.N != 2) {
        printf("  test3: expected sys.N = 2, got %d\n", sys.N);
        ok = 0;
    }

    if (fabsf(M_before - M_after) > 1e-4f) {
        printf("  test3: total mass changed %.6f -> %.6f\n", M_before, M_after);
        ok = 0;
    }

    float dPx = fabsf(P_before.x - P_after.x);
    float dPy = fabsf(P_before.y - P_after.y);
    float dPz = fabsf(P_before.z - P_after.z);
    if (dPx > 1e-4f || dPy > 1e-4f || dPz > 1e-4f) {
        printf("  test3: total momentum changed dP=(%.6e,%.6e,%.6e)\n", dPx, dPy, dPz);
        ok = 0;
    }

    free(sys.positions);
    free(sys.velocities);
    free(sys.masses);
    free(sys.types);

    print_result("No collision when star outside radius", ok);
    return ok;
}

/* ---------------- Test 4: BH¨CBH merge inside radius (no stars) ---------- */

static int test_bh_bh_merge(void)
{
    ParticleSystem sys;
    sys.N = 3;
    sys.positions  = (Vector3*)malloc(sys.N * sizeof(Vector3));
    sys.velocities = (Vector3*)malloc(sys.N * sizeof(Vector3));
    sys.masses     = (float*)  malloc(sys.N * sizeof(float));
    sys.types      = (int*)    malloc(sys.N * sizeof(int));

    if (!sys.positions || !sys.velocities || !sys.masses || !sys.types) {
        fprintf(stderr, "Allocation failed in test 4\n");
        return 0;
    }

    /* BH0 */
    sys.types[0] = TYPE_BH;
    sys.masses[0] = 5.0f;
    sys.positions[0].x = 50.0f;
    sys.positions[0].y = 50.0f;
    sys.positions[0].z = 50.0f;
    sys.velocities[0].x = 1.0f;
    sys.velocities[0].y = 0.0f;
    sys.velocities[0].z = 0.0f;

    /* BH1 (within BH_BH_COLLISION_RADIUS) */
    sys.types[1] = TYPE_BH;
    sys.masses[1] = 3.0f;
    sys.positions[1].x = 50.0f + 0.5f * BH_BH_COLLISION_RADIUS;
    sys.positions[1].y = 50.0f;
    sys.positions[1].z = 50.0f;
    sys.velocities[1].x = -1.0f;
    sys.velocities[1].y =  0.0f;
    sys.velocities[1].z =  0.0f;

    /* Far BH2 */
    sys.types[2] = TYPE_BH;
    sys.masses[2] = 2.0f;
    sys.positions[2].x = 50.0f + 10.0f * BH_BH_COLLISION_RADIUS;
    sys.positions[2].y = 50.0f;
    sys.positions[2].z = 50.0f;
    sys.velocities[2].x = 0.0f;
    sys.velocities[2].y = 1.0f;
    sys.velocities[2].z = 0.0f;

    int bh_indices[3] = {0, 1, 2};
    int n_bh = 3;

    float M_before;
    Vector3 P_before;
    total_mass_momentum(&sys, &M_before, &P_before);

    bh_collision_step(&sys, bh_indices, &n_bh);

    float M_after;
    Vector3 P_after;
    total_mass_momentum(&sys, &M_after, &P_after);

    int ok = 1;

    if (sys.N != 2) {
        printf("  test4: expected sys.N = 2, got %d\n", sys.N);
        ok = 0;
    }
    if (n_bh != 2) {
        printf("  test4: expected n_bh = 2, got %d\n", n_bh);
        ok = 0;
    }
    if (fabsf(M_before - M_after) > 1e-4f) {
        printf("  test4: total mass changed %.6f -> %.6f\n", M_before, M_after);
        ok = 0;
    }

    int count_mass8 = 0, count_mass2 = 0;
    for (int i = 0; i < sys.N; ++i) {
        if (sys.types[i] != TYPE_BH) {
            printf("  test4: non-BH particle left after merge (type=%d)\n", sys.types[i]);
            ok = 0;
        }
        if (fabsf(sys.masses[i] - 8.0f) < 1e-3f) count_mass8++;
        if (fabsf(sys.masses[i] - 2.0f) < 1e-3f) count_mass2++;
    }
    if (count_mass8 != 1 || count_mass2 != 1) {
        printf("  test4: expected one BH with mass 8, one with mass 2; got (8x%d, 2x%d)\n",
               count_mass8, count_mass2);
        ok = 0;
    }

    float dPx = fabsf(P_before.x - P_after.x);
    float dPy = fabsf(P_before.y - P_after.y);
    float dPz = fabsf(P_before.z - P_after.z);
    if (dPx > 1e-4f || dPy > 1e-4f || dPz > 1e-4f) {
        printf("  test4: total momentum changed dP=(%.6e,%.6e,%.6e)\n", dPx, dPy, dPz);
        ok = 0;
    }

    free(sys.positions);
    free(sys.velocities);
    free(sys.masses);
    free(sys.types);

    print_result("BH¨CBH merge inside radius (no stars)", ok);
    return ok;
}

/* ------- Test 5: Plummer init + 64^3 mesh + BH collisions (integration) ------- */

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
    assign_mass_cic(pm, sys);
    double grid_mass_before = 0.0;
    for (int i = 0; i < pm->N; ++i)
        for (int j = 0; j < pm->N; ++j)
            for (int k = 0; k < pm->N; ++k)
                grid_mass_before += pm->rho[i][j][k];

    float M_before;
    Vector3 P_before;
    total_mass_momentum(sys, &M_before, &P_before);

    printf("  Particle mass before collisions     = %.6f\n", M_before);
    printf("  Grid mass from CIC (before)         = %.6f\n", grid_mass_before);

    /* One BH collision step */
    bh_collision_step(sys, bh_indices, &n_bh);

    /* Mass on grid after collisions */
    assign_mass_cic(pm, sys);
    double grid_mass_after = 0.0;
    for (int i = 0; i < pm->N; ++i)
        for (int j = 0; j < pm->N; ++j)
            for (int k = 0; k < pm->N; ++k)
                grid_mass_after += pm->rho[i][j][k];

    float M_after;
    Vector3 P_after;
    total_mass_momentum(sys, &M_after, &P_after);

    printf("  Particle mass after collisions      = %.6f\n", M_after);
    printf("  Grid mass from CIC (after)          = %.6f\n", grid_mass_after);

    int ok = 1;

    if (fabsf(M_before - M_after) > 1e-3f) {
        printf("  test5: particle mass not conserved!\n");
        ok = 0;
    }
    if (fabs(grid_mass_before - grid_mass_after) > 1e-2) {
        printf("  test5: grid mass changed (before %.6f, after %.6f)\n",
               grid_mass_before, grid_mass_after);
        ok = 0;
    }

    float dPx = fabsf(P_before.x - P_after.x);
    float dPy = fabsf(P_before.y - P_after.y);
    float dPz = fabsf(P_before.z - P_after.z);
    if (dPx > 1e-2f || dPy > 1e-2f || dPz > 1e-2f) {
        printf("  test5: total momentum changed dP=(%.6e,%.6e,%.6e)\n", dPx, dPy, dPz);
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

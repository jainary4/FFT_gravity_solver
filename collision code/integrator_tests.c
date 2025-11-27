#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include "constants.h"   // has PI, L, etc.
#include "structs.h"
#include "integrator.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* For comparing 2nd vs 4th order on the same problem */
static double last_dE_leapfrog = -1.0;

/* ---------- small helpers for tests ---------- */

static ParticleSystem* make_test_system(int N)
{
    ParticleSystem *sys = (ParticleSystem*)malloc(sizeof(ParticleSystem));
    sys->N = N;
    sys->positions     = (Vector3*)calloc(N, sizeof(Vector3));
    sys->velocities    = (Vector3*)calloc(N, sizeof(Vector3));
    sys->accelerations = (Vector3*)calloc(N, sizeof(Vector3));
    sys->masses        = (double*)calloc(N, sizeof(double));
    sys->types         = (int*)calloc(N, sizeof(int));
    return sys;
}

static void free_test_system(ParticleSystem *sys)
{
    if (!sys) return;
    free(sys->positions);
    free(sys->velocities);
    free(sys->accelerations);
    free(sys->masses);
    free(sys->types);
    free(sys);
}

/* Kinetic energy */
static double kinetic_energy(const ParticleSystem *sys)
{
    const int N = sys->N;
    double K = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:K) schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
        double vx = sys->velocities[i].x;
        double vy = sys->velocities[i].y;
        double vz = sys->velocities[i].z;
        double v2 = vx*vx + vy*vy + vz*vz;
        K += 0.5 * sys->masses[i] * v2;
    }
    return K;
}

/* Potential energy for 3D harmonic oscillator
 * centered at box center (L/2,L/2,L/2):
 *   V = 0.5 m ¦Ø^2 |x - x_c|^2
 */
static double potential_energy_harmonic(const ParticleSystem *sys, double omega)
{
    const int N = sys->N;
    double V = 0.0;
    double om2 = omega * omega;
    const double xc = 0.5 * L;
    const double yc = 0.5 * L;
    const double zc = 0.5 * L;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:V) schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
        double dx = sys->positions[i].x - xc;
        double dy = sys->positions[i].y - yc;
        double dz = sys->positions[i].z - zc;
        double r2 = dx*dx + dy*dy + dz*dz;
        V += 0.5 * sys->masses[i] * om2 * r2;
    }
    return V;
}

/* Potential for Kepler problem with central mass at box center:
 *   V = -GM m / |x - x_c|
 */
static double potential_energy_kepler(const ParticleSystem *sys, double GM)
{
    const int N = sys->N;
    double V = 0.0;
    const double xc = 0.5 * L;
    const double yc = 0.5 * L;
    const double zc = 0.5 * L;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:V) schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
        double dx = sys->positions[i].x - xc;
        double dy = sys->positions[i].y - yc;
        double dz = sys->positions[i].z - zc;
        double r  = sqrt(dx*dx + dy*dy + dz*dz);
        V += -GM * sys->masses[i] / r;
    }
    return V;
}

/* Angular momentum magnitude for 1-particle system about box center */
static double angular_momentum_mag_1p(const ParticleSystem *sys)
{
    if (sys->N <= 0) return 0.0;

    const double xc = 0.5 * L;
    const double yc = 0.5 * L;
    const double zc = 0.5 * L;

    double m  = sys->masses[0];
    double x  = sys->positions[0].x - xc;
    double y  = sys->positions[0].y - yc;
    double z  = sys->positions[0].z - zc;
    double vx = sys->velocities[0].x;
    double vy = sys->velocities[0].y;
    double vz = sys->velocities[0].z;

    double Lx = m * (y * vz - z * vy);
    double Ly = m * (z * vx - x * vz);
    double Lz = m * (x * vy - y * vx);
    return sqrt(Lx*Lx + Ly*Ly + Lz*Lz);
}

/* ---------- force callbacks for tests ---------- */

/* Harmonic oscillator: a = -¦Ø^2 (x - x_c) */
typedef struct {
    double omega;
} HarmonicCtx;

static void force_harmonic(ParticleSystem *sys, void *vctx)
{
    HarmonicCtx *ctx = (HarmonicCtx*)vctx;
    double om2 = ctx->omega * ctx->omega;
    const int N = sys->N;
    const double xc = 0.5 * L;
    const double yc = 0.5 * L;
    const double zc = 0.5 * L;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
        double dx = sys->positions[i].x - xc;
        double dy = sys->positions[i].y - yc;
        double dz = sys->positions[i].z - zc;
        sys->accelerations[i].x = -om2 * dx;
        sys->accelerations[i].y = -om2 * dy;
        sys->accelerations[i].z = -om2 * dz;
    }
}

/* Kepler problem with central mass at box center:
 *   a = -GM (x - x_c) / |x - x_c|^3
 */
typedef struct {
    double GM;
} KeplerCtx;

static void force_kepler(ParticleSystem *sys, void *vctx)
{
    KeplerCtx *ctx = (KeplerCtx*)vctx;
    double GM = ctx->GM;
    const int N = sys->N;
    const double xc = 0.5 * L;
    const double yc = 0.5 * L;
    const double zc = 0.5 * L;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
        double dx = sys->positions[i].x - xc;
        double dy = sys->positions[i].y - yc;
        double dz = sys->positions[i].z - zc;
        double r2 = dx*dx + dy*dy + dz*dz;
        double r3 = r2 * sqrt(r2);

        sys->accelerations[i].x = -GM * dx / r3;
        sys->accelerations[i].y = -GM * dy / r3;
        sys->accelerations[i].z = -GM * dz / r3;
    }
}

/* ---------- TEST 1: Harmonic oscillator energy (leapfrog) ---------- */

static void test_harmonic_leapfrog()
{
    printf("TEST: harmonic oscillator, leapfrog\n");

    HarmonicCtx ctx = { .omega = 1.0 };

    ParticleSystem *sys = make_test_system(1);
    sys->masses[0] = 1.0;

    /* Start 1 kpc away from box center along x, zero velocity. */
    sys->positions[0].x = 0.5 * L + 1.0;
    sys->positions[0].y = 0.5 * L;
    sys->positions[0].z = 0.5 * L;
    sys->velocities[0].x = 0.0;
    sys->velocities[0].y = 0.0;
    sys->velocities[0].z = 0.0;

    /* initial acceleration */
    force_harmonic(sys, &ctx);

    double E0 = kinetic_energy(sys) + potential_energy_harmonic(sys, ctx.omega);

    double dt = 0.01;
    int n_steps = (int)(2.0 * PI / dt) * 100; /* 100 periods */

    double E_max = E0;
    double E_min = E0;

    for (int n = 0; n < n_steps; ++n) {
        leapfrog_step(sys, dt, force_harmonic, &ctx);

        double E = kinetic_energy(sys) + potential_energy_harmonic(sys, ctx.omega);
        if (E > E_max) E_max = E;
        if (E < E_min) E_min = E;
    }

    double dE_rel = (E_max - E_min) / E0;
    printf("  Relative energy oscillation: %.3e\n", dE_rel);

    /* Store for comparison with symplectic4 */
    last_dE_leapfrog = dE_rel;

    if (dE_rel < 1e-3) {
        printf("  PASS (leapfrog energy bounded)\n");
    } else {
        printf("  FAIL (leapfrog energy drift too large)\n");
    }

    free_test_system(sys);
}

/* ---------- TEST 2: Harmonic oscillator energy (symplectic 4) ---------- */

static void test_harmonic_symplectic4()
{
    printf("TEST: harmonic oscillator, symplectic4\n");

    HarmonicCtx ctx = { .omega = 1.0 };

    ParticleSystem *sys = make_test_system(1);
    sys->masses[0] = 1.0;
    sys->positions[0].x = 0.5 * L + 1.0;
    sys->positions[0].y = 0.5 * L;
    sys->positions[0].z = 0.5 * L;
    sys->velocities[0].x = 0.0;
    sys->velocities[0].y = 0.0;
    sys->velocities[0].z = 0.0;

    /* initial acceleration */
    force_harmonic(sys, &ctx);

    double E0 = kinetic_energy(sys) + potential_energy_harmonic(sys, ctx.omega);

    /* Use the SAME dt as leapfrog test so we can compare fairly */
    double dt = 0.01;
    int n_steps = (int)(2.0 * PI / dt) * 100; /* 100 periods */

    printf("  Debug: dt = %.3e, n_steps = %d, E0 = %.6e\n", dt, n_steps, E0);

    double E_max = E0;
    double E_min = E0;

    for (int n = 0; n < n_steps; ++n) {
        symplectic4_step(sys, dt, force_harmonic, &ctx);

        double E = kinetic_energy(sys) + potential_energy_harmonic(sys, ctx.omega);
        if (E > E_max) E_max = E;
        if (E < E_min) E_min = E;
    }

    double dE_rel = (E_max - E_min) / E0;
    printf("  E_min = %.6e, E_max = %.6e\n", E_min, E_max);
    printf("  Relative energy oscillation (sym4): %.3e\n", dE_rel);

    /* Compare with leapfrog if available */
    if (last_dE_leapfrog > 0.0) {
        printf("  Relative energy oscillation (leapfrog): %.3e\n",
               last_dE_leapfrog);
        printf("  Ratio sym4 / leapfrog: %.3f\n", dE_rel / last_dE_leapfrog);
    } else {
        printf("  (No stored leapfrog error to compare against)\n");
    }

    /* Criteria:
       1) absolute error not crazy (here < 1e-3, same scale as leapfrog test)
       2) if leapfrog error is known, symplectic4 should not be worse
          than leapfrog by more than ~5%.
    */
    int ok = 1;
    if (dE_rel > 1e-3) {
        ok = 0;
    }
    if (last_dE_leapfrog > 0.0 && dE_rel > last_dE_leapfrog * 1.05) {
        ok = 0;
    }

    if (ok) {
        printf("  PASS (4th-order energy conservation as good or better than leapfrog)\n");
    } else {
        printf("  FAIL (4th-order energy conservation not better than leapfrog)\n");
    }

    free_test_system(sys);
}

/* ---------- TEST 3: Kepler orbit (energy + angular momentum) ---------- */

static void test_kepler_orbit()
{
    printf("TEST: Kepler orbit (central mass), symplectic4\n");

    KeplerCtx ctx = { .GM = 1.0 };

    const double xc = 0.5 * L;
    const double yc = 0.5 * L;
    const double zc = 0.5 * L;

    /* Circular orbit of radius 1 kpc around box center in x¨Cy plane. */
    double r0 = 1.0;
    double v0 = 1.0;  /* for GM=1, circular orbit at r=1 has v=1 */

    ParticleSystem *sys2 = make_test_system(1);
    sys2->masses[0] = 1.0;
    sys2->positions[0].x = xc + r0;
    sys2->positions[0].y = yc;
    sys2->positions[0].z = zc;
    sys2->velocities[0].x = 0.0;
    sys2->velocities[0].y = v0;
    sys2->velocities[0].z = 0.0;

    force_kepler(sys2, &ctx);

    double E0    = kinetic_energy(sys2) + potential_energy_kepler(sys2, ctx.GM);
    double Lmag0 = angular_momentum_mag_1p(sys2);

    double dt = 0.01;
    double T  = 2.0 * PI;   /* orbital period */
    int n_steps = (int)(100.0 * T / dt);  /* 100 orbits */

    double E_max = E0,    E_min = E0;
    double L_max = Lmag0, L_min = Lmag0;

    for (int n = 0; n < n_steps; ++n) {
        symplectic4_step(sys2, dt, force_kepler, &ctx);

        double E    = kinetic_energy(sys2) + potential_energy_kepler(sys2, ctx.GM);
        double Lmag = angular_momentum_mag_1p(sys2);

        if (E    > E_max) E_max = E;
        if (E    < E_min) E_min = E;
        if (Lmag > L_max) L_max = Lmag;
        if (Lmag < L_min) L_min = Lmag;
    }

    double dE_rel = (E_max - E_min) / fabs(E0);
    double dL_rel = (L_max - L_min) / fabs(Lmag0);

    printf("  Relative E oscillation: %.3e\n", dE_rel);
    printf("  Relative L oscillation: %.3e\n", dL_rel);

    if (dE_rel < 1e-4 && dL_rel < 1e-4) {
        printf("  PASS (Kepler invariants well conserved)\n");
    } else {
        printf("  FAIL (Kepler invariants drift too large)\n");
    }

    free_test_system(sys2);
}

/* ---------- TEST 4: Time reversibility (symplectic check) ---------- */

static void test_time_reversibility()
{
    printf("TEST: time-reversibility of symplectic4\n");

    HarmonicCtx ctx = { .omega = 1.0 };

    ParticleSystem *sys = make_test_system(1);
    sys->masses[0] = 1.0;
    sys->positions[0].x = 0.5 * L + 0.7;
    sys->positions[0].y = 0.5 * L;
    sys->positions[0].z = 0.5 * L;
    sys->velocities[0].x = 0.3;
    sys->velocities[0].y = 0.0;
    sys->velocities[0].z = 0.0;

    force_harmonic(sys, &ctx);

    Vector3 x0 = sys->positions[0];
    Vector3 v0 = sys->velocities[0];

    double dt = 0.05;
    int n_steps = 1000;

    for (int n = 0; n < n_steps; ++n) {
        symplectic4_step(sys, dt, force_harmonic, &ctx);
    }
    for (int n = 0; n < n_steps; ++n) {
        symplectic4_step(sys, -dt, force_harmonic, &ctx);
    }

    double dx = sys->positions[0].x - x0.x;
    double dv = sys->velocities[0].x - v0.x;

    printf("  |dx| = %.3e, |dv| = %.3e\n", fabs(dx), fabs(dv));

    if (fabs(dx) < 1e-8 && fabs(dv) < 1e-8) {
        printf("  PASS (time reversibility good)\n");
    } else {
        printf("  FAIL (time reversibility degraded)\n");
    }

    free_test_system(sys);
}

/* ---------- TEST 5: Edge cases to try to break things ---------- */

static void test_edge_cases()
{
    printf("TEST: edge cases\n");

    /* N = 0 system */
    ParticleSystem *empty = make_test_system(0);
    HarmonicCtx ctx = { .omega = 1.0 };
    printf("  Calling integrators on empty system...\n");
    leapfrog_step(empty, 0.1, force_harmonic, &ctx);
    symplectic4_step(empty, 0.1, force_harmonic, &ctx);
    printf("  (no crash == PASS)\n");
    free_test_system(empty);

    /* dt = 0 */
    ParticleSystem *sys = make_test_system(1);
    sys->masses[0] = 1.0;
    sys->positions[0].x = 0.5 * L + 0.5;
    sys->positions[0].y = 0.5 * L;
    sys->positions[0].z = 0.5 * L;
    sys->velocities[0].x = 0.5;
    sys->velocities[0].y = 0.0;
    sys->velocities[0].z = 0.0;
    force_harmonic(sys, &ctx);

    Vector3 x_before = sys->positions[0];
    Vector3 v_before = sys->velocities[0];

    leapfrog_step(sys, 0.0, force_harmonic, &ctx);
    symplectic4_step(sys, 0.0, force_harmonic, &ctx);

    Vector3 x_after = sys->positions[0];
    Vector3 v_after = sys->velocities[0];

    printf("  dt = 0 => positions/vel unchanged? dx=%.3e, dv=%.3e\n",
           fabs(x_after.x - x_before.x),
           fabs(v_after.x - v_before.x));

    /* Particle near boundary to test periodic wrap (we *want* wrapping here) */
    sys->positions[0].x = L - 1e-6;
    sys->velocities[0].x = 1.0;
    force_harmonic(sys, &ctx);
    leapfrog_step(sys, 1.0, force_harmonic, &ctx);
    printf("  Position after crossing boundary: x=%.6f (should be in [0,L))\n",
           sys->positions[0].x);

    free_test_system(sys);
}

/* ---------- main() for test executable ---------- */

int main(void)
{
#ifdef _OPENMP
    printf("Running with OpenMP, max threads = %d\n", omp_get_max_threads());
#endif

    test_harmonic_leapfrog();
    printf("\n");
    test_harmonic_symplectic4();
    printf("\n");
    test_kepler_orbit();
    printf("\n");
    test_time_reversibility();
    printf("\n");
    test_edge_cases();
    return 0;
}

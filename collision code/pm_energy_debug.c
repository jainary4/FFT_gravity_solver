#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "constants.h"
#include "structs.h"
#include "particles.h"
#include "mesh.h"
#include "fft_solver.h"
#include "poisson.h"
#include "force.h"

/* ------------------------------------------------------------------
 * Simple helpers: total mass, net force, etc.
 * ------------------------------------------------------------------ */

static double total_mass(const ParticleSystem *sys)
{
    double M = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:M) schedule(static)
#endif
    for (int i = 0; i < sys->N; ++i) {
        M += sys->masses[i];
    }
    return M;
}

static void net_force(const ParticleSystem *sys, double F[3])
{
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:Fx,Fy,Fz) schedule(static)
#endif
    for (int i = 0; i < sys->N; ++i) {
        Fx += sys->masses[i] * sys->accelerations[i].x;
        Fy += sys->masses[i] * sys->accelerations[i].y;
        Fz += sys->masses[i] * sys->accelerations[i].z;
    }
    F[0] = Fx; F[1] = Fy; F[2] = Fz;
}

/* Minimum-image periodic distance in [0,L)^3. */
static double distance_periodic(const Vector3 *a, const Vector3 *b)
{
    double dx = a->x - b->x;
    double dy = a->y - b->y;
    double dz = a->z - b->z;

    if (dx >  0.5 * L) dx -= L;
    if (dx < -0.5 * L) dx += L;
    if (dy >  0.5 * L) dy -= L;
    if (dy < -0.5 * L) dy += L;
    if (dz >  0.5 * L) dz -= L;
    if (dz < -0.5 * L) dz += L;

    return sqrt(dx*dx + dy*dy + dz*dz);
}

/* Direct-sum potential energy for a small subset of particles.
 * Uses G from constants.h and minimum-image periodic distance.
 * This is O(N_sub^2), so keep N_sub modest (e.g. 500-2000).
 */
static double potential_energy_direct_subset(const ParticleSystem *sys, int N_sub)
{
    if (N_sub > sys->N) N_sub = sys->N;
    double W = 0.0;

    for (int i = 0; i < N_sub; ++i) {
        for (int j = i + 1; j < N_sub; ++j) {
            double r = distance_periodic(&sys->positions[i], &sys->positions[j]);
            if (r > 0.0) {
                W -= G * sys->masses[i] * sys->masses[j] / r;
            }
        }
    }
    return W;
}

/* Mesh-based potential energy:
 *   W = 0.5 * sum_{i,j,k} rho(x_ijk) * phi(x_ijk) * dV
 * over the full padded grid.
 * This assumes rho and phi are the same ones used in Poisson/forces.
 */
static double potential_energy_from_mesh(double ***rho_pad,
                                         double ***phi_pad,
                                         int Np,
                                         double h)
{
    double W  = 0.0;
    double dV = h * h * h;

#ifdef _OPENMP
#pragma omp parallel for collapse(3) reduction(+:W) schedule(static)
#endif
    for (int i = 0; i < Np; ++i) {
        for (int j = 0; j < Np; ++j) {
            for (int k = 0; k < Np; ++k) {
                W += 0.5 * rho_pad[i][j][k] * phi_pad[i][j][k] * dV;
            }
        }
    }
    return W;
}

/* Mass represented on mesh:
 *   M_mesh = sum rho(x_ijk) * dV
 * Helps check normalization of assign_mass_cic_padded.
 */
static double mass_from_mesh(double ***rho_pad, int Np, double h)
{
    double M  = 0.0;
    double dV = h * h * h;
#ifdef _OPENMP
#pragma omp parallel for collapse(3) reduction(+:M) schedule(static)
#endif
    for (int i = 0; i < Np; ++i) {
        for (int j = 0; j < Np; ++j) {
            for (int k = 0; k < Np; ++k) {
                M += rho_pad[i][j][k] * dV;
            }
        }
    }
    return M;
}

/* ------------------------------------------------------------------
 * Single PM gravity solve + force gather, WITHOUT moving particles.
 * This uses your existing pipeline:
 *  - zero rho_pad
 *  - assign_mass_cic_padded
 *  - create_laplacian_equation
 *  - solve_poisson_fftw
 *  - compute_forces_from_potential
 *  - gather_forces_to_particles
 * It returns W_mesh for this snapshot, and overwrites accelerations in sys.
 * ------------------------------------------------------------------ */
static double pm_solve_and_forces(ParticleSystem *sys,
                                  ParticleMesh    *pm,
                                  double         ***rho_pad,
                                  double         ***force_x,
                                  double         ***force_y,
                                  double         ***force_z)
{
    int    Np    = NMESH_PADDED;
    int    offset = 0;             /* physical mesh at corner of padded grid */
    double h     = pm->cell_size;

    /* 1) Zero padded density grid */
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for (int i = 0; i < Np; ++i) {
        for (int j = 0; j < Np; ++j) {
            for (int k = 0; k < Np; ++k) {
                rho_pad[i][j][k] = 0.0;
            }
        }
    }

    /* 2) Assign mass to mesh */
    assign_mass_cic_padded(rho_pad, Np, sys, pm->N, h, offset);

    /* 3) Build Laplacian RHS and solve Poisson */
    double ***laplacian_phi_pad = create_laplacian_equation(Np, rho_pad);
    if (!laplacian_phi_pad) {
        fprintf(stderr, "pm_solve_and_forces: create_laplacian_equation returned NULL\n");
        return NAN;
    }

    double ***phi_pad = solve_poisson_fftw(laplacian_phi_pad, Np, h);
    if (!phi_pad) {
        fprintf(stderr, "pm_solve_and_forces: solve_poisson_fftw returned NULL\n");
        free_3d_array(laplacian_phi_pad, Np);
        return NAN;
    }

    /* 4) Compute forces on mesh */
    compute_forces_from_potential(phi_pad, Np, h, force_x, force_y, force_z);

    /* 5) Gather forces to particles => sys->accelerations */
    gather_forces_to_particles(force_x, force_y, force_z,
                               Np, sys, pm->N, h, offset);

    /* 6) Potential energy from mesh */
    double W_mesh = potential_energy_from_mesh(rho_pad, phi_pad, Np, h);

    /* 7) Clean up temporaries */
    free_3d_array(laplacian_phi_pad, Np);
    free_3d_array(phi_pad, Np);

    return W_mesh;
}

/* ------------------------------------------------------------------
 * main: debug program to test PM gravity energy consistency
 * ------------------------------------------------------------------ */
int main(void)
{
#ifdef _OPENMP
    printf("Running pm_energy_debug with OpenMP, max threads = %d\n", omp_get_max_threads());
#endif

    /* 1) Build particle system with your existing IC generator */
    printf("Initializing particle system...\n");
    ParticleSystem *sys = initialize_particle_system();
    if (!sys) {
        fprintf(stderr, "ERROR: initialize_particle_system() returned NULL\n");
        return 1;
    }
    printf("  N = %d particles\n", sys->N);

    /* 2) Create mesh */
    printf("Creating particle mesh...\n");
    ParticleMesh *pm = create_particle_mesh();
    if (!pm) {
        fprintf(stderr, "ERROR: create_particle_mesh() returned NULL\n");
        destroy_particle_system(sys);
        return 1;
    }
    printf("  Mesh: %d^3, cell size h = %.6f (code units)\n", pm->N, pm->cell_size);

    /* 3) Allocate padded grids */
    int Np = NMESH_PADDED;
    if (Np <= 0) {
        fprintf(stderr, "ERROR: NMESH_PADDED must be > 0\n");
        destroy_particle_mesh(pm);
        destroy_particle_system(sys);
        return 1;
    }

    double ***rho_pad = allocate_3d_array(Np);
    double ***force_x = allocate_3d_array(Np);
    double ***force_y = allocate_3d_array(Np);
    double ***force_z = allocate_3d_array(Np);

    if (!rho_pad || !force_x || !force_y || !force_z) {
        fprintf(stderr, "ERROR: allocate_3d_array failed for padded grids\n");
        if (rho_pad) free_3d_array(rho_pad, Np);
        if (force_x) free_3d_array(force_x, Np);
        if (force_y) free_3d_array(force_y, Np);
        if (force_z) free_3d_array(force_z, Np);
        destroy_particle_mesh(pm);
        destroy_particle_system(sys);
        return 1;
    }

    /* 4) One PM solve to get baseline W and mesh mass */
    printf("\n--- Static PM check (no particle motion) ---\n");
    double W0 = pm_solve_and_forces(sys, pm, rho_pad, force_x, force_y, force_z);
    if (!isfinite(W0)) {
        fprintf(stderr, "ERROR: W0 is not finite\n");
    }

    double M_particles = total_mass(sys);
    double M_mesh      = mass_from_mesh(rho_pad, Np, pm->cell_size);

    double F_net[3];
    net_force(sys, F_net);
    double F_net_mag = sqrt(F_net[0]*F_net[0] + F_net[1]*F_net[1] + F_net[2]*F_net[2]);

    printf("Initial total mass (particles): M      = %.6e\n", M_particles);
    printf("Mass from mesh (rho*h^3 sum):  M_mesh = %.6e\n", M_mesh);
    printf("Initial potential energy from mesh: W0 = %.6e\n", W0);
    printf("Initial net force: |F_net| = %.6e\n", F_net_mag);

    /* 5) Repeat PM solve several times without moving particles */
    int n_repeat = 5;
    for (int rep = 1; rep <= n_repeat; ++rep) {
        double W = pm_solve_and_forces(sys, pm, rho_pad, force_x, force_y, force_z);
        net_force(sys, F_net);
        F_net_mag = sqrt(F_net[0]*F_net[0] + F_net[1]*F_net[1] + F_net[2]*F_net[2]);

        printf("Repeat %d: W = %.6e, |F_net| = %.6e\n", rep, W, F_net_mag);
    }

    /* 6) Direct-sum check on a subset of particles */
    int N_sub = (sys->N > 1000) ? 1000 : sys->N;
    printf("\n--- Direct-sum potential energy on first %d particles ---\n", N_sub);
    double W_direct = potential_energy_direct_subset(sys, N_sub);
    printf("Direct-sum W_direct (subset) = %.6e\n", W_direct);
    printf("Mesh W0 (full system)        = %.6e\n", W0);
    printf("Ratio W_direct/W0            = %.6e (NOTE: subset vs full)\n",
           (W0 != 0.0) ? (W_direct / W0) : 0.0);

    /* 7) Cleanup */
    free_3d_array(rho_pad, Np);
    free_3d_array(force_x, Np);
    free_3d_array(force_y, Np);
    free_3d_array(force_z, Np);
    destroy_particle_mesh(pm);
    destroy_particle_system(sys);

    printf("\npm_energy_debug completed.\n");
    return 0;
}
